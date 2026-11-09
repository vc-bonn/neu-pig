import torch
from src.network.time_encodings import *
from src.network.network_base import MLP, Encoder_Decoder


class Network(torch.nn.Module):
    def __init__(
        self,
        args: dict,
        point_dim=3,
        time_dim=64,
        hidden=512,
        layers=3,
        outdim=7,
        time_encoding="fourier",
        network="mlp",
    ):
        """A wrapper for a small MLP network that takes points and time as input"""
        super().__init__()
        # self.time_dim = time_dim
        # shared MLP for each point: input = (point + time_feat)
        in_dim = point_dim + time_dim

        if network == "mlp":
            self.network = MLP(
                input_dim=in_dim, hidden=hidden, layers=layers, outdim=outdim
            )
        elif network == "encoder_decoder":
            self.network = Encoder_Decoder(
                encoder_layers=layers,
                decoder_layers=layers,
                input_dim=in_dim,
                outdim=outdim,
            )

        assert time_dim % 2 == 0
        if time_encoding == "random_fourier":
            self.time_encoding = RandomFourierEncoding(args, time_dim)
        elif time_encoding == "fourier_features1":
            self.time_encoding = FourierFeatures1(args, time_dim)
        elif time_encoding == "fourier_features2":
            self.time_encoding = FourierFeatures2(args, time_dim)
        elif time_encoding == "poly":
            self.time_encoding = Poly(args, time_dim)
        elif time_encoding == "mlp":
            self.time_encoding = MLP(args, time_dim)
        elif time_encoding == "sinus_mlp":
            self.time_encoding = SinusMLP(args, time_dim)
        elif time_encoding == "gaussian_fourier":
            self.time_encoding = GaussianFourier(args, time_dim)

    def enable_compile(self, performance):
        self.compile_bucket = int(performance.get("compile_vertex_bucket", 128))
        if self.compile_bucket < 1:
            raise ValueError("compile_vertex_bucket must be positive")
        # Keep one registered set of parameters/state-dict entries. The compiled
        # wrapper refers to the same module and is reused by Dynamo's cache.
        mode = performance.get("compile_mlp_mode", "max-autotune")
        if mode not in ("default", "max-autotune"):
            raise ValueError("compile_mlp_mode must be default or max-autotune")
        # Keep intermediate BF16/FP16 casts explicit. Compiled GEMM choices
        # can still change rounding near activation boundaries.
        options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
        if mode == "max-autotune":
            options.update(max_autotune=True, coordinate_descent_tuning=True)
        object.__setattr__(
            self,
            "_compiled_network",
            torch.compile(
                self.network,
                fullgraph=True,
                dynamic=False,
                options=options,
            ),
        )

    def prepare_compile(self, frames, vertices, device, dtype):
        if not hasattr(self, "_compiled_network"):
            return
        count = (
            (vertices + self.compile_bucket - 1)
            // self.compile_bucket
            * self.compile_bucket
        )
        features = self.network.mlp[0].in_features
        example = torch.zeros(
            frames, count, features, device=device, requires_grad=True
        )
        # Compile forward and backward before timing/capture without an update.
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            self._compiled_network(example).sum().backward()
        self.zero_grad(set_to_none=True)

    def forward(self, x, t_scalar, time_embedding=None):
        # points: (B, P, 3)
        # t_scalar: (B,) or float
        if not isinstance(t_scalar, torch.Tensor):
            t_scalar = torch.tensor([t_scalar] * x.shape[0], device=x.device)
        if t_scalar.dim() == 0:
            t_scalar = t_scalar.unsqueeze(0).repeat(x.shape[0])
        time_emb = (
            self.time_encoding(t_scalar) if time_embedding is None else time_embedding
        )
        # broadcast to points
        B, P, _ = x.shape
        tf = time_emb[:, None, :].expand(B, P, -1)  # (B, P, hidden)
        x = torch.cat([x, tf], dim=-1)  # (B, P, 3+hidden)
        if hasattr(self, "_compiled_network"):
            padding = (-P) % self.compile_bucket
            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            return self._compiled_network(x)[:, :P]
        return self.network(x)
