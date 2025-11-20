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

    def forward(self, x, t_scalar):
        # points: (B, P, 3)
        # t_scalar: (B,) or float
        if not isinstance(t_scalar, torch.Tensor):
            t_scalar = torch.tensor([t_scalar] * x.shape[0], device=x.device)
        if t_scalar.dim() == 0:
            t_scalar = t_scalar.unsqueeze(0).repeat(x.shape[0])
        time_emb = self.time_encoding(t_scalar)
        # broadcast to points
        B, P, _ = x.shape
        tf = time_emb[:, None, :].expand(B, P, -1)  # (B, P, hidden)
        x = torch.cat([x, tf], dim=-1)  # (B, P, 3+hidden)
        return self.network(x)
