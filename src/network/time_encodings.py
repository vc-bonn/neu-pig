import torch
import torch.nn.functional as F
import math


class RandomFourierEncoding(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.B = torch.randn(dim // 2, device=args.device)

    def forward(self, t):
        x = t.unsqueeze(-1) * self.B
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class FourierFeatures1(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.time_dim = dim
        half = self.time_dim // 2
        self.freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=args.device)
            * (torch.log(torch.tensor(1e4)) / (half - 1))
        )

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class FourierFeatures2(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.freqs = (2 ** (torch.arange(dim // 2, device=args.device)))[None, :]

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        x = t * self.freqs * torch.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class GaussianFourier(torch.nn.Module):
    def __init__(self, args: dict, dim=64, scale=10.0):
        super().__init__()
        self.B = torch.randn(dim // 2, device=args.device) * scale

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        x = (t * 2 * torch.pi) @ self.B[None]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class Poly(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.degree = dim

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        return torch.cat([t**i for i in range(1, self.degree + 1)], dim=-1)


class MLP(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, dim),
        ).to(args.device)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.time_embed(t)


class SinusMLP(torch.nn.Module):
    def __init__(self, args: dict, dim=64):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim // 2, dim),
        ).to(args.device)
        half = dim // 2
        self.freqs = torch.exp(
            -torch.arange(half, device=args.device) * (math.log(10000.0) / (half - 1))
        )

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.time_embed(t)
