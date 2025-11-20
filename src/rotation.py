import torch
from pytorch3d.transforms import (
    so3_exp_map,
    quaternion_to_matrix,
    axis_angle_to_quaternion,
)


class Rotation(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.method = self.args.method_args["rotation_method"]
        if self.method not in ["quaternions", "exp", "cayley", "quaternion_axis"]:
            raise Exception("Rotation Method not implemented")

    @property
    def rotation_dim(self) -> int:
        if self.method == "quaternions":
            return 4
        elif self.method in ["exp", "cayley", "quaternion_axis"]:
            return 3

    def reshape(self, t: torch.Tensor):
        shape = t.shape
        t = t.view(-1, self.rotation_dim)
        return t, shape

    def quaternion_rotation(self, t: torch.Tensor) -> torch.Tensor:
        t[..., 0] = t[..., 0] + 1
        t = t / (torch.norm(t, dim=-1, keepdim=True) + 1e-8)

        return quaternion_to_matrix(t)

    def exp_rotation(self, t: torch.Tensor) -> torch.Tensor:
        return so3_exp_map(t.flatten(end_dim=-2))

    def cayley_rotation(self, t: torch.Tensor) -> torch.Tensor:
        batch_size = t.shape[0]
        A = torch.zeros((batch_size, 3, 3), device=t.device)
        A[:, 0, 1] = -t[:, 2]
        A[:, 0, 2] = t[:, 1]
        A[:, 1, 0] = t[:, 2]
        A[:, 1, 2] = -t[:, 0]
        A[:, 2, 0] = -t[:, 1]
        A[:, 2, 1] = t[:, 0]

        I = torch.eye(3, device=t.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.matmul(I + A, torch.linalg.inv(I - A))

    def quaternion_axis_rotation(self, t: torch.Tensor) -> torch.Tensor:
        q = axis_angle_to_quaternion(t)
        return self.quaternion_rotation(q)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t, shape = self.reshape(t)
        if self.method == "quaternions":
            R = self.quaternion_rotation(t)
        elif self.method == "exp":
            R = self.exp_rotation(t)
        elif self.method == "cayley":
            R = self.cayley_rotation(t)
        elif self.method == "quaternion_axis":
            R = self.quaternion_axis_rotation(t)
        return R.view(*shape[:-1], 3, 3)
