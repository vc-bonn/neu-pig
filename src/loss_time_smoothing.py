import torch
import math


class CatchupScaling(torch.nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args

    @property
    def method(self) -> str:
        return self.args.method_args["optimization"].get("catchup_scaling", "pdg")

    @property
    def epochs(self) -> int:
        return self.args.method_args["optimization"]["epochs"]

    def forward(self, chamfer_distance: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the catchup scaling loss.
        Args:
            chamfer_distance (torch.Tensor): Chamfer distances of shape (T,)
        Returns:
            torch.Tensor: Catchup scaling loss
        """
        if self.method == "pdg":
            return chamfer_distance ** (1 - (epoch / self.epochs) ** 0.5)
        elif self.method == "none":
            return torch.ones_like(chamfer_distance)
        elif self.method == "linear":
            return chamfer_distance ** (1 - (epoch / self.epochs))
        elif self.method == "exponential":
            return chamfer_distance ** math.exp(-5 * (epoch / self.epochs))
        elif self.method == "lerp":
            return (1 - (epoch / self.epochs)) * chamfer_distance + (
                epoch / self.epochs
            )
        raise NotImplementedError(
            f"Catchup scaling method {self.method} not implemented."
        )


class LossTimeSmoothing(torch.nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.catchup = CatchupScaling(args)

    @property
    def method(self) -> str:
        return self.args.method_args["optimization"].get("time_smoothing", "PDG")

    @property
    def K(self) -> int:
        return self.args.method_args["keyframe_index"]

    @property
    def epochs(self) -> int:
        return self.args.method_args["optimization"]["epochs"]

    @property
    def device(self) -> torch.device:
        return self.args.device

    def _pdg_smoothing(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the PDG time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: PDG time smoothing loss
        """

        K = self.K
        L = chamfer_losses[K].detach()

        N = chamfer_losses.numel()
        left = chamfer_losses[:K].detach()
        right = (
            chamfer_losses[K + 1 :].detach()
            if K + 1 < N
            else torch.tensor([], device=self.device)
        )
        left = torch.flip(left, dims=(0,))
        kernel = torch.ones(N, device=self.device)
        left_ = 1 / (1 + (left - L).clamp(min=0))
        right_ = 1 / (1 + (right - L).clamp(min=0))
        if left.numel() > 0:
            kernel[:K] = self.catchup(left_, epoch=epoch)
            kernel[:K] = torch.flip(kernel[:K].cumprod(dim=0), dims=(0,))

        if right.numel() > 0:
            kernel[K + 1 :] = self.catchup(right_, epoch=epoch)
            kernel[K + 1 :] = kernel[K + 1 :].cumprod(dim=0)

        losses_smoothed = kernel.detach() * chamfer_losses
        return losses_smoothed.mean()

    def _exp_average(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the PDG time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: PDG time smoothing loss
        """

        K = self.K
        L = chamfer_losses[K].detach()

        N = chamfer_losses.numel()
        left = chamfer_losses[:K].detach()
        right = (
            chamfer_losses[K + 1 :].detach()
            if K + 1 < N
            else torch.tensor([], device=self.device)
        )
        left = torch.flip(left, dims=(0,))
        kernel = torch.ones(N, device=self.device)
        left_ = (left - L).clamp(min=0)
        right_ = (right - L).clamp(min=0)
        if left.numel() > 0:
            kernel[:K] = torch.flip(left_.cumsum(dim=0), dims=(0,)) / left_.numel()

        if right.numel() > 0:
            kernel[K + 1 :] = right_.cumsum(dim=0) / right_.numel()
        kernel = torch.exp(-kernel)
        kernel = self.catchup(kernel, epoch=epoch)

        losses_smoothed = kernel * chamfer_losses
        return losses_smoothed.mean()

    def _direct_average(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the direct average time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: Direct average time smoothing loss
        """

        N = chamfer_losses.numel()
        L = chamfer_losses.mean().detach()
        losses_smoothed = 1 / ((chamfer_losses / L) + 1)
        losses_smoothed = self.catchup(losses_smoothed, epoch=epoch)
        losses_smoothed = losses_smoothed.detach() * chamfer_losses

        return losses_smoothed.mean()

    def _delta_based(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the delta-based time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: Delta-based time smoothing loss
        """
        losses_smoothed = self.catchup(
            (torch.ones_like(chamfer_losses) * 0.5), epoch=epoch
        )
        losses_smoothed = losses_smoothed.detach() * chamfer_losses
        return losses_smoothed.mean()

    def _constant(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the constant time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: Constant time smoothing loss
        """
        losses_smoothed = chamfer_losses
        return losses_smoothed.mean()

    def _direct(self, chamfer_losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the direct time smoothing loss.

        Args:
            chamfer_losses (torch.Tensor): Chamfer losses of shape (T,)

        Returns:
            torch.Tensor: Direct time smoothing loss
        """
        K = self.K
        L = chamfer_losses[K].detach()

        losses_smoothed = 1 / (1 + (chamfer_losses - L).clamp(min=0))
        losses_smoothed = self.catchup(losses_smoothed, epoch=epoch)
        losses_smoothed = losses_smoothed.detach() * chamfer_losses
        return losses_smoothed.mean()

    def forward(self, chamfer_distance: torch.Tensor, epoch: int) -> torch.Tensor:
        """Computes the time smoothing loss.
        Args:
            chamfer_distance (torch.Tensor): Chamfer distances of shape (T,)
        Returns:
            torch.Tensor: Time smoothing loss
        """
        if self.method == "pdg":
            return self._pdg_smoothing(chamfer_distance, epoch)
        elif self.method == "direct":
            return self._direct(chamfer_distance, epoch)
        elif self.method == "delta_based":
            return self._delta_based(chamfer_distance, epoch)
        elif self.method == "exp_average":
            return self._exp_average(chamfer_distance, epoch)
        elif self.method == "direct_average":
            return self._direct_average(chamfer_distance, epoch)
        elif self.method == "constant":
            return self._constant(chamfer_distance, epoch)
        raise NotImplementedError(
            f"Time smoothing method {self.method} not implemented."
        )
