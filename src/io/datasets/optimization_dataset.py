from torch.utils.data.dataset import Dataset
import torch


class OptimizationDataset(Dataset):
    def __init__(self, args: dict, data: dict) -> None:  # T x P x 3
        super().__init__()
        self.args = args
        self.points = data["points"].to(self.device)
        self.epochs = 0

    @property
    def device(self):
        return self.args.device

    @property
    def io_args(self):
        return self.args.io_args

    def __len__(self) -> int:
        if self.args.sequence:
            return 1
        return self.points.shape[0]

    def __getitem__(self, index) -> dict:
        self.epochs += 1
        if self.args.sequence:
            index = self.epochs // 30

        return {
            "target": self.points[index],
            "target_index": torch.tensor([index], device=self.device),
        }
