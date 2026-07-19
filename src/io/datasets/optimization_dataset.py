import torch
from torch.utils.data import Dataset


class OptimizationDataset(Dataset):
    def __init__(self, args, data: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.points = data["points"].to(args.device, dtype=torch.float32)
        self.indices = torch.arange(len(self.points), device=args.device)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target": self.points[index],
            "target_index": self.indices[index],
        }
