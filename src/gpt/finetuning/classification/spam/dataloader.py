from .dataset import SpamDataset
from os import PathLike
from torch.utils.data import DataLoader


def create_dataloader(path: str | PathLike[str], batch_size: int = 8, num_workers: int = 0, shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset=SpamDataset.from_csv(path),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle
    )
