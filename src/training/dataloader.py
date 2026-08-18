from os import PathLike
from torch.utils.data import DataLoader
from training.dataset import TrainingDataset


def create_dataloader(path: str | PathLike[str], batch_size: int = 2) -> DataLoader:
    with open(path) as file:
        text = file.read()

    return DataLoader(TrainingDataset(text),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
