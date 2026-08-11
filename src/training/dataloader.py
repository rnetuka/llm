from os import PathLike
from torch.utils.data import DataLoader
from training.dataset import TrainingDataset

##
# Creates two dataloaders from file the-verdict.txt, one for training and one for verification
#
# split_ratio - ratio between training and validation data
#               default value is 0.9 which means 90% data available goes for training, 10% goes for validation
#
# context_length - context length of the data windows
#                  normally specified by CONTEXT_LENGTH, it is scaled down to 256 for performance purpose
#                  see TrainingDataset for details
#
# stride - stride of the data window
#          max value is the same as context_length, which means the data windows do not overlap. Smaller value is used
#          for overlapping windows
#          see TrainingDataset for details
#
def dataloaders(split_ratio: float = 0.9, batch_size: int = 2, context_length: int = 256, stride: int = 256) -> tuple[DataLoader, DataLoader]:
    with open('../resources/texts/the-verdict.txt') as file:
        text = file.read()
        split_index = int(split_ratio * len(text))
        training_dataset = TrainingDataset(text[:split_index], context_length, stride)
        validation_dataset = TrainingDataset(text[split_index:], context_length, stride)
        return (
            DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
            DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        )

def create_dataloader(path: str | PathLike[str], batch_size: int = 2, context_length: int = 256, stride: int = 256) -> DataLoader:
    with open(path) as file:
        text = file.read()
        validation_dataset = TrainingDataset(text, context_length, stride)
        return DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
