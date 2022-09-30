
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_training_data(ts,
batch_size,
shuffle=True,
num_workers=1,
download=True, train=True, 
):
      # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=download, train=train, transform=ts),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return training_dataloader


def load_validation_data(ts,
batch_size,DATA_DIR,
shuffle=False,
num_workers=1,
download=True, train=False,

):
    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=download, train=train, transform=ts),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return validation_dataloader