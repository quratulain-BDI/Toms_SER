from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import typer
from ser.model import load_model
from ser.transforms import load_transformer
from ser.data import load_training_data, load_validation_data
from ser.train import train_func
import json


main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from dataclasses import dataclass

@dataclass
class parameters:
    name: str
    learning_rate: float
    epochs: int
    batch_size: int 


    
@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", "--learning_rate", help="set the Learning Rate Parameter"
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="The number of epochs to run"
    ),
    batch_size: int = typer.Option(
        ..., "-b", "--batch_size", help="The number of batch_size to run"
    ),
):
    """Takes epochs batch size and learning rates as input and runs out model"""
    params = parameters(name, learning_rate, epochs, batch_size)
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = load_model(device)
    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load transformer
    ts = load_transformer()

    # dataloaders
    training_dataloader = load_training_data(ts, batch_size)
    validation_dataloader = load_validation_data(ts, batch_size, DATA_DIR)

    best_model_params, param_list = train_func(
        params,
        training_dataloader,
        validation_dataloader,
        device,
        model,
        optimizer,
    )


@main.command()
def infer():
    print("This is where the inference code will go")
