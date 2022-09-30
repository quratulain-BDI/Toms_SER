from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import load_model
from transforms import load_transformer
import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    learning_rate: int = typer.Option(
        ...,'-lr','--learning_rate',help="set the Learning Rate Parameter"
        ),
    epochs: int = typer.Option(
        ...,'-e','--epochs',help="The number of epochs to run"
        ),
    batch_size: int = typer.Option(
        ...,'-b','--batch_size',help="The number of batch_size to run"
        )
    ):
    '''Takes epochs batch size and learning rates as input and runs out model'''

    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model =load_model(device)
     # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load transformer
    ts = load_transformer()