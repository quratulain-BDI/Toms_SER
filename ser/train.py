from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from model import load_model
from transforms import load_transformer
import data
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

     # dataloaders 
    training_dataloader = data.load_training_data(ts,batch_size)
    validation_dataloader = data.load_validation_data(ts,batch_size,DATA_DIR)

     # training loop
    for epoch in range(epochs): #full dataset 
        for i, (images, labels) in enumerate(training_dataloader): #1 batch
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )

