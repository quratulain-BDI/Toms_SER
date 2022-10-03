from datetime import datetime
from pathlib import Path

import typer
import torch
import git
import json
from ser.model import Net
from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize
from ser.visual_helper import *
main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
     label: int = typer.Option(
        ..., "-l", "--label", help="Enter the label to pick images."
    )
):
    run_path = Path("./results/2022-10-03 13:27:51.640216/")
    

    # TODO load the parameters from the run_path so we can print them out!
    f = open(run_path/'run_params.json')
    params = data = json.load(f)
    print('Rum paramaeters are:\n \n',params)

    
    f = open(run_path/'best_params.json')
    params = data = json.load(f)
    print('Hyper paramaeters are:\n \n',params)

 # load the model
   # model = torch.load(run_path / "model.pt")

    model = Net()
    model.load_state_dict(torch.load(run_path / "model.pt"))

    # select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        #print(labels[0].item())
        images, labels = next(iter(dataloader))

   
    # run inference
    model.eval()
    output = model(images)
    print(output)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")