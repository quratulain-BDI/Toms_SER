from datetime import datetime
from pathlib import Path

import typer
import torch
import git
from typing  import List, Optional

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.infer import infer as run_infer
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize, flip
import torchvision
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
    run_path: Path = typer.Option(
        ..., "-p", "--path", help="Path to run from which you want to infer."
    ),
    label: int = typer.Option(
        6, "-l", "--label", help="Label of image to show to the model"
    ),

    ts: Optional[List[str]] = typer.Option(...,"-ts"
    ,help = 'Enter the list of transformations to perform')

):

    print(ts)
    """Run the inference code"""
    params = load_params(run_path)
    model = torch.load(run_path / "model.pt")
    image = _select_test_image(ts, label)
    run_infer(run_path,params, model, image, label)


def _select_test_image(ts,label
):
    # TODO `ts` is a list of transformations that will be applied to the loaded
    # image. This works... but in order to add a transformation, or change one,
    # we now have to come and edit the code... which sucks. What if we could
    # configure the transformations via the cli?

    #ts = [normalize, flip]
    dataloader = test_dataloader(1, transforms(*ts))
    images, labels = next(iter(dataloader))
    print(torchvision.transforms.functional.get_image_size(images))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images
