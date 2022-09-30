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
        ),