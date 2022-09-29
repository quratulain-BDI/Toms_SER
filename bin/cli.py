import typer

main = typer.Typer()
a new change haha

@main.command()
def train():
    print("This is where the training code will go")


@main.command()
def infer():
    pass
