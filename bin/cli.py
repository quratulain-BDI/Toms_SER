import typer
from ser.infer import def_infer

main = typer.Typer()
<<<<<<< HEAD
a new change haha
=======
here's a change to create a conflict
>>>>>>> new

@main.command()
def train():
    print("This is where the training code will go")


@main.command()
def infer():
    def_infer()
