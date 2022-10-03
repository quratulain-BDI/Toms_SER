from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import typer
from datetime import datetime
import time
import os
import json 

# datetime object containing current date and time
now_var = str(datetime.now())

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train_func(
    params,
    training_dataloader,
    validation_dataloader,
    device,
    model,
    optimizer
):

    """Takes epochs batch size and learning rates as input and runs out model"""
    RESULTS_DIR = PROJECT_ROOT/params.name/now_var
    #os.makedir(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents = True, exist_ok=True) #path lib version of makedir

    param_list = []
    # training loop
    for epoch in range(params.epochs):  # full dataset
        start_time = time.time()

        for i, (images, labels) in enumerate(training_dataloader):
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
        # Do some stuff

        end_time = time.time()
       
        run_dict = { 
                "epoch": params.epochs,
                "time_taken": end_time - start_time,
                "Avg Loss": val_loss,
                "Accuracy": val_acc,
        }
        param_list.append(run_dict)
        if epoch ==0:
            best_model = torch.save(model.state_dict(), RESULTS_DIR / "model.pt")
            best_accuracy = val_acc

            best_model_params= run_dict
        elif val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model =  torch.save(model.state_dict(),RESULTS_DIR/'model.pt')

            best_model_params= run_dict
    run_params = { 
                "run_id": now_var,
                "model_name": params.name,
                "lr": params.learning_rate,
                "batch_size": params.batch_size,
        }

    
    #params_json = json.dumps(run_params)
    with open(RESULTS_DIR / 'run_params.json', 'w') as f:
        json.dump(run_params,f)

    print(best_model_params)

    #best_json = json.dumps(best_model_params)
    with open(RESULTS_DIR/'best_params.json', 'w') as f:
        json.dump(best_model_params,f)


    return best_model_params, param_list



