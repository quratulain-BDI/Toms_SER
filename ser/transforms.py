
from torchvision import  transforms

def load_transformer():
      # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return ts
    