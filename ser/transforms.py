from torchvision import transforms as torch_transforms


def transforms(*ts):
    
    transformations = {
        "normalize": normalize,
        "flip": flip,
    }
    transformation_list=[]
   
    # for key in transformations.keys():
    #     for trx in ts:
    #         if trx in key:
    #             transformation_list.append(transformations[key])  
    transformation_list=[ transformations[t] for t in ts]  
    print(transformation_list)
    return torch_transforms.Compose(
        [
            torch_transforms.ToTensor(),
            *(stage() for stage in transformation_list),
        ]
    )


def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def flip():
    """
    Flip a tensor both vertically and horizontally
    """
    return torch_transforms.Compose(
        [
            torch_transforms.RandomHorizontalFlip(p=1.0),
            torch_transforms.RandomVerticalFlip(p=1.0),
        ]
    )

def rotate():
    ''' Rotates the image'''
    
