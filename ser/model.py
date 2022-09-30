def load_model(device):
    '''It will load our model '''
    model = Net().to(device)
    return model 