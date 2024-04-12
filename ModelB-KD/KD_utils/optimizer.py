# Created by George Chang at 4/12/24

import torch

def make_optimizer(config, model, state):
    func = getattr(torch.optim, config['type'])
    optimizer = func(model.parameters(), **config['args'])
    if state:
        optimizer.load_state_dict(state['optimizer'])
    return optimizer
