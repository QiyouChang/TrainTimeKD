# Created by George Chang at 4/12/24

import torch

def make_scheduler(config, optimizer, state):
    func = getattr(torch.optim.lr_scheduler, config['type'])
    scheduler = func(optimizer, **config['args'])
    if state:
        scheduler.load_state_dict(state['scheduler'])
    return scheduler
