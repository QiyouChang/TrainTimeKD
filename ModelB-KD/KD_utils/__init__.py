# Created by George Chang at 4/12/24
import os.path

from .dataset import make_dataloaders
from .model import make_model
from .optimizer import make_optimizer
from .scheduler import make_scheduler
import torch

__all__ = ['prepare']

def prepare(config, gpus):
    train_loader, test_loader = make_dataloaders(config['dataset'])
    if config['resume']:
        state = torch.load(os.path.join(config['output_dir'], config['experiment'], 'checkpoints', f'epoch_{config["resume"]}.pth'))
    else:
        state = None
    
    device = torch.device(f'cuda:{gpus[0]}') if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu' 
    model = make_model(config['model'], gpus, state, device)
    optimizer = make_optimizer(config['optimizer'], model, state)
    scheduler = make_scheduler(config['scheduler'], optimizer, state)
    return train_loader, test_loader, model, optimizer, scheduler, device