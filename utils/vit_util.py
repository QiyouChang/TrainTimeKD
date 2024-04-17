# Created by Baole Fang at 4/2/24
import os.path

from .dataset import make_dataloaders
from .model import make_model
from .optimizer import make_optimizer
from .scheduler import make_scheduler
import torch

def prepare(config, gpus):
    train_loader, test_loader = make_dataloaders(config['dataset'], config['batch_size'])
    n_classes = len(train_loader.dataset.classes)
    config['model']['args']['num_classes'] = n_classes
    if config['resume']:
        state = torch.load(os.path.join(config['output_dir'], 'checkpoints', f'epoch_{config["resume"]}.pth'))
    else:
        state = None
    device = torch.device(f'cuda:{gpus[0]}')
    model = make_model(config['model'], gpus, state, device, os.path.join(config['output_dir'], 'log.txt'))
    optimizer = make_optimizer(config['optimizer'], model, state)
    scheduler = make_scheduler(config['scheduler'], optimizer, state)
    return train_loader, test_loader, model, optimizer, scheduler, device
