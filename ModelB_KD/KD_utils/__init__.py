# Created by George Chang at 4/12/24
import os.path

from .dataset import make_dataloaders
from .model import make_model
from .optimizer import make_optimizer
import torch

__all__ = ['prepare']

def prepare(config, gpus):
    train_loader, test_loader = make_dataloaders(config['dataset'])
    if config['resume']:
        state = torch.load(os.path.join(config['output_dir'], config['experiment'], 'checkpoints', f'epoch_{config["resume"]}.pth'))
    else:
        state = None
    
    device = torch.device(f'cuda:{gpus[0]}') if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu' 
    teacher_model = make_model(config['teacher_model'], gpus, state, device, os.path.join(config['output_dir'], config['experiment'], 'teacher_log.txt'), "teacher_model")
    student_model = make_model(config['student_model'], gpus, state, device, os.path.join(config['output_dir'], config['experiment'], 'student_log.txt'), "student_model")
    teacher_optimizer = make_optimizer(config['teacher_optimizer'], teacher_model, state)
    student_optimizer = make_optimizer(config['student_optimizer'], student_model, state)
    return teacher_model, student_model, train_loader, test_loader, teacher_optimizer, student_optimizer