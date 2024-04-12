# Created by George Chang at 4/12/24
import argparse
import os.path

import wandb
import yaml
from vit_utils import prepare
import torch.nn as nn
import torch
from tqdm import tqdm
import shutil
from KD import VanillaKD

def main(config, gpus):
    # init wandb
    if config['resume']:
        with open(os.path.join(config['output_dir'], config['experiment'], 'wandb.txt'), 'r') as f:
            wandb_id = f.read().strip()
        run = wandb.init(config=config, project=config['experiment'], resume="allow", id=wandb_id)
    else:
        run = wandb.init(config=config, project=config['experiment'], resume="allow")
        with open(os.path.join(config['output_dir'], config['experiment'], 'wandb.txt'), 'w') as f:
            f.write(run.id)

    with open(os.path.join(config['output_dir'], config['experiment'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    teacher_model, student_model, train_loader, test_loader, teacher_optimizer, student_optimizer = prepare(config, gpus)

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                      teacher_optimizer, student_optimizer)
    
    distiller.train_teacher(epochs=100, plot_losses=True, save_model=True)    # Train the teacher network
    distiller.train_student(epochs=100, plot_losses=True, save_model=True)    # Train the student network
    distiller.evaluate(teacher=False)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar100.yaml', help='path to the configuration file')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_root = os.path.join(config['output_dir'], config['experiment'])
    if os.path.exists(output_root) and not config['resume']:
        if not config['experiment'].endswith('_'):
            print(f'Experiment {config["experiment"]} already exists. Enter y to overwrite.')
            choice = input()
            if choice != 'y':
                exit()
        shutil.rmtree(output_root)
    os.makedirs(os.path.join(output_root, 'checkpoints'), exist_ok=True)
    main(config, list(map(int, args.gpus.split(','))))
