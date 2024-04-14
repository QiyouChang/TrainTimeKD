# Created by Baole Fang at 4/2/24
import argparse
import os
import re

import wandb
import yaml
from vit_utils import prepare
import torch.nn as nn
import torch
from tqdm import tqdm
import shutil


def train(train_loader, model, optimizer, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0
    phar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    for data, target in phar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        curr_correct = (pred.argmax(1) == target).sum().item()
        correct += curr_correct
        phar.set_postfix(loss=loss.item(), auc=curr_correct / len(data))
    scheduler.step()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)


def evaluate(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            correct += (pred.argmax(1) == target).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


def main(config, gpus):
    train_loader, test_loader, model, optimizer, scheduler, device = prepare(config, gpus)

    # init wandb
    if config['resume']:
        with open(os.path.join(config['output_dir'], 'wandb.txt'), 'r') as f:
            wandb_id = f.read().strip()
        run = wandb.init(config=config, project=config['dataset']['type'], name=config['experiment'], resume="allow",
                         id=wandb_id)
    else:
        run = wandb.init(config=config, project=config['dataset']['type'], name=config['experiment'], resume="allow")
        with open(os.path.join(config['output_dir'], 'wandb.txt'), 'w') as f:
            f.write(run.id)

    with open(os.path.join(config['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    criterion = nn.CrossEntropyLoss()
    start = 1 if not config['resume'] else config['resume'] + 1
    # train your model
    for epoch in range(start, config['epochs'] + 1):
        lr = scheduler.get_last_lr()[0]
        # train your model
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, criterion, epoch, device)
        # validate your model
        test_loss, test_acc = evaluate(test_loader, model, criterion, device)
        # log your results
        with open(os.path.join(config['output_dir'], 'log.txt'), 'a') as f:
            f.write(
                f'Epoch {epoch}: train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}, lr: {lr}\n')
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                   'test_acc': test_acc, 'lr': lr})
        # save your model
        if epoch % config['save_interval'] == 0:
            output_path = os.path.join(config['output_dir'], 'checkpoints', f'epoch_{epoch}.pth')
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            state = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, output_path)
            output_path = os.path.join(config['output_dir'], 'checkpoints', 'epoch_latest.pth')
            torch.save(state, output_path)
            wandb.save(output_path)
    run.finish()


def parse(cfg, params):
    for param in params:
        if '=' not in param:
            raise ValueError('Invalid argument: {}'.format(param))
        key, value = param.split('=')
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        curr = cfg
        keys = key.split('.')
        for key in keys[:-1]:
            curr = curr[key]
        curr[keys[-1]] = value

    words = re.split('[{}]', cfg['experiment'])
    for i, word in enumerate(words):
        if i % 2 == 1:
            curr = cfg
            for key in word.split('.'):
                curr = curr[key]
            words[i] = str(curr)
    cfg['experiment'] = ''.join(words)
    cfg['experiment'] = '-'.join([cfg['dataset']['type'], cfg['experiment']])
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='configs/vit/vit_base.yaml',
                        help='path to the model configuration file')
    parser.add_argument('--data', type=str, default='configs/data/cifar10.yaml',
                        help='path to the dataset configuration file')
    parser.add_argument('--set', type=str, nargs='+', help='override configuration file')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use')
    parser.add_argument('--force', action='store_true', help='overwrite existing experiment')
    args = parser.parse_args()

    with open(args.model, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.data, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    config = {**model_config, **data_config}
    config = parse(config, args.set)
    output_root = os.path.join(config['output_dir'], config['experiment'])
    if os.path.exists(output_root) and not config['resume']:
        if not args.force:
            print(f'Experiment {config["experiment"]} already exists. Enter y to overwrite.')
            choice = input()
            if choice != 'y':
                exit()
        shutil.rmtree(output_root)
    os.makedirs(os.path.join(output_root, 'checkpoints'), exist_ok=True)
    config['output_dir'] = output_root
    main(config, list(map(int, args.gpus.split(','))))
