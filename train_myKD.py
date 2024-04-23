# Created by George Chang at 4/21/24
import argparse
import os.path
import wandb
import yaml
from utils.kd_util import prepare
from utils import parse
import shutil
from ModelC.my_kd import VanillaKD

def main(config, gpus):
    # init wandb
    if config['resume']:
        with open(os.path.join(config['output_dir'], 'wandb.txt'), 'r') as f:
            wandb_id = f.read().strip()
        run = wandb.init(config=config, project=config['dataset']['type'], name=config['experiment'], resume="allow", id=wandb_id)
    else:
        run = wandb.init(config=config, project=config['dataset']['type'], name=config['experiment'], resume="allow")
        with open(os.path.join(config['output_dir'], 'wandb.txt'), 'w') as f:
            f.write(run.id)

    with open(os.path.join(config['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    teacher_model, student_model, train_loader, test_loader, teacher_optimizer, student_optimizer = prepare(config, gpus)

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, teacher_optimizer, student_optimizer, **config['kd_model']['args'])
    
    distiller.train_model(epochs_teacher=config['epochs'], epochs_student=config['epochs'], plot_losses=False, \
                          save_teacher_model_pth=config['teacher_model_pth'],\
                          save_student_model_pth=os.path.join(config['output_dir'], 'checkpoints', 'epoch_latest.pth'),\
                          save_teacher_model=True, save_student_model=True)    # Train the student network
    distiller.evaluate(teacher=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/kd/vkd.yaml', help='path to the configuration file')
    parser.add_argument('--data', type=str, default='configs/dataset/cifar10.yaml', help='path to the dataset configuration file')
    parser.add_argument('--set', type=str, nargs='+', default=[], help='override configuration file')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use')
    parser.add_argument('--force', action='store_true', help='overwrite existing experiment')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        kd_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.data, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    config = {**kd_config, **data_config}
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
    try: 
        config['dataset']['train']['loader']['num_workers'] = min(config['dataset']['train']['loader']['num_workers'],
                                                                len(os.sched_getaffinity(0)))
        config['dataset']['test']['loader']['num_workers'] = min(config['dataset']['test']['loader']['num_workers'],
                                                                len(os.sched_getaffinity(0)))
    except: # when running in non-Unix system
        config['dataset']['train']['loader']['num_workers'] = 0 
        config['dataset']['test']['loader']['num_workers'] = 0

    main(config, list(map(int, args.gpus.split(','))))
