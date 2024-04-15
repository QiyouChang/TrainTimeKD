# Created by Baole Fang at 4/2/24
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def make_transforms(config):
    transforms_list = []
    for transform in config:
        func = getattr(transforms, transform['type'])
        if 'args' in transform:
            func = func(**transform['args'])
        else:
            func = func()
        transforms_list.append(func)
    return transforms.Compose(transforms_list)


def make_datasets(config):
    train_transforms = make_transforms(config['train']['transforms'])
    test_transforms = make_transforms(config['train']['transforms'])
    func = getattr(datasets, config['type'])
    train_dataset = func(transform=train_transforms, **config['train']['args'])
    test_dataset = func(transform=test_transforms, **config['test']['args'])
    return train_dataset, test_dataset


def make_dataloaders(config):
    train_dataset, test_dataset = make_datasets(config)
    train_loader = DataLoader(train_dataset, **config['train']['loader'])
    test_loader = DataLoader(test_dataset, **config['test']['loader'])
    return train_loader, test_loader
