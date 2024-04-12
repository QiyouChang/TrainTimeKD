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
    train_transforms = make_transforms(config['train_transforms'])
    test_transforms = make_transforms(config['test_transforms'])
    func = getattr(datasets, config['type'])
    train_dataset = func(train=True, transform=train_transforms, **config['args'])
    test_dataset = func(train=False, transform=test_transforms, **config['args'])
    return train_dataset, test_dataset


def make_dataloaders(config):
    train_dataset, test_dataset = make_datasets(config)
    train_loader = DataLoader(train_dataset, **config['train_loader'])
    test_loader = DataLoader(test_dataset, **config['test_loader'])
    return train_loader, test_loader
