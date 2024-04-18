# Created by Baole Fang at 4/2/24
# Modified by George Chang at 4/13/24
import vit_pytorch
from torch.nn.parallel import DataParallel

def make_model(config, gpus, state, device, path, model_name=''):
    model = getattr(vit_pytorch, config['type'])(**config['args'])
    parameters = sum(p.numel() for p in model.parameters())
    if model_name == '' or model_name == 'student':
        config['parameters'] = parameters
        msg = f'parameters: {parameters:,}'
    if model_name:
        config[f'{model_name}_parameters'] = parameters
        msg = f'{model_name} parameters: {parameters:,}'
    print(msg)
    with open(path, 'a') as f:
        f.write(f'{msg}\n')
    if len(gpus) > 1:
        model = DataParallel(model, device_ids=gpus, output_device=device)
    if state:
        if len(gpus) > 1:
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
    return model.to(device)

