# Created by Baole Fang at 4/2/24
# Modified by George Chang at 4/13/24
# from ModelA_Vit.vit_pytorch import vit_pytorch
import ModelA_Vit.vit_pytorch as vit_pytorch
from torch.nn.parallel import DataParallel

def make_model(config, gpus, state, device, path, model_type):
    model = getattr(vit_pytorch, config['type'])(**config['args'])
    print(f'{model_type} parameters: {sum(p.numel() for p in model.parameters()):,}')
    with open(path, 'a') as f:
        f.write(f'{model_type} parameters: {sum(p.numel() for p in model.parameters()):,}\n')
    if len(gpus) > 1:
        model = DataParallel(model, device_ids=gpus, output_device=device)
    if state:
        if len(gpus) > 1:
            model.module.load_state_dict(state[model_type])
        else:
            model.load_state_dict(state[model_type])
    return model.to(device)
