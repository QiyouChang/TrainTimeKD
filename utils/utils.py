# Created by Baole Fang at 4/17/24

import re

def parse(cfg, params):
    if params:
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
