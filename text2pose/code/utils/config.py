import logging

import numpy as np
from easydict import EasyDict as edict

logger = logging.getLogger()

__C = edict()
cfg = __C

__C.ANN_NAME = 'mpii.json'
__C.DATA_DIR = 'data'
__C.CONFIG_NAME = ''
__C.CUDA = True
__C.WORKERS = 6
__C.SEED = -1
__C.DEBUG = False

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = [24]
__C.TRAIN.MAX_EPOCH = 120
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''

# Modal options
__C.GAN = edict()


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                logger.info('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

