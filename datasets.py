import pandas as pd
import numpy as np
import logging

import config, utils, data_transforms

logger = logging.getLogger(__name__)

def load_dataset(name):
    try:
        dataset = utils.load_from_cache(name)
        logger.info('Loaded dataset %s from cache' % name)
    except IOError:
        dataset = make_dataset(name)
        utils.save_to_cache(dataset, name)
    return dataset
    
#~ def make_dataset(name):
    #~ cfgs = config.dataset_configs(name)
    #~ input_data = [load_dataset(ds) for ds in cfgs['input_data']]
    #~ try:
        #~ transform = getattr(data_transforms, cfgs['transform'])
    #~ except AttributeError:
        #~ raise AttributeError('Unable to find transform \
                               #~ %s in data_transforms.py' % cfgs['transform'])
    #~ data = transform(*input_data, **cfgs['args'])
    #~ return data

def make_dataset(name):
    cfgs = config.dataset_configs(name)
    data = [load_dataset(ds) for ds in cfgs['input_data']]
    
    if len(data) == 1:
        data = data[0]
    
    logger.info('Creating dataset %s' % name)
    for tname, args in cfgs['transforms']:
        try:
            transform = getattr(data_transforms, tname)
        except AttributeError:
            raise AttributeError('Unable to find transform \
                                   %s in data_transforms.py' % tname)
        logger.info('Applying %s on %s' % (tname, name))
        data = transform(data, **args)
    return data
