import numpy as np
from matplotlib.mlab import find
from data_transforms import create_weight 

def apply_bound(pred, bounds=(0,1,0)):
    """
    This ensures that all views and comments >= 0 and all votes >= 1
    """
    pred = (pred >= bounds) * pred + bounds * (pred < bounds)
    return pred

def apply_scales(pred, categories, scales):
    """
    Applies scales to a prediction given a dict containing scales indexed
    by category name and a list of categories
    
    len(categories) == pred.shape[0]
    """
    
# Scales for when CV is 20% of the train set and keep=22500
scales = {'Chicago': (0.78, 0.88, 1),
          'Chicago_rac': (0.6, 0.8, 0.1),
          'New_Haven': (0.89, 1, 0.68),
          'New_Haven_rac': (1, 1, 1),
          'Oakland': (0.84, 1, 0.51),
          'Oakland_rac': (1, 0.94, 0.23),
          'Richmond': (0.64, 1, 1),
          'Richmond_rac': (1, 1, 1)}
          
#~ scales = {'Chicago': (1, 1, 1),
          #~ 'Chicago_rac': (1, 1, 1),
          #~ 'New_Haven': (1, 1, 1),
          #~ 'New_Haven_rac': (1, 1, 1),
          #~ 'Oakland': (1, 1, 1),
          #~ 'Oakland_rac': (1, 1, 1),
          #~ 'Richmond': (1, 1, 1),
          #~ 'Richmond_rac': (1, 1, 1)}
