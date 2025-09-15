"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .IS2020B import *
from .SHVPLI import *

datasets = {
    'is2020b': IS2020B,
    'shvpli': SHVPLI,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
