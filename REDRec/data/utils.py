import importlib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

def bulid_dataloader(config, local_rank=None, world_size=None):
    '''
    split dataset, generate user history sequence, train/valid/test dataset
    '''
    dataset_dict = {
        'REDRec': ('REDRecDataset'),
    }
    if config.data.get('dataset_type', None) == 'precomputed_embedding':
        dataset_dict['REDRec'] = 'REDRecPrecomputedEmbeddingDataset'
    # V0-aligned dataset: V0Simple data protocol (label=seq -> input,
    # label=pos -> target, V0 memmap embedding directory). The model
    # sees the SAME batch-dict schema as the official precomputed path,
    # so model.forward_precomputed_embedding can be reused unchanged.
    if config.data.get('dataset_type', None) == 'v0_aligned':
        dataset_dict['REDRec'] = 'REDRecV0AlignedDataset'
    
    model_name = config.model.model_name
    dataset_module = importlib.import_module('REDRec.data.dataset')
    train_set_name = dataset_dict[model_name]

    train_set_class = getattr(dataset_module, train_set_name)
    
    train_dataset = train_set_class(local_rank, world_size, config)
    train_num_workers = config.data.train_num_workers
    # train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=None, num_workers=train_num_workers, shuffle=False, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=None, num_workers=train_num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader