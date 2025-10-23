import torch
import numpy as np
import random
import itertools

from g4satbench.data.dataset import SATDataset,  SATDataset_prediction
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


def collate_fn(batch):
    return Batch.from_data_list([s for s in list(itertools.chain(*batch))])


def get_dataloader(data_dir, splits, sample_size, opts, mode, use_contrastive_learning=False):
    dataset = SATDataset(data_dir, splits, sample_size, use_contrastive_learning, opts)
    batch_size = opts.batch_size // len(splits) if opts.data_fetching == 'parallel' else opts.batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )

def get_dataloader_prediction(cnf_file, splits, opts, mode, use_contrastive_learning=False):
    dataset = SATDataset_prediction(cnf_file, splits, use_contrastive_learning, opts)
    batch_size = 1

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode=='train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )