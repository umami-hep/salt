import numpy as np
import torch
from torch.utils.data import Sampler


class RandomBatchSampler(Sampler):
    """Batch sampler for an h5 dataset.

    The batch sampler performs weak shuffling. Objects are batched first,
    and then batches are shuffled.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Input dataset
    batch_size : int
        Number of objects to batch
    shuffle : bool, optional
        Shuffle the batches, by default False
    drop_last : bool, optional
        Drop the last incomplete batch (if present), by default False
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.nonzero_last_batch = int(self.n_batches) < self.n_batches
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        return int(self.n_batches) + int(not self.drop_last and self.nonzero_last_batch)

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(int(self.n_batches))
        else:
            self.batch_ids = torch.arange(int(self.n_batches))
        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start, stop = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            yield np.s_[int(start) : int(stop)]

        # in case the batch size is not a perfect multiple of the number of samples,
        # yield the remaining samples
        if not self.drop_last and self.nonzero_last_batch:
            start, stop = int(self.n_batches) * self.batch_size, self.dataset_length
            yield np.s_[int(start) : int(stop)]
