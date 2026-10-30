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
        n_full = int(self.n_batches)
        self.batch_ids = torch.randperm(n_full) if self.shuffle else torch.arange(n_full)
        for batch_id in self.batch_ids:  # full batches, in (shuffled) batch order
            yield np.s_[int(batch_id * self.batch_size) : int((batch_id + 1) * self.batch_size)]
        if not self.drop_last and self.nonzero_last_batch:  # the ragged remainder, last
            yield np.s_[int(n_full * self.batch_size) : int(self.dataset_length)]
