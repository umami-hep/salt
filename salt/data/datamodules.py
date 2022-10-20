import pytorch_lightning as pl
from torch.utils.data import DataLoader

from salt.data.datasets import SimpleJetDataset
from salt.data.samplers import RandomBatchSampler
from salt.utils.collate import collate


class JetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        inputs: dict,
        tasks: dict,
        batched_read: bool,
        batch_size: int,
        num_workers: int,
        num_jets_train: int,
        num_jets_val: int,
        num_jets_test: int,
    ):
        """h5 jet datamodule.

        Parameters
        ----------
        train_file : str
            Training file path
        val_file : str
            Validation file path
        test_file : str
            Test file path
        inputs : dict
            Input dataset name for each input type
        tasks : dict
            Dict of tasks to perform
        batched_read : bool
            If true, read from h5 in batches
        batch_size : int
            Number of jets to process in each step
        num_workers : int
            Number of worker processes to load batches
        num_jets_train : int
            Total number of training jets
        num_jets_val : int
            Total number of validation jets
        num_jets_test : int
            Total number of testing jets
        """
        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.inputs = inputs
        self.tasks = tasks
        self.batched_read = batched_read
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_jets_train = num_jets_train
        self.num_jets_val = num_jets_val
        self.num_jets_test = num_jets_test

    def setup(self, stage: str):
        print("-" * 100)

        # create training and validation datasets
        if stage == "fit":
            self.train_dset = SimpleJetDataset(
                filename=self.train_file,
                inputs=self.inputs,
                tasks=self.tasks,
                num_jets=self.num_jets_train,
            )
            print(f"Created training dataset with {len(self.train_dset):,} jets")

            self.val_dset = SimpleJetDataset(
                filename=self.val_file,
                inputs=self.inputs,
                tasks=self.tasks,
                num_jets=self.num_jets_val,
            )
            print(f"Created validation dataset with {len(self.val_dset):,} jets")

            # log dataset sizes
            if self.trainer.logger:
                self.trainer.logger.experiment.log_parameter(
                    "num_jets_train", len(self.train_dset)
                )
                self.trainer.logger.experiment.log_parameter(
                    "num_jets_val", len(self.val_dset)
                )

        print("-" * 100, "\n")

    def get_dataloader(self, dataset, shuffle):
        # batched reads from h5 (weak shuffling)
        if self.batched_read:
            sampler = RandomBatchSampler(
                dataset, self.batch_size, shuffle, drop_last=True
            )
            batch_size = None
            collate_fn = None
            shuffle = False
        # automatic batching with true shuffling
        else:
            sampler = None
            batch_size = self.batch_size
            collate_fn = collate

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, shuffle=False)
