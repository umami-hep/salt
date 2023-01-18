import pytorch_lightning as pl
from torch.utils.data import DataLoader

import salt.utils.fileutils as fu
from salt.data.datasets import TestJetDataset, TrainJetDataset
from salt.data.samplers import RandomBatchSampler
from salt.utils.collate import collate


class JetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        inputs: dict,
        batched_read: bool,
        batch_size: int,
        num_workers: int,
        num_jets_train: int,
        num_jets_val: int,
        num_jets_test: int,
        exclude: dict = None,
        labels: dict = None,
        move_files_temp: str = None,
        scale_dict: str = None,
        test_file: str = None,
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
        exclude :
            Dict of variables in the input datasets not to consider for training
        labels : dict
            Mapping from task name to label name
        move_files_temp : str
            Directory to move training files to, default is None,
            which will result in no copying of files
        scale_dict : str
            Path to umami preprocessing scale dict file
        """
        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.inputs = inputs
        self.labels = labels
        self.batched_read = batched_read
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_jets_train = num_jets_train
        self.num_jets_val = num_jets_val
        self.num_jets_test = num_jets_test
        self.exclude = exclude
        self.scale_dict = scale_dict
        self.move_files_temp = move_files_temp

    def prepare_data(self):
        if self.move_files_temp and not self.trainer.fast_dev_run:
            print("-" * 100)
            print(f"Moving train files to {self.move_files_temp} ")
            print("-" * 100)
            fu.move_files_temp(self.move_files_temp, self.train_file, self.val_file)

    def setup(self, stage: str):
        if self.trainer.is_global_zero:
            print("-" * 100)

        if stage == "fit" and self.move_files_temp and not self.trainer.fast_dev_run:
            # Set the training/validation file to the temp path
            self.train_file = fu.get_temp_path(self.move_files_temp, self.train_file)
            self.val_file = fu.get_temp_path(self.move_files_temp, self.val_file)

        # create training and validation datasets
        if stage == "fit" or stage == "test":
            self.train_dset = TrainJetDataset(
                filename=self.train_file,
                inputs=self.inputs,
                labels=self.labels,
                num_jets=self.num_jets_train,
                exclude=self.exclude,
            )

        if stage == "fit":
            self.val_dset = TrainJetDataset(
                filename=self.val_file,
                inputs=self.inputs,
                labels=self.labels,
                num_jets=self.num_jets_val,
                exclude=self.exclude,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} jets")
            print(f"Created validation dataset with {len(self.val_dset):,} jets")

        if stage == "test":
            assert self.test_file is not None, "No test file specified, see --data.test_file"
            assert self.scale_dict is not None, "No scale dict specified, see --data.scale_dict"
            self.test_dset = TestJetDataset(
                filename=self.test_file,
                inputs=self.inputs,
                scale_dict=self.scale_dict,
                num_jets=self.num_jets_test,
                exclude=self.exclude,
            )
            print(f"Created test dataset with {len(self.test_dset):,} jets")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: TrainJetDataset, shuffle: bool):
        drop_last = True if stage == "fit" else False

        # batched reads from h5 (weak shuffling)
        if self.batched_read:
            sampler = RandomBatchSampler(dataset, self.batch_size, shuffle, drop_last=drop_last)
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
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def teardown(self, stage: str = None):
        """Remove temporary files."""
        if (
            stage == "fit"
            and self.move_files_temp
            and not self.trainer.fast_dev_run
            and self.trainer.is_global_zero
        ):
            print("-" * 100)
            print(f"Removing training files: \n\t{self.train_file}\n\t{self.val_file}")
            fu.remove_files_temp(self.train_file, self.val_file)
            print("-" * 100)
