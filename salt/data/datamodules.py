import lightning as L
from torch.utils.data import DataLoader

import salt.utils.fileutils as fu
from salt.data.datasets import JetDataset
from salt.data.samplers import RandomBatchSampler


class JetDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        batch_size: int,
        num_workers: int,
        num_jets_train: int,
        num_jets_val: int,
        num_jets_test: int,
        move_files_temp: str | None = None,
        class_dict: str | None = None,
        test_file: str | None = None,
        test_suff: str | None = None,
        **kwargs,
    ):
        """h5 jet datamodule.

        Parameters
        ----------
        train_file : str
            Training file path
        val_file : str
            Validation file path
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
        move_files_temp : str
            Directory to move training files to, default is None,
            which will result in no copying of files
        class_dict : str
            Path to umami preprocessing scale dict file
        test_file : str
            Test file path, default is None
        test_suff : str
            Test file suffix, default is None
        **kwargs
            Additional arguments to pass to the Dataset class
        """
        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.test_suff = test_suff
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_jets_train = num_jets_train
        self.num_jets_val = num_jets_val
        self.num_jets_test = num_jets_test
        self.class_dict = class_dict
        self.move_files_temp = move_files_temp
        self.kwargs = kwargs

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
        if stage == "fit":
            self.train_dset = JetDataset(
                filename=self.train_file,
                num_jets=self.num_jets_train,
                **self.kwargs,
            )
            self.val_dset = JetDataset(
                filename=self.val_file,
                num_jets=self.num_jets_val,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} jets")
            print(f"Created validation dataset with {len(self.val_dset):,} jets")

        if stage == "test":
            assert self.test_file is not None, "No test file specified, see --data.test_file"
            self.test_dset = JetDataset(
                filename=self.test_file,
                num_jets=self.num_jets_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} jets")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: JetDataset, shuffle: bool):
        drop_last = stage == "fit"
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=RandomBatchSampler(dataset, self.batch_size, shuffle, drop_last=drop_last),
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def teardown(self, stage: str | None = None):
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
