import lightning as L
from torch.utils.data import DataLoader

import salt.utils.file_utils as fu
from salt.data.datasets import SaltDataset
from salt.data.samplers import RandomBatchSampler


class SaltDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        move_files_temp: str | None = None,
        class_dict: str | None = None,
        test_file: str | None = None,
        test_suff: str | None = None,
        pin_memory: bool = True,
        config_S3: dict | None = None,
        **kwargs,
    ):
        """H5 datamodule, wraps a [`salt.data.SaltDataset`][salt.data.SaltDataset] for training,
        validation and testing.

        This datamodule will load data from h5 files. The training, validation and test files
        are specified by the `train_file`, `val_file` and `test_file` arguments.

        The arguments of this class can be set from the YAML config file or from the command line
        using the `data` key. For example, to set the `batch_size` from the command line, use
        `--data.batch_size=1000`.

        Parameters
        ----------
        train_file : str
            Training file path
        val_file : str
            Validation file path
        batch_size : int
            Number of samples to process in each training step
        num_workers : int
            Number of CPU worker processes to load batches from disk
        num_train : int
            Total number of training samples
        num_val : int
            Total number of validation samples
        num_test : int
            Total number of testing samples
        move_files_temp : str
            Directory to move training files to, default is None,
            which will result in no copying of files
        class_dict : str
            Path to umami preprocessing scale dict file
        test_file : str
            Test file path, default is None
        test_suff : str
            Test file suffix, default is None
        pin_memory: bool
            Pin memory for faster GPU transfer, default is True
        config_S3: dict, optional
            Some parameters for the S3 access
        **kwargs
            Keyword arguments for [`salt.data.SaltDataset`][salt.data.SaltDataset]
        """
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.test_suff = test_suff
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.class_dict = class_dict
        self.move_files_temp = move_files_temp
        self.pin_memory = pin_memory
        self.config_S3 = config_S3
        self.kwargs = kwargs

    def prepare_data(self):
        if self.move_files_temp and not self.trainer.fast_dev_run:
            print("-" * 100)
            print(f"Moving train files to {self.move_files_temp} ")
            print("-" * 100)
            fu.move_files_temp(self.move_files_temp, self.train_file, self.val_file)

    def setup(self, stage: str):
        if self.trainer is not None and self.trainer.is_global_zero:
            print("-" * 100)

        if stage == "fit" and self.move_files_temp and not self.trainer.fast_dev_run:
            # Set the training/validation file to the temp path
            self.train_file = fu.get_temp_path(self.move_files_temp, self.train_file)
            self.val_file = fu.get_temp_path(self.move_files_temp, self.val_file)

        # create training and validation datasets
        if stage == "fit":
            self.train_dset = SaltDataset(
                filename=self.train_file,
                num=self.num_train,
                stage=stage,
                **self.kwargs,
            )
            self.val_dset = SaltDataset(
                filename=self.val_file,
                num=self.num_val,
                stage=stage,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} entries")
            print(f"Created validation dataset with {len(self.val_dset):,} entries")

        if stage == "test":
            assert self.test_file is not None, "No test file specified, see --data.test_file"
            self.test_dset = SaltDataset(
                filename=self.test_file,
                num=self.num_test,
                stage=stage,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} entries")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: SaltDataset, shuffle: bool):
        drop_last = stage == "fit"
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=RandomBatchSampler(dataset, self.batch_size, shuffle, drop_last=drop_last),
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def teardown(self, stage: str | None = None):
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
