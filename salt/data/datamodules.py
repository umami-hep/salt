from pathlib import Path

import lightning
import torch
from torch.utils.data import DataLoader

import salt.utils.file_utils as fu
from salt.data.datasets import SaltDataset
from salt.data.samplers import RandomBatchSampler


class SaltDataModule(lightning.LightningDataModule):
    """Datamodule wrapping a [`salt.data.SaltDataset`][salt.data.SaltDataset] for training,
    validation and testing.

    This datamodule will load data from h5 files. The training, validation and test files
    are specified by the `train_file`, `val_file` and `test_file` arguments.

    The arguments of this class can be set from the YAML config file or from the command line
    using the `data` key. For example, to set the `batch_size` from the command line, use
    `--data.batch_size=1000`.

    Parameters
    ----------
    train_file : str | Path
        Training file path
    val_file : str | Path
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
    move_files_temp : str | None, optional
        Directory to move training files to, by default None,
        which will result in no copying of files
    class_dict : str | None, optional
        Path to umami preprocessing scale dict file, by default None
    test_file : str | None, optional
        Test file path, by default None
    train_vds_path : str | None, optional
        Path to where the VDS train file will be created, if wildcards are used for this.
    val_vds_path : str | None, optional
        Path to where the VDS validation file will be created, if wildcards are used for this.
    test_vds_path : str | None, optional
        Path to where the VDS test file will be created, if wildcards are used for this.
    test_suff : str | None, optional
        Test file suffix, by default None
    pin_memory : bool, optional
        Pin memory for faster GPU transfer, by default True
    persistent_workers : bool, optional
        Whether to keep DataLoader worker processes alive between epochs. Defaults to True.
        This is strongly recommended when reading HDF5 / VDS files, since it ensures that each
        worker keeps its own HDF5 file handle open for the lifetime of training. Repeated worker
        teardown and recreation can otherwise lead to excessive file-open churn and increased risk
        of multiprocessing-related HDF5 issues. Ignored if ``num_workers == 0``.
    prefetch_factor : int, optional
        Number of batches loaded in advance by each worker. Defaults to 2.
        Lower values reduce concurrent HDF5 reads and metadata pressure, which can improve stability
        and reduce I/O contention when using Virtual Datasets (VDS) or many DataLoader workers. Only
        used when ``num_workers > 0``.
    multiprocessing_context : str | None, optional
        Multiprocessing start method for DataLoader workers. Typical values are ``"fork"``,
        ``"spawn"``, or ``"forkserver"``. Defaults to None (PyTorch default).
        When reading HDF5 files in parallel, using ``"spawn"`` can improve robustness by preventing
        file handle inheritance across processes. This is particularly relevant for DDP training on
        Linux systems where the default start method is ``"fork"``. Note that ``"spawn"`` may
        slightly increase worker startup time.
    config_s3 : dict | None, optional
        Some parameters for the S3 access, by default None
    **kwargs
        Keyword arguments for [`salt.data.SaltDataset`][salt.data.SaltDataset]
    """

    def __init__(
        self,
        train_file: str | Path,
        val_file: str | Path,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        move_files_temp: str | None = None,
        class_dict: str | None = None,
        test_file: str | None = None,
        train_vds_path: str | None = None,
        val_vds_path: str | None = None,
        test_vds_path: str | None = None,
        test_suff: str | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        multiprocessing_context: str | None = None,
        config_s3: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.test_file = test_file
        self.train_vds_path = train_vds_path
        self.val_vds_path = val_vds_path
        self.test_vds_path = test_vds_path
        self.test_suff = test_suff
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.class_dict = class_dict
        self.move_files_temp = move_files_temp
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.config_s3 = config_s3
        self.kwargs = kwargs

    @staticmethod
    def _dist_barrier() -> None:
        """Synchronize all distributed ranks (no-op if not running distributed).

        This helper calls :func:`torch.distributed.barrier` if and only if the
        default process group is initialized.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _precreate_vds_rank0(self, stage: str) -> None:
        """Pre-create VDS files on global rank 0 to avoid stampede at startup.

        In DDP/multi-node training, each rank may instantiate datasets. When
        wildcards are used, dataset initialization can trigger VDS creation.
        This method runs dataset initialization once on global rank 0 and then
        returns. Other ranks should call :meth:`_dist_barrier` to wait for the
        VDS to exist.

        Notes
        -----
        - Correctness is enforced by the VDS lock inside :class:`salt.data.SaltDataset`.
          This method is a performance/robustness optimization to avoid redundant
          work and metadata pressure.
        - This method should be called before constructing the "real" datasets
          that will be used by DataLoaders.

        Parameters
        ----------
        stage : str
            The Lightning stage (e.g. ``"fit"`` or ``"test"``).
        """
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        if stage == "fit":
            _ = SaltDataset(
                filename=self.train_file,
                num=self.num_train,
                stage=stage,
                vds_path=self.train_vds_path,
                **self.kwargs,
            )
            _ = SaltDataset(
                filename=self.val_file,
                num=self.num_val,
                stage=stage,
                vds_path=self.val_vds_path,
                **self.kwargs,
            )

        elif stage == "test":
            assert self.test_file is not None, "No test file specified, see --data.test_file"
            _ = SaltDataset(
                filename=self.test_file,
                num=self.num_test,
                stage=stage,
                vds_path=self.test_vds_path,
                **self.kwargs,
            )

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

        # Rank-0 precreate + barrier
        if stage in {"fit", "test"}:
            self._precreate_vds_rank0(stage)
            self._dist_barrier()

        # create training and validation datasets
        if stage == "fit":
            self.train_dset = SaltDataset(
                filename=self.train_file,
                num=self.num_train,
                stage=stage,
                vds_path=self.train_vds_path,
                **self.kwargs,
            )
            self.val_dset = SaltDataset(
                filename=self.val_file,
                num=self.num_val,
                stage=stage,
                vds_path=self.val_vds_path,
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
                vds_path=self.test_vds_path,
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
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context=self.multiprocessing_context,
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
            fu.remove_files_temp(Path(self.train_file), Path(self.val_file))
            print("-" * 100)
