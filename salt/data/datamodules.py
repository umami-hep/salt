import pytorch_lightning as pl
from torch.utils.data import DataLoader

from salt.data.datasets import SimpleJetDataset
from salt.utils.collate import collate


class JetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        tracks_name: str,
        labels: dict,
        batch_size: int,
        num_workers: int,
        num_jets_train: int,
        num_jets_val: int,
        num_jets_test: int,
        jet_class_dict: dict,
    ):
        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tracks_name = tracks_name
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_jets_train = num_jets_train
        self.num_jets_val = num_jets_val
        self.num_jets_test = num_jets_test
        self.jet_class_dict = jet_class_dict

    def setup(self, stage: str):
        print("-" * 100)

        # create training and validation datasets
        if stage == "fit":
            self.train_dset = SimpleJetDataset(
                filename=self.train_file,
                tracks_name=self.tracks_name,
                labels=self.labels,
                num_jets=self.num_jets_train,
                jet_class_dict=self.jet_class_dict,
            )
            print(f"Created training dataset with {len(self.train_dset):,} jets")

            self.val_dset = SimpleJetDataset(
                filename=self.val_file,
                tracks_name=self.tracks_name,
                labels=self.labels,
                num_jets=self.num_jets_val,
                jet_class_dict=self.jet_class_dict,
            )
            print(f"Created validation dataset with {len(self.val_dset):,} jets")

            # if self.trainer.logger:
            #    self.trainer.logger.experiment.log_parameter(
            #        "num_jets_train", len(self.train_dset)
            #    )
            #    self.trainer.logger.experiment.log_parameter(
            #        "num_jets_valid", len(self.val_dset)
            #    )

        print("-" * 100, "\n")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True,
        )
