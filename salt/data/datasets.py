import h5py
import torch
from torch.utils.data import Dataset


class SimpleJetDataset(Dataset):
    """A simple map-style dataset for loading jets."""

    def __init__(self, filename: str, jet_class_dict: dict, num_jets: int = -1):
        """_summary_

        Parameters
        ----------
        filename : str
            Input h5 filepath
        jet_class_dict : dict
            mapping of flavour label to training label
        num_jets : int, optional
            number of jets to use, by default -1
        """
        super().__init__()

        # get datasets
        self.file = h5py.File(filename, "r")
        self.inputs = self.file["X_tracks_loose_train"]
        self.labels = self.file["flavour"]

        self.num_jets = num_jets

        # map jet flavour labels to classes
        self.jet_class_dict = {
            int(k): {"label": int(v["label"]), "weight": float(v["weight"])}
            for k, v in jet_class_dict.items()
        }

    def __len__(self):
        return int(self.num_jets)

    def __getitem__(self, jet_idx):
        inputs = torch.FloatTensor(self.inputs[jet_idx])
        label = torch.tensor(self.jet_class_dict[self.labels[jet_idx]]["label"])
        return inputs, label
