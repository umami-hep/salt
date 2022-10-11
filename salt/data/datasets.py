import h5py
import torch
from torch.utils.data import Dataset


class SimpleJetDataset(Dataset):
    """A simple map-style dataset for loading jets."""

    def __init__(
        self,
        filename: str,
        tracks_name: str,
        labels: dict,
        jet_class_dict: dict,
        num_jets: int = -1,
    ):
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
        self.inputs = self.file[tracks_name]
        self.jet_class_labels = self.file[labels["jet_classification"]]
        self.track_class_labels = self.file[labels["track_classification"]]

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
        jet_class_label = torch.tensor(
            self.jet_class_dict[self.jet_class_labels[jet_idx]]["label"]
        )
        # TODO: update umami to allow for named label accessing
        track_class_label = torch.tensor(self.track_class_labels[jet_idx][:, 0])
        labels = {
            "jet_classification": jet_class_label,
            "track_classification": track_class_label,
        }
        return inputs, labels
