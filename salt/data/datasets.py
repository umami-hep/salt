import h5py
import torch
from torch.utils.data import Dataset


class SimpleJetDataset(Dataset):
    def __init__(
        self,
        filename: str,
        inputs: dict,
        tasks: dict,
        num_jets: int = -1,
    ):
        """A simple map-style dataset for loading jets.

        Parameters
        ----------
        filename : str
            Input h5 filepath
        inputs : dict
            Names of the h5 group to access for each type of input
        tasks : dict
            Dict containing information about each aux task, used to
            load the labels for each task
        num_jets : int, optional
            number of jets to use, by default -1
        """
        super().__init__()

        # open file
        self.filename = filename
        self.file = h5py.File(self.filename, "r")

        # make sure the input file looks okay
        self.check_file(inputs)

        # get datasets
        self.inputs = {n: self.file[f"{g}/inputs"] for n, g in inputs.items()}
        self.valid = {"track": self.file[f"{inputs['track']}/valid"]}
        self.labels = {n: self.file[f"{g}/labels"] for n, g in inputs.items()}

        # set number of jets
        self.set_num_jets(num_jets)

    def __len__(self):
        return int(self.num_jets)

    def __getitem__(self, jet_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        jet_idx
            Either an int corresponding to a single jet to load,
            or a numpy slice corresponding to a batch of jets.

        Returns
        -------
        tuple
            Inputs (tensor) and labels (dict of tensor). Each tensor
            will contain a single element or a batch of elements.
        """

        # read inputs (assume we have already concatenated jet features in umami)
        track_inputs = torch.FloatTensor(self.inputs["track"][jet_idx])
        track_mask = ~torch.tensor(self.valid["track"][jet_idx])
        track_mask[:, 0] = False  # hack to make the MHA work

        # read labels
        jet_class_label = torch.tensor(self.labels["jet"][jet_idx]).long()
        track_labels = torch.LongTensor(self.labels["track"][jet_idx])

        labels = {
            "jet_classification": jet_class_label,
            "track_classification": track_labels[..., 0],
        }

        return track_inputs, track_mask, labels

    def set_num_jets(self, num_jets_requested):
        num_jets_available = len(self.inputs["jet"])

        # not enough jets
        if num_jets_requested > num_jets_available:
            raise ValueError(
                f"You asked for {num_jets_requested:,} jets, but only"
                f" {num_jets_available:,} jets are available in the file"
                f" {self.filename}. Please modify `num_jets_train` in your training"
                " config."
            )

        # use all jets
        if num_jets_requested < 0:
            self.num_jets = num_jets_available

        # use requested jets
        else:
            self.num_jets = num_jets_requested

    def check_file(self, inputs):
        def check_group(g: h5py.Group):
            if not isinstance(g, h5py.Group):
                raise TypeError(
                    f"The input file {self.filename} contains a top level object"
                    f" {g} which is not a group, suggesting you have specified an"
                    " old-style umami training file. Please make sure you use an input"
                    " file which is produced using a version of umami which includes"
                    " !648, i.e. versions >=0.15."
                )

            if "inputs" not in g.keys():
                raise ValueError(
                    f"The group {g} in file {self.filename} does not contain an inputs"
                    " dataset. This is unexpected."
                )

            if inputs["track"] in g.name:
                track_vars = list(g["inputs"].attrs.values())[0]
                if not any(["jet" in v for v in track_vars]):
                    raise ValueError(
                        "Expected to find some variables called jet_* in the inputs"
                        f" dataset for the group {g}. You should make sure to specify"
                        " `concat_jet_tracks: True` in your umami preprocessing"
                        " config."
                    )

        [check_group(self.file[g]) for g in self.file]
