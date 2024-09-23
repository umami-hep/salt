import torch
from ftag import get_mock_file

from salt.data import SaltDataset


def test_salt_dataset():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    dataset = SaltDataset(f, norm_dict, variables, "train")
    for i in range(0, len(dataset), 10):
        dataset[i : i + 10]


def test_track_selector():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    selections = {"tracks": ["d0 < 3.5"]}
    no_sel = SaltDataset(f, norm_dict, variables, "train")
    sel = SaltDataset(f, norm_dict, variables, "train", selections=selections)
    for i in range(0, len(no_sel), 10):
        inputs, masks, _ = no_sel[i : i + 10]
        assert torch.any(inputs["tracks"][~masks["tracks"]] > 3.5)
        inputs, masks, _ = sel[i : i + 10]
        assert not torch.any(inputs["tracks"][~masks["tracks"]] > 3.5)
