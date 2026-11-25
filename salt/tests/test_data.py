from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ftag import get_mock_file

from salt.data.datamodules import SaltDataModule
from salt.data.samplers import RandomBatchSampler
from salt.data.transforms import GaussianNoise


def make_structured(n=100):
    rng = np.random.default_rng(42)
    data = np.zeros(n, dtype=[("pt", "f4"), ("eta", "f4")])
    data["pt"] = rng.random(n) + 1
    data["eta"] = rng.random(n)
    return data


def test_gaussian_noise_no_spec_is_noop():
    data = make_structured()
    out = GaussianNoise()(data.copy(), "jets")
    np.testing.assert_array_equal(out, data)


def test_gaussian_noise_unknown_input_type_untouched():
    data = make_structured()
    noise = GaussianNoise({"tracks": [{"variable": "pt", "mean": 1.0, "std": 0.5}]})
    out = noise(data.copy(), "jets")
    np.testing.assert_array_equal(out, data)


def test_gaussian_noise_unit_mean_zero_std_unchanged():
    data = make_structured()
    noise = GaussianNoise({"jets": [{"variable": "pt", "mean": 1.0, "std": 0.0}]})
    out = noise(data.copy(), "jets")
    np.testing.assert_allclose(out["pt"], data["pt"], rtol=1e-6)


def test_gaussian_noise_changes_only_named_field():
    data = make_structured()
    noise = GaussianNoise({"jets": [{"variable": "pt", "mean": 1.0, "std": 0.5}]})
    out = noise(data.copy(), "jets")
    assert not np.allclose(out["pt"], data["pt"])
    np.testing.assert_array_equal(out["eta"], data["eta"])


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("length", [10, 11])
def test_random_batch_sampler_len_and_tiling(length, drop_last):
    dataset = list(range(length))
    sampler = RandomBatchSampler(dataset, batch_size=5, drop_last=drop_last)
    batches = list(sampler)
    assert len(batches) == len(sampler)

    seen = [i for s in batches for i in dataset[s]]
    if drop_last and length % 5:
        assert seen == dataset[: length - length % 5]
    else:
        assert seen == dataset


def test_random_batch_sampler_shuffle_permutes_batches():
    dataset = list(range(20))
    sampler = RandomBatchSampler(dataset, batch_size=5, shuffle=True)
    found = False
    for seed in range(10):
        torch.manual_seed(seed)
        batches = [dataset[s] for s in sampler]
        assert sorted(b[0] for b in batches) == [0, 5, 10, 15]
        if [b[0] for b in batches] != [0, 5, 10, 15]:
            found = True
    assert found, "shuffling never permuted the batch order"


def test_salt_datamodule_setup_and_batch():
    fname, f = get_mock_file()
    f.close()
    dm = SaltDataModule(
        train_file=fname,
        val_file=fname,
        batch_size=10,
        num_workers=0,
        num_train=50,
        num_val=20,
        num_test=0,
        pin_memory=False,
        norm_dict={},
        variables={"jets": ["pt", "eta"], "tracks": ["d0"]},
    )
    dm.trainer = SimpleNamespace(is_global_zero=True, fast_dev_run=False)
    dm.setup("fit")

    assert len(dm.train_dset) == 50
    assert len(dm.val_dset) == 20

    inputs, pad_masks, labels = next(iter(dm.train_dataloader()))
    assert set(inputs) == {"jets", "tracks"}
    assert inputs["jets"].shape[0] == 10
    assert inputs["tracks"].shape[0] == 10
    assert "tracks" in pad_masks
    assert isinstance(labels, dict)


def test_salt_datamodule_sign_flip_augmentation_train_only():
    fname, f = get_mock_file()
    f.close()
    flip = {"prob": 1.0, "variables": {"tracks": ["d0"]}}
    dm = SaltDataModule(
        train_file=fname,
        val_file=fname,
        batch_size=10,
        num_workers=0,
        num_train=50,
        num_val=20,
        num_test=0,
        pin_memory=False,
        norm_dict={},
        variables={"jets": ["pt", "eta"], "tracks": ["d0"]},
        sign_flip_augmentation=flip,
    )
    dm.trainer = SimpleNamespace(is_global_zero=True, fast_dev_run=False)
    dm.setup("fit")
    assert dm.train_dset.sign_flip_augmentation == flip
    assert not dm.val_dset.sign_flip_augmentation
