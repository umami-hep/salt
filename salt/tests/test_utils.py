from ftag import get_mock_file

from salt.utils import compare_models


def test_compare_models():
    fname_A = get_mock_file()[0]
    fname_B = get_mock_file()[0]
    args = ["--file_A", fname_A, "--file_B", fname_B, "--tagger_A", "MockTagger"]

    compare_models.main(args)
