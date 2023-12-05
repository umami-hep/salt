import pytest
from ftag import get_mock_file

from salt.utils import compare_models
from salt.utils.clean_logs import delete_dirs_without_subdir, main


def test_compare_models():
    fname_A = get_mock_file()[0]
    fname_B = get_mock_file()[0]
    args = ["--file_A", fname_A, "--file_B", fname_B, "--tagger_A", "MockTagger"]
    compare_models.main(args)


# Fixture to create a temporary directory with a specified subdirectory
@pytest.fixture
def temp_directory_with_subdir(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir3 = tmp_path / "dir3"

    dir1.mkdir()
    dir2.mkdir()
    dir3.mkdir()

    subdir_path = dir2 / "specified_subdirectory"
    subdir_path.mkdir()

    return tmp_path


# Test delete_dirs_without_subdir function
def test_delete_dirs_without_subdir(temp_directory_with_subdir):
    # Only dir1 and dir3 should be kept after calling the function
    delete_dirs_without_subdir(temp_directory_with_subdir, "specified_subdirectory")

    # Check if the correct directories were deleted
    assert not (temp_directory_with_subdir / "dir1").exists()
    assert (temp_directory_with_subdir / "dir2").exists()
    assert not (temp_directory_with_subdir / "dir3").exists()


# Test the main function by capturing stdout
def test_main(temp_directory_with_subdir):
    print(temp_directory_with_subdir)
    args = [
        "--folder_path",
        str(temp_directory_with_subdir),
        "--subdirectory",
        "specified_subdirectory",
    ]
    main(args=args)
