import shutil
from pathlib import Path


def get_temp_path(move_files_temp: str, in_path: str):
    """Create full temp file path."""
    return Path(Path(move_files_temp) / Path(in_path).name)


def copy_file(in_path: Path, out_path: Path):
    """Copy a file unless it already exists."""
    if in_path == out_path or out_path.is_file():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_path, out_path)


def remove_file(path: Path):
    """Remove a file if it exists."""
    if path.is_file():
        path.unlink()
    else:
        print(f"No file to delete at {path}")


def remove_files_temp(train_temp_path, val_temp_path):
    remove_file(train_temp_path)
    remove_file(val_temp_path)
    val_temp_path.parent.rmdir()


def move_files_temp(move_files_temp: str, train_path: str, val_path: str):
    """Move training files to a temporary location before training.

    Set move_files_temp to "/dev/shm/your/path" for RAM.
    """
    temp_train_path = get_temp_path(move_files_temp, train_path)
    temp_val_path = get_temp_path(move_files_temp, val_path)

    copy_file(Path(train_path), temp_train_path)
    copy_file(Path(val_path), temp_val_path)
