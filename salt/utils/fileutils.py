import pathlib
import shutil


def copy_file(in_path, out_path, verbose=False):
    """Copys the file from the 'in_path' to the 'out_path', including required
    checks.

    Returns 0 if the file was moved succesfully, and 1 if a file already
    existed in the desired path
    """

    path = pathlib.Path(out_path)

    # Skip copying if file already exists or is in the same place
    if in_path == out_path or path.is_file():
        return

    # Make the directory for the output file
    pathlib.Path(out_path).parents[0].mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_path, out_path)


def remove_file(file_path):
    path = pathlib.Path(file_path)
    if path.is_file():
        path.unlink()
    else:
        print(f"No file was found to delete at {file_path}")


def get_temp_path(move_files_temp, in_path):
    """"""
    path = pathlib.Path(in_path)
    return pathlib.Path(move_files_temp).joinpath(path.name)


def move_files_temp(move_files_temp, train_path, val_path):
    """Automatically moves required files to the desired directory as defined
    by the config variable:

    move_files_temp: /dev/shm/xzcappon # Example to store train files in shared memory

    If this variable is empty, no files will be copied over
    """
    # Get the modified paths
    temp_train_path = get_temp_path(move_files_temp, train_path)
    temp_val_path = get_temp_path(move_files_temp, val_path)

    copy_file(train_path, temp_train_path)
    copy_file(val_path, temp_val_path)


def remove_files_temp(train_temp_path, val_temp_path):
    remove_file(train_temp_path)
    remove_file(val_temp_path)
