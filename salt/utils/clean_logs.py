import argparse
import shutil
from pathlib import Path


def delete_dirs_without_subdir(folder_path, subdir):
    folder_path = Path(folder_path)

    count = 0
    for directory in folder_path.iterdir():
        if directory.is_dir() and not (directory / subdir).is_dir():
            print(f"Deleting directory: {directory}")
            shutil.rmtree(directory)
            count += 1
    print(f"Deleted {count} directories")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Delete directories which do not contain a specified subdirectory"
    )
    parser.add_argument(
        "--folder_path", required=True, type=str, help="Path to the folder to clean"
    )
    parser.add_argument(
        "--subdirectory",
        required=True,
        type=str,
        help="Name of the subdirectory to check for in each directory",
    )
    args = parser.parse_args(args)
    delete_dirs_without_subdir(args.folder_path, args.subdirectory)


if __name__ == "__main__":
    main()
