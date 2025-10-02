import argparse
import shutil
from pathlib import Path


def delete_dirs_without_subdir(folder_path: str | Path, subdir: str) -> None:
    """Delete directories that do not contain a specified subdirectory.

    Parameters
    ----------
    folder_path : str | Path
        Path to the parent folder to scan.
    subdir : str
        Name of the subdirectory to check for in each directory.
    """
    folder_path = Path(folder_path)

    count = 0
    for directory in folder_path.iterdir():
        if directory.is_dir() and not (directory / subdir).is_dir():
            print(f"Deleting directory: {directory}")
            shutil.rmtree(directory)
            count += 1
    print(f"Deleted {count} directories")


def main(args: list[str] | None = None) -> None:
    """Parse CLI arguments and delete directories without a specified subdirectory.

    Parameters
    ----------
    args : list[str] | None, optional
        Command-line arguments. If ``None``, uses ``sys.argv``. Default is ``None``.
    """
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
    parsed_args = parser.parse_args(args)
    delete_dirs_without_subdir(parsed_args.folder_path, parsed_args.subdirectory)


if __name__ == "__main__":
    main()
