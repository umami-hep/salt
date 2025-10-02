"""Utilities for temporary file handling and S3 downloads.

This module provides helpers to:
- build temporary file paths (e.g., on RAM disks),
- copy/move/remove files safely,
- convert various S3 path formats to canonical S3 URLs,
- download files from S3 (optionally in parallel),
- and patch configuration dictionaries/paths after downloading.
"""

import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

try:
    import boto3 as _boto3

except ImportError:
    _boto3 = None


def get_temp_path(move_files_temp: str, in_path: str | Path) -> Path:
    """Create the full temporary path for a file.

    Parameters
    ----------
    move_files_temp : str
        Root directory where temporary files should live (e.g. ``/dev/shm/user/tmp``).
    in_path : str | Path
        Original absolute or relative input file path.

    Returns
    -------
    Path
        Temporary path composed as ``Path(move_files_temp) / Path(in_path).name``.
    """
    return Path(Path(move_files_temp) / Path(in_path).name)


def copy_file(in_path: Path, out_path: Path) -> None:
    """Copy a file to a destination unless the destination already exists.

    Parameters
    ----------
    in_path : Path
        Source file path.
    out_path : Path
        Destination file path. Parent directories are created if necessary.
    """
    if in_path == out_path or out_path.is_file():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_path, out_path)


def remove_file(path: Path) -> None:
    """Remove a file if it exists.

    Parameters
    ----------
    path : Path
        Path to the file to remove.
    """
    if path.is_file():
        path.unlink()
    else:
        print(f"No file to delete at {path}")


def remove_files_temp(train_temp_path: Path, val_temp_path: Path) -> None:
    """Remove the temporary train/validation files and the temp directory (if empty).

    Parameters
    ----------
    train_temp_path : Path
        Temporary path for the training file.
    val_temp_path : Path
        Temporary path for the validation file.
    """
    remove_file(train_temp_path)
    remove_file(val_temp_path)
    # Best-effort: remove the parent directory (succeeds only if empty)
    val_temp_path.parent.rmdir()


def move_files_temp(move_files_temp: str, train_path: str | Path, val_path: str | Path) -> None:
    """Copy training/validation files to a temporary location before training.

    This is useful when the temporary location is a RAM disk (e.g. ``/dev/shm``).
    The original files are not deleted.

    Parameters
    ----------
    move_files_temp : str
        Root temporary directory (e.g. ``/dev/shm/your/path``).
    train_path : str | Path
        Path to the training file on disk.
    val_path : str | Path
        Path to the validation file on disk.
    """
    temp_train_path = get_temp_path(move_files_temp, train_path)
    temp_val_path = get_temp_path(move_files_temp, val_path)

    copy_file(Path(train_path), temp_train_path)
    copy_file(Path(val_path), temp_val_path)


def convert_path_to_S3url(path: Path | str) -> str:
    """Normalize a path into a canonical ``s3://`` URL.

    Accepts several forms (e.g., ``s3:/bucket/key`` or ``prefix...s3:/bucket/key``)
    and converts them into ``s3://bucket/key``.

    Parameters
    ----------
    path : Path | str
        Input path or URL.

    Returns
    -------
    str
        Canonical S3 URL starting with ``s3://``.
    """
    path = str(path)
    s3_start = "s3://"
    s3_inc_start = "s3:/"

    # If there's something before the "s3:/", remove it
    if s3_inc_start in path and s3_inc_start != path[: len(s3_inc_start)]:
        assert (
            len(path.split(s3_inc_start)) <= 2
        ), "path is invalid: do not set 's3:/' in the paths!"
        path = path.split(s3_inc_start)[-1]

    if s3_start == path[: len(s3_start)]:
        return path
    if s3_inc_start == path[: len(s3_inc_start)]:
        return s3_start + path[len(s3_inc_start) :]
    return s3_start + path


def download_S3(
    session: Any,
    bucket: str,
    file_to_load: str,
    store_path: Path,
    count: int,
) -> None:
    """Download a single S3 object to a local path with a progress bar.

    Parameters
    ----------
    session : Any
        A ``boto3.client('s3')``-like object (duck-typed; must support ``head_object`` and
        ``download_file``).
    bucket : str
        Name of the S3 bucket.
    file_to_load : str
        Object key inside the bucket (may start with ``/``, which is stripped).
    store_path : Path
        Local output path to save the object to.
    count : int
        Position index used by ``tqdm`` so multiple progress bars can render concurrently.
    """
    file_to_load = file_to_load[1:] if file_to_load and file_to_load[0] == "/" else file_to_load
    meta_data = session.head_object(Bucket=bucket, Key=file_to_load)
    total_length = int(meta_data.get("ContentLength", 0))
    with tqdm(
        total=total_length,
        desc=f"Downloading s3://{bucket}/{file_to_load}",
        position=count,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as t:
        session.download_file(bucket, file_to_load, str(store_path), Callback=t.update)


def download_script_S3(
    bucket: str,
    local_path: Path | str,
    key: str,
    file: str,
    count: int,
) -> tuple[str, str]:
    """Download an S3 object if not present locally, returning the updated mapping.

    The function is intended to be launched in parallel via ``multiprocessing.Pool``.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    local_path : Path | str
        Local directory to store the downloaded file.
    key : str
        Key name used in the configuration (returned unchanged).
    file : str
        Full S3 URL or object key to download.
    count : int
        Position for the progress bar (``tqdm``).

    Returns
    -------
    tuple[str, str]
        A pair ``(key, local_file_path)`` to be used to update configs.

    Raises
    ------
    ValueError
        If boto3 is not available
    """
    if _boto3:
        target_path = Path(local_path, file.split("/")[-1])
        if not target_path.is_file():
            session = _boto3.client("s3")
            download_S3(session, bucket, file, target_path, count)
        else:
            print(f'- "{file}" found locally and not downloaded.')
        return key, str(target_path)

    raise ValueError("boto3 is not installed!")


def import_data_S3(config_path: str | Path) -> str:
    """Optionally download S3 data referenced in a YAML config and write a local copy.

    If the config contains a ``data.config_s3`` section with ``download_S3: true``,
    all files listed under ``download_files`` are fetched to ``download_path`` in parallel,
    and the paths in the config are updated to the downloaded local files. A local copy of the
    (now patched) config is written next to the downloads and its path is returned.

    Parameters
    ----------
    config_path : str | Path
        Path to the input configuration YAML file.

    Returns
    -------
    str
        Path to the (possibly new) local configuration file to use going forward.
    """
    with open(Path(config_path)) as file:
        cfg = yaml.safe_load(file)

    config_s3 = cfg["data"]["config_s3"]
    os.environ["AWS_ACCESS_KEY_ID"] = config_s3["pubKey"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config_s3["secKey"]
    os.environ["AWS_ENDPOINT_URL"] = config_s3["url"]

    if config_s3.get("download_S3"):
        local_path = Path(config_s3["download_path"])
        local_path.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"S3 download in progress at local path: {local_path}")
        args = [
            (config_s3["bucket"], local_path, key, cfg["data"][key], count)
            for count, key in enumerate(config_s3["download_files"])
        ]
        with Pool() as pool:
            output = pool.starmap(download_script_S3, args)
        for file, result in output:  # type: ignore[assignment]
            print(f'Downloaded {file} as {result.split("/")[-1]} at local path')
            cfg["data"][file] = str(result)

        local_config = Path(local_path, "local_base.yaml")
        with open(local_config, "w") as file:
            yaml.dump(cfg, file, sort_keys=False)
        print("Stored a local version of the config.")
        print("-" * 100, "\n")
    else:
        local_config = Path(config_path)

    return str(local_config)


def setup_S3_CLI(sc_data: dict) -> dict:
    """Prepare environment and optionally download S3 data (CLI-friendly path).

    Similar to :func:`import_data_S3`, but operates directly on an in-memory
    configuration dictionary (e.g., the ``data`` section of a larger config).

    Parameters
    ----------
    sc_data : dict
        The ``data`` sub-dictionary containing a ``config_s3`` section.

    Returns
    -------
    dict
        The (possibly updated) ``data`` dictionary with local file paths after download.
    """
    """Setting up salt to use S3."""
    config_s3 = sc_data["config_s3"]
    os.environ["AWS_ACCESS_KEY_ID"] = config_s3["pubKey"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config_s3["secKey"]
    os.environ["AWS_ENDPOINT_URL"] = config_s3["url"]

    if config_s3.get("download_S3"):
        local_path = Path(config_s3["download_path"])
        local_path.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"S3 download in progress at local path: {local_path}")

        # Parallelise the download
        args = [
            (config_s3["bucket"], local_path, key, sc_data[key], count)
            for count, key in enumerate(config_s3["download_files"])
        ]
        with Pool() as pool:
            output = pool.starmap(download_script_S3, args)

        # Update the config
        for file, result in output:
            print(f'Downloaded {file} as {result.split("/")[-1]} at local path')
            sc_data[file] = str(result)

        print("Data part of the config updated to track the downloaded files.")
        print("-" * 100, "\n")
    return sc_data


def require_S3(path: Path | str) -> bool:
    """Return whether the YAML config at ``path`` requires S3 access.

    The config is considered to require S3 if
    ``data.config_s3.use_S3 == true``.

    Parameters
    ----------
    path : Path | str
        Path to a YAML configuration file.

    Returns
    -------
    bool
        ``True`` if S3 is required, ``False`` otherwise.
    """
    with open(path) as file:
        print("Doign this")
        cfg = yaml.safe_load(file)
        return (
            "config_s3" in cfg["data"]
            and "use_S3" in cfg["data"]["config_s3"]
            and cfg["data"]["config_s3"]["use_S3"]
        )


def require_S3_CLI(config_s3: dict | None) -> bool:
    """Return whether S3 is required based on a ``config_s3`` dictionary.

    Parameters
    ----------
    config_s3 : dict | None
        The ``config_s3`` sub-dictionary (or ``None``).

    Returns
    -------
    bool
        ``True`` if either ``use_S3`` or ``download_S3`` is enabled; otherwise ``False``.
    """
    """Checking whether salt requires s3."""
    if config_s3 is None:
        return False
    if config_s3.get("use_S3"):
        return True
    return config_s3.get("download_S3") is not None


def download_from_S3() -> None:
    """Convenience entry-point: use the default base config and import data from S3.

    This locates ``configs/base.yaml`` relative to this file, and delegates to
    :func:`import_data_S3`.
    """
    config_dir = Path(__file__).parent.parent / "configs"
    config = f"{config_dir}/base.yaml"
    config = import_data_S3(config)
