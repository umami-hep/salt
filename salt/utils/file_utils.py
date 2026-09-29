"""File helpers: `copy_file`, plus the S3 download chain driven by `import_data_S3`
(`download_script_S3` -> `download_S3`), which fetches config-listed files in parallel.
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

from salt.utils.logging import get_logger

_LOG = get_logger(__name__)


def copy_file(in_path: Path, out_path: Path) -> None:
    """Copy a file to a destination unless the destination already exists.

    Parent directories of ``out_path`` are created if necessary.
    """
    if in_path == out_path or out_path.is_file():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_path, out_path)


def download_S3(
    session: Any,
    bucket: str,
    file_to_load: str,
    store_path: Path,
    count: int,
) -> None:
    """Download a single S3 object to a local path with a progress bar.

    ``session`` is duck-typed: it must support ``head_object`` and
    ``download_file`` (a ``boto3.client('s3')``-like object). ``count`` positions
    the ``tqdm`` bar so multiple downloads can render concurrently.
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
    """Download an S3 object if not present locally, returning ``(key, local_file_path)``.

    Intended to be launched in parallel via ``multiprocessing.Pool``; ``key`` is
    the configuration key name, returned unchanged for updating configs.

    Raises
    ------
    ValueError
        If boto3 is not available.
    """
    if _boto3:
        target_path = Path(local_path, file.rsplit("/", maxsplit=1)[-1])
        if not target_path.is_file():
            session = _boto3.client("s3")
            download_S3(session, bucket, file, target_path, count)
        else:
            _LOG.info(f'- "{file}" found locally and not downloaded.')
        return key, str(target_path)

    raise ValueError("boto3 is not installed!")


def import_data_S3(config_path: str | Path) -> str:
    """Optionally download S3 data referenced in a YAML config and write a local copy.

    If the config contains a ``data.config_s3`` section with ``download_S3: true``,
    all files listed under ``download_files`` are fetched to ``download_path`` in
    parallel, and the paths in the config are updated to the downloaded local
    files. A local copy of the (now patched) config is written next to the
    downloads. Returns the path to the (possibly new) local configuration file
    to use going forward.
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
        _LOG.info("-" * 100)
        _LOG.info(f"S3 download in progress at local path: {local_path}")
        args = [
            (config_s3["bucket"], local_path, key, cfg["data"][key], count)
            for count, key in enumerate(config_s3["download_files"])
        ]
        with Pool() as pool:
            output = pool.starmap(download_script_S3, args)
        for file, result in output:  # type: ignore[assignment]
            _LOG.info(f"Downloaded {file} as {result.split('/')[-1]} at local path")
            cfg["data"][file] = str(result)

        local_config = Path(local_path, "local_base.yaml")
        with open(local_config, "w") as file:
            yaml.dump(cfg, file, sort_keys=False)
        _LOG.info("Stored a local version of the config.")
        _LOG.info("-" * 100 + " \n")
    else:
        local_config = Path(config_path)

    return str(local_config)
