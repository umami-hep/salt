import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import boto3
import yaml
from tqdm import tqdm


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


def convert_path_to_S3url(path: Path | str):
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


def download_S3(session, bucket, file_to_load, store_path, count):
    file_to_load = file_to_load[1:] if file_to_load[0] == "/" else file_to_load
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


def download_script_S3(bucket, local_path, key, file, count):
    target_path = Path(local_path, file.split("/")[-1])
    if not target_path.is_file():
        session = boto3.client("s3")
        download_S3(session, bucket, file, target_path, count)
    else:
        print(f'- "{file}" found locally and not downloaded.')
    return key, str(target_path)


def import_data_S3(config_path):
    with open(config_path) as file:
        cfg = yaml.safe_load(file)

    config_S3 = cfg["data"]["config_S3"]
    os.environ["AWS_ACCESS_KEY_ID"] = config_S3["pubKey"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config_S3["secKey"]
    os.environ["AWS_ENDPOINT_URL"] = config_S3["url"]

    if "download_S3" in config_S3 and config_S3["download_S3"]:
        local_path = Path(config_S3["download_path"])
        local_path.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"S3 download in progress at local path: {local_path}")
        args = [
            (config_S3["bucket"], local_path, key, cfg["data"][key], count)
            for count, key in enumerate(config_S3["download_files"])
        ]
        with Pool() as pool:
            output = pool.starmap(download_script_S3, args)
        for file, result in output:
            print(f'Downloaded {file} as {result.split("/")[-1]} at local path')
            cfg["data"][file] = str(result)

        local_config = Path(local_path, "local_base.yaml")
        with open(local_config, "w") as file:
            yaml.dump(cfg, file, sort_keys=False)
        print("Stored a local version of the config.")
        print("-" * 100, "\n")
    else:
        local_config = config_path

    return str(local_config)


def setup_S3_CLI(sc_data):
    """Setting up salt to use S3."""
    config_S3 = sc_data["config_S3"]
    os.environ["AWS_ACCESS_KEY_ID"] = config_S3["pubKey"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config_S3["secKey"]
    os.environ["AWS_ENDPOINT_URL"] = config_S3["url"]

    if "download_S3" in config_S3 and config_S3["download_S3"]:
        local_path = Path(config_S3["download_path"])
        local_path.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"S3 download in progress at local path: {local_path}")

        # Parallelise the download
        args = [
            (config_S3["bucket"], local_path, key, sc_data[key], count)
            for count, key in enumerate(config_S3["download_files"])
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


def require_S3(path):
    with open(path) as file:
        print("Doign this")
        cfg = yaml.safe_load(file)
        if (
            "config_S3" in cfg["data"]
            and "use_S3" in cfg["data"]["config_S3"]
            and cfg["data"]["config_S3"]["use_S3"]
        ):
            return True
        return False


def require_S3_CLI(config_S3):
    """Checking whether salt requires s3."""
    if config_S3 is None:
        return False
    if "use_S3" in config_S3 and config_S3["use_S3"]:
        return True
    if "download_S3" in config_S3 and config_S3["download_S3"]:
        return True
    return False


def download_from_S3() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    config = f"{config_dir}/base.yaml"
    config = import_data_S3(config)
