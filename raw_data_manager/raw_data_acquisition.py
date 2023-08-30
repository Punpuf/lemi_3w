import sys

sys.path.append("..")  # Allows imports from sibling directories

from absl import logging
from constants import config, utils
import pandas as pd
import pathlib
import tempfile
import shutil
from git.repo.base import Repo
import parallelbar
import configparser
import urllib.request
import re
import os
from typing import Tuple


URL_3W_REPO = config.URL_3W_REPO
URL_3W_CONFIG_FILE = config.URL_3W_DATASET_CONFIG_FILE
DIR_DATA = config.DIR_PROJECT_DATA
DIR_DOWNLOADED_REPO = config.DIR_DOWNLOADED_REPO
DIR_RAW_DATASET = config.DIR_RAW_DATASET

CSV_EXTENSION = ".csv"
PARQUET_EXTENSION = ".parquet"


def acquire_dataset_if_needed(
    url_3w_repo: str = URL_3W_REPO,
    url_3w_dataset_config_file=URL_3W_CONFIG_FILE,
    dir_3w_repo: str = DIR_DOWNLOADED_REPO,
    dir_data: str = DIR_DATA,
    dir_raw_dataset: str = DIR_RAW_DATASET,
    require_latest_version: bool = True,
) -> None:
    """Downloads dataset, and converts its format if the latest isn't already available"""
    pathlib.Path(dir_data).mkdir(parents=True, exist_ok=True)
    dir_latest_local, version_latest_local = get_latest_local_converted_data_version(
        dir_data
    )
    logging.info(f"Latest local version is {version_latest_local}")

    latest_dataset_version_online = fetch_latest_online_dataset_version(
        url_3w_dataset_config_file
    )
    logging.info(f"Latest online version is {latest_dataset_version_online}")

    if latest_dataset_version_online is None:
        logging.error("Unable to get latest of dataset available online.")
        return

    if has_valid_converted_dataset(
        dir_latest_local,
        version_latest_local,
        latest_dataset_version_online,
        require_latest_version,
    ):
        logging.info(
            f"Found existing converted data with dataset version of {version_latest_local}"
        )
        return

    logging.info(
        "No existing converted data with the latest version. Attempting to aquire it."
    )

    dir_converted_dataset = DIR_DATA / (
        config.DIR_CONVERTED_PREFIX + latest_dataset_version_online
    )

    download_3w_repo(url_3w_repo, dir_3w_repo)
    create_output_directories(dir_raw_dataset, dir_converted_dataset)
    convert_csv_to_parquet(dir_raw_dataset, dir_converted_dataset)
    delete_3w_repo(dir_3w_repo)


def has_valid_converted_dataset(
    dir_latest_local,
    version_latest_local,
    version_latest_online,
    is_latest_version_required,
):
    """Checks if version is the latest, and if data is present"""
    if dir_latest_local is not None and version_latest_online is not None:
        if (
            is_latest_version_required == False
            or version_latest_online == version_latest_local
        ):
            if has_converted_data(dir_latest_local):
                return True
    return False


def get_dataset_version_from_config_file(dataset_config_file_path: str) -> str:
    """Returns the dataset version present in a local .ini file"""
    parser = configparser.ConfigParser()
    parser.read(dataset_config_file_path)
    try:
        return parser["Versions"]["DATASET"]
    except KeyError:
        logging.error("Config file has no dataset version data.")
        return


def fetch_latest_online_dataset_version(url_dataset_config_file: str) -> str:
    """Fetches the latest dataset version present in the online repository"""
    dataset_config_file_path = pathlib.Path.cwd() / tempfile.mkdtemp() / "config.ini"
    logging.info(f"Going to fetch config file from ${url_dataset_config_file}")

    result = os.popen(f"curl {url_dataset_config_file}").read()
    with open(dataset_config_file_path, "w") as f:
        f.write(result)

    latest_dataset_version = get_dataset_version_from_config_file(
        dataset_config_file_path
    )

    pathlib.Path(dataset_config_file_path).unlink()

    return utils.version_string_to_number(latest_dataset_version)


def extract_directory_dataset_version(directory_name: str) -> str:
    """Extracts dataset version from directory name"""
    # matches any version with digits and dots
    regex_pattern = rf"{config.DIR_CONVERTED_PREFIX}(\d+(\.\d+)*)"

    matches = re.search(regex_pattern, directory_name)
    if matches:
        return matches.group(1)

    return None


def get_latest_local_converted_data_version(dir_data: str) -> Tuple[pathlib.Path, str]:
    base_directory = pathlib.Path(dir_data)
    directories = [dir for dir in base_directory.iterdir() if dir.is_dir()]

    max_version = None
    max_version_directory = None

    # Iterate over directories and find the one with the biggest version
    for directory in directories:
        version = extract_directory_dataset_version(directory.name)
        if version:
            if max_version is None or version > max_version:
                max_version = version
                max_version_directory = directory

    if max_version_directory:
        logging.info(f"Directory with the biggest version: {max_version_directory}")
        logging.info(f"Version: {max_version}")
        return max_version_directory, max_version
    else:
        logging.info("No directory with matching version found.")
        return None, None


def has_converted_data(path_converted_data: str) -> bool:
    """Checks for .parquet file existence in the converted raw data folder"""
    data_converted_dir = pathlib.Path(path_converted_data)
    if not data_converted_dir.is_dir():
        return False

    for anomaly_folder in data_converted_dir.iterdir():
        if anomaly_folder.stem.isnumeric():
            for anomaly_file in anomaly_folder.iterdir():
                if anomaly_file.suffix == PARQUET_EXTENSION:
                    return True

    return False


def download_3w_repo(download_url: str, path_downloaded_repo: str) -> None:
    """Downloads the 3W repo from GitHub"""
    logging.debug("Downloading 3W repo")
    Repo.clone_from(
        url=download_url,
        to_path=path_downloaded_repo,
        progress=utils.GitRemoteProgress(),
    )


def create_output_directories(path_raw_dataset: str, path_converted_dataset: str):
    """Create output directories for converted data"""
    data_raw_dir = pathlib.Path(path_raw_dataset)
    data_converted_dir = pathlib.Path(path_converted_dataset)

    for item in data_raw_dir.iterdir():
        if item.stem.isnumeric():
            item_output_dir = data_converted_dir / item.stem
            item_output_dir.mkdir(parents=True, exist_ok=True)


def convert_file(anomaly_file, item_output_dir):
    if anomaly_file.suffix == CSV_EXTENSION:
        file_name = anomaly_file.stem
        file_path = item_output_dir / f"{file_name}{PARQUET_EXTENSION}"
        pd.read_csv(anomaly_file).to_parquet(file_path)


def convert_csv_to_parquet(path_raw_dataset: str, path_converted_dataset: str) -> None:
    """Converts downloaded 'dataset' from .csv to .parquet"""
    logging.debug("Converting 3W data from .csv to .parquet")

    data_raw_dir = pathlib.Path(path_raw_dataset)
    data_converted_dir = pathlib.Path(path_converted_dataset)

    tasks_input = []
    tasks_output = []
    for anomaly_folder in data_raw_dir.iterdir():
        if anomaly_folder.stem.isnumeric():
            item_output_dir = data_converted_dir / anomaly_folder.stem
            for anomaly_file in anomaly_folder.iterdir():
                tasks_input.append(anomaly_file)
                tasks_output.append(item_output_dir)

    total_tasks = len(tasks_input)
    parallelbar.progress_starmap(
        convert_file, zip(tasks_input, tasks_output), total=total_tasks
    )
    logging.info("Conversion to parquet has been completed")


def delete_3w_repo(path_downloaded_repo: str):
    """Deletes the original downloaded repo files, as no longer needed"""
    logging.debug("Deleting unconverted .csv dataset data")
    shutil.rmtree(path_downloaded_repo)
