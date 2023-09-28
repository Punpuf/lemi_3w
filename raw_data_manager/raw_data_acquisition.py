import sys

sys.path.append(".")  # Allows imports from sibling directories

from absl import logging
from git.repo.base import Repo
from typing import Tuple
import pandas as pd
import pathlib
import tempfile
import shutil
import parallelbar
import configparser
import re
import os

from constants import storage_config
from raw_data_manager.git_remote_progress import GitRemoteProgress

URL_3W_REPO = storage_config.URL_3W_REPO
URL_3W_CONFIG_FILE = storage_config.URL_3W_DATASET_CONFIG_FILE
DIR_DATA = storage_config.DIR_PROJECT_DATA
DIR_DOWNLOADED_REPO = storage_config.DIR_DOWNLOADED_REPO
DIR_RAW_DATASET = storage_config.DIR_RAW_DATASET

CSV_EXTENSION = ".csv"
PARQUET_EXTENSION = ".parquet"


def acquire_dataset_if_needed(
    url_3w_repo: str = URL_3W_REPO,
    url_3w_dataset_config_file: str = URL_3W_CONFIG_FILE,
    dir_3w_repo: str = DIR_DOWNLOADED_REPO,
    dir_data: str = DIR_DATA,
    dir_raw_dataset: str = DIR_RAW_DATASET,
    require_latest_version: bool = True,
) -> None:
    """
    Downloads the dataset and converts its format if the latest version isn't already available.

    Parameters
    ----------
    url_3w_repo : str, optional
        The URL of the 3W repository, by default URL_3W_REPO.
    url_3w_dataset_config_file : str, optional
        The URL of the 3W dataset config file, by default URL_3W_CONFIG_FILE.
    dir_3w_repo : str, optional
        The directory where the 3W repository will be downloaded, by default DIR_DOWNLOADED_REPO.
    dir_data : str, optional
        The parent directory where the dataset will be stored, by default DIR_DATA.
    dir_raw_dataset : str, optional
        The directory for the raw dataset, by default DIR_RAW_DATASET.
    require_latest_version : bool, optional
        Flag to specify if the latest version is required, by default True.
    """

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
        storage_config.DIR_CONVERTED_PREFIX + latest_dataset_version_online
    )

    download_3w_repo(url_3w_repo, dir_3w_repo)
    create_output_directories(dir_raw_dataset, dir_converted_dataset)
    convert_csv_dataset_to_parquet(dir_raw_dataset, dir_converted_dataset)
    delete_3w_repo(dir_3w_repo)


def has_valid_converted_dataset(
    dir_latest_local,
    version_latest_local,
    version_latest_online,
    is_latest_version_required: bool,
) -> bool:
    """
    Checks if the provided version is the latest and if converted data is present.

    Parameters
    ----------
    dir_latest_local
        The latest local directory.
    version_latest_local
        The version of the latest local dataset.
    version_latest_online
        The latest online version of the dataset.
    is_latest_version_required
        Flag indicating if the latest version is required.

    Returns
    -------
    bool
        True if the version is valid and data is present, False otherwise.
    """

    if dir_latest_local is not None and version_latest_online is not None:
        if (
            is_latest_version_required == False
            or version_latest_online == version_latest_local
        ):
            if has_converted_data(dir_latest_local):
                return True
    return False


def get_dataset_version_from_config_file(dataset_config_file_path: str) -> str:
    """
    Returns the dataset version present in a local .ini file.

    Parameters
    ----------
    dataset_config_file_path : str
        The path to the dataset configuration file.

    Returns
    -------
    str
        The dataset version.
    """
    parser = configparser.ConfigParser()
    parser.read(dataset_config_file_path)
    try:
        return parser["Versions"]["DATASET"]
    except KeyError:
        logging.error("Config file has no dataset version data.")
        return


def fetch_latest_online_dataset_version(url_dataset_config_file: str) -> str:
    """
    Fetches the latest dataset version present in the online repository.

    Parameters
    ----------
    url_dataset_config_file : str
        The URL of the dataset configuration file.

    Returns
    -------
    str
        The latest dataset version.
    """
    dataset_config_file_path = pathlib.Path.cwd() / tempfile.mkdtemp() / "config.ini"
    logging.info(f"Going to fetch config file from ${url_dataset_config_file}")

    result = os.popen(f"curl {url_dataset_config_file}").read()
    with open(dataset_config_file_path, "w") as f:
        f.write(result)

    latest_dataset_version = get_dataset_version_from_config_file(
        dataset_config_file_path
    )

    pathlib.Path(dataset_config_file_path).unlink()

    return version_string_to_number(latest_dataset_version)


def extract_directory_dataset_version(directory_name: str) -> str:
    """
    Extracts the dataset version from a directory name.

    Parameters
    ----------
    directory_name : str
        The name of the directory.

    Returns
    -------
    str
        The extracted dataset version.
    """

    # matches any version with digits and dots
    regex_pattern = rf"{storage_config.DIR_CONVERTED_PREFIX}(\d+(\.\d+)*)"

    matches = re.search(regex_pattern, directory_name)
    if matches:
        return matches.group(1)

    return None


def get_latest_local_converted_data_version(dir_data: str) -> Tuple[pathlib.Path, str]:
    """
    Gets the latest local converted data version.

    Parameters
    ----------
    dir_data : str
        The directory containing the data.

    Returns
    -------
    Tuple[pathlib.Path, str]
        A tuple containing the directory and the version of the latest local converted data.
    """

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
    """
    Checks if converted data exists in the specified path.

    Parameters
    ----------
    path_converted_data : str
        The path to the converted data.

    Returns
    -------
    bool
        True if converted data is present, False otherwise.
    """
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
    """
    Downloads the 3W repository from GitHub.

    Parameters
    ----------
    download_url : str
        The URL of the 3W repository on GitHub.
    path_downloaded_repo : str
        The local path where the repository will be downloaded.
    """
    logging.debug("Downloading 3W repo")
    Repo.clone_from(
        url=download_url,
        to_path=path_downloaded_repo,
        progress=GitRemoteProgress(),
    )


def create_output_directories(
    path_raw_dataset: str, path_converted_dataset: str
) -> None:
    """
    Creates output directories for converted data.

    This function iterates through the raw dataset directory and creates output directories
    for each numeric subdirectory within the converted dataset directory.

    Parameters
    ----------
    path_raw_dataset : str
        The path to the directory containing the raw dataset.
    path_converted_dataset : str
        The path to the directory where the converted dataset will be stored.
    """
    data_raw_dir = pathlib.Path(path_raw_dataset)
    data_converted_dir = pathlib.Path(path_converted_dataset)

    for item in data_raw_dir.iterdir():
        if item.stem.isnumeric():
            item_output_dir = data_converted_dir / item.stem
            item_output_dir.mkdir(parents=True, exist_ok=True)


def convert_csv_file_to_parquet(event_file_path, output_dir) -> None:
    """
    Converts a CSV anomaly file to Parquet format.

    Parameters
    ----------
    event_file_path
        The path of the CSV event file to be converted.
    output_dir
        The output directory for the converted Parquet file.
    """
    if event_file_path.suffix == CSV_EXTENSION:
        file_name = event_file_path.stem
        file_path = output_dir / f"{file_name}{PARQUET_EXTENSION}"
        pd.read_csv(event_file_path).to_parquet(file_path)


def convert_csv_dataset_to_parquet(
    path_raw_dataset: str, path_converted_dataset: str
) -> None:
    """
    Converts the downloaded dataset from .csv to .parquet format.

    Parameters
    ----------
    path_raw_dataset : str
        The path to the directory containing the raw dataset in CSV format.
    path_converted_dataset : str
        The path to the directory where the converted dataset in Parquet format will be stored.
    """

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
        convert_csv_file_to_parquet, zip(tasks_input, tasks_output), total=total_tasks
    )
    logging.info("Conversion to parquet has been completed")


def delete_3w_repo(path_downloaded_repo: str) -> None:
    """
    Deletes the original downloaded repo files as they are no longer needed.

    Parameters
    ----------
    path_downloaded_repo : str
        The path to the downloaded repository to be deleted.
    """
    logging.debug("Deleting unconverted .csv dataset data")
    shutil.rmtree(path_downloaded_repo)


def get_directory_size_bytes(directory) -> int:
    """
    Calculates the size of a directory in bytes.

    Parameters
    ----------
    directory
        The directory for which the size will be calculated.

    Returns
    -------
    int
        The size of the directory in bytes.
    """
    total_size = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_directory_size_bytes(entry.path)
    return total_size


def get_event(path) -> pd.DataFrame:
    """
    Reads and processes an event from a Parquet file.

    Parameters
    ----------
    path
        The path to the Parquet file containing the event.

    Returns
    -------
    pd.DataFrame
        The processed event as a DataFrame.
    """
    event = pd.read_parquet(path)
    event["timestamp"] = pd.to_datetime(event["timestamp"])
    event = event.set_index("timestamp", drop=True)
    return event


def version_string_to_number(version: str) -> str:
    """
    Converts a version string to a numeric representation.

    Parameters
    ----------
    version : str
        The version string to be converted.

    Returns
    -------
    str
        The numeric representation of the version string.
    """
    components = version.split(".")
    version_number = (
        int(components[0]) * 10000 + int(components[1]) * 100 + int(components[2])
    )
    return str(version_number)
