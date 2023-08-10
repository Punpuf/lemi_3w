from absl import logging
from constants import config, utils
import pandas as pd
import pathlib
import shutil
from git.repo.base import Repo
import parallelbar


URL_3W_REPO = config.URL_3W_REPO
DIR_DOWNLOADED_REPO = config.DIR_DOWNLOADED_REPO
DIR_RAW_DATASET = config.DIR_RAW_DATASET
DIR_CONVERTED_DATASET = config.DIR_CONVERTED_DATASET

CSV_EXTENSION = ".csv"
PARQUET_EXTENSION = ".parquet"


def acquire_dataset_if_needed(
    url_3w_repo: str = URL_3W_REPO,
    dir_3w_repo: str = DIR_DOWNLOADED_REPO,
    dir_raw_dataset: str = DIR_RAW_DATASET,
    dir_converted_dataset: str = DIR_CONVERTED_DATASET,
) -> None:
    """Downloads dataset, and converts its format if not already available"""
    if has_converted_data(dir_converted_dataset):
        logging.info("Found existing converted data.")
        return

    logging.info("No existing converted data. Attempting to aquire it.")
    download_3w_repo(url_3w_repo, dir_3w_repo)
    create_output_directories(dir_raw_dataset, dir_converted_dataset)
    convert_csv_to_parquet(dir_raw_dataset, dir_converted_dataset)
    delete_3w_repo(dir_3w_repo)


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
