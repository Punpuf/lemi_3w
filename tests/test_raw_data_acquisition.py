from lemi_3w.raw_data_acquisition import (
    has_converted_data,
    create_output_directories,
    convert_csv_to_parquet,
    delete_3w_repo,
)
from constants import config

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path


def create_sample_csv(file_path):
    """Helper function to create a sample CSV file"""
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print("path do .csv é ", file_path)
    assert Path(file_path).exists()


def assert_parquet_files_exist(directory):
    """Helper function to assert the existence of .parquet files"""
    for folder in directory.iterdir():
        if folder.stem.isnumeric():
            for file in folder.iterdir():
                if file.suffix == ".parquet":
                    assert file.exists()


class TestDataset:
    """Test cases for raw_data_acquisition.py functions"""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        # Store original constants.DIR_DOWNLOADED_REPO value and update it to the temp_dir
        self.original_dir_downloaded_repo = config.DIR_DOWNLOADED_REPO
        config.DIR_DOWNLOADED_REPO = self.temp_dir
        yield
        # Clean up after each test
        config.DIR_DOWNLOADED_REPO = self.original_dir_downloaded_repo
        shutil.rmtree(self.temp_dir)

    def test_has_converted_data_no_data(self):
        assert not has_converted_data(self.temp_dir)

    def test_has_converted_data_with_directory(self):
        # Create a temp directory with no files inside
        sample_dir = Path(self.temp_dir) / "1"
        sample_dir.mkdir(parents=True, exist_ok=True)
        assert not has_converted_data(self.temp_dir)

    def test_has_converted_data_with_data(self):
        # Create a sample .parquet file in the temp directory
        sample_file = Path(self.temp_dir) / "1" / "sample.parquet"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.touch()
        assert has_converted_data(self.temp_dir)

    def test_create_output_directories(self):
        # Create 2 sample directories in the temp directory
        raw_dir = Path(self.temp_dir) / "raw"
        raw_dir.mkdir()
        sample_folder1 = raw_dir / "11"
        sample_folder1.mkdir()
        sample_folder2 = raw_dir / "12"
        sample_folder2.mkdir()

        create_output_directories(raw_dir, self.temp_dir)

        converted_dir1 = Path(self.temp_dir) / "11"
        converted_dir2 = Path(self.temp_dir) / "12"
        assert converted_dir1.exists()
        assert converted_dir2.exists()

    def test_convert_csv_to_parquet(self):
        raw_dir = Path(self.temp_dir) / "raw"
        converted_dir = Path(self.temp_dir) / "converted"

        # Create a sample CSV file in the temp directory
        sample_file = raw_dir / "8" / "sample.csv"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        create_sample_csv(sample_file)

        # Convert the CSV file to parquet
        create_output_directories(raw_dir, converted_dir)
        convert_csv_to_parquet(raw_dir, converted_dir)

        # Check if the parquet file exists
        sample_file_converted = converted_dir / "8" / "sample.parquet"
        assert sample_file_converted.exists()

        # Check if .csv file has the same contents as the .parquet file
        original_file_contents = pd.read_csv(sample_file)
        converted_file_contents = pd.read_parquet(sample_file_converted)
        assert original_file_contents.equals(converted_file_contents)

    def test_delete_3w_repo(self):
        # Create a temporary directory to store the downloaded repo
        download_dir = tempfile.mkdtemp()

        # Create a dummy .git folder in the temp directory
        (Path(download_dir) / ".git").mkdir()

        # Delete the 3W repo
        delete_3w_repo(download_dir)

        # Check if the .git folder is deleted
        assert not Path(download_dir).exists()
