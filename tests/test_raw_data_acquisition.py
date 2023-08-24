import sys

sys.path.append("..")  # Allows imports from sibling directories

from raw_data_manager.raw_data_acquisition import (
    has_valid_converted_dataset,
    get_dataset_version_from_config_file,
    has_converted_data,
    extract_directory_dataset_version,
    get_latest_local_converted_data_version,
    create_output_directories,
    convert_csv_to_parquet,
    delete_3w_repo,
)
from raw_data_manager import raw_data_acquisition
from constants import config

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import configparser


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


class TestDataAcquisition:
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

    def test_get_dataset_version_from_config_file_with_data(self):
        # Creating test config data
        config_file_path = Path(self.temp_dir) / "config.ini"
        dataset_version = "12.3.4"
        config = configparser.ConfigParser()
        config["Versions"] = {"DATASET": dataset_version}
        with open(str(config_file_path), "w") as configfile:
            config.write(configfile)

        # Testing function
        result_version = get_dataset_version_from_config_file(str(config_file_path))
        assert result_version == dataset_version

    def test_get_dataset_version_from_config_file_no_data(self):
        # Creating test config data
        config_file_path = Path(self.temp_dir) / "config.ini"
        config = configparser.ConfigParser()
        config["Weather"] = {"RAINS": "Today"}
        with open(str(config_file_path), "w") as configfile:
            config.write(configfile)

        # Testing function
        result_version = get_dataset_version_from_config_file(str(config_file_path))
        assert result_version == None

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


class TestHasValidConvertedDataset:
    """Test cases for the has_valid_converted_dataset function"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        def mock_has_converted_data(directory):
            return True

        # Monkey-patching the original function with the mock
        monkeypatch.setattr(
            raw_data_acquisition, "has_converted_data", mock_has_converted_data
        )

    # Define test cases
    @pytest.mark.parametrize(
        "dir_latest_local, version_latest_local, version_latest_online, is_latest_version_required, expected_result",
        [
            ("/path/to/data", "1.0", "1.0", True, True),
            ("/path/to/data", "1.0", "1.1", False, True),
            ("/path/to/data", "1.0", "1.1", True, False),
            (None, None, "1.0", False, False),
        ],
    )
    def test_has_valid_converted_dataset(
        self,
        dir_latest_local,
        version_latest_local,
        version_latest_online,
        is_latest_version_required,
        expected_result,
    ):
        # Call the function being tested
        result = has_valid_converted_dataset(
            dir_latest_local,
            version_latest_local,
            version_latest_online,
            is_latest_version_required,
        )

        # Check the result against the expected outcome
        assert result == expected_result


class TestExtractDirectoryDatasetVersion:
    """Test cases for the extract_directory_dataset_version function"""

    @pytest.mark.parametrize(
        "directory, expected_result",
        [
            (f"{config.DIR_CONVERTED_PREFIX}2.1.3", "2.1.3"),
            (f"{config.DIR_CONVERTED_PREFIX}3.10.5", "3.10.5"),
            (f"{config.DIR_CONVERTED_PREFIX}1.11", "1.11"),
            (f"{config.DIR_CONVERTED_PREFIX}2", "2"),
            ("olha_a_pamonha", None),
            ("pamonha_caseira_v2.0.4", None),
            ("pamonha_fresquinha_v1.0", None),
        ],
    )
    def test_extract_directory_dataset_version(self, directory, expected_result):
        result = extract_directory_dataset_version(directory)
        assert result == expected_result


class TestGetLatestLocalConvertedDataVersion:
    """Test cases for the get_latest_local_converted_data_version function"""

    @pytest.mark.parametrize(
        "dir_data, directories, expected_directory, expected_version",
        [
            ("/path/to/data", [], None, None),
            (
                "/path/to/data",
                [Path("converted_dataset_1.0")],
                Path("converted_dataset_1.0"),
                "1.0",
            ),
            (
                "/path/to/data",
                [Path("converted_dataset_1.0"), Path("converted_dataset_2.0")],
                Path("converted_dataset_2.0"),
                "2.0",
            ),
            ("/path/to/data", [Path("some_other_directory")], None, None),
        ],
    )
    def test_get_latest_local_converted_data_version(
        self, dir_data, directories, expected_directory, expected_version, monkeypatch
    ):
        def mock_extract_directory_dataset_version(directory_name):
            if directory_name == "converted_dataset_1.0":
                return "1.0"
            elif directory_name == "converted_dataset_2.0":
                return "2.0"
            else:
                return None

        monkeypatch.setattr(
            raw_data_acquisition,
            "extract_directory_dataset_version",
            mock_extract_directory_dataset_version,
        )

        def mock_iterdir(_):
            return directories

        def mock_is_dir(_):
            return True

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)
        monkeypatch.setattr(Path, "is_dir", mock_is_dir)

        # Call the function being tested
        directory, version = get_latest_local_converted_data_version(dir_data)

        # Check the result against the expected outcome
        assert directory == expected_directory
        assert version == expected_version
