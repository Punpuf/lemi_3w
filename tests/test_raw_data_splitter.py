import sys

sys.path.append("..")  # Allows imports from sibling directories

import pytest
from raw_data_manager import models
from raw_data_manager.raw_data_splitter import RawDataSplitter
from sklearn.model_selection import train_test_split
import pandas as pd
import pathlib
import shutil
from unittest.mock import patch, MagicMock

dataset_version = "10103"


class TestRawDataSplitter:
    @pytest.fixture
    def mock_metadata_table(self):
        metadata = {
            "path": [
                "path1",
                "path2",
                "path3",
                "path4",
                "path5",
                "path6",
                "path7",
                "path8",
            ],
            "source": [
                models.EventSource.REAL,
                models.EventSource.SIMULATED,
                models.EventSource.HAND_DRAWN,
                models.EventSource.REAL,
                models.EventSource.REAL,
                models.EventSource.SIMULATED,
                models.EventSource.HAND_DRAWN,
                models.EventSource.REAL,
            ],
            "class_type": [
                models.EventClassType.ABRUPT_INCREASE_BSW,
                models.EventClassType.FLOW_INSTABILITY,
                models.EventClassType.FLOW_INSTABILITY,
                models.EventClassType.SEVERE_SLUGGING,
                models.EventClassType.ABRUPT_INCREASE_BSW,
                models.EventClassType.FLOW_INSTABILITY,
                models.EventClassType.FLOW_INSTABILITY,
                models.EventClassType.SEVERE_SLUGGING,
            ],
        }
        return pd.DataFrame(metadata)

    @pytest.fixture
    def raw_data_splitter(self, mock_metadata_table):
        return RawDataSplitter(mock_metadata_table, dataset_version)

    @pytest.mark.parametrize(
        "sources, expected",
        [
            ([models.EventSource.REAL], "r"),
            ([models.EventSource.SIMULATED, models.EventSource.HAND_DRAWN], "s-d"),
            ([models.EventSource.HAND_DRAWN, models.EventSource.SIMULATED], "s-d"),
            (
                [
                    models.EventSource.HAND_DRAWN,
                    models.EventSource.REAL,
                    models.EventSource.SIMULATED,
                ],
                "all",
            ),
            (None, "all"),
        ],
    )
    def test_get_source_name(self, raw_data_splitter, sources, expected):
        print("sources", sources)
        print("expected", expected)
        print(raw_data_splitter)

        # = raw_data_splitter()
        assert raw_data_splitter._RawDataSplitter__get_source_name(sources) == expected

    @pytest.mark.parametrize(
        "class_types, expected",
        [
            (
                [models.EventClassType.ABRUPT_INCREASE_BSW],
                str(models.EventClassType.ABRUPT_INCREASE_BSW.value),
            ),
            (
                [models.EventClassType.SEVERE_SLUGGING],
                str(models.EventClassType.SEVERE_SLUGGING.value),
            ),
            (
                [models.EventClassType.FLOW_INSTABILITY, models.EventClassType.NORMAL],
                str(models.EventClassType.NORMAL.value)
                + "-"
                + str(models.EventClassType.FLOW_INSTABILITY.value),
            ),
            (None, "all"),
        ],
    )
    def test_get_class_type_name(self, raw_data_splitter, class_types, expected):
        assert (
            raw_data_splitter._RawDataSplitter__get_class_type_name(class_types)
            == expected
        )

    @pytest.mark.parametrize(
        "well_ids, expected",
        [([6], "6"), ([1, 2], "1-2"), ([3, 12, 1], "1-3-12"), (None, "all")],
    )
    def test_get_well_ids_name(self, raw_data_splitter, well_ids, expected):
        assert (
            raw_data_splitter._RawDataSplitter__get_well_ids_name(well_ids) == expected
        )

    @pytest.mark.parametrize(
        "test_size, class_types, sources, well_ids, expected_train, expected_test",
        [
            (
                0.2,
                [models.EventClassType.NORMAL],
                [models.EventSource.REAL],
                [1, 2],
                f"dataset_converted_v{dataset_version}_split-20_source-r_class-0_well-1-2_train",
                f"dataset_converted_v{dataset_version}_split-20_source-r_class-0_well-1-2_test",
            ),
            (
                0.25,
                [
                    models.EventClassType.NORMAL,
                    models.EventClassType.ABRUPT_INCREASE_BSW,
                ],
                [models.EventSource.REAL, models.EventSource.SIMULATED],
                [1, 2],
                f"dataset_converted_v{dataset_version}_split-25_source-r-s_class-0-1_well-1-2_train",
                f"dataset_converted_v{dataset_version}_split-25_source-r-s_class-0-1_well-1-2_test",
            ),
            (
                0.3,
                None,
                None,
                None,
                f"dataset_converted_v{dataset_version}_split-30_source-all_class-all_well-all_train",
                f"dataset_converted_v{dataset_version}_split-30_source-all_class-all_well-all_test",
            ),
        ],
    )
    def test_get_split_name(
        self,
        raw_data_splitter,
        test_size,
        class_types,
        sources,
        well_ids,
        expected_train,
        expected_test,
    ):
        train_name, test_name = raw_data_splitter.get_split_name(
            test_size, class_types, sources, well_ids
        )
        assert train_name == expected_train
        assert test_name == expected_test

    @patch("shutil.copy")
    def test_move_file(self, mock_copy, raw_data_splitter):
        raw_data_splitter.move_file(
            "path/0/to_file.feather", pathlib.Path("output_dir")
        )
        mock_copy.assert_called_once_with(
            pathlib.Path("path/0/to_file.feather"),
            pathlib.Path("output_dir/0/to_file.feather"),
        )
