import sys

sys.path.append("..")  # Allows imports from sibling directories

from raw_data_manager import models, raw_data_acquisition
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import shutil
import parallelbar
from typing import List, Tuple
from absl import logging


class RawDataSplitter:
    """Manages splitting converted data into train/eval folders, following filters."""

    def __init__(self, metadata_table: pd.DataFrame, data_version: str) -> None:
        self.__metadata_table = metadata_table
        self.__data_version = data_version
        return

    def stratefy_split_of_data(
        self,
        data_dir: pathlib.Path,
        test_size: float,
        class_types: list[models.EventClassType] = None,
        sources: list[models.EventSource] = None,
        well_ids: list[int] = None,
    ):
        train_path_list, test_path_list = self.get_split_path_division(
            test_size, class_types, sources, well_ids
        )
        logging.debug(
            f"size of train data: {len(train_path_list)} --- size of test data: {len(test_path_list)}"
        )

        split_name_train, split_name_test = self.get_split_name(
            test_size, class_types, sources, well_ids
        )

        train_dir_path = data_dir / split_name_train
        test_dir_path = data_dir / split_name_test
        logging.debug(f"train path {train_dir_path} --- test path {test_dir_path}")

        # creates folders for later data
        self.create_class_type_subdirectories(train_dir_path)
        self.create_class_type_subdirectories(test_dir_path)

        self.copy_files_to_path(train_path_list, train_dir_path)
        self.copy_files_to_path(test_path_list, test_dir_path)

        return train_dir_path, test_dir_path

    def get_split_path_division(
        self,
        test_size: float,
        class_types: list[models.EventClassType] = None,
        sources: list[models.EventSource] = None,
        well_ids: list[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Splits input metadata into train/eval in a stratefied way, also applies filters"""
        train, test, _, _ = train_test_split(
            self.__metadata_table,
            self.__metadata_table,
            test_size=test_size,
            random_state=1331,
            stratify=self.__metadata_table[["source", "class_type"]],
        )

        return train["path"].tolist(), test["path"].tolist()

    @staticmethod
    def __get_source_name(sources: list[models.EventSource] = None) -> str:
        """Returns the section of the split name regarding the filters for data source"""

        if sources is not None:
            text = ""

            if models.EventSource.REAL in sources:
                text += "r-"
            if models.EventSource.SIMULATED in sources:
                text += "s-"
            if models.EventSource.HAND_DRAWN in sources:
                text += "d-"
            text = text.removesuffix("-")

            # Contains all data sources, same as not have any filter
            if text == "r-s-d":
                return "all"

            return text

        return "all"

    @staticmethod
    def __get_class_type_name(class_types: list[models.EventClassType] = None) -> str:
        """Returns the section of the split name regarding the filters for data class type"""
        if class_types is not None:
            selected_class_types_value = [
                class_type.value for class_type in class_types
            ]
            selected_class_types_value = sorted(selected_class_types_value)

            if len(selected_class_types_value) == len(models.EventClassType):
                return "all"

            return "-".join(map(str, selected_class_types_value))

        return "all"

    @staticmethod
    def __get_well_ids_name(well_ids: list[int] = None) -> str:
        """Returns the section of the split name regarding the filters for data well ids"""
        if well_ids is not None:
            well_ids = sorted(well_ids)
            return "-".join(map(str, well_ids))

        return "all"

    def get_split_name(
        self,
        test_size: float,
        class_types: list[models.EventClassType] = None,
        sources: list[models.EventSource] = None,
        well_ids: list[int] = None,
    ) -> (str, str):
        """Returns a name describing the modifications made to the data, used as output folder name"""
        split_name = "dataset_converted_v" + self.__data_version + "_"
        split_name += "split-" + str(int(test_size * 100)) + "_"
        split_name += "source-" + self.__get_source_name(sources) + "_"
        split_name += "class-" + self.__get_class_type_name(class_types) + "_"
        split_name += "well-" + self.__get_well_ids_name(well_ids) + "_"

        split_name_train = split_name + "train"
        split_name_test = split_name + "test"

        return split_name_train, split_name_test

    @staticmethod
    def create_class_type_subdirectories(directory_path: str):
        """Create output directories for converted data"""
        data_converted_dir = pathlib.Path(directory_path)

        for i in [class_type.value for class_type in models.EventClassType]:
            item_output_dir = data_converted_dir / str(i)
            item_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy_file(file_path: str, output_dir_path: pathlib.Path) -> None:
        """Moves single file to new path"""
        # logging.debug(f"copy file -> {file_path}, {output_dir_path}")
        input_file_path = pathlib.Path(file_path)
        output_file_path = (
            output_dir_path
            / str(input_file_path.parent.name)
            / str(input_file_path.name)
        )
        shutil.copy(input_file_path, output_file_path)

    def copy_files_to_path(
        self, input_path_list: list[str], output_dir_path: pathlib.Path
    ) -> None:
        """Parallel copy of files to a new destination"""
        parallelbar.progress_starmap(
            self.copy_file,
            [(input_path, output_dir_path) for input_path in input_path_list],
        )
