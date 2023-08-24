import sys

sys.path.append("..")  # Allows imports from sibling directories

from raw_data_manager import models
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import shutil
import parallelbar
from typing import List, Tuple


class RawDataSplitter:
    """Manages splitting converted data into train/eval folders, following filters."""

    def __init__(self, metadata_table: pd.DataFrame, data_version: str) -> None:
        self.__metadata_table = metadata_table
        self.__data_version = data_version
        return

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
    def move_file(file_path: str, output_dir_path: pathlib.Path) -> None:
        """Moves single file to new path"""
        input_file_path = pathlib.Path(file_path)
        output_file_path = (
            output_dir_path / input_file_path.parent.name / input_file_path.name
        )
        print("copy with:", input_file_path, output_file_path)
        shutil.copy(input_file_path, output_file_path)

    def move_files_to_path(
        self, input_path_list: list[str], output_dir_path: pathlib.Path
    ) -> None:
        parallelbar.progress_starmap(
            self.move_file,
            [(input_path, output_dir_path) for input_path in input_path_list],
        )
