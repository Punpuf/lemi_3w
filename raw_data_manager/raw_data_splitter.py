import sys

sys.path.append("..")  # Allows imports from sibling directories

from raw_data_manager import models
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import shutil
import parallelbar


class RawDataSplitter:
    """Manages splitting converted data into train/eval folders, following filters."""

    def __init__(self, metadata_table: pd.DataFrame, data_version: str) -> None:
        self.__metadata_table = metadata_table
        self.__data_version = data_version
        return

    def get_split_path_division(
        self,
        test_size: float,
        class_types: list[EventClassType] = None,
        sources: list[EventSource] = None,
        well_ids: list[int] = None,
    ) -> (list(str), list(str)):
        """Splits input metadata into train/eval in a stratefied way, also applies filters"""
        train, test, _, _ = train_test_split(
            self.__metadata_table,
            self.__metadata_table,
            test_size=test_size,
            random_state=1331,
            stratify=self.__metadata_table[["source", "class_type"]],
        )

        return train["path"].tolist(), test["path"].tolist()

    def __get_source_name(sources: list[models.EventSource] = None) -> str:
        """Returns the section of the split name regarding the filters for data source"""

        if sources is not None:
            text = ""

            if models.EventSource.REAL in sources:
                text += "r"
            if models.EventSource.SIMULATED in sources:
                text += "s"
            if models.EventSource.DRAWN in sources:
                text += "d"

            # Contains all data sources, same as not have any filter
            if text == "rsd":
                return "all"

            return text

        return "all"

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
        split_name += "source-" + __get_source_name(sources) + "_"
        split_name += "class-" + __get_class_type_name(class_types) + "_"
        split_name += "well-" + __get_well_ids_name(well_ids) + "_"

        split_name_train = split_name + "_train"
        split_name_test = split_name + "_test"

        return split_name_train, split_name_test

    def __move_file(file_path: str, output_dir_path: pathlib.Path) -> None:
        """Moves single file to new path"""
        input_file_path = pathlib.Path(file_path)
        output_file_path = (
            output_dir_path / input_file_path.parent.name / input_file_path.name
        )
        shutil.copy(input_file_path, output_file_path)

    def move_files_to_path(
        input_path_list: list[str], output_dir_path: pathlib.Path
    ) -> None:
        """Moves a list of files to a new directory, with parallelization"""
        total_tasks = len(tasks_input)
        parallelbar.progress_starmap(
            convert_file, zip(tasks_input, tasks_output), total=total_tasks
        )
