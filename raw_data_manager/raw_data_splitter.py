import sys

sys.path.append("..")  # Allows imports from sibling directories

from sklearn.model_selection import train_test_split
from typing import List, Tuple
from absl import logging
import pandas as pd
import pathlib
import shutil
import parallelbar

from raw_data_manager import models


class RawDataSplitter:
    """
    Manages splitting converted data into train/eval folders, following filters.

    This class provides methods to split data into train and test sets based on various
    filters and save them in appropriate directories.

    Attributes
    ----------
    __metadata_table : pd.DataFrame
        The metadata table containing event information.
    __data_version : str
        The version of the dataset being split.
    """

    def __init__(self, metadata_table: pd.DataFrame, data_version: str) -> None:
        """
        Initializes a RawDataSplitter instance.

        Parameters
        ----------
        metadata_table : pd.DataFrame
            The metadata table containing event information.
        data_version : str
            The version of the dataset being split.
        """

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
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Stratified splitting of data into train and test sets, following specified filters.

        Parameters
        ----------
        data_dir : pathlib.Path
            The root directory where the split data will be saved.
        test_size : float
            The proportion of the dataset to include in the test set.
        class_types : list of models.EventClassType, optional
            The list of event class types to filter, by default None.
        sources : list of models.EventSource, optional
            The list of event sources to filter, by default None.
        well_ids : list of int, optional
            The list of well IDs to filter, by default None.

        Returns
        -------
        Tuple[pathlib.Path, pathlib.Path]
            Paths to the train and test directories.
        """

        filtered_metadata_table = self.__metadata_table

        # apply filters
        if class_types is not None and len(class_types) > 0:
            filtered_metadata_table = filtered_metadata_table[
                filtered_metadata_table["class_type"].isin(class_types)
            ]
        if sources is not None and len(sources) > 0:
            filtered_metadata_table = filtered_metadata_table[
                filtered_metadata_table["source"].isin(sources)
            ]

        if well_ids is not None and len(well_ids) > 0:
            filtered_metadata_table = filtered_metadata_table[
                filtered_metadata_table["well_id"].isin(well_ids)
            ]

        # divide filtered data
        train_path_list, test_path_list = self.get_split_path_division(
            filtered_metadata_table, test_size
        )
        logging.debug(
            f"size of train data: {len(train_path_list)} --- size of test data: {len(test_path_list)}"
        )

        # define name of output directories
        split_name_train, split_name_test = self.get_split_name(
            test_size, class_types, sources, well_ids
        )
        train_dir_path = data_dir / split_name_train
        test_dir_path = data_dir / split_name_test
        logging.debug(f"train path {train_dir_path} --- test path {test_dir_path}")

        # creates folders for coming event files
        self.create_class_type_subdirectories(train_dir_path)
        self.create_class_type_subdirectories(test_dir_path)

        # copy files to new directories
        self.copy_files_to_path(train_path_list, train_dir_path)
        self.copy_files_to_path(test_path_list, test_dir_path)

        return train_dir_path, test_dir_path

    @staticmethod
    def get_split_path_division(
        metadata_table: pd.DataFrame,
        test_size: float,
    ) -> Tuple[List[str], List[str]]:
        """
        Splits the metadata into train and test paths in a stratified manner.

        Parameters
        ----------
        metadata_table : pd.DataFrame
            The metadata table containing event information.
        test_size : float
            The proportion of the dataset to include in the test set.

        Returns
        -------
        Tuple[List[str], List[str]]
            Paths for train and test data.
        """

        train, test, _, _ = train_test_split(
            metadata_table,
            metadata_table,
            test_size=test_size,
            random_state=1331,
            stratify=metadata_table[["source", "class_type"]],
        )

        return train["path"].tolist(), test["path"].tolist()

    @staticmethod
    def __get_source_name(sources: list[models.EventSource] = None) -> str:
        """
        Returns the section of the split name regarding the filters for data source.

        Parameters
        ----------
        sources : list of models.EventSource, optional
            The list of event sources to filter, by default None.

        Returns
        -------
        str
            Section of the split name regarding data source filters.
        """

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
        """
        Returns the section of the split name regarding the filters for data class types.

        Parameters
        ----------
        class_types : list of models.EventClassType, optional
            The list of event class types to filter, by default None.

        Returns
        -------
        str
            Section of the split name regarding data class types.
        """
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
        """
        Returns the section of the split name regarding the filters for data well IDs.

        Parameters
        ----------
        well_ids : list of int, optional
            The list of well IDs to filter, by default None.

        Returns
        -------
        str
            Section of the split name regarding data well IDs.
        """
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
    ) -> Tuple[str, str]:
        """
        Returns a name describing the modifications made to the data, used as output folder name.

        Parameters
        ----------
        test_size : float
            The proportion of the dataset to include in the test set.
        class_types : list of models.EventClassType, optional
            The list of event class types to filter, by default None.
        sources : list of models.EventSource, optional
            The list of event sources to filter, by default None.
        well_ids : list of int, optional
            The list of well IDs to filter, by default None.

        Returns
        -------
        Tuple[str, str]
            Names for the train and test splits.
        """

        split_name = "dataset_converted_v" + self.__data_version + "_"
        split_name += "split-" + str(int(test_size * 100)) + "_"
        split_name += "source-" + self.__get_source_name(sources) + "_"
        split_name += "class-" + self.__get_class_type_name(class_types) + "_"
        split_name += "well-" + self.__get_well_ids_name(well_ids) + "_"

        split_name_train = split_name + "train"
        split_name_test = split_name + "test"

        return split_name_train, split_name_test

    @staticmethod
    def create_class_type_subdirectories(directory_path: str) -> None:
        """
        Create output directories for converted data based on class types.

        This method creates subdirectories for each event class type in the specified directory.

        Parameters
        ----------
        directory_path : str
            The directory path where subdirectories will be created.
        """

        data_converted_dir = pathlib.Path(directory_path)

        for i in [class_type.value for class_type in models.EventClassType]:
            item_output_dir = data_converted_dir / str(i)
            item_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy_file(file_path: str, output_dir_path: pathlib.Path) -> None:
        """
        Moves a single file to a new path.

        Parameters
        ----------
        file_path : str
            The path of the file to be moved.
        output_dir_path : pathlib.Path
            The target directory path.
        """

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
        """
        Parallel copy of files to a new destination.

        Parameters
        ----------
        input_path_list : list of str
            List of input file paths to be copied.
        output_dir_path : pathlib.Path
            The target directory path.
        """

        parallelbar.progress_starmap(
            self.copy_file,
            [(input_path, output_dir_path) for input_path in input_path_list],
        )
