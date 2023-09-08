from absl import logging
import pandas as pd
import numpy as np
from pathlib import Path
from constants import module_constants
from typing import Tuple
from tensorflow import keras


class TransformationManager:
    """Manages transformation of events present in a metadata table"""

    # class that receives metadata table

    # can call a prepration function
    # fc will make a folder into which it will save the files after applying transformations
    # fc will run a function over all the paths

    # each processing function will apply the 3 transformation functions
    # then save file to path
    def __init__(self, metadata_table: pd.DataFrame, folder_name: str) -> None:
        self.__metadata_table = metadata_table
        self.folder_name = folder_name
        logging.debug(
            f"""TransformationManager initialized with {metadata_table.shape[0]} items.
            Folder name is {folder_name}."""
        )
        return

    def apply_transformations_to_table(self, output_parent_dir: Path) -> None:
        TRANSFORMATION_NAME_PREFIX = "transform-imp-std_tim-"
        output_dir: Path = (
            output_parent_dir / TRANSFORMATION_NAME_PREFIX + self.folder_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # get all paths
        # apply function to all of them

    @staticmethod
    def apply_transformations_to_event(
        event_path: Path,
        sample_interval_seconds,
        avg_variable_mean: pd.Series,
        avg_variable_std_dev: pd.Series,
    ) -> None:
        # get item
        event = pd.read_parquet(event_path)

        # lower sample rate
        downampled_event = TransformationManager.transform_event_with_downsample(
            event, sample_interval_seconds
        )

        # imput item
        event_class_type = event_path.parent.stem
        imputed_event = TransformationManager.transform_event_with_imputation(
            downampled_event, event_class_type
        )

        # standardize item
        standardized_event = TransformationManager.transform_event_with_standardization(
            imputed_event, avg_variable_mean, avg_variable_std_dev
        )

        # add sin and cos transformation of the time stamp regarding time of day and year

        # get time windows

        # store results
        new_path = event_path.replace(input_folder, output_folder).replace(
            ".feather", ".npy"
        )

    @staticmethod
    def transform_event_with_imputation(
        event_data: pd.DataFrame, event_class_type: int
    ) -> pd.DataFrame:
        """Processes an event by imputation of its null values

        Imputation is divided by variable type
        ----------
        numerical attributes: such as temperature, pressure, flow
            (1) linear interpolation;
            (2) forward then backwards fill;
            (3) fill remaining values with zero (when the entire variable is null).
        class type attribute: represents the anomaly type (or normal)
            (1) forward then backwards fill;
            (2) fill remaining values with file path folder
            (events are stored divided by anomally type).

        Parameters
        ----------
        event_data: pandas.DataFrame
            Data of an event (corresponds to a single .csv file).
        event_class_type: int
            Class (anomaly) type of the event_data.

        Returns
        -------
        pandas.DataFrame
            Data of the event transformed by imputing its null values.
        """

        event_data[module_constants.event_num_attribs] = (
            event_data[module_constants.event_num_attribs]
            .interpolate()
            .ffill()
            .bfill()
            .fillna(0)
        )
        event_data[module_constants.event_class_attrib] = (
            event_data[module_constants.event_class_attrib]
            .ffill()
            .bfill()
            .fillna(event_class_type)
        )
        return event_data

    @staticmethod
    def transform_event_with_standardization(
        event_data: pd.DataFrame,
        avg_variable_mean: pd.Series,
        avg_variable_std_dev: pd.Series,
    ) -> pd.DataFrame:
        """Processes an event by standardizing its values

        The mean and std. dev. values are provided.
        They've been calculated previously on a sample of the event population.

        Parameters
        ----------
        event_data: pandas.DataFrame
            Data of an event (corresponds to a single .csv file).
        avg_variable_mean: pd.Series
            Calculated mean values for each variable, based on all* events.
        avg_variable_std_dev: pd.Series
            Calculated std deviation values for each variable, based on all* events.

        Returns
        -------
        pandas.DataFrame
            Data of the event transformed by standardizing its null values.
        """

        def standardize(x):
            return (x - avg_variable_mean[x.name]) / avg_variable_std_dev[x.name]

        event_data[module_constants.event_num_attribs] = event_data[
            module_constants.event_num_attribs
        ].apply(lambda x: standardize(x), axis=0)
        return event_data

    @staticmethod
    def transform_event_with_downsample(
        event_data: pd.DataFrame,
        sample_interval_seconds: int,
    ) -> pd.DataFrame:
        """Process an event by downsampling its values

        Downsampling is done by using mean of non null values.

        Parameters
         ----------
        event_data: pandas.DataFrame
            Data of an event (corresponds to a single .csv file).
        sample_interval_seconds: int
            Desired interval between each sample of data.

        Returns
        -------
        pandas.DataFrame
            Data of the event transformed by downsampling its values.
        """

        return event_data.resample(
            f"{sample_interval_seconds}s", origin="start", closed="left"
        ).mean()

    @staticmethod
    def transform_event_with_timestep_windows(
        event_data: pd.DataFrame, num_timesteps: int
    ) -> Tuple[np.array, np.array]:
        """Process an event by generating windowed samples

        Parameters
         ----------
        event_data: pandas.DataFrame
            Data of an event (corresponds to a single .csv file).
        num_timesteps: int
            Number of timesteps to use for each window sample.

        Returns
        -------
        np.array
            X: Array of window samples with input features.
        np.array
            y: Array of corresponding targets.
        """

        num_rows = event_data.shape[0]
        numeric_column_name_list = event_data[
            module_constants.event_num_attribs
        ].columns

        input_sequences = np.hstack(
            [
                np.array(event_data[c]).reshape((num_rows, 1))
                for c in numeric_column_name_list
            ]
        )
        output_sequece = np.array(event_data[module_constants.event_class_attrib])

        return TransformationManager.split_sequences(
            input_sequences, output_sequece, num_timesteps
        )

    @staticmethod
    def split_sequences(
        sequences_input: np.array, sequence_output: np.array, num_timesteps: int
    ) -> Tuple[np.array, np.array]:
        """Splits a multivariate sequence into windowed samples

        Parameters
         ----------
        sequences_input: np.array
            Sequence with input features (temperature, pressure, flow).
        sequence_output: np.array
            Sequence with the output (class types), target.
        num_timesteps: int
            Number of timesteps to use for each window sample.

        Returns
        -------
        np.array
            X: Array of window samples with input features.
        np.array
            y: Array of corresponding targets.
        """
        X, y = list(), list()

        for i in range(len(sequences_input)):
            # find the end of this pattern
            end_ix = i + num_timesteps

            # check if we are beyond the sequence size
            if end_ix > len(sequences_input):
                break

            # gather input and output parts of the pattern
            seq_x, seq_y = sequences_input[i:end_ix], sequence_output[end_ix - 1]
            X.append(seq_x)
            y.append(seq_y)

        # converts labels from integer vector to binary class matrix
        y = keras.utils.to_categorical(y, num_classes=module_constants.num_class_types)

        return np.array(X), np.array(y)
