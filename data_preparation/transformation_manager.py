from absl import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from tensorflow import keras
from parallelbar import progress_starmap
from itertools import repeat

from raw_data_manager import raw_data_acquisition
from raw_data_manager.models import EventClassType, EventParameters


class TransformationManager:
    """Manages transformation of events present in a metadata table"""

    __valid_pressure_range = (-1e10, 1e10)
    __valid_temperature_range = (-1e3, 1e3)
    __valid_flow_range = (-3, 3)

    valid_num_attribs_range = {
        "P-PDG": __valid_pressure_range,
        "P-TPT": (-1e9, 1e9),
        "T-TPT": __valid_temperature_range,
        "P-MON-CKP": (-1e9, 1e9),
        "T-JUS-CKP": __valid_temperature_range,
        "P-JUS-CKGL": __valid_pressure_range,
        "QGL": __valid_flow_range,
    }

    TRANSFORMATION_NAME_PREFIX = "transform-isdt-"

    def __init__(
        self, metadata_table: pd.DataFrame, output_folder_base_name: str
    ) -> None:
        """Initialize the TransformationManager.

        Parameters
        ----------
        metadata_table: pd.DataFrame
            The metadata table containing event information.
        output_folder_base_name: str
            The base name for the output folder where transformed files will be saved.
        """
        self.metadata_table = metadata_table
        self.folder_name = output_folder_base_name
        logging.debug(
            f"""TransformationManager initialized with {metadata_table.shape[0]} items.
            Folder name is {output_folder_base_name}."""
        )
        return

    def apply_transformations_to_table(
        self,
        output_parent_dir: Path,
        sample_interval_seconds: int,
        num_timesteps_for_window: int,
        avg_variable_mean: pd.Series,
        avg_variable_std_dev: pd.Series,
    ) -> None:
        """Processes each event in the metadata table by imputing null values,
        standardizing, and creating time window samples.

        Parameters
        ----------
        output_parent_dir: Path
            The parent directory where the transformed event data will be stored.
        sample_interval_seconds: int
            Desired interval for downsampling the event data.
        num_timesteps_for_window: int
            Number of timesteps to use for creating windowed samples.
        avg_variable_mean: pd.Series
            Calculated mean values for each variable based on a sample of events.
        avg_variable_std_dev: pd.Series
            Calculated std. deviation values for each variable based on a sample of events.
        """

        # create output directories
        output_dir: Path = (
            output_parent_dir / f"{self.TRANSFORMATION_NAME_PREFIX}{self.folder_name}"
        )

        for class_type in range(len(EventClassType)):
            class_type_dir = output_dir / str(class_type)
            class_type_dir.mkdir(parents=True, exist_ok=True)

        # get all paths
        events_path = self.metadata_table["path"].tolist()

        # apply function to all of them
        progress_starmap(
            TransformationManager.apply_transformations_to_event,
            zip(
                events_path,
                repeat(output_dir),
                repeat(sample_interval_seconds),
                repeat(num_timesteps_for_window),
                repeat(avg_variable_mean),
                repeat(avg_variable_std_dev),
            ),
            total=len(events_path),
        )

    @staticmethod
    def apply_transformations_to_event(
        event_input_path: str,
        event_grandparent_output_dir: Path,
        sample_interval_seconds: int,
        num_timesteps_for_window: int,
        avg_variable_mean: pd.Series,
        avg_variable_std_dev: pd.Series,
        should_return: bool,
    ) -> None:
        """Processes a single event by imputing null values, standardizing, and creating time window samples.

        Parameters
        ----------
        event_input_path: str
            Path to the event data.
        event_grandparent_output_dir: Path
            The grandparent directory where the transformed event data will be stored.
        sample_interval_seconds: int
            Desired interval for downsampling the event data.
        num_timesteps_for_window: int
            Number of timesteps to use for creating windowed samples.
        avg_variable_mean: pd.Series
            Calculated mean values for each variable based on a sample of events.
        avg_variable_std_dev: pd.Series
            Calculated standard deviation values for each variable based on a sample of events.
        should_return: bool
            Flag indicating whether to return the windowed event data.
        """
        # get item
        event_input_path = Path(event_input_path)
        event = raw_data_acquisition.get_event(event_input_path)

        # skip event if its values aren't valid
        if not TransformationManager.is_event_values_valid(event):
            logging.info(
                f"Skipping the following event due to invalid data: {event_input_path}"
            )
            return

        # lower sample rate
        downampled_event = TransformationManager.transform_event_with_downsample(
            event, sample_interval_seconds
        )

        # imput item
        event_class_type = event_input_path.parent.stem
        imputed_event = TransformationManager.transform_event_with_imputation(
            downampled_event, event_class_type
        )

        # standardize item
        standardized_event = TransformationManager.transform_event_with_standardization(
            imputed_event, avg_variable_mean, avg_variable_std_dev
        )

        # sin and cos transformation of the time stamp regarding time of day and year
        # TODO implement

        # get time windows
        try:
            (
                windowed_event_X,
                windowed_event_y,
            ) = TransformationManager.transform_event_with_timestep_windows(
                standardized_event, num_timesteps_for_window
            )
        except Exception:
            raise ValueError(
                f"Exception while to split_sequences_into_windows. Path is: {event_input_path}"
            )

        if should_return:
            return windowed_event_X, windowed_event_y

        # store results
        file_name = event_input_path.stem
        output_path = (
            event_grandparent_output_dir
            / event_input_path.parent.stem
            / f"{file_name}.npz"
        )

        TransformationManager.store_pair_array(
            windowed_event_X, windowed_event_y, output_path
        )

    @staticmethod
    def store_pair_array(
        array_1: np.array, array_2: np.array, storage_file_path: Path
    ) -> None:
        """Joins and stores a pair of same size array to storage

        Parameters
        ----------
        array_1: np.array
            First array, represents X.
        array_2: np.array
            Second array, represents y.
        storage_file_path: Path
            Path where the array pair was stored.
        """

        np.savez(storage_file_path, X=array_1, y=array_2)

    @staticmethod
    def retrieve_pair_array(storage_file_path: Path) -> Tuple[np.array, np.array]:
        """Retrieves a previously stored array pair

        Parameters
        ----------
        storage_file_path: Path
            Path where the array pair was stored.

        Returns
        -------
        Tuple[np.array, np.array]
            Pair of arrays stored at file path. Represents [X, y].
        """

        arrays = np.load(storage_file_path)
        return arrays["X"], arrays["y"]

    @staticmethod
    def is_event_values_valid(event_data: pd.DataFrame) -> bool:
        is_num_attribs_valid = all(
            event_data[var].fillna(value=0).between(interval[0], interval[1]).all()
            for var, interval in TransformationManager.valid_num_attribs_range.items()
        )

        # check if class is in permanent (0-8) or transient regime (101-108)
        is_class_attrib_valid = (
            event_data[EventParameters.event_class_attrib]
            .fillna(value=0)
            .between(0, 8, inclusive="both")
            + event_data[EventParameters.event_class_attrib]
            .fillna(value=0)
            .between(101, 108, inclusive="both")
        ).all()

        is_timestamp_valid = (
            pd.to_datetime(
                event_data.index, format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )
            .notnull()
            .all()
        )
        return is_num_attribs_valid and is_class_attrib_valid and is_timestamp_valid

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

        event_data[EventParameters.event_num_attribs] = (
            event_data[EventParameters.event_num_attribs]
            .interpolate()
            .ffill()
            .bfill()
            .fillna(0)
        )
        event_data[EventParameters.event_class_attrib] = (
            event_data[EventParameters.event_class_attrib]
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

        event_data[EventParameters.event_num_attribs] = event_data[
            EventParameters.event_num_attribs
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

        resampled_numeric_data = (
            event_data[EventParameters.event_num_attribs]
            .resample(f"{sample_interval_seconds}s", origin="start", closed="left")
            .mean()
        )

        resampled_class_data = (
            event_data[EventParameters.event_class_attrib]
            .resample(f"{sample_interval_seconds}s", origin="start", closed="left")
            .min()
        )

        return pd.concat(
            [resampled_numeric_data, resampled_class_data],
            axis=1,
        )

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
        numeric_column_name_list = event_data[EventParameters.event_num_attribs].columns

        input_sequences = np.hstack(
            [
                np.array(event_data[c]).reshape((num_rows, 1))
                for c in numeric_column_name_list
            ]
        )
        output_sequece = np.array(event_data[EventParameters.event_class_attrib])

        return TransformationManager.split_sequences_into_windows(
            input_sequences, output_sequece, num_timesteps
        )

    @staticmethod
    def split_sequences_into_windows(
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
        y = np.array(y).astype(int)
        y[y >= 100] = y[y >= 100] - 100  # 100 is the constant for transient annomalies

        try:
            y = keras.utils.to_categorical(y, num_classes=len(EventClassType))
        except Exception:
            raise ValueError(
                f"Exception while to categorical. Present values: {np.unique(y)}"
            )

        return np.array(X), np.array(y)
