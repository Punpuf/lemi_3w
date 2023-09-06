# class that receives metadata table

# can call a prepration function
# fc will make a folder into which it will save the files after applying transformations
# fc will run a function over all the paths

# each processing function will apply the 3 transformation functions
# then save file to path
from absl import logging
import pandas as pd
import numpy as np
from pathlib import Path
from constants import module_constants


class TransformationManager:
    def __init__(self, metadata_table: pd.DataFrame, folder_name: str) -> None:
        self.__metadata_table = metadata_table
        self.folder_name = folder_name
        logging.debug(
            f"""TransformationManager initialized with {metadata_table.shape[0]} items.
            Folder name is {folder_name}."""
        )
        return

    def tranform_table_with_imput_std_time(self, output_parent_dir: Path) -> None:
        TRANSFORMATION_NAME_PREFIX = "transform-imp-std_tim-"
        output_dir: Path = (
            output_parent_dir / TRANSFORMATION_NAME_PREFIX + self.folder_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # get all paths
        # apply function to all of them

    @staticmethod
    def transform_event_with_imput_std_time(
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

        Explain how it will handle null values.

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
    ) -> pd.DataFrame:
        # input data: multivariate
        # output: X window of multiple steps of multivariate data
        # y is class of current step
        X = get_sequence_in_from_event(event_data)
        y = get_prediction_out_from_event(event_data)

        # if y is None:
        #     print('there was error, bad class data, with event @ ', event_path)
        #     return

        X_split, y_split = split_sequences(X, y, num_timesteps)

        new_path = event_path.replace(input_folder, output_folder).replace(
            ".feather", ".npy"
        )


# input is event data
# output is the data of features that will be used for making predictions
# output is formatted for split_sequences function
def get_sequence_in_from_event(event):
    in_array_list = []
    x = 1
    for col_name, col_data in event[module_constants.event_num_attribs].iteritems():
        in_seq = np.array(col_data)
        in_array_list.append(in_seq.reshape((len(in_seq), 1)))

    return np.hstack(tuple(in_array_list))


# input is event data
# output is the data of targets that will be used for verifying predictions
# output is formatted for split_sequences function
def get_prediction_out_from_event(event):
    out_seq = np.array(event[module_constants.event_class_attrib])
    out_seq = out_seq.reshape((len(out_seq), 1))
    out_seq[out_seq >= 100] -= 100

    # bad class input data
    # if max(out_seq) > len(out_seq):
    #     return None

    out_seq = to_categorical(out_seq, num_classes)
    return out_seq


# split a multivariate sequence into samples
# sequence_in: array in the format [[a, b, c], [a, b, c], ...]
# prediction_out: [[1., ..., 0.], [1., ..., 0.]]
def split_sequences(sequence_in, prediction_out, n_steps):
    X, y = list(), list()
    for i in range(len(sequence_in)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequence_in):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence_in[i:end_ix,], prediction_out[end_ix - 1, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
