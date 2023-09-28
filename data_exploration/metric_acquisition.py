import sys

sys.path.append("..")  # Allows imports from sibling directories

from absl import logging
from parallelbar import progress_map, progress_starmap
from itertools import repeat
from typing import Callable
import pandas as pd
import numpy as np
import pickle

from constants import storage_config
from raw_data_manager.models import EventParameters


class MetricAcquisition:
    """Acquires metrics regarding a group of events."""

    def __init__(self, metadata_table: pd.DataFrame) -> None:
        """
        Initializes an instance of MetricAcquisition.

        Parameters
        ----------
        metadata_table: pd.DataFrame
            DataFrame containing metadata for events.
        """
        self.__metadata_table = metadata_table
        return

    def get_mean_and_std_metric(
        self, cache_file_name: str, use_cache: bool
    ) -> pd.DataFrame:
        """Returns global mean and standard deviation metrics for each variable.

        Parameters
        ----------
        cache_file_name: str
            The cache file name.
        use_cache: bool
            Flag indicating whether to use cached data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing mean and standard deviation metrics for each variable.
        """
        mean_of_means = self.save_retrieve_object(
            f"{cache_file_name}-mean_of_means",
            self.get_mean_of_means,
            [self.__metadata_table],
            use_cache,
        )
        logging.debug(f"Mean of means was adquired\n{mean_of_means}")

        processed_means = self.save_retrieve_object(
            f"{cache_file_name}-processed_means",
            self.get_table_all_columns_mean,
            [mean_of_means],
            use_cache,
        )
        logging.debug(f"Processed means was adquired\n{processed_means}")

        mean_of_stds = self.save_retrieve_object(
            f"{cache_file_name}-mean_of_stds",
            self.get_table_all_columns_std_dev,
            [self.__metadata_table, processed_means],
            use_cache,
        )
        logging.debug(f"Mean of stds was adquired\n{mean_of_stds}")

        processed_stds = self.save_retrieve_object(
            f"{cache_file_name}-processed_stds",
            self.get_mean_std_dev,
            [mean_of_stds],
            use_cache,
        )
        logging.debug(f"Processed stds was adquired\n{processed_stds}")

        pmm = processed_means.to_frame()
        pmm.columns = ["mean_of_means"]

        psm = processed_stds.to_frame()
        psm.columns = ["mean_of_stds"]

        return pd.concat([pmm, psm], axis=1)

    @staticmethod
    def get_mean_and_std_metric_from_cache(cache_file_name: str) -> pd.DataFrame:
        """Retrieve mean and standard deviation metrics from cache.

        Parameters
        ----------
        cache_file_name: str
            The cache file name.

        Returns
        -------
        pd.DataFrame
            DataFrame containing mean and standard deviation metrics for each variable.
        """
        processed_means = MetricAcquisition.save_retrieve_object(
            f"{cache_file_name}-processed_means",
            None,
            [None],
            True,
        )

        processed_stds = MetricAcquisition.save_retrieve_object(
            f"{cache_file_name}-processed_stds",
            None,
            [None],
            True,
        )

        pmm = processed_means.to_frame()
        pmm.columns = ["mean_of_means"]

        psm = processed_stds.to_frame()
        psm.columns = ["mean_of_stds"]

        return pd.concat([pmm, psm], axis=1)

    def process_event_mean(self, path: str) -> pd.Series:
        """Returns the mean of each variable for a single event.

        Parameters
        ----------
        path: str
            The path to the event data.

        Returns
        -------
        pd.Series
            Series containing the mean of each variable for the event.
        """
        event = pd.read_parquet(path)
        event_mean = event[EventParameters.event_num_attribs].mean()
        return event_mean

    def get_mean_of_means(self, events_metadata: pd.DataFrame) -> pd.DataFrame:
        """Generates a table concatenating each event's mean variables.

        Parameters
        ----------
        events_metadata: pd.DataFrame
            DataFrame containing event metadata.

        Returns
        -------
        pd.DataFrame
            DataFrame containing mean of means for each variable.
        """
        events_paths = events_metadata.path.tolist()
        event_num_means = progress_map(self.process_event_mean, events_paths)
        return pd.DataFrame(event_num_means)

    def get_column_mean(self, column: pd.Series, extreme_index_range: int) -> float:
        """Calculates the mean of a DataFrame column, after disregarding outliers.

        Parameters
        ----------
        column: pd.Series
            The column to calculate the mean for.
        extreme_index_range: int
            The range of extreme indices to disregard.

        Returns
        -------
        float
            The mean value of the column.
        """
        smallest_indexes = column.nsmallest(extreme_index_range).index.tolist()
        largest_indexes = column.nlargest(extreme_index_range).index.tolist()
        extreme_indexes = smallest_indexes + largest_indexes
        return column.drop(extreme_indexes).mean()

    def get_table_all_columns_mean(
        self, table: pd.DataFrame, extreme_index_range: int = 0
    ) -> pd.Series:
        """Calculates the mean of each variable of a DataFrame.

        Parameters
        ----------
        table: pd.DataFrame
            DataFrame containing event data.
        extreme_index_range: int, optional
            The range of extreme indices to disregard, by default 0.

        Returns
        -------
        pd.Series
            Series containing the mean of each variable.
        """
        return table.apply(
            self.get_column_mean, extreme_index_range=extreme_index_range
        )

    def get_column_std_dev(self, column: pd.Series, columns_mean: pd.Series) -> float:
        """Calculates the standard deviation for a column using processed_means for mean.

        Parameters
        ----------
        column: pd.Series
            The column to calculate the standard deviation for.
        columns_mean: pd.Series
            The mean values for each column.

        Returns
        -------
        float
            The standard deviation value for the column.
        """
        column_mean = float(columns_mean[column.name])
        diff = column - column_mean
        total_squared_diff = np.nansum(diff**2)

        count_non_null_items = np.count_nonzero(~np.isnan(column))
        divider = count_non_null_items - 1

        return np.sqrt(total_squared_diff / divider)

    def get_event_std_dev(self, path: str, columns_mean: pd.Series) -> pd.Series:
        """Calculates the standard deviation for each variable of an event.

        Parameters
        ----------
        path: str
            The path to the event data.
        columns_mean: pd.Series
            The mean values for each variable.

        Returns
        -------
        pd.Series
            Series containing the standard deviation for each variable.
        """
        event = pd.read_parquet(path)
        return event[EventParameters.event_num_attribs].apply(
            self.get_column_std_dev, columns_mean=columns_mean
        )

    def get_table_all_columns_std_dev(
        self, events_metadata: pd.DataFrame, columns_mean: pd.Series
    ) -> pd.DataFrame:
        """Generates a table with the standard deviation for each variable of each event.

        Parameters
        ----------
        events_metadata: pd.DataFrame
            DataFrame containing event metadata.
        columns_mean: pd.Series
            The mean values for each variable.

        Returns
        -------
        pd.DataFrame
            DataFrame containing standard deviation for each variable of each event.
        """
        events_paths = events_metadata.path.tolist()

        event_std_means = progress_starmap(
            self.get_event_std_dev,
            zip(events_paths, repeat(columns_mean)),
            total=len(events_paths),
        )
        return pd.DataFrame(event_std_means)

    def get_mean_std_dev(
        self, table: pd.DataFrame, extreme_index_range: int = 0
    ) -> pd.Series:
        """Calculates a mean value for the standard deviation of each variable.

        Parameters
        ----------
        table: pd.DataFrame
            DataFrame containing standard deviation data.
        extreme_index_range: int, optional
            The range of extreme indices to disregard, by default 0.

        Returns
        -------
        pd.Series
            Series containing the mean standard deviation for each variable.
        """
        return table.apply(
            self.get_column_mean, extreme_index_range=extreme_index_range
        )

    @staticmethod
    def save_retrieve_object(
        file_name: str, funct: Callable, arg_list: list = [], use_cached: bool = True
    ) -> any:
        """Intermediates a function call, its results can be quickly retrieved from cache.

        Parameters
        ----------
        file_name: str
            Name of the file to store the cached object.
        funct: Callable
            Function to be cached.
        arg_list: list, optional
            List of arguments for the function, by default [].
        use_cached: bool, optional
            Flag to use cached data if available, by default True.

        Returns
        -------
        any
            Cached result of the function.
        """
        full_path = storage_config.DIR_SAVED_OBJECTS / f"{file_name}.pickle"
        if (not use_cached) or (not full_path.exists()):
            full_path.unlink(missing_ok=True)

            result = funct(*arg_list)

            full_path.parent.mkdir(parents=True, exist_ok=True)
            dbfile = open(full_path, "ab")
            pickle.dump(result, dbfile)
            dbfile.close()

        # load saved data
        dbfile = open(full_path, "rb")
        result = pickle.load(dbfile)
        dbfile.close()
        return result
