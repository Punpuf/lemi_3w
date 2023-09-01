import sys

sys.path.append("..")  # Allows imports from sibling directories

import pandas as pd
import numpy as np
from absl import logging
from parallelbar import progress_map, progress_starmap
from itertools import repeat
from constants import module_constants, utils


class MetricAcquisition:
    """Acquires metrics regarding a group of events."""

    def __init__(self, metadata_table: pd.DataFrame) -> None:
        self.__metadata_table = metadata_table
        return

    def get_mean_and_std_metric(
        self, cache_file_name: str, use_cache: bool
    ) -> pd.DataFrame:
        """Returns global mean and standard deviation metrics for each variable."""
        mean_of_means = utils.save_retrieve_object(
            f"{cache_file_name}-mean_of_means",
            self.get_mean_of_means,
            [self.__metadata_table],
            use_cache,
        )
        logging.debug(f"Mean of means was adquired\n{mean_of_means}")

        processed_means = utils.save_retrieve_object(
            f"{cache_file_name}-processed_means",
            self.get_table_all_columns_mean,
            [mean_of_means],
            use_cache,
        )
        logging.debug(f"Processed means was adquired\n{processed_means}")

        mean_of_stds = utils.save_retrieve_object(
            f"{cache_file_name}-mean_of_stds",
            self.get_table_all_columns_std_dev,
            [self.__metadata_table, processed_means],
            use_cache,
        )
        logging.debug(f"Mean of stds was adquired\n{mean_of_stds}")

        processed_stds = utils.save_retrieve_object(
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

    def process_event_mean(self, path: str) -> pd.Series:
        """Returns the mean of each variable, for a single event."""
        event = pd.read_parquet(path)
        event_mean = event[module_constants.event_num_attribs].mean()
        return event_mean

    def get_mean_of_means(self, events_metadata: pd.DataFrame) -> pd.DataFrame:
        """Generates table concating each event's mean variables."""
        events_paths = events_metadata.path.tolist()
        event_num_means = progress_map(self.process_event_mean, events_paths)
        return pd.DataFrame(event_num_means)

    def get_column_mean(self, column: pd.Series, extreme_index_range: int) -> float:
        """Calculates the mean of a DataFrame column, after disconsidering outliers."""
        smallest_indexes = column.nsmallest(extreme_index_range).index.tolist()
        largest_indexes = column.nlargest(extreme_index_range).index.tolist()
        extreme_indexes = smallest_indexes + largest_indexes
        return column.drop(extreme_indexes).mean()

    def get_table_all_columns_mean(
        self, table: pd.DataFrame, extreme_index_range: int = 7
    ) -> pd.Series:
        """Calculates the mean of each variable of a DataFrame."""
        return table.apply(
            self.get_column_mean, extreme_index_range=extreme_index_range
        )

    def get_column_std_dev(self, column: pd.Series, columns_mean: pd.Series) -> float:
        """Calculates the std. dev. for a column using processed_means_ for mean."""
        column_mean = float(columns_mean[column.name])
        diff = column - column_mean
        total_squared_diff = np.nansum(diff**2)

        count_non_null_items = np.count_nonzero(~np.isnan(column))
        divider = count_non_null_items - 1

        return np.sqrt(total_squared_diff / divider)

    def get_event_std_dev(self, path: str, columns_mean: pd.Series) -> pd.Series:
        """Calculates the std. dev. for each variable of an event."""
        event = pd.read_parquet(path)
        return event[module_constants.event_num_attribs].apply(
            self.get_column_std_dev, columns_mean=columns_mean
        )

    def get_table_all_columns_std_dev(
        self, events_metadata: pd.DataFrame, columns_mean: pd.Series
    ) -> pd.DataFrame:
        """Generates table with the std. dev. for each variable of each event."""
        events_paths = events_metadata.path.tolist()

        event_std_means = progress_starmap(
            self.get_event_std_dev,
            zip(events_paths, repeat(columns_mean)),
            total=len(events_paths),
        )
        return pd.DataFrame(event_std_means)

    def get_mean_std_dev(
        self, table: pd.DataFrame, extreme_index_range: int = 7
    ) -> pd.Series:
        """Calculates a mean value for the std. dev. of each variable."""
        return table.apply(
            self.get_column_mean, extreme_index_range=extreme_index_range
        )
