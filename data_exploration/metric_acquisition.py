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
        pass

    def get_mean_and_std_metric(
        self, cache_file_name: str, use_cache: bool
    ) -> pd.DataFrame:
        mean_of_means = utils.save_retrieve_object(
            f"{cache_file_name}-mean_of_means",
            self.get_mean_of_means,
            [self.__metadata_table],
            use_cache,
        )
        logging.debug("Mean of means was adquired")
        logging.debug(mean_of_means)

        processed_means = utils.save_retrieve_object(
            f"{cache_file_name}-processed_means",
            self.process_mean_all_columns,
            [mean_of_means],
            use_cache,
        )
        logging.debug("Processed means was adquired")
        logging.debug(processed_means)

        mean_of_stds = utils.save_retrieve_object(
            f"{cache_file_name}-mean_of_stds",
            self.get_mean_of_stds,
            [self.__metadata_table, processed_means],
            use_cache,
        )
        logging.debug("Mean of stds was adquired")
        logging.debug(mean_of_stds)

        processed_stds = utils.save_retrieve_object(
            f"{cache_file_name}-processed_stds",
            self.process_std_all_columns,
            [mean_of_stds],
            use_cache,
        )
        logging.debug("Processed stds was adquired")
        logging.debug(processed_stds)

        pmm = processed_means.to_frame()
        pmm.columns = ["mean_of_means"]

        psm = processed_stds.to_frame()
        psm.columns = ["mean_of_stds"]

        return pd.concat([pmm, psm], axis=1)

    def process_event_mean(self, path):
        # print(f"process_event_mean path is {path}")
        event = pd.read_parquet(path)
        event_mean = event[module_constants.event_num_attribs].mean()
        return event_mean

    def get_mean_of_means(self, events_metadata):
        events_paths = events_metadata.path.tolist()
        event_num_means = progress_map(self.process_event_mean, events_paths)
        return pd.DataFrame(event_num_means)

    def process_mean_column(self, column, extreme_index_range):
        smallest_indexes = column.nsmallest(extreme_index_range).index.tolist()
        largest_indexes = column.nlargest(extreme_index_range).index.tolist()
        extreme_indexes = smallest_indexes + largest_indexes
        return column.drop(extreme_indexes).mean()

    def process_mean_all_columns(self, df, extreme_index_range=0):
        return df.apply(
            self.process_mean_column, extreme_index_range=extreme_index_range
        )

    """
    Conjundo de funções responsáveis por calcular o desvio padrão das variáveis numéricas.
    É descosiderado x=7 valores extremos "globais" no cálculo de cada variável (outliers).
    Usado para a padronização.
    """

    def get_event_column_std(self, column, processed_means_):
        return np.sqrt(np.sum(column - processed_means_[column.name]) ** 2) / (
            len(column) - 1
        )

    def process_event_std(self, path, processed_means_):
        event = pd.read_parquet(path)
        return event[module_constants.event_num_attribs].apply(
            self.get_event_column_std, processed_means_=processed_means_
        )

    def get_mean_of_stds(self, events_metadata, processed_means_):
        events_paths = events_metadata.path.tolist()
        event_std_means = progress_starmap(
            self.process_event_std,
            zip(events_paths, repeat(processed_means_)),
            total=len(events_paths),
        )
        return pd.DataFrame(event_std_means)

    def process_std_column(self, column, extreme_index_range):
        smallest_indexes = column.nsmallest(extreme_index_range).index.tolist()
        largest_indexes = column.nlargest(extreme_index_range).index.tolist()
        extreme_indexes = smallest_indexes + largest_indexes

        return column.drop(extreme_indexes).mean()

    def process_std_all_columns(self, df, extreme_index_range=7):
        return df.apply(
            self.process_std_column, extreme_index_range=extreme_index_range
        )
