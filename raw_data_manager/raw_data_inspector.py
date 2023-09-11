import pandas as pd
from constants import utils
from raw_data_manager.models import EventMetadata, EventSource, EventClassType
import re
import pathlib
import parallelbar
from itertools import repeat
from absl import logging


PARQUET_EXTENSION = ".parquet"
NUMPY_ZIP_EXTENTION = ".npz"
HASH_LENGTH = 7


class RawDataInspector:
    """Manages metadata aquisition and its internal organization. Includes caching system."""

    def __init__(
        self,
        dataset_dir: pathlib.Path,
        cache_file_path: pathlib.Path,
        use_cached: bool = True,
    ):
        self.__dataset_dir = dataset_dir
        self.__cache_file_path = cache_file_path
        self.__events_metadata = self.__load_data(use_cached)
        return

    def __process_data(self):
        """Maps all metadata in the given directory, caching result as a DataFrame"""

        events = []

        # Cycles through all directories mapping event files
        for anomaly_folder in pathlib.Path(self.__dataset_dir).iterdir():
            if not anomaly_folder.stem.isnumeric():
                continue

            anomaly_files = list(anomaly_folder.iterdir())
            total_tasks = len(anomaly_files)
            logging.info(
                f"Processing {total_tasks} events of class type {anomaly_folder.stem}."
            )

            if total_tasks == 0:
                continue

            events.append(
                parallelbar.progress_starmap(
                    get_anomaly_metadata,
                    zip(anomaly_folder.iterdir(), repeat(int(anomaly_folder.stem))),
                    total=total_tasks,
                )
            )

        if len(events) == 0:
            logging.warning("Found no event files.")
            return

        logging.info(f"Found {len(events)}.")

        # Apply processing to all event metadata entries
        flatten_event = [item for sublist in events for item in sublist]

        events_df = pd.DataFrame.from_records([e.to_dict() for e in flatten_event])
        events_df.class_type = pd.Categorical(
            events_df.class_type,
            categories=[e.name for e in EventClassType],
            ordered=True,
        )
        events_df.source = pd.Categorical(
            events_df.source, categories=[e.name for e in EventSource], ordered=True
        )

        events_df["hash_id"] = events_df["hash_id"].str.slice(stop=HASH_LENGTH)
        result_data = events_df.set_index("hash_id", drop=True)

        # save result to cache
        self.__cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        result_data.to_parquet(self.__cache_file_path)

    def __load_data(self, use_cached):
        """Returns data, loading new data if not available or if resquested"""

        # Process data and save to cache if needed
        if (not use_cached) or (not self.__cache_file_path.is_file()):
            self.__process_data()

        # Return cached data
        return pd.read_parquet(self.__cache_file_path)

    def get_metadata_table(
        self,
        class_types: list[EventClassType] = None,
        sources: list[EventSource] = None,
        well_ids: list[int] = None,
    ) -> pd.DataFrame:
        """Returns DataFrame with metadata, optionally filtered by class type or source"""

        filtered_data = self.__events_metadata

        if class_types is not None:
            filtered_data = filtered_data[filtered_data.class_type.isin(class_types)]

        if sources is not None:
            filtered_data = filtered_data[filtered_data.source.isin(sources)]

        if well_ids is not None:
            filtered_data = filtered_data[filtered_data.well_id.isin(well_ids)]

        return filtered_data

    @staticmethod
    def generate_table_by_anomaly_source(metadata_table: pd.DataFrame) -> pd.DataFrame:
        """Given metadata table, returns metrics relating anomaly and source type"""
        anomaly = []
        real_count = []
        simul_count = []
        drawn_count = []
        soma = []

        for anomaly_type in EventClassType:
            anomaly.append(anomaly_type.name)

            real_count.append(
                len(
                    metadata_table[
                        (metadata_table["class_type"] == anomaly_type.name)
                        & (metadata_table["source"] == "REAL")
                    ]
                )
            )
            simul_count.append(
                len(
                    metadata_table[
                        (metadata_table["class_type"] == anomaly_type.name)
                        & (metadata_table["source"] == "SIMULATED")
                    ]
                )
            )
            drawn_count.append(
                len(
                    metadata_table[
                        (metadata_table["class_type"] == anomaly_type.name)
                        & (metadata_table["source"] == "HAND_DRAWN")
                    ]
                )
            )
            soma.append(
                len(metadata_table[metadata_table["class_type"] == anomaly_type.name])
            )

        anomaly.append("Total")
        real_count.append(sum(real_count))
        simul_count.append(sum(simul_count))
        drawn_count.append(sum(drawn_count))
        soma.append(sum(soma))

        data = {
            "anomaly": anomaly,
            "real_count": real_count,
            "simul_count": simul_count,
            "drawn_count": drawn_count,
            "soma": soma,
        }

        # Create the DataFrame
        df_source = pd.DataFrame(data)
        df_source.set_index("anomaly", inplace=True)
        return df_source


def get_anomaly_metadata(anomaly_file: pathlib.Path, class_type: int) -> EventMetadata:
    """Returns EventMetadata for a single event given its path"""

    if anomaly_file.suffix == PARQUET_EXTENSION:
        return get_parquet_metadata(anomaly_file, class_type)

    if anomaly_file.suffix == NUMPY_ZIP_EXTENTION:
        return get_numpy_zip_metadata(anomaly_file, class_type)

    return


def get_parquet_metadata(anomaly_file: pathlib.Path, class_type: int) -> EventMetadata:
    """Returns EventMetadata for a single event given its path"""

    event_metadata = EventMetadata(
        class_type=EventClassType(class_type).name,
        path=str(anomaly_file),
    )
    event_source = re.search("^[^_]+(?=_)", anomaly_file.stem)[0]

    event_metadata.hash_id = utils.sha256sum(
        f"{anomaly_file.parent.stem}/{anomaly_file.stem}"
    )
    event_metadata.file_size = anomaly_file.stat().st_size
    # TODO, check if - 1 is needed
    event_metadata.num_timesteps = utils.get_event(str(anomaly_file)).shape[0]

    if event_source.startswith("WELL"):
        event_metadata.source = EventSource.REAL.name
        event_metadata.well_id = int(event_source.removeprefix("WELL-"))
        event_metadata.timestamp = pd.Timestamp(
            anomaly_file.stem.removeprefix(f"{event_source}_")
        )
    elif event_source.startswith("SIMULATED"):
        event_metadata.source = EventSource.SIMULATED.name
    elif event_source.startswith("DRAWN"):
        event_metadata.source = EventSource.HAND_DRAWN.name

    return event_metadata


def get_numpy_zip_metadata(
    anomaly_file: pathlib.Path, class_type: int
) -> EventMetadata:
    """Returns EventMetadata for a single event given its path"""

    event_metadata = EventMetadata(
        class_type=EventClassType(class_type).name,
        path=str(anomaly_file),
    )
    event_source = re.search("^[^_]+(?=_)", anomaly_file.stem)[0]

    event_metadata.hash_id = utils.sha256sum(
        f"{anomaly_file.parent.stem}/{anomaly_file.stem}"
    )
    event_metadata.file_size = anomaly_file.stat().st_size
    # TODO, check if - 1 is needed
    # event_metadata.num_timesteps = utils.get_event(str(anomaly_file)).shape[0]

    if event_source.startswith("WELL"):
        event_metadata.source = EventSource.REAL.name
        event_metadata.well_id = int(event_source.removeprefix("WELL-"))
        event_metadata.timestamp = pd.Timestamp(
            anomaly_file.stem.removeprefix(f"{event_source}_")
        )
    elif event_source.startswith("SIMULATED"):
        event_metadata.source = EventSource.SIMULATED.name
    elif event_source.startswith("DRAWN"):
        event_metadata.source = EventSource.HAND_DRAWN.name

    return event_metadata
