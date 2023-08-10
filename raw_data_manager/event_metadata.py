from dataclasses import dataclass
from enum import Enum
import pandas as pd
import hashlib


class EventSource(Enum):
    REAL = 0
    SIMULATED = 1
    HAND_DRAWN = 2


class EventType(Enum):
    NORMAL = 0
    ABRUPT_INCREASE_BSW = 1
    SPURIOUS_CLOSURE_DHSV = 2
    SEVERE_SLUGGING = 3
    FLOW_INSTABILITY = 4
    RAPID_PRODUCTIVITY_LOSS = 5
    QUICK_RESTRICTION_PCK = 6
    SCALING_IN_PCK = 7
    HYDRATE_IN_PRODUCTION_LINE = 8

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return self.value <= other


@dataclass
class EventMetadata:
    """Class representing the metadata of an event in the 3W dataset."""

    hash_id: str = None
    type_: str = None
    source: str = None
    well_id: int = None
    path: str = None
    timestamp: pd.Timestamp = None
    file_size: str = None
    num_timesteps: int = None

    def to_dict(self):
        return {
            "hash_id": self.hash_id,
            "type_": self.type_,
            "source": self.source,
            "well_id": self.well_id,
            "path": self.path,
            "timestamp": self.timestamp,
            "file_size": self.file_size,
            "num_timesteps": self.num_timesteps,
        }


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


class EventMetadataAccess:
    __CACHE_PATH__ = f"{path_saved}/cache_metadata_access_"

    def __init__(self, dataset_path, use_cached=True):
        self.dataset_path = dataset_path
        self.cache_path_full = EventMetadataAccess.__CACHE_PATH__ + self.dataset_path
        self.events_metadata = self.__load_data(use_cached)
        return

    def __load_data(self, use_cached):
        if (use_cached) and (os.path.exists(self.cache_path_full)):
            dbfile = open(self.cache_path_full, "rb")
            cached_data = pickle.load(dbfile)
            dbfile.close()
            return cached_data

        events = []

        for item in pathlib.Path(self.dataset_path).iterdir():
            if not item.stem.isnumeric():
                continue

            for subitem in item.iterdir():
                if not subitem.suffix == ".feather":
                    continue

                event_metadata = EventMetadata(
                    type_=EventType(int(item.stem)).name, path=str(subitem)
                )
                event_source = re.search("^[^_]+(?=_)", subitem.stem)[0]

                event_metadata.hash_id = sha256sum(str(subitem))
                event_metadata.file_size = subitem.stat().st_size
                event_metadata.num_timesteps = get_event(str(subitem)).shape[0] - 1

                if event_source.startswith("WELL"):
                    event_metadata.source = EventSource.REAL.name
                    event_metadata.well_id = int(event_source.removeprefix("WELL-"))
                    event_metadata.timestamp = pd.Timestamp(
                        subitem.stem.removeprefix(f"{event_source}_")
                    )
                elif event_source.startswith("SIMULATED"):
                    event_metadata.source = EventSource.SIMULATED.name
                elif event_source.startswith("DRAWN"):
                    event_metadata.source = EventSource.HAND_DRAWN.name

                events.append(event_metadata)

        events_df = pd.DataFrame.from_records([e.to_dict() for e in events])
        events_df.type_ = pd.Categorical(
            events_df.type_, categories=[e.name for e in EventType], ordered=True
        )
        events_df.source = pd.Categorical(
            events_df.source, categories=[e.name for e in EventSource], ordered=True
        )

        events_df["hash_id"] = events_df["hash_id"].str.slice(stop=6)

        result_data = events_df.set_index("hash_id", drop=True)

        # save result to cache
        os.makedirs(path_saved, exist_ok=True)
        dbfile = open(self.cache_path_full, "ab")
        pickle.dump(result_data, dbfile)
        dbfile.close()

        return result_data

    def load_data(self, types=None, sources=None):
        filtered_data = self.events_metadata

        if types is not None:
            filtered_data = filtered_data[filtered_data.type_.isin(types)]

        if sources is not None:
            filtered_data = filtered_data[filtered_data.source.isin(sources)]

        return filtered_data
