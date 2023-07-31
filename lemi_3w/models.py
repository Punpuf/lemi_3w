from dataclasses import dataclass
import pandas as pd
from enum import Enum


@dataclass
class EventMetadata:
    """Class representing the metadata of an event in the 3W dataset."""

    hash_id: str = None
    class_type: str = None
    source: str = None
    well_id: int = None
    path: str = None
    timestamp: pd.Timestamp = None
    file_size: str = None
    num_timesteps: int = None

    def to_dict(self):
        return {
            "hash_id": self.hash_id,
            "class_type": self.class_type,
            "source": self.source,
            "well_id": self.well_id,
            "path": self.path,
            "timestamp": self.timestamp,
            "file_size": self.file_size,
            "num_timesteps": self.num_timesteps,
        }

    def __eq__(self, other):
        if not isinstance(other, EventMetadata):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            self.hash_id == other.hash_id
            and self.class_type == other.class_type
            and self.source == other.source
            and self.well_id == other.well_id
            and self.path == other.path
            and self.timestamp == other.timestamp
            and self.file_size == other.file_size
            and self.num_timesteps == other.num_timesteps
        )


class EventSource(Enum):
    REAL = 0
    SIMULATED = 1
    HAND_DRAWN = 2


class EventClassType(Enum):
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
