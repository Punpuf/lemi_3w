from lemi_3w.raw_data_inspector import RawDataInspector, __get_anomaly_metadata
from lemi_3w.models import EventMetadata, EventSource, EventClassType
from constants import utils
from pathlib import Path
import tempfile
import pandas as pd
import shutil
from absl import logging


def get_dummy_data() -> pd.DataFrame:
    # Create dummy data
    data = {
        "timestamp": ["20140124093303", "20140124093304", "20140124093305"],
        "B": [4, 5, 6],
    }
    return pd.DataFrame(data)


def test_get_anomaly_metadata():
    # Create dummy data
    df = get_dummy_data()
    temp_dir = tempfile.mkdtemp()

    # Create temporary anomaly files for each sourcy type
    anomaly_class_type = 7
    anomaly_parent_dir = Path(temp_dir) / str(anomaly_class_type)
    anomaly_parent_dir.mkdir()

    anomaly_file_real = anomaly_parent_dir / "WELL-00001_20140124093303.parquet"
    anomaly_file_simu = anomaly_parent_dir / "SIMULATED_00103.parquet"
    anomaly_file_draw = anomaly_parent_dir / "DRAWN_00004.parquet"
    df.to_parquet(anomaly_file_real, index=False)
    df.to_parquet(anomaly_file_simu, index=False)
    df.to_parquet(anomaly_file_draw, index=False)
    file_size = anomaly_file_simu.stat().st_size

    # Call the function and check if it returns an instance of EventMetadata
    event_metadata_real = __get_anomaly_metadata(anomaly_file_real, anomaly_class_type)
    event_metadata_simu = __get_anomaly_metadata(anomaly_file_simu, anomaly_class_type)
    event_metadata_draw = __get_anomaly_metadata(anomaly_file_draw, anomaly_class_type)

    assert isinstance(event_metadata_real, EventMetadata)
    assert isinstance(event_metadata_simu, EventMetadata)
    assert isinstance(event_metadata_draw, EventMetadata)

    # Check if metadata contents are correct
    expected_metadata_real = EventMetadata(
        hash_id=utils.sha256sum(f"{anomaly_class_type}/{anomaly_file_real.stem}"),
        class_type=EventClassType(anomaly_class_type).name,
        source=EventSource.REAL.name,
        well_id=1,
        path=str(anomaly_file_real),
        timestamp=pd.Timestamp("20140124093303"),
        file_size=file_size,
        num_timesteps=3,
    )
    expected_metadata_simu = EventMetadata(
        hash_id=utils.sha256sum(f"{anomaly_class_type}/{anomaly_file_simu.stem}"),
        class_type=EventClassType(anomaly_class_type).name,
        source=EventSource.SIMULATED.name,
        path=str(anomaly_file_simu),
        file_size=file_size,
        num_timesteps=3,
    )
    expected_metadata_draw = EventMetadata(
        hash_id=utils.sha256sum(f"{anomaly_class_type}/{anomaly_file_draw.stem}"),
        class_type=EventClassType(anomaly_class_type).name,
        source=EventSource.HAND_DRAWN.name,
        path=str(anomaly_file_draw),
        file_size=file_size,
        num_timesteps=3,
    )

    assert event_metadata_real == expected_metadata_real
    assert event_metadata_simu == expected_metadata_simu
    assert event_metadata_draw == expected_metadata_draw

    # Clean up the temporary anomaly file
    anomaly_file_real.unlink()
    anomaly_file_simu.unlink()
    anomaly_file_draw.unlink()
    shutil.rmtree(temp_dir)


def test_get_metadata_table_no_data():
    # Create a temporary directory for dataset_path and cache_path
    anomaly_parent_dir = Path(tempfile.mkdtemp())
    dataset_path = anomaly_parent_dir / "temp_dataset_path"
    cache_path = anomaly_parent_dir / "temp_cache_dir"

    dataset_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Initialize the RawDataInspector with temporary paths
    inspector = RawDataInspector(dataset_path, cache_path, use_cached=False)
    metadata_table = inspector.get_metadata_table()

    # Check if the returned table is a pandas DataFrame
    assert isinstance(metadata_table, pd.DataFrame)
    assert metadata_table.size == 0

    # Clean up the temporary directory and cache file
    shutil.rmtree(anomaly_parent_dir)


def test_get_metadata_table_with_data():
    logging.set_verbosity(logging.DEBUG)

    # Create a temporary directory for dataset_path and cache_path
    temp_dir = Path(tempfile.mkdtemp())
    dataset_path = temp_dir / "temp_dataset_path"
    cache_path = temp_dir / "temp_cache_dir" / "data_inspector_cache.parquet"

    # Create a test anomaly folder with an anomaly file]
    anomaly_class_type1 = 0
    anomaly_class_type2 = 4
    anomaly_class_type3 = 5

    anomaly_folder1 = dataset_path / str(anomaly_class_type1)
    anomaly_folder2 = dataset_path / str(anomaly_class_type2)
    anomaly_folder1.mkdir(parents=True, exist_ok=True)
    anomaly_folder2.mkdir(parents=True, exist_ok=True)

    anomaly_file_real1 = anomaly_folder1 / "WELL-00001_20140124093303.parquet"
    anomaly_file_real2 = anomaly_folder2 / "WELL-00002_20140124093303.parquet"
    anomaly_file_simu1 = anomaly_folder1 / "SIMULATED_00101.parquet"
    anomaly_file_simu2 = anomaly_folder2 / "SIMULATED_00102.parquet"
    anomaly_file_drawn = anomaly_folder1 / "DRAWN_00004.parquet"

    df = get_dummy_data()
    df.to_parquet(anomaly_file_real1)
    df.to_parquet(anomaly_file_real2)
    df.to_parquet(anomaly_file_simu1)
    df.to_parquet(anomaly_file_simu2)
    df.to_parquet(anomaly_file_drawn)

    # Initialize the RawDataInspector with temporary paths
    inspector = RawDataInspector(dataset_path, cache_path, use_cached=False)

    # Test filtering by type and source
    metadata_table = inspector.get_metadata_table()

    # Check if the returned table is a pandas DataFrame
    assert isinstance(metadata_table, pd.DataFrame)
    assert metadata_table.shape[0] == 5

    assert inspector.get_metadata_table(sources=[EventSource.REAL.name]).shape[0] == 2
    assert (
        inspector.get_metadata_table(sources=[EventSource.SIMULATED.name]).shape[0] == 2
    )
    assert (
        inspector.get_metadata_table(sources=[EventSource.HAND_DRAWN.name]).shape[0]
        == 1
    )

    assert (
        inspector.get_metadata_table(
            class_types=[EventClassType(anomaly_class_type1).name]
        ).shape[0]
        == 3
    )
    assert (
        inspector.get_metadata_table(
            class_types=[EventClassType(anomaly_class_type2).name]
        ).shape[0]
        == 2
    )
    assert (
        inspector.get_metadata_table(
            class_types=[EventClassType(anomaly_class_type3).name]
        ).shape[0]
        == 0
    )

    # Clean up the temporary directory and cache file
    shutil.rmtree(dataset_path)
    shutil.rmtree(cache_path.parent)
