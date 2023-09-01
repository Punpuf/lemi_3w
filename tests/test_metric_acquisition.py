import pandas as pd
import numpy as np
from unittest.mock import patch
import pytest
from data_exploration.metric_acquisition import MetricAcquisition


@pytest.fixture
def sample_metadata_table():
    data = {
        "hash_id": ["74203bb", "9fbd6f9", "28804c5"],
        "class_type": ["NORMAL", "NORMAL", "NORMAL"],
        "source": ["REAL", "REAL", "REAL"],
        "well_id": [1.0, 2.0, 6.0],
        "timestamp": [
            "2017-05-24 03:00:00",
            "2017-08-09 06:00:00",
            "2017-05-08 09:00:31",
        ],
        "path": [
            "event1.parquet",
            "event2.parquet",
            "event3.parquet",
        ],
        "file_size": [491415, 520154, 349162],
        "num_timesteps": [17885, 17933, 17970],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_utils():
    with patch("your_module.utils") as mock_utils:
        yield mock_utils


@pytest.fixture
def mock_pd_read_parquet():
    with patch("pandas.read_parquet") as mock_read_parquet:

        def mock_function(path, **kwargs):
            if path == "event1.parquet":
                return pd.DataFrame(
                    {
                        "P-PDG": [1, 2, 3],
                        "P-TPT": [4, 5, 6],
                        "T-TPT": [7, 8, 9],
                        "P-MON-CKP": [10, 11, 12],
                        "T-JUS-CKP": [13, 14, 15],
                        "P-JUS-CKGL": [16, 17, 18],
                        "QGL": [19, 20, 21],
                    }
                )
            elif path == "event2.parquet":
                return pd.DataFrame(
                    {
                        "P-PDG": [2, 3, 4],
                        "P-TPT": [5, 6, 7],
                        "T-TPT": [8, 9, 10],
                        "P-MON-CKP": [11, 12, 13],
                        "T-JUS-CKP": [14, 15, 16],
                        "P-JUS-CKGL": [np.nan, np.nan, np.nan],
                        "QGL": [np.nan, np.nan, np.nan],
                    }
                )

            elif path == "event3.parquet":
                return pd.DataFrame(
                    {
                        "P-PDG": [3, 4, 5],
                        "P-TPT": [6, 7, 8],
                        "T-TPT": [9, 10, 11],
                        "P-MON-CKP": [12, 13, 14],
                        "T-JUS-CKP": [15, 16, 17],
                        "P-JUS-CKGL": [np.nan, np.nan, 19],
                        "QGL": [21, 22, 23],
                    }
                )

            else:
                raise ValueError(f"Requested path ({path}) not in test section.")

        mock_read_parquet.side_effect = mock_function
        yield mock_read_parquet


def test_process_event_mean(mock_pd_read_parquet, sample_metadata_table):
    metric_acquisition = MetricAcquisition(sample_metadata_table)
    result = metric_acquisition.process_event_mean("event1.parquet")

    assert result["P-PDG"] == pytest.approx(2.0)
    assert result["P-TPT"] == pytest.approx(5.0)
    assert result["T-TPT"] == pytest.approx(8.0)
    assert result["P-MON-CKP"] == pytest.approx(11.0)
    assert result["T-JUS-CKP"] == pytest.approx(14.0)
    assert result["P-JUS-CKGL"] == pytest.approx(17.0)
    assert result["QGL"] == pytest.approx(20.0)


def test_get_mean_of_means(mock_pd_read_parquet, sample_metadata_table):
    metric_acquisition = MetricAcquisition(sample_metadata_table)
    result = metric_acquisition.get_mean_of_means(sample_metadata_table)

    assert result["P-PDG"].tolist() == pytest.approx([2, 3, 4])
    assert result["P-TPT"].tolist() == pytest.approx([5, 6, 7])
    assert result["T-TPT"].tolist() == pytest.approx([8, 9, 10])
    assert result["P-MON-CKP"].tolist() == pytest.approx([11, 12, 13])
    assert result["T-JUS-CKP"].tolist() == pytest.approx([14, 15, 16])

    for i, (obt, exp) in enumerate(zip(result["T-JUS-CKP"].tolist(), [14, 15, 16])):
        if np.isnan(exp):
            assert np.isnan(obt)
        else:
            assert obt == pytest.approx(exp)

    for i, (obt, exp) in enumerate(zip(result["QGL"].tolist(), [20.0, np.nan, 22.0])):
        if np.isnan(exp):
            assert np.isnan(obt)
        else:
            assert obt == pytest.approx(exp)


def test_get_table_column_mean():
    metric_acquisition = MetricAcquisition(sample_metadata_table)

    column1 = pd.Series([1, 2, 3, 4, 5])
    result1 = metric_acquisition.get_column_mean(column1, extreme_index_range=0)

    column2 = pd.Series([-10, 2, 3, 4, 50])
    result2 = metric_acquisition.get_column_mean(column2, extreme_index_range=1)

    column3 = pd.Series([2, 3, 4, np.nan])
    result3 = metric_acquisition.get_column_mean(column3, extreme_index_range=1)

    assert result1 == pytest.approx(3.0)
    assert result2 == pytest.approx(3.0)
    assert result3 == pytest.approx(3.0)


def test_get_table_all_columns_mean():
    metric_acquisition = MetricAcquisition(sample_metadata_table)
    df = pd.DataFrame(
        {
            "col1": [1, 2, 30],
            "col2": [4, 5, 60],
        }
    )
    result = metric_acquisition.get_table_all_columns_mean(df, extreme_index_range=1)

    assert result["col1"] == pytest.approx(2.0)
    assert result["col2"] == pytest.approx(5.0)


def test_get_column_std_dev():
    metric_acquisition = MetricAcquisition(sample_metadata_table)

    processed_means = pd.DataFrame([{"T-TPT": 3.0, "P-TPT": 2.5}])

    column1 = pd.Series([1, 2, 3, 4, 5], name="T-TPT")
    result1 = metric_acquisition.get_column_std_dev(column1, processed_means)

    column2 = pd.Series([1, 2, 3, 4, np.nan], name="P-TPT")
    result2 = metric_acquisition.get_column_std_dev(column2, processed_means)

    # Calculated manually
    assert result1 == pytest.approx(1.5811388300842)
    assert result2 == pytest.approx(1.2909944487358)


def test_process_event_std(mock_pd_read_parquet):
    processed_means = pd.DataFrame(
        [
            {
                "P-PDG": 3,
                "P-TPT": 6,
                "T-TPT": 9,
                "P-MON-CKP": 12,
                "T-JUS-CKP": 15,
                "P-JUS-CKGL": 18,
                "QGL": 21,
            }
        ]
    )

    metric_acquisition = MetricAcquisition(sample_metadata_table)
    result = metric_acquisition.get_event_std_dev("event3.parquet", processed_means)

    # Calculated manually
    assert result["P-PDG"] == pytest.approx(1.58113883)
    assert result["P-TPT"] == pytest.approx(1.58113883)


def test_process_std_all_columns(sample_metadata_table):
    metric_acquisition = MetricAcquisition(sample_metadata_table)
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        }
    )
    result = metric_acquisition.get_mean_std_dev(df, extreme_index_range=1)

    assert result["col1"] == pytest.approx(2)
    assert result["col2"] == pytest.approx(5.0)
