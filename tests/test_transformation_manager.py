import pandas as pd
import numpy as np
from unittest.mock import patch
import pathlib
import pytest
from data_preparation.transformation_manager import TransformationManager
import tempfile
import shutil


def test_transform_event_with_imputation():
    event_1_original = pd.DataFrame(
        {
            "P-PDG": [10, 11, 12, 13, 12, 11],
            "P-TPT": [10, 11, 12, 13, 12, 11],
            "T-TPT": [0, 1, 2, 3, 2, 1],
            "P-MON-CKP": [30, 31, 32, 33, 32, 31],
            "T-JUS-CKP": [30, 31, 32, 33, 32, 31],
            "P-JUS-CKGL": [30, 31, 32, 33, 32, 31],
            "QGL": [20, 21, 22, 23, 22, 21],
            "class": [np.nan, 0, np.nan, 0, 4, np.nan],
        }
    )
    event_1_expected = pd.DataFrame(
        {
            "P-PDG": [10, 11, 12, 13, 12, 11],
            "P-TPT": [10, 11, 12, 13, 12, 11],
            "T-TPT": [0, 1, 2, 3, 2, 1],
            "P-MON-CKP": [30, 31, 32, 33, 32, 31],
            "T-JUS-CKP": [30, 31, 32, 33, 32, 31],
            "P-JUS-CKGL": [30, 31, 32, 33, 32, 31],
            "QGL": [20, 21, 22, 23, 22, 21],
            "class": [0, 0, 0, 0, 4, 4],
        }
    )
    event_1_transformed = TransformationManager.transform_event_with_imputation(
        event_data=event_1_original, event_class_type=4
    )
    assert event_1_expected.astype(float).equals(event_1_transformed.astype(float))

    event_2_class_type = 6
    event_2_original = pd.DataFrame(
        {
            "P-PDG": [10, 11, 12, 13, 12, 11],
            "P-TPT": [10, 11, 12, 13, 12, 11],
            "T-TPT": [0, np.nan, 2, np.nan, 2, 1],
            "P-MON-CKP": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "T-JUS-CKP": [30, 31, 32, 33, 32, 31],
            "P-JUS-CKGL": [30, 31, 32, 33, 32, 31],
            "QGL": [np.nan, 21, 22, 23, 22, np.nan],
            "class": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    event_2_expected = pd.DataFrame(
        {
            "P-PDG": [10, 11, 12, 13, 12, 11],
            "P-TPT": [10, 11, 12, 13, 12, 11],
            "T-TPT": [0, 1, 2, 2, 2, 1],
            "P-MON-CKP": [0, 0, 0, 0, 0, 0],
            "T-JUS-CKP": [30, 31, 32, 33, 32, 31],
            "P-JUS-CKGL": [30, 31, 32, 33, 32, 31],
            "QGL": [21, 21, 22, 23, 22, 22],
            "class": [
                event_2_class_type,
                event_2_class_type,
                event_2_class_type,
                event_2_class_type,
                event_2_class_type,
                event_2_class_type,
            ],
        }
    )
    event_2_transformed = TransformationManager.transform_event_with_imputation(
        event_data=event_2_original, event_class_type=event_2_class_type
    )
    assert event_2_expected.astype(float).equals(event_2_transformed.astype(float))


def test_transform_event_with_standardization():
    event_1_original = pd.DataFrame(
        {
            "P-PDG": [10, 20, 30, 40, 50, 60],
            "P-TPT": [20, 40, 60, 80, 100, 120],
            "T-TPT": [100, 200, 300, 400, 500, 600],
            "P-MON-CKP": [10, 20, 30, 40, 50, 60],
            "T-JUS-CKP": [20, 40, 60, 80, 100, 120],
            "P-JUS-CKGL": [100, 200, 300, 400, 500, 600],
            "QGL": [1, 2, 3, 4, 5, 6],
            "class": [0, 0, 0, 0, 4, 4],
        }
    )
    event_1_expected = pd.DataFrame(
        {
            "P-PDG": [0, 1, 2, 3, 4, 5],
            "P-TPT": [-0.5, 0, 0.5, 1, 1.5, 2],
            "T-TPT": [-0.8, -0.6, -0.4, -0.2, 0, 0.2],
            "P-MON-CKP": [-0.5, 0, 0.5, 1, 1.5, 2],
            "T-JUS-CKP": [0, 1, 2, 3, 4, 5],
            "P-JUS-CKGL": [-0.5, 0, 0.5, 1, 1.5, 2],
            "QGL": [0, 1, 2, 3, 4, 5],
            "class": [0, 0, 0, 0, 4, 4],
        }
    )

    avg_variable_mean = pd.Series(
        {
            "P-PDG": 10,
            "P-TPT": 40,
            "T-TPT": 500,
            "P-MON-CKP": 20,
            "T-JUS-CKP": 20,
            "P-JUS-CKGL": 200,
            "QGL": 1,
        }
    )
    avg_variable_std_dev = pd.Series(
        {
            "P-PDG": 10,
            "P-TPT": 40,
            "T-TPT": 500,
            "P-MON-CKP": 20,
            "T-JUS-CKP": 20,
            "P-JUS-CKGL": 200,
            "QGL": 1,
        }
    )

    event_1_transformed = TransformationManager.transform_event_with_standardization(
        event_1_original, avg_variable_mean, avg_variable_std_dev
    )
    assert event_1_transformed.astype(float).equals(event_1_expected.astype(float))


def test_transform_event_with_downsample():
    event_1_original = pd.DataFrame(
        {
            "P-PDG": [38265830.0] * 10,
            "P-TPT": [13654450.0] * 10,
            "T-TPT": [
                117.8,
                118.0,
                119.0,
                118.0,
                117.0,
                116.0,
                115.0,
                114.0,
                113.0,
                float("NaN"),
            ],
            "P-MON-CKP": [6029680.0] * 10,
            "T-JUS-CKP": [
                69.0,
                70.0,
                float("NaN"),
                float("NaN"),
                74.0,
                74.0,
                75.0,
                76.0,
                77.0,
                78.0,
            ],
            "P-JUS-CKGL": [3283309.0] * 10,
            "T-JUS-CKGL": [float("NaN")] * 10,
            "QGL": [0.0] * 10,
            "class": [4] * 10,
        }
    )
    event_1_original.index = pd.to_datetime(
        [
            "2017-03-16 12:02:03",
            "2017-03-16 12:02:04",
            "2017-03-16 12:02:05",
            "2017-03-16 12:02:06",
            "2017-03-16 12:02:07",
            "2017-03-16 12:02:08",
            "2017-03-16 12:02:09",
            "2017-03-16 12:02:10",
            "2017-03-16 12:02:11",
            "2017-03-16 12:02:12",
        ]
    )

    event_1_expected_inteval_2s = pd.DataFrame(
        {
            "P-PDG": [38265830.0] * 5,
            "P-TPT": [13654450.0] * 5,
            "T-TPT": [
                117.5,
                118.5,
                116.5,
                114.5,
                113.0,
            ],
            "P-MON-CKP": [6029680.0] * 5,
            "T-JUS-CKP": [
                69.5,
                float("NaN"),
                74.0,
                75.5,
                77.5,
            ],
            "P-JUS-CKGL": [3283309.0] * 5,
            "T-JUS-CKGL": [float("NaN")] * 5,
            "QGL": [0.0] * 5,
            "class": [4] * 5,
        }
    )
    event_1_expected_inteval_2s.index = pd.to_datetime(
        [
            "2017-03-16 12:02:03",
            "2017-03-16 12:02:05",
            "2017-03-16 12:02:07",
            "2017-03-16 12:02:09",
            "2017-03-16 12:02:11",
        ]
    )
    event_1_transformed_inteval_2s = (
        TransformationManager.transform_event_with_downsample(event_1_original, 2)
    )
    print(event_1_transformed_inteval_2s)
    print(event_1_expected_inteval_2s)
    differences = event_1_transformed_inteval_2s == event_1_expected_inteval_2s

    # Now, you can filter the rows where differences exist
    diff_rows = differences[differences.any(axis=1)]

    # Print the rows with differences
    print(diff_rows)
    assert event_1_transformed_inteval_2s.astype(float).equals(
        event_1_expected_inteval_2s.astype(float)
    )

    event_1_expected_inteval_5s = pd.DataFrame(
        {
            "P-PDG": [38265830.0] * 2,
            "P-TPT": [13654450.0] * 2,
            "T-TPT": [
                117.8,
                114.5,
            ],
            "P-MON-CKP": [6029680.0] * 2,
            "T-JUS-CKP": [
                71.0,
                76.0,
            ],
            "P-JUS-CKGL": [3283309.0] * 2,
            "T-JUS-CKGL": [float("NaN")] * 2,
            "QGL": [0.0] * 2,
            "class": [4] * 2,
        }
    )
    event_1_expected_inteval_5s.index = pd.to_datetime(
        [
            "2017-03-16 12:02:03",
            "2017-03-16 12:02:08",
        ]
    )
    event_1_transformed_inteval_5s = (
        TransformationManager.transform_event_with_downsample(event_1_original, 5)
    )
    assert event_1_transformed_inteval_5s.astype(float).equals(
        event_1_expected_inteval_5s.astype(float)
    )


def test_transform_event_with_timestep_windows():
    event_1_original = pd.DataFrame(
        {
            "P-PDG": [10, 20, 30, 40, 50, 60],
            "P-TPT": [20, 40, 60, 80, 100, 120],
            "T-TPT": [100, 200, 300, 400, 500, 600],
            "P-MON-CKP": [10, 20, 30, 40, 50, 60],
            "T-JUS-CKP": [20, 40, 60, 80, 100, 120],
            "P-JUS-CKGL": [100, 200, 300, 400, 500, 600],
            "QGL": [1, 2, 3, 4, 5, 6],
            "class": [0, 0, 0, 0, 104, 4],
        }
    )
    (
        X_event1_2step_window,
        y_event1_2step_window,
    ) = TransformationManager.transform_event_with_timestep_windows(event_1_original, 2)
    (
        X_event1_3step_window,
        y_event1_3step_window,
    ) = TransformationManager.transform_event_with_timestep_windows(event_1_original, 3)
    (
        X_event1_5step_window,
        y_event1_5step_window,
    ) = TransformationManager.transform_event_with_timestep_windows(event_1_original, 5)

    print(f"2 steps: {X_event1_2step_window, y_event1_2step_window}\n\n")
    print(f"3 steps: {X_event1_3step_window, y_event1_3step_window}\n\n")
    print(f"5 steps: {X_event1_5step_window, y_event1_5step_window}\n\n")

    assert len(X_event1_2step_window) == 5
    assert len(y_event1_2step_window) == 5
    assert (
        y_event1_2step_window[-1].astype(float)
        == np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
    ).all()

    assert len(X_event1_3step_window) == 4
    assert len(y_event1_3step_window) == 4

    assert len(X_event1_5step_window) == 2
    assert len(y_event1_5step_window) == 2
    assert (
        X_event1_5step_window
        == np.array(
            [
                [
                    [10, 20, 100, 10, 20, 100, 1],
                    [20, 40, 200, 20, 40, 200, 2],
                    [30, 60, 300, 30, 60, 300, 3],
                    [40, 80, 400, 40, 80, 400, 4],
                    [50, 100, 500, 50, 100, 500, 5],
                ],
                [
                    [20, 40, 200, 20, 40, 200, 2],
                    [30, 60, 300, 30, 60, 300, 3],
                    [40, 80, 400, 40, 80, 400, 4],
                    [50, 100, 500, 50, 100, 500, 5],
                    [60, 120, 600, 60, 120, 600, 6],
                ],
            ]
        )
    ).all()


def test_store_and_retrieve_pair_array():
    array1 = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    array2 = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    # define file location
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    temp_file = temp_dir / "temp_file.npz"

    # store and retrieve file
    TransformationManager.store_pair_array(array1, array2, temp_file)
    out_array1, out_array2 = TransformationManager.retrieve_pair_array(temp_file)

    # clean up
    shutil.rmtree(str(temp_dir))

    # assert equality
    assert (array1 == out_array1).all()
    assert (array2 == out_array2).all()
