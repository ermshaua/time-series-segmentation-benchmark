import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import paired_euclidean_distances


def load_time_series_segmentation_datasets(names=None):
    '''
    Loads and parses the TSSB dataset as a pandas dataframe.
    Parameters
    -----------
    :param names: dataset names to load, default: all
    :return: a pandas dataframe with (TS name, window size, CPs, TS) rows
    Examples
    -----------
    >>> tssb = load_time_series_segmentation_datasets()
    '''
    desc_filename = os.path.join(ABS_PATH, "datasets", "desc.txt")
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc_file.append(line)

    df = []

    for row in desc_file:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts = np.loadtxt(fname=os.path.join(ABS_PATH, "datasets", ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change points", "time_series"])


def relative_change_point_distance(cps_true, cps_pred, ts_len):
    '''
    Calculates the relative CP distance between ground truth and predicted change points.
    Parameters
    -----------
    :param cps_true: an array of true change point positions
    :param cps_pred: an array of predicted change point positions
    :param ts_len: the length of the associated time series
    :return: relative distance between cps_true and cps_pred considering ts_len
    Examples
    -----------
    >>> score = relative_change_point_distance(cps, found_cps, ts.shape[0])
    '''
    assert len(cps_true) == len(cps_pred), "true/predicted cps must have the same length."
    differences = 0

    for cp_pred in cps_pred:
        distances = paired_euclidean_distances(
            np.array([cp_pred]*len(cps_true)).reshape(-1,1),
            cps_true.reshape(-1,1)
        )
        cp_true_idx = np.argmin(distances, axis=0)
        cp_true = cps_true[cp_true_idx]
        differences += np.abs(cp_pred-cp_true)

    return np.round(differences / (len(cps_true) * ts_len), 6)