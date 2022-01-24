import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_color_codes()

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


def visualize_time_series(ts, ts_name=None, cps_true=None, cps_pred=None, fontsize=18):
    '''
    Visualizes a time series and its predicted segmentation (if provided).
    Parameters
    -----------
    :param ts: an array of time series data points
    :param ts_name: the time series name
    :param cps_true: an array of true change point positions
    :param cps_pred: an array of predicted change point positions
    :param fontsize: the font size used for displayed text
    :return: a (Figure, Axes) tuple
    Examples
    -----------
    >>> fig, ax = visualize_time_series(ts, ts_name, cps, found_cps)
    '''
    fig, ax = plt.subplots(1, figsize=(20, 5))

    if cps_true is None:
        cps_true = np.zeros(0)

    if cps_pred is None:
        cps_pred = np.zeros(0)

    segments = [0] + cps_true.tolist() + [ts.shape[0]]

    for idx in np.arange(0, len(segments) - 1):
        ax.plot(np.arange(segments[idx], segments[idx + 1]), ts[segments[idx]:segments[idx + 1]])

    for idx, cp in enumerate(cps_pred):
        ax.axvline(x=cp, linewidth=5, color='black', label=f'Predicted Change Point' if idx == 0 else None)

    if ts_name is not None:
        ax.set_title(ts_name, fontsize=fontsize)

    if cps_pred.shape[0] > 0:
        ax.legend(prop={'size': fontsize})

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    return fig, ax
