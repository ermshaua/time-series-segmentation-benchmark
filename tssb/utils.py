import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import daproli as dp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_color_codes()

from scipy.stats import zscore


def generate_time_series_segmentation_dataset(X, y, labels, resample_rate=1, label_cut=0):
    '''
    Generates a TSSB dataset from a sktime dataframe (with cols "dim_0" and "class_val").
    Parameters
    -----------
    :param X: 1d time series data
    :param y: target variables (classes)
    :param labels: a list of (potentially cut) labels used to create the dataset (determines the segment order)
    :param resample_rate: the number of data points that are mean-aggregated (controls the TS resolution)
    :param label_cut: the number of cuts for a label (enables the use of sub-segments, possible values: 0, 1, 2)

    :return: a tuple of TS and a CP np.array
    Examples
    -----------
    >>> from aeon.datasets import load_from_tsv_file
    >>> df = load_from_tsv_file(os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.tsv"))
    >>> ts, cps = generate_time_series_segmentation_dataset(df, labels=[0,1], resample_rate=2)
    '''
    X_concat, y_concat = [], []

    df = pd.DataFrame()
    df["dim_0"] = [pd.Series(ts[0]) for ts in X]
    df["class_val"] = y

    segment_splits = {
        1: [.4, .6],
        2: [.25, .5, .75]
    }

    # group and concatenate TS by label
    for label, df_group in df.groupby("class_val"):
        label_ts = np.concatenate(df_group["dim_0"].apply(zscore).to_numpy())

        if label_cut == 0:
            X_concat.append(label_ts)
            y_concat.append(label)
            continue

        np.random.seed(label_ts.shape[0])

        segment_borders = np.concatenate((
            [0],
            np.asarray(
                np.sort(np.random.choice(segment_splits[label_cut], label_cut, replace=False)) * label_ts.shape[0],
                np.int64),
            [label_ts.shape[0]]
        ))

        for idx in range(1, len(segment_borders)):
            X_concat.append(label_ts[segment_borders[idx - 1]:segment_borders[idx]])
            y_concat.append(label)

    # reduce TS to relevant labels
    X_seg = np.array(X_concat, dtype=np.object)[labels]

    # resample TS
    X_seg = dp.map(lambda seg: dp.windowed(seg, resample_rate, step=resample_rate, ret_type=list), X_seg, ret_type=list,
                   expand_args=False)
    X_seg = dp.map(lambda seg: np.mean(seg, axis=1), X_seg, ret_type=list, expand_args=False)

    # create CP offsets
    y_seg = dp.map(len, X_seg, ret_type=list, expand_args=False)

    # create final TS and offsets
    ts = np.concatenate(X_seg)
    cps = np.cumsum(y_seg)[:-1]

    return ts, cps


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

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "time_series"])


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
        tick.label1.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    return fig, ax
