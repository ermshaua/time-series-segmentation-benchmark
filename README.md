# Time Series Segmentation Benchmark (TSSB)
The problem of time series segmentation (TSS) is to find a meaningful segmentation of a time series (TS) that captures a data-generating process with distinct states and transitions. We consider a segmentation meaningful, if the change points (CPs) between two consecutive segments correspond to state transitions in the associated process. 

This repository contains the time series segmentation benchmark (TSSB). It currently contains 66 annotated TS with 2-7 segments. Each TS is constructed from one of the <a target="_blank" href="http://timeseriesclassification.com/">UEA & UCR time series classification</a> datasets. We group TS by label and concatenate them to create segments with distinctive temporal patterns and statistical properties. We annotate the offsets at which we concatenated the segments as CPs. Addtionally, we apply resampling to control the dataset resolution and add approximate, hand-selected window sizes that are able to capture temporal patterns.   

## Installation
You can install the TSSB with PyPi:
`python -m pip install git+https://github.com/ermshaua/time-series-segmentation-benchmark` 

## Citation
If you use the TSSB in your scientific publication, we would appreciate the following <a target="_blank" href="https://dl.acm.org/doi/abs/10.1145/3459637.3482240">citation</a>:

```
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
```

## Results

We have evaluated multiple time series segmentation algorithms using the TSSB. The following table summarises the mean relative CP distance error (smaller is better) and the corresponding mean ranks. Evaluation details are in the <a target="_blank" href="https://dl.acm.org/doi/abs/10.1145/3459637.3482240">paper</a>. The raw result sheet and an evaluation notebook are in the <a target="_blank" href="https://github.com/ermshaua/time-series-segmentation-benchmark/tree/main/tssb/notebooks">notebooks</a> folder.

| Segmentation Algorithm | Mean Error | Mean Rank | Wins & Ties |
| ---------------------- | ---------- | --------- | ---------
| ClaSP                  | 0.00676    | 1.1       | 59/66       |
| FLOSS                  | 0.03796    | 2.1       | 5/66        |
| Window-L2              | 0.14442    | 3.4       | 1/66        |
| BOCD                   | 0.17803    | 3.5       | 1/66        |
| BinSeg-L2              | 0.31853    | 4.9       | 0/66        |

## Basic Usage
Let's first import methods to load TS from the benchmark and to evaluate TSS algorithms. As an example, we also import our segmentation algorithm ClaSP from <a target="_blank" href="https://github.com/alan-turing-institute/sktime/">sktime</a>. 

```python3
>>> from tssb.utils import load_time_series_segmentation_datasets, relative_change_point_distance
>>> from sktime.annotation.clasp import ClaSPSegmentation
```

We can now load the entire benchmark (66 TS) as a pandas dataframe using 

```python3
>>> tssb = load_time_series_segmentation_datasets()
```

or a selection of TS by specifying the `names` attribute using

```python3
>>> tssb = load_time_series_segmentation_datasets(names=["ArrowHead", "InlineSkate", "Plane"])
```

The dataframe `tssb` contains (TS name, window size, CPs, TS) rows and can now be iterated to evaluate a TSS algorithm.

```python3
>>> for _, (ts_name, window_size, cps, ts) in tssb.iterrows():
>>>   found_cps = ClaSPSegmentation(window_size, n_cps=len(cps)).fit_predict(ts)
>>>   score = relative_change_point_distance(cps, found_cps, ts.shape[0])
>>>   print(f"Time Series: {ts_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score}")
```

In a similar fashion, you can evaluate your TSS algorithm and compare results. For more details, see the example <a href="https://github.com/ermshaua/time-series-segmentation-benchmark/tree/main/tssb/notebooks">notebooks</a>.

## Visualizations

See the following example TS to get an overview of the TSSB. You can find more images in the <a href="https://github.com/ermshaua/time-series-segmentation-benchmark/tree/main/tssb/visualizations">visualizations</a> folder.

![image](tssb/visualizations/ArrowHead.png)

![image](tssb/visualizations/InlineSkate.png)

![image](tssb/visualizations/Plane.png)

In order to create a visualization of your predicted segmentation, you can use the following code.

```python3
>>> import matplotlib.pyplot as plt
>>> from tssb.utils import visualize_time_series
>>> fig, ax = visualize_time_series(ts, ts_name, cps, found_cps)
>>> plt.show()
```

![image](tssb/plane_example_segmentation.png)
