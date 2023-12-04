
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from typing import Optional
# from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from scipy import signal
# from scipy import interpolate
# from scipy.ndimage import uniform_filter1d
from scipy import stats

from typing import Tuple, List, Dict
from numpy.typing import NDArray

from pybaselines import Baseline

from peak_analysis import (
    ModPolyBaseLineFitter,
    IModPolyBaseLineFitter,
    ScipyPeakDetector,
    TraceProcessor
)


def process_data(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_poly_order: int,
    prominence_in_stdev: float,
    wlen: int,
    s_stdev: pd.Series,
    min_peak_amplitude: float = 0,
    earliest_peak_idx: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process the data

    Args:
        df_input (pd.DataFrame): Input dataframe with time index and the cell traces as columns
            (time_colname, 0, 1, ...).
        time_colname (str): Column name for the time index
        baseline_poly_order (int): Polynomial order for the fitted baseline
        prominence_in_sd (float): Prominence threshold for peak detection in ROBUST standard deviations.
            Robust SD is calculated based on interquartile distance.
        wlen (int): Window length in which the prominence for peak detetction is applied.
        min_peak_amplitude (float): Additional threshold for peak detection in absolute amplitude above baseline.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: First dataframe has the processed traces, the second dataframe has the
            peaks themselves.
    """
    baseline_fitter = ModPolyBaseLineFitter(poly_order=baseline_poly_order)
    peak_detector = ScipyPeakDetector(
        prominence_in_stdev=prominence_in_stdev,
        wlen=wlen,
        min_peak_amplitude=min_peak_amplitude,
        earliest_peak_idx=earliest_peak_idx
    )
    trace_processor = TraceProcessor(
        baseline_fitter=baseline_fitter,
        peak_detector=peak_detector
    )
    cell_nr_list: List[str] = [c for c in df_input.columns if c != time_colname]
    df_proc = df_input[[time_colname]].copy()
    # dict for holding intermediate result over the loop
    _dict_peaks: Dict[str, list] = {'cell_nr': [], 'peak_row_idx': [], time_colname: [], 'amplitude': []}
    # go over the cell traces, remove baseline and identify peaks based on prominence
    for cell_nr in cell_nr_list:
        output: Dict[str, NDArray] = trace_processor.process_trace(s_trace=df_input[cell_nr], stdev=s_stdev.loc[cell_nr])
        df_proc[cell_nr] = output["trace_proc"]
        df_proc[f"{cell_nr}_baseline"] = output["baseline"]
        _peak_row_idx = output["peak_row_idx"]
        _dict_peaks['peak_row_idx'] += _peak_row_idx.tolist()
        _dict_peaks[time_colname] += df_proc[time_colname].iloc[_peak_row_idx].tolist()
        _dict_peaks['amplitude'] += output["amplitudes"].tolist()
        _dict_peaks['cell_nr'] += [cell_nr] * len(_peak_row_idx)
        
    return df_proc, pd.DataFrame(_dict_peaks)


def plot_processing(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_poly_order: int,
    prominence_in_stdev: float,
    wlen: int,
    min_peak_amplitude: float,
    earliest_peak_idx: int,
    s_stdev: pd.Series,
):
    """Plot the data processing to check results.

    Args:
        df_input (pd.DataFrame): Input dataframe with time index and the cell traces as columns
            (time_colname, 0, 1, ...).
        time_colname (str): Column name for the time index
        baseline_poly_order (int): Polynomial order for the fitted baseline
        prominence_in_sd (float): Prominence threshold for peak detection in ROBUST standard deviations.
            Robust SD is calculated based on interquartile distance.
        wlen (int): Window length in which the prominence for peak detetction is applied.
        min_peak_amplitude (float): Additional threshold for peak detection in absolute amplitude above baseline.
    """
    cell_nr_list = [c for c in df_input.columns if c != time_colname]
    # We run the processing but don't yet filter based on min_peak_amplitude!
    df_proc, df_peaks = process_data(
        df_input=df_input,
        time_colname=time_colname,
        baseline_poly_order=baseline_poly_order,
        prominence_in_stdev=prominence_in_stdev,
        wlen=wlen,
        min_peak_amplitude=0,
        earliest_peak_idx=earliest_peak_idx,
        s_stdev=s_stdev
    )
    df_peaks = df_peaks.set_index('cell_nr')
    for cell_nr in cell_nr_list:
        # plot removed baseline
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(df_input[time_colname], df_input[cell_nr])
        axes[0].plot(df_proc[time_colname], df_proc[f"{cell_nr}_baseline"])

        # plot peaks on traces after baseline removal
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].plot(df_proc[time_colname], df_proc[cell_nr])
        if cell_nr in df_peaks.index:
            _peak_idx: pd.Series = df_peaks.loc[[cell_nr], 'peak_row_idx']
            _peak_amplitudes = df_peaks.loc[[cell_nr], 'amplitude'].to_list()
            # We will mark in green the traces that are removed by min_peak_amplitude!
            _color_list = ['red' if x >= min_peak_amplitude else 'green' for x in _peak_amplitudes]
            # axes[1].plot(
            #     df_proc[time_colname].iloc[_peak_idx],
            #     df_proc[cell_nr].iloc[_peak_idx],
            #     "o",
            #     markersize=2,
            #     c=_color_list
            # )
            axes[1].vlines(
                x=df_proc[time_colname].iloc[_peak_idx],
                ymin=np.zeros(shape=len(_peak_idx)),
                ymax=df_proc[cell_nr].iloc[_peak_idx],
                colors=_color_list,
                linewidths=1
            )
            # mark earliest_peak_idx cutoff on the plot
            axes[1].axvline(
                x=df_proc[time_colname].iloc[earliest_peak_idx],
                color="black"
            )
            
        fig.show()


def calculate_peak_aggregates(df_peaks: pd.DataFrame) -> pd.DataFrame:
    # finding the time index for the maximal peaks is more complex than the rest
    meax_peak_time = df_peaks.loc[
        df_peaks.groupby(["cell_nr"])["amplitude"].idxmax(),
        ["cell_nr", "time(m)"]
    ].rename({"time(m)": "max_peak_time"}, axis=1)
    df_aggregates = (
        df_peaks
        .groupby("cell_nr")
        # most of the aggregations are simple
        .agg(
            # if there is no peak, the time should be nan
            peak_count=("time(m)", "count"),
            first_peak_time=("time(m)", "min"),
            mean_peak_amplitude=("amplitude", "mean"),
            max_peak_amplitude=("amplitude", "max")
        )
        .reset_index()
        # join the time for the max peak for every cell
        .merge(
            meax_peak_time,
            on="cell_nr",
            how='inner'
        )
    )
    return df_aggregates


def format_peak_data(
    df_peaks: pd.DataFrame,
    condition_str: str,
    file_str: str,
) -> pd.DataFrame:
    id_prefix: str = f"{condition_str}_{file_str}"
    df_peaks["peak_ID"] = df_peaks["cell_nr"].apply(lambda x: f"{id_prefix}_{x}")
    return df_peaks.drop(["cell_nr", "peak_row_idx"], axis=1)


def format_agg_data(
    df_agg: pd.DataFrame,
    condition_str: str,
    file_str: str,
) -> pd.DataFrame:
    id_prefix: str = f"{condition_str}_{file_str}"
    df_agg["cell_ID"] = df_agg["cell_nr"].apply(lambda x: f"{id_prefix}_{x}")
    # we drop the cell_nr as a new one will be assigned that is unique over all the files
    return df_agg.drop(["cell_nr"], axis=1)


def two_sample_t_test(sample0, sample1, equal_var: bool):
    return stats.ttest_ind(
        a=sample0,
        b=sample1,
        equal_var=equal_var
    ).pvalue
