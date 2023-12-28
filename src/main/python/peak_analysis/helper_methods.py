import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, MaxNLocator, AutoMinorLocator)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# from scipy import interpolate
# from scipy.ndimage import uniform_filter1d
from scipy import stats

from typing import Tuple, List, Dict, Optional, Union
from numpy.typing import NDArray

from statsmodels.stats.proportion import (
    proportion_confint,
    proportions_ztest
)

from peak_analysis import (
    BaseLineFitter,
    ModPolyBaseLineFitter,
    ModPolyCustomBaseLineFitter,
    IModPolyBaseLineFitter,
    PeakDetector,
    ScipyPeakDetector,
    TraceProcessor,
)
# 189, 184, 168, 164, 151, 149, 145, 144


def process_data(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter,
    peak_detector: PeakDetector,
    s_stdev: Optional[pd.Series] = None,
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

    trace_processor = TraceProcessor(
        baseline_fitter=baseline_fitter, peak_detector=peak_detector
    )
    cell_nr_list: List[str] = [c for c in df_input.columns if c != time_colname]
    df_proc = df_input[[time_colname]].copy()
    # dict for holding intermediate result over the loop
    _dict_peaks: Dict[str, list] = {
        "cell_nr": [],
        "peak_row_idx": [],
        time_colname: [],
        "amplitude": [],
    }
    # go over the cell traces, remove baseline and identify peaks based on prominence
    for cell_nr in cell_nr_list:
        # this is hacky, TODO: clean this up
        output: Dict[str, NDArray]
        if s_stdev is not None:
            output = trace_processor.process_trace(
                s_trace=df_input[cell_nr], stdev_lower_bound=s_stdev.loc[cell_nr]
            )
        else:
            output = trace_processor.process_trace(
                s_trace=df_input[cell_nr]
            )
        df_proc[cell_nr] = output["trace_proc"]
        df_proc[f"{cell_nr}_baseline"] = output["baseline"]
        _peak_row_idx = output["peak_row_idx"]
        _dict_peaks["peak_row_idx"] += _peak_row_idx.tolist()
        _dict_peaks[time_colname] += df_proc[time_colname].iloc[_peak_row_idx].tolist()
        _dict_peaks["amplitude"] += output["amplitudes"].tolist()
        _dict_peaks["cell_nr"] += [cell_nr] * len(_peak_row_idx)

    return df_proc, pd.DataFrame(_dict_peaks)


def plot_agg_output_comparison(
    agg_outputs: List[pd.DataFrame],
    comparisons: Dict[str, List[str]],
    n_major_ticks: Optional[int] = None,
    n_minor_to_major: Optional[int] = None,
    figsize: Tuple[float, float] = (3, 4),
    colors: Tuple[str, str] = ("tab:red", "tab:blue"),
    alpha: float = 0.8,
    x_rotation: float = 0.0,
    width: float = 0.8,
    condition_alias_dict: Dict[str, str] = {},
    colname_alias_dict: Dict[str, str] = {},
    errorbar_type: str = "se",
    errorbar_alpha: float = 0.05,
    savepath: Optional[str] = None
):
    """Plot the response rate.
    
    agg_output files are structured as follows:
        cell_nr  peak_count  first_peak_time  mean_peak_amplitude  max_peak_amplitude  max_peak_time            cell_ID
    0         1           4          7.70257          163.673578         314.156600           7.70257  EVs_20230901_02_3


    Args:
        agg_outputs (List[pd.DataFrame]): _description_
        major_multiple (Optional[float], optional): _description_, if None the automatic setting is used.
            Defaults to 10.
        minor_multiple (Optional[float], optional): _description_, if None the automatic setting is used.
            Defaults to 10.

    Args:
        agg_outputs (List[pd.DataFrame]): _description_
        comparisons (Dict[str, str]): {"column in agg_output": ["function_name0", "function_name1"]}
        n_major_ticks (Optional[int], optional): _description_, if None the automatic setting is used.
            Defaults to 10.
        n_minor_to_major (Optional[int], optional): _description_, if None the automatic setting is used.
            Defaults to 10.
        figsize (Tuple[float, float], optional): _description_. Defaults to (3, 4).
        colors (Tuple[str, str]): The color for the benchmark and the rest as a tuple of strings.
            Defaults to ("tab:red", "tab:blue").
        alpha (float): The opacity for the bars, float between 0 and 1. Defaults to 0.8.
        x_rotation (float): Degree to rotate x axis tick labels with.
        width (float): Scaling factor for the bar width, a value of 1 means there is no gap between the bars.
            Defaults to 0.8.
        condition_alias_dict (Dict[str, str]): Dict for renaming the ticklabels with an alias name for certain conditions.
            Defaults to {}.
        colname_alias_dict (Dict[str, str]): _description_
        errorbar_type (str): If "se", the errorbar is +- standard error; otherwise the jeffreys confidence interval
            with 1 - errorbar_alpha width. Defaults to "se".
        errorbar_alpha (float): The width of the confidence interval is +- (1 - alpha / 2), unless errorbar_type is
            "se". Defaults to 0.05.
    """
    assert (errorbar_alpha > 0.0 and errorbar_alpha < 0.5)
    color_list: List[str] = [colors[0]] + [colors[1]] * (len(agg_outputs) - 1)
    for colname, comp_type_list in comparisons.items():
        for comp_type in comp_type_list:
            errorbars_list: List[NDArray] = []
            pvalue_list: List[Optional[float]] = []
            metric_name: str
            if comp_type == "response_rate":
                metric_name = "% Responders"
            elif comp_type == "mean":
                if colname in colname_alias_dict:
                    metric_name = f"Mean {colname_alias_dict[colname]}"
                else:
                    metric_name = f"Mean {colname}"
            else:
                raise ValueError()
            _dict_plot: Dict[str, list] = {"condition": [], metric_name: []}
            for i, agg_output in enumerate(agg_outputs):
                condition: str = _get_condition_from_cell_ID(agg_output)
                if condition in condition_alias_dict:
                    condition = condition_alias_dict[condition]
                if comp_type == "response_rate":
                    if i > 0:
                        pct_resp, errorbars, p_value = _calculate_response_rate_metrics(
                            s_agg_output=agg_output[colname],
                            s_agg_output_bench=agg_outputs[0][colname],
                            errorbar_type=errorbar_type,
                            errorbar_alpha=errorbar_alpha
                        )
                    else:
                        pct_resp, errorbars, p_value = _calculate_response_rate_metrics(
                            s_agg_output=agg_output[colname],
                            errorbar_type=errorbar_type,
                            errorbar_alpha=errorbar_alpha
                        )
                elif comp_type == "mean":
                    if i > 0:
                        pct_resp, errorbars, p_value = _calculate_mean_metrics(
                            s_agg_output=agg_output[colname],
                            s_agg_output_bench=agg_outputs[0][colname],
                            errorbar_type=errorbar_type,
                            errorbar_alpha=errorbar_alpha
                        )
                    else:
                        pct_resp, errorbars, p_value = _calculate_mean_metrics(
                            s_agg_output=agg_output[colname],
                            errorbar_type=errorbar_type,
                            errorbar_alpha=errorbar_alpha
                        )
                else:
                    raise ValueError()
                _dict_plot[metric_name].append(pct_resp)
                errorbars_list.append(errorbars)
                _dict_plot["condition"].append(condition)
                pvalue_list.append(p_value)
            # print(pvalue_list)
            df_plot = pd.DataFrame(_dict_plot)
            # print(df_plot)
            # convert to array shaped as (2, N)
            errorbars_arr = np.array(errorbars_list).T
            fig, ax = _make_barplot(
                df_plot=df_plot,
                x="condition",
                y=metric_name,
                colors_for_bars=color_list,
                errorbars_arr=errorbars_arr,
                alpha=alpha,
                width=width,
                linewidth=1,
                edgecolor="black",
                figsize=figsize,
            )
            # add significance asterisks, offset vertically by the upper errorbar length + offset
            offset = df_plot[metric_name].max() / 20
            _add_signif_markers_to_bars(
                ax=ax, pvalue_list=pvalue_list, y_offset=errorbars_arr[1, :] + offset, fontsize=15
            )
            ylims: Optional[Tuple[float, float]] = None
            if comp_type == "response_rate":
                ylims = (0.0, 100.0)  
            _format_ax(ax=ax, n_major_ticks=n_major_ticks, n_minor_to_major=n_minor_to_major, x_rotation=x_rotation, ylims=ylims)
            if savepath is not None:
                fname = f"{comp_type}_{colname}.png"
                print(f"Saving plot to: {savepath}/{fname}")
                fig.savefig(f"{savepath}/{fname}", bbox_inches='tight', transparent=True)
            fig.show() 


def _calculate_response_rate_metrics(
    s_agg_output: pd.Series,
    errorbar_type: str,
    errorbar_alpha: float = 0.05,
    s_agg_output_bench: Optional[pd.Series] = None
) -> Tuple[float, NDArray, Optional[float]]:
    """_summary_

    Args:
        agg_output (pd.DataFrame): _description_
        agg_output_bench (pd.DataFrame): _description_
        errorbar_type (str): _description_
        errorbar_alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        Tuple[float, NDArray]: _description_
    """
    errorbars: NDArray
    pct_resp: float
    p_value: Optional[float]
    # if the agg data contains no cells
    if len(s_agg_output) == 0:
        pct_resp = 0.0
        errorbars = np.array([0.0, 0.0])
        p_value = None
    else:
        count: int = (s_agg_output > 0).sum()
        nobs: int = len(s_agg_output)
        pct_resp = count / nobs * 100
        if errorbar_type == "se":
            _se: float = ((count / nobs) * (1 - count / nobs) / nobs) ** 0.5
            errorbars = np.array((_se, _se)) * 100
        else:
            # convert from tuple to array so we can do arithmetic on the values
            confint: NDArray = np.array(proportion_confint(count, nobs, alpha=errorbar_alpha, method='jeffreys'))
            # convert to percentage and errorbar format
            errorbars: NDArray = np.abs(confint * 100 - pct_resp)
        if s_agg_output_bench is None:
            p_value = None
        else:
            # benchmark values for 2 sample significance testing
            count_bm: int = (s_agg_output_bench > 0).sum()
            nobs_bm: int = len(s_agg_output_bench)
            p_value = proportions_ztest(
                count=[count, count_bm],
                nobs=[nobs, nobs_bm],
                value=0.0,
                alternative="two-sided"
            )[1]
    return pct_resp, errorbars, p_value


def _calculate_mean_metrics(
    s_agg_output: pd.Series,
    errorbar_type: str,
    errorbar_alpha: float = 0.05,
    s_agg_output_bench: Optional[pd.Series] = None
) -> Tuple[float, NDArray, Optional[float]]:
    """_summary_

    Args:
        agg_output (pd.DataFrame): _description_
        agg_output_bench (pd.DataFrame): _description_
        errorbar_type (str): _description_
        errorbar_alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        Tuple[float, NDArray]: _description_
    """
    # if the agg data contains no cells
    mean: float
    errorbars: NDArray
    p_value: Optional[float]
    if len(s_agg_output) == 0:
        mean = 0.0
        errorbars = np.array([0.0, 0.0])
        p_value = None
    else:
        mean = s_agg_output.mean()
        se: float = s_agg_output.std() / (len(s_agg_output - 1) ** 0.5)
        if errorbar_type == "se":
            errorbars = np.array((se, se))
        else:
            ci_width_upper: float
            # if stdev is 0
            if se == 0:
                ci_width_upper = 0.0
            else:
                ci_width_upper = stats.t.ppf(1 - (errorbar_alpha / 2), (len(s_agg_output) - 1), loc=0, scale=se)
            errorbars = np.array((ci_width_upper, ci_width_upper))
        if s_agg_output_bench is None:
            p_value = None
        else:
            # benchmark values for 2 sample significance testing
            p_value = two_sample_t_test(
                sample0=s_agg_output_bench,
                sample1=s_agg_output,
                equal_var=False
            )
    return mean, errorbars, p_value


def plot_output_distribution(
    agg_outputs,
    columns_to_plot: List[str],
    id_colname: str = "peak_ID",
    n_major_ticks: Optional[int] = None,
    n_minor_to_major: Optional[int] = None,
    x_rotation: float = 0.0,
    width: float = 0.75,
    figsize: Optional[Tuple[float, float]] = None,
    condition_alias_dict: Dict[str, str] = {},
    colname_alias_dict: Dict[str, str] = {},
):
    df_plot: pd.DataFrame = pd.concat(agg_outputs, axis=0)
    df_plot["condition"] = df_plot[id_colname].str.split("_").str[0]
    df_plot["condition"] = df_plot["condition"].replace(condition_alias_dict)
    df_plot = df_plot.rename(columns=colname_alias_dict)
    for c in columns_to_plot:
        c = colname_alias_dict.get(c, c)
        fig, ax = _make_boxplot(
            df_plot=df_plot,
            x="condition",
            y=c,
            width=width,
            figsize=figsize
        )
        _format_ax(
            ax=ax, n_major_ticks=n_major_ticks, n_minor_to_major=n_minor_to_major, x_rotation=x_rotation
        )
        _add_counts_to_boxes(ax=ax, df_plot=df_plot, y=c)
        fig.show()


def _make_barplot(
    df_plot: pd.DataFrame,
    x: str,
    y: str,
    colors_for_bars: List[str],
    errorbars_arr: NDArray,
    alpha: float = 0.8,
    width: float = 0.8,
    linewidth: Union[float, int] = 1,
    edgecolor: str = "black",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Axes, Figure]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        df_plot,
        x=x,
        y=y,
        palette=colors_for_bars,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        width=width,
        ax=ax
    )
    ax.errorbar(x=df_plot[x], y=df_plot[y], yerr=errorbars_arr, fmt="none", elinewidth=2, capsize=0, c="k")
    return fig, ax


def _make_boxplot(
    df_plot: pd.DataFrame,
    x: str,
    y: str,
    width: float = 0.75,
    figsize: Optional[Tuple[float, float]] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        df_plot,
        x=x,
        y=y,
        boxprops={'facecolor': 'none', 'edgecolor': 'black'},
        medianprops={'color': "#D67B77"},
        whiskerprops={'color': 'black'},  # whicsker stems
        capprops={'color': 'black'},  # whiscep caps
        flierprops={"marker": 'o', "markerfacecolor": 'None', "markersize": 3, "markeredgecolor": 'black'},
        # palette=colors_for_bars,
        # alpha=alpha,
        # linewidth=linewidth,
        # edgecolor=edgecolor,
        width=width,
        ax=ax
    )
    return fig, ax


def _add_counts_to_boxes(ax: Axes, df_plot: pd.DataFrame, y: str):
    xtick_loc: Dict[str, int] = {v.get_text(): v.get_position()[0] for v in ax.get_xticklabels()}
    counts_by_group: pd.Series = df_plot.groupby("condition")[y].count()
    ymax_by_group: pd.Series = df_plot.groupby("condition")[y].max()
    offset: float = ymax_by_group.max() * 0.1
    for group, count in counts_by_group.items():
        ax.text(
            x=xtick_loc[group], y=ymax_by_group.loc[group] + offset, s=str(count), ha="center"
        )


def _add_signif_markers_to_bars(
    ax: Axes, pvalue_list: List[float], y_offset: Union[NDArray, List[float]], fontsize: Union[float, int] = 15
):
    for i, _bar, _pval in zip(range(len(pvalue_list)), ax.patches, pvalue_list):
        if _pval is None:
            continue
            # signif_str = ""
        elif np.isnan(_pval):
            signif_str = ''
        elif _pval >= 0.05:
            signif_str = ''
        # elif _pval < 0.001:
        #     signif_str = '****'
        # elif _pval < 0.005:
        #     signif_str = '***'
        elif _pval < 0.01:
            signif_str = '**'
        else:
            signif_str = '*'
        ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + y_offset[i],
            signif_str,
            ha='center',
            va='center',
            fontsize=fontsize
        )


def _format_ax(
    ax: Axes,
    # major_multiple: Optional[float] = None,
    # minor_multiple: Optional[float] = None,
    n_major_ticks: Optional[int] = None,
    n_minor_to_major: Optional[int] = None,
    x_rotation: float = 0.0,
    ylims: Optional[Tuple[float, float]] = None
):
    ax.tick_params(axis='x', rotation=x_rotation)
    # add minor ticks
    ax.minorticks_on()
    # turn it off on x axis
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    # format axis ticks
    ax.tick_params(axis='y', which='major', direction='out', length=7, width=2)
    ax.tick_params(axis='y', which='minor', direction='out', length=3.5, width=2)
    ax.tick_params(axis='x', which='major', direction='out', length=7, width=2)
    # define the multiples at which minor and major ticks are placed
    # if major_multiple is not None:
    #     ax.yaxis.set_major_locator(MultipleLocator(major_multiple))
    # if minor_multiple is not None:
    #     ax.yaxis.set_minor_locator(MultipleLocator(minor_multiple))
    if n_major_ticks is not None:
        major_locator = MaxNLocator(nbins=n_major_ticks, min_n_ticks=n_major_ticks)
        ax.yaxis.set_major_locator(major_locator)
    if n_minor_to_major is not None:
        # minor_locator = MaxNLocator(nbins=n_minor_ticks, min_n_ticks=n_minor_ticks)
        minor_locator = AutoMinorLocator(n_minor_to_major)
        ax.yaxis.set_minor_locator(minor_locator)
    # make axis end aligned with the last major tick, ax.get_yticks() returns the major ticks
    # we also assume the minimal y value is 0!
    ticks = [tick for tick in ax.get_yticks()]
    if ylims is None:
        ax.set_ylim(0.0, ticks[-1])
    else:
        ax.set_ylim(ylims[0], ylims[1])
    # turn off top spine of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # hide x axis label
    ax.set_xlabel('')


def _get_condition_from_cell_ID(agg_output: pd.DataFrame) -> str:
    if len(agg_output) == 0:
        return "Unknown"
    else:
        s_condition: pd.Series = agg_output["cell_ID"].str.split("_").str[0]
        assert (s_condition == s_condition.iloc[0]).all(), f"Not all conditions are the same!\n{s_condition}"
        return s_condition.iloc[0]


def plot_processing(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter,
    peak_detector: PeakDetector,
    s_stdev: Optional[pd.Series] = None
):
    """Plot the data processing to check results.
    """
    cell_nr_list = [c for c in df_input.columns if c != time_colname]
    df_proc, df_peaks = process_data(
        df_input=df_input,
        time_colname='time(m)',
        baseline_fitter=baseline_fitter,
        peak_detector=peak_detector,
        s_stdev=s_stdev
    )
    df_peaks = df_peaks.set_index("cell_nr")
    for cell_nr in cell_nr_list:
        # plot removed baseline
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(df_input[time_colname], df_input[cell_nr])
        axes[0].plot(df_proc[time_colname], df_proc[f"{cell_nr}_baseline"])
        plot_text: str = f"nr {str(cell_nr)}"
        if s_stdev is not None:
            _stdev = str(round(s_stdev.loc[cell_nr], 2))
            plot_text += f": noise_stdev={_stdev}"
        axes[0].text(
            0.01,
            0.99,
            plot_text,
            color="darkred",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=axes[0].transAxes,
        )
        # plot the index and amplitude cutoffs we used to filter the peaks
        peak_detector.plot_filters_on_ax(ax=axes[1], x_series=df_proc[time_colname])
        # plot the detected peaks on the plot, use green for the ones that are filtered out
        if s_stdev is not None:
            peak_detector.plot_detection_on_ax(
                ax=axes[1],
                trace_arr=df_proc[cell_nr],
                x_series=df_proc[time_colname],
                stdev_lower_bound=s_stdev.loc[cell_nr]
            )
        else:
            peak_detector.plot_detection_on_ax(
                ax=axes[1], trace_arr=df_proc[cell_nr], x_series=df_proc[time_colname],
            )
        fig.show()


def plot_peak_detection(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter,
    peak_detectors: List[PeakDetector],
    plot_names: List[str],
    stdev: float,
):
    assert df_input.shape[1] == 2
    cell_nr: int = [c for c in df_input.columns if c != time_colname][0]
    # We run the processing but don't yet filter based on min_peak_amplitude!
    _dict_peaks: Dict[str, list] = {
        "peak_row_idx": [],
        time_colname: [],
        "amplitude": [],
        "plot_name": []
    }
    for detector, plot_name in zip(peak_detectors, plot_names):
        trace_processor = TraceProcessor(
            baseline_fitter=baseline_fitter, peak_detector=detector
        )
        df_proc = df_input[[time_colname]].copy()
        # dict for holding intermediate result over the loop
        
        output: Dict[str, NDArray] = trace_processor.process_trace(
            s_trace=df_input[cell_nr], stdev_lower_bound=stdev
        )
        df_proc[cell_nr] = output["trace_proc"]
        df_proc[f"{cell_nr}_baseline"] = output["baseline"]
        _peak_row_idx = output["peak_row_idx"]
        _dict_peaks["peak_row_idx"] += _peak_row_idx.tolist()
        _dict_peaks[time_colname] += df_proc[time_colname].iloc[_peak_row_idx].tolist()
        _dict_peaks["amplitude"] += output["amplitudes"].tolist()
        _dict_peaks["plot_name"] += [plot_name] * len(_peak_row_idx)
    df_peaks = pd.DataFrame(_dict_peaks).set_index("plot_name")
    for detector, plot_name in zip(peak_detectors, plot_names):
        earliest_peak_idx: int = detector.earliest_peak_idx
        # plot removed baseline
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(df_input[time_colname], df_input[cell_nr])
        axes[0].plot(df_proc[time_colname], df_proc[f"{cell_nr}_baseline"])

        # plot peaks on traces after baseline removal
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[1].plot(df_proc[time_colname], df_proc[cell_nr])
        axes[1].axvline(x=df_proc[time_colname].iloc[earliest_peak_idx], color="black")
        _stdev = str(round(stdev, 2))
        axes[0].text(
            0.01,
            0.99,
            f"nr {str(cell_nr)}: noise_stdev={_stdev}",
            color="darkred",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=axes[0].transAxes,
        )
        # min_peak_amplitude: float = detector.min_peak_amplitude
        _peak_idx: pd.Series = df_peaks.loc[[plot_name], "peak_row_idx"]
      
        axes[1].vlines(
            x=df_proc[time_colname].iloc[_peak_idx],
            ymin=np.zeros(shape=len(_peak_idx)),
            ymax=df_proc[cell_nr].iloc[_peak_idx],
            # colors=_color_list,
            color="red",
            linewidths=1,
        )
        # mark earliest_peak_idx cutoff on the plot

        fig.show()


def calculate_peak_aggregates(df_peaks: pd.DataFrame) -> pd.DataFrame:
    # finding the time index for the maximal peaks is more complex than the rest
    meax_peak_time = df_peaks.loc[
        df_peaks.groupby(["cell_nr"])["amplitude"].idxmax(), ["cell_nr", "time(m)"]
    ].rename({"time(m)": "max_peak_time"}, axis=1)
    df_aggregates = (
        df_peaks.groupby("cell_nr")
        # most of the aggregations are simple
        .agg(
            # if there is no peak, the time should be nan
            peak_count=("time(m)", "count"),
            first_peak_time=("time(m)", "min"),
            mean_peak_amplitude=("amplitude", "mean"),
            # median_peak_amplitude=("amplitude", "median"),
            max_peak_amplitude=("amplitude", "max"),
        ).reset_index()
        # join the time for the max peak for every cell
        .merge(meax_peak_time, on="cell_nr", how="inner")
    )
    return df_aggregates


def format_peak_data(
    df_peaks: pd.DataFrame,
    condition_str: str,
    file_str: str,
) -> pd.DataFrame:
    id_prefix: str = f"{condition_str}_{file_str}"
    df_peaks["peak_ID"] = df_peaks["cell_nr"].apply(lambda x: f"{id_prefix}_{x}")
    # return df_peaks.drop(["cell_nr", "peak_row_idx"], axis=1)
    return df_peaks.drop(["peak_row_idx"], axis=1)


def format_agg_data(
    df_agg: pd.DataFrame,
    condition_str: str,
    file_str: str,
) -> pd.DataFrame:
    id_prefix: str = f"{condition_str}_{file_str}"
    df_agg["cell_ID"] = df_agg["cell_nr"].apply(lambda x: f"{id_prefix}_{x}")
    # we drop the cell_nr as a new one will be assigned that is unique over all the files
    # return df_agg.drop(["cell_nr"], axis=1)
    return df_agg


def two_sample_t_test(sample0, sample1, equal_var: bool) -> Dict[str, float]:
    return stats.ttest_ind(a=sample0, b=sample1, equal_var=equal_var).pvalue
