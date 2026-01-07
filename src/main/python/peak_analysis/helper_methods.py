import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy import stats

from typing import Literal
from numpy.typing import NDArray
from itertools import product

from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from peak_analysis import (
    BaseLineFitter,
    PeakDetector,
    TraceProcessor,
)


def process_data(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter | None,
    peak_detector: PeakDetector,
    s_stdev: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the data.

    Args:
        df_input (pd.DataFrame): Input dataframe with time index and the cell traces as columns
            (time_colname, 0, 1, ...).
        time_colname (str): Column name for the time index
        baseline_fitter (BaseLineFitter | None): The object for fitting the baseline of the traces.
        peak_detector (PeakDetector): The object for detecting peaks.
        s_stdev (pd.Series | None, optional): A series of exogenous standard deviation values, one per trace. If
            specified, it acts as a lower bound for the stdev. estimate used for defining the prominences for the peak
            detection. Defaults to None, in which case the stdev. is estimated using the trace.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of dataframes, one with the traces after the removal of the
            baselines, the other with the detected peaks
    """
    trace_processor = TraceProcessor(
        baseline_fitter=baseline_fitter, peak_detector=peak_detector
    )
    cell_ID_list: list[str] = [c for c in df_input.columns if c != time_colname]
    df_proc = df_input[[time_colname]].copy()
    # dict for holding intermediate result over the loop
    _dict_peaks: dict[str, list] = {
        "cell_ID": [],
        "peak_pos": [],
        time_colname: [],
        "amplitude": [],
    }
    # go over the cell traces, remove baseline and identify peaks based on prominence
    for cell_ID in cell_ID_list:
        output: dict[str, NDArray]
        # There is often an error when processing traces, but the error won't return the actual trace that causes the
        # error by default. -> print the culprit before throwing the error to make debugging faster
        try:
            # this is hacky, TODO: clean this up
            if s_stdev is not None:
                output = trace_processor.process_trace(
                    s_trace=df_input[cell_ID], stdev_lower_bound=s_stdev.loc[cell_ID]
                )
            else:
                output = trace_processor.process_trace(s_trace=df_input[cell_ID])
        except Exception as e:
            print(f"Error when processing trace {cell_ID}")
            raise e
        df_proc[cell_ID] = output["trace_proc"]
        df_proc[f"{cell_ID}_baseline"] = output["baseline"]
        _peak_pos = output["peak_pos"]
        _dict_peaks["peak_pos"] += _peak_pos.tolist()
        _dict_peaks[time_colname] += df_proc[time_colname].iloc[_peak_pos].tolist()
        _dict_peaks["amplitude"] += output["amplitudes"].tolist()
        _dict_peaks["cell_ID"] += [cell_ID] * len(_peak_pos)

    return df_proc, pd.DataFrame(_dict_peaks)


def plot_agg_output_comparison(
    df_agg: pd.DataFrame,
    comparisons: dict[str, list[str]],
    primary_group_column: str,
    secondary_group_column: str | None = None,
    benchmark_group: str | None = None,
    primary_group_order: list[str] | None = None,
    n_major_ticks: int | None = None,
    n_minor_to_major: int | None = None,
    figsize: tuple[float, float] = (3, 4),
    colors: tuple[str, str] = ("tab:red", "tab:blue"),
    alpha: float = 0.8,
    x_rotation: float = 0.0,
    width: float = 0.8,
    group_alias_dict: dict[str, str] = {},
    colname_alias_dict: dict[str, str] = {},
    errorbar_type: str = "se",
    errorbar_alpha: float = 0.05,
    savepath: Path | None = None,
):
    """Plot the response rate.

    agg_output files are structured as follows:
        cell_nr  peak_count  first_peak_time  mean_peak_amplitude  max_peak_amplitude  max_peak_time            cell_ID
    0         1           4          7.70257          163.673578         314.156600           7.70257  EVs_20230901_02_3

    Args:
        df_agg (pd.DataFrame): Dataframe with summary statistics of each trace.
        comparisons (dict[str, str]): {"column in agg_output": ["function_name0", "function_name1"]}
        primary_group_column (str): The main column for creating the comparison groups.
        secondary_group_column (Optional[str]): The secondar column for creating the comparison groups, nested under
            the primary. Defaults to None, in which case only primary grouping is used.
        benchmark_group (Optional[str]): The benchmark group (from the primary group column), plotted first and in
            different color. Defaults to None, in which case a radnom group is selected.
        n_major_ticks (Optional[int], optional): Number of major ticks to use on the y axis, if None the automatic
            setting is used. Defaults to None.
        n_minor_to_major (Optional[int], optional): Number of minor ticks to use for every major one on the y axis,
            if None the automatic setting is used. Defaults to None.
        figsize (tuple[float, float], optional): The figure size for the plot. Defaults to (3, 4).
        colors (tuple[str, str]): The color for the benchmark and the rest as a tuple of strings.
            Defaults to ("tab:red", "tab:blue").
        alpha (float): The opacity for the bars, float between 0 and 1. Defaults to 0.8.
        x_rotation (float): Degree to rotate x axis tick labels with.
        width (float): Scaling factor for the bar width, a value of 1 means there is no gap between the bars.
            Defaults to 0.8.
        group_alias_dict (dict[str, str]): A dict for renaming the ticklabels with an alias name for certain conditions.
            Defaults to {}.
        colname_alias_dict (dict[str, str]): A dict for renaming the labels for certain variables on the plots.
        errorbar_type (str): If "se", the errorbar is +- standard error; otherwise the jeffreys confidence interval
            with 1 - errorbar_alpha width. Defaults to "se".
        errorbar_alpha (float): The width of the confidence interval is +- (1 - alpha / 2), unless errorbar_type is
            "se". Defaults to 0.05.
        savepath: Path | None: The path where the plot should be saved. Defaults to None, in which case the plot isn't
            saved.
    """
    assert errorbar_alpha > 0.0 and errorbar_alpha < 0.5
    group_columns: list[str] = [primary_group_column]
    if secondary_group_column is not None:
        group_columns.append(secondary_group_column)
    # rename all the groups that need renaming
    df_agg[group_columns] = df_agg[group_columns].replace(group_alias_dict)
    # convert the primary_group_column to pd.Categorical so that we can use a custom sort order
    custom_sort_oder: list[str]
    if primary_group_order is not None:
        # rename the sort order with the group aliases too
        primary_group_order = [group_alias_dict.get(c, c) for c in primary_group_order]
        custom_sort_oder = primary_group_order
    else:
        custom_sort_oder = sorted(df_agg[primary_group_column].astype(str).unique())
        if benchmark_group is not None:
            custom_sort_oder = [benchmark_group] + [
                c for c in custom_sort_oder if c != benchmark_group
            ]
    df_agg[primary_group_column] = pd.Categorical(
        df_agg[primary_group_column], custom_sort_oder
    )
    df_agg.sort_values(group_columns, inplace=True)
    df_agg[group_columns] = df_agg[group_columns].astype("str")
    # get the unique group combinations
    group_list: list[tuple[str, ...]] = list(
        df_agg[group_columns].drop_duplicates().itertuples(index=False, name=None)
    )
    # set the groups as the index for easier filtering
    if secondary_group_column is not None:
        df_agg = df_agg.copy().set_index(
            [primary_group_column, secondary_group_column], drop=True
        )
    else:
        df_agg = df_agg.copy().set_index([primary_group_column], drop=True)
    bm_group_chosen: str
    if benchmark_group is None:
        # use the first groups as benchmark
        bm_group_chosen = group_list[0]
    else:
        bm_group_chosen = str(benchmark_group)
    # separate color for the benchmark group
    # TODO: Figure this out for the dual grouping setup!
    if secondary_group_column is None:
        color_list: list[str] = [
            colors[0] if c == benchmark_group else colors[1] for c in custom_sort_oder
        ]
    else:
        color_list: list[str] = [colors[0]] + [colors[1]] * (len(group_list))
    # go over the different comparisons we need to do, for a given column such as mean amplutide
    # multiple comparison types such as mean or max are possible
    for colname, comp_type_list in comparisons.items():
        for comp_type in comp_type_list:
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
            _dict_errorbars: dict[str, list] = {"groups": [], "errorbar": []}
            _dict_plot: dict[str, list] = {"groups": [], metric_name: []}
            # calculate the point estimates and confidence intervals in a single pass
            # over the groups (which can be tuple combinations!)
            group: tuple[str, ...] | str
            for group in group_list:
                group_alias: str = group
                # for single rows df.loc[index, columns] will return pd.Series unless df.loc[[index], columns] is
                # passed, but tuples of length one will cause an error
                if len(group) == 1:
                    group = group[0]
                if comp_type == "response_rate":
                    pct_resp, errorbars = _calculate_proportion_metrics(
                        s_agg_output=df_agg.loc[[group], colname],
                        errorbar_type=errorbar_type,
                        errorbar_alpha=errorbar_alpha,
                    )
                elif comp_type == "mean":
                    pct_resp, errorbars = _calculate_mean_metrics(
                        s_agg_output=df_agg.loc[[group], colname],
                        errorbar_type=errorbar_type,
                        errorbar_alpha=errorbar_alpha,
                    )
                else:
                    raise ValueError()
                _dict_plot[metric_name].append(pct_resp)
                _dict_plot["groups"].append(group_alias)
                _dict_errorbars["errorbar"].append(errorbars)
                _dict_errorbars["groups"].append(group_alias)
            df_plot = pd.DataFrame(_dict_plot)
            df_plot[group_columns] = pd.DataFrame(
                df_plot["groups"].tolist(), index=df_plot.index
            )
            df_errorbars = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    _dict_errorbars["groups"], names=group_columns
                ),
                data=_dict_errorbars["errorbar"],
            )

            fig, ax = _make_barplot(
                df_plot=df_plot,
                x=primary_group_column,
                hue=secondary_group_column,
                y=metric_name,
                colors_for_bars=color_list,
                df_errorbars=df_errorbars,
                alpha=alpha,
                width=width,
                linewidth=1,
                edgecolor="black",
                figsize=figsize,
            )

            # the p value involves multiple combinations
            if secondary_group_column is None:
                pvalue_list: list[float | None] = []
                for group in group_list:
                    # same reason for conversion as above
                    if len(group) == 1:
                        group = group[0]
                    if comp_type == "response_rate":
                        if group != bm_group_chosen:
                            p_value = _calculate_p_value_of_proportions(
                                s_agg_output=df_agg.loc[[group], colname],
                                s_agg_output_bench=df_agg.loc[[bm_group_chosen], colname],
                            )
                        else:
                            p_value = None
                    elif comp_type == "mean":
                        if group != bm_group_chosen:
                            p_value = _calculate_p_value_of_means(
                                s_agg_output=df_agg.loc[[group], colname],
                                s_agg_output_bench=df_agg.loc[[bm_group_chosen], colname],
                            )
                        else:
                            p_value = None
                    else:
                        raise ValueError()
                    pvalue_list.append(p_value)

            if secondary_group_column is None:
                # add significance asterisks, offset vertically by the upper errorbar length + offset
                offset = df_plot[metric_name].max() / 20
                _add_signif_markers_to_bars(
                    ax=ax,
                    pvalue_list=pvalue_list,
                    y_offset=df_errorbars.values.T[1, :] + offset,
                    fontsize=15,
                )
                ylims: tuple[float, float] | None = None
                if comp_type == "response_rate":
                    ylims = (0.0, 100.0)
                _format_ax(
                    ax=ax,
                    n_major_ticks=n_major_ticks,
                    n_minor_to_major=n_minor_to_major,
                    x_rotation=x_rotation,
                    ylims=ylims,
                )
            if savepath is not None:
                fname = f"{comp_type}_{colname}.png"
                print(f"Saving plot to: {savepath / fname}")
                fig.savefig(
                    savepath / fname, bbox_inches="tight", transparent=True
                )
            fig.show()


def _calculate_proportion_metrics(
    s_agg_output: pd.Series,
    errorbar_type: str,
    errorbar_alpha: float = 0.05,
) -> tuple[float, NDArray]:
    errorbars: NDArray
    pct_resp: float
    # if the agg data contains no cells
    if len(s_agg_output) == 0:
        pct_resp = 0.0
        errorbars = np.array([0.0, 0.0])
    else:
        count: int = (s_agg_output > 0).sum()
        nobs: int = len(s_agg_output)
        pct_resp = count / nobs * 100
        if errorbar_type == "se":
            _se: float = ((count / nobs) * (1 - count / nobs) / nobs) ** 0.5
            errorbars = np.array((_se, _se)) * 100
        else:
            # convert from tuple to array so we can do arithmetic on the values
            confint: NDArray = np.array(
                proportion_confint(count, nobs, alpha=errorbar_alpha, method="jeffreys")
            )
            # convert to percentage and errorbar format
            errorbars: NDArray = np.abs(confint * 100 - pct_resp)
    return pct_resp, errorbars


def _calculate_p_value_of_proportions(
    s_agg_output: pd.Series,
    s_agg_output_bench: pd.Series | None = None,
) -> float | None:
    p_value: float | None = None
    # benchmark values for 2 sample significance testing
    if len(s_agg_output) > 0:
        count: int = (s_agg_output > 0).sum()
        nobs: int = len(s_agg_output)
        count_bm: int = (s_agg_output_bench > 0).sum()
        nobs_bm: int = len(s_agg_output_bench)
        p_value = proportions_ztest(
            count=[count, count_bm],
            nobs=[nobs, nobs_bm],
            value=0.0,
            alternative="two-sided",
        )[1]
    return p_value


def _calculate_mean_metrics(
    s_agg_output: pd.Series,
    errorbar_type: Literal["se", "ci"],
    errorbar_alpha: float = 0.05,
) -> tuple[float, NDArray]:
    """Calculate the mean and the errorbars for the mean estimate.

    Args:
        s_agg_output (pd.Series): A series of values, normally 1 per trace.
        errorbar_type (Literal["se", "ci"]): The type of errorbar, "se": standard error, "ci": confidence interval
            (using the `errorbar_alpha` argument).
        errorbar_alpha (float): The confidence interval, ignored if `errorbar_type` is "se". Defaults to 0.05.

    Returns:
        tuple[float, NDArray]: A tuple of the estimated mean and the lower and upper errorbar values in an array.
    """
    # if the agg data contains no cells
    mean: float
    errorbars: NDArray
    if len(s_agg_output) == 0:
        mean = 0.0
        errorbars = np.array([0.0, 0.0])
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
                ci_width_upper = stats.t.ppf(
                    1 - (errorbar_alpha / 2), (len(s_agg_output) - 1), loc=0, scale=se
                )
            errorbars = np.array((ci_width_upper, ci_width_upper))

    return mean, errorbars


def _calculate_p_value_of_means(
    s_agg_output: pd.Series,
    s_agg_output_bench: pd.Series | None = None,
) -> float | None:
    """Calculate the p value for a 2 sample t-test with unequal variance.

    Often there are no traces with any peaks in a group, so None is returned as a fallback mechanism.

    Args:
        s_agg_output (pd.Series): A series of values, normally 1 per trace.
        s_agg_output (pd.Series): A series of benchmark values, normally 1 per trace.

    Returns:
        float | None: The p-value, if the value can be calculated, otherwise None.
    """
    p_value: float | None = None
    if len(s_agg_output) > 0:
        p_value = two_sample_t_test(
            sample0=s_agg_output_bench, sample1=s_agg_output, equal_var=False
        )
    return p_value


def plot_output_distribution(
    agg_outputs,
    columns_to_plot: list[str],
    id_colname: str = "peak_ID",
    n_major_ticks: int | None = None,
    n_minor_to_major: int | None = None,
    x_rotation: float = 0.0,
    width: float = 0.75,
    figsize: tuple[float, float] | None = None,
    condition_alias_dict: dict[str, str] = {},
    colname_alias_dict: dict[str, str] = {},
):
    df_plot: pd.DataFrame = pd.concat(agg_outputs, axis=0)
    df_plot["condition"] = df_plot[id_colname].str.split("_").str[0]
    df_plot["condition"] = df_plot["condition"].replace(condition_alias_dict)
    df_plot = df_plot.rename(columns=colname_alias_dict)
    for c in columns_to_plot:
        c = colname_alias_dict.get(c, c)
        fig, ax = _make_boxplot(
            df_plot=df_plot, x="condition", y=c, width=width, figsize=figsize
        )
        _format_ax(
            ax=ax,
            n_major_ticks=n_major_ticks,
            n_minor_to_major=n_minor_to_major,
            x_rotation=x_rotation,
        )
        _add_counts_to_boxes(ax=ax, df_plot=df_plot, y=c)
        fig.show()


def _make_barplot(
    df_plot: pd.DataFrame,
    x: str,
    y: str,
    colors_for_bars: list[str],
    df_errorbars: pd.DataFrame,
    hue: str | None = None,
    alpha: float = 0.8,
    width: float = 0.8,
    linewidth: float | int = 1,
    edgecolor: str = "black",
    figsize: tuple[float, float] | None = None,
) -> tuple[Axes, Figure]:
    # TODO: Put benchmark first here
    # NOTE: pd.Series.unique() preserves order!
    x_order: list[str] = df_plot[x].unique().tolist()
    hue_order: list[str] | None = None
    if hue is not None:
        hue_order = df_plot[hue].unique().tolist()
        x_hue_order: list[tuple[str, ...]] = list(
            df_plot[[x, hue]].drop_duplicates().itertuples(index=False, name=None)
        )
    else:
        x_hue_order: list[tuple[str, ...]] = list(
            df_plot[[x]].drop_duplicates().itertuples(index=False, name=None)
        )
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        df_plot,
        x=x,
        y=y,
        hue=hue,
        palette=colors_for_bars,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        width=width,
        ax=ax,
        order=x_order,
        hue_order=hue_order,
    )
    # Get the bar width of the plot
    bar_width: float = ax.patches[0].get_width()
    # n_hues = len(hue_order)
    # offset: NDArray = np.linspace(-n_hues / 2, n_hues / 2, n_hues) * bar_width / 1.5

    # patches are not sorted left to right, because they are nested under hue as [x0hue0, x1hue0, ...]
    # -> we need to sort them to get the x_hue_order we are using
    err_pos_x: list[float] = sorted([p.get_x() + bar_width / 2 for p in ax.patches])

    err_pos_y: pd.Series
    err_lims: pd.DataFrame
    # if there is no hue grouping things are very straightforward
    if hue is None:
        err_pos_y = df_plot.set_index([x], drop=True).loc[x_order, y]
        err_lims = df_errorbars.loc[x_hue_order]
    # otherwise some combinations might be missing -> we need to fill in with dummy values
    else:
        err_pos_y: pd.Series = pd.Series(
            index=pd.MultiIndex.from_tuples(product(x_order, hue_order)),
            data=[0] * (len(x_order) * len(hue_order)),
            name=y,
        )
        err_pos_y.loc[x_hue_order] = df_plot.set_index([x, hue], drop=True).loc[
            x_hue_order, y
        ]
        err_lims: pd.DataFrame = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(product(x_order, hue_order)),
            data=np.full([len(x_order) * len(hue_order), 2], 0),
        )
        err_lims.loc[x_hue_order] = df_errorbars.loc[x_hue_order].values
    ax.errorbar(
        x=err_pos_x,
        y=err_pos_y,
        # convert to array shaped as (2, N)
        yerr=err_lims.values.T,
        fmt="none",
        elinewidth=2,
        capsize=0,
        c="k",
    )
    return fig, ax


def _make_boxplot(
    df_plot: pd.DataFrame,
    x: str,
    y: str,
    width: float = 0.75,
    figsize: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        df_plot,
        x=x,
        y=y,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        medianprops={"color": "#D67B77"},
        whiskerprops={"color": "black"},  # whicsker stems
        capprops={"color": "black"},  # whiscep caps
        flierprops={
            "marker": "o",
            "markerfacecolor": "None",
            "markersize": 3,
            "markeredgecolor": "black",
        },
        # palette=colors_for_bars,
        # alpha=alpha,
        # linewidth=linewidth,
        # edgecolor=edgecolor,
        width=width,
        ax=ax,
    )
    return fig, ax


def _add_counts_to_boxes(ax: Axes, df_plot: pd.DataFrame, y: str):
    xtick_loc: dict[str, int] = {
        v.get_text(): v.get_position()[0] for v in ax.get_xticklabels()
    }
    counts_by_group: pd.Series = df_plot.groupby("condition")[y].count()
    ymax_by_group: pd.Series = df_plot.groupby("condition")[y].max()
    offset: float = ymax_by_group.max() * 0.1
    for group, count in counts_by_group.items():
        ax.text(
            x=xtick_loc[group],
            y=ymax_by_group.loc[group] + offset,
            s=str(count),
            ha="center",
        )


def _add_signif_markers_to_bars(
    ax: Axes,
    pvalue_list: list[float],
    y_offset: NDArray | list[float],
    fontsize: float | int = 15,
):
    for i, _bar, _pval in zip(range(len(pvalue_list)), ax.patches, pvalue_list):
        if _pval is None:
            continue
            # signif_str = ""
        elif np.isnan(_pval):
            signif_str = ""
        elif _pval >= 0.05:
            signif_str = ""
        # elif _pval < 0.001:
        #     signif_str = '****'
        # elif _pval < 0.005:
        #     signif_str = '***'
        elif _pval < 0.01:
            signif_str = "**"
        else:
            signif_str = "*"
        ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + y_offset[i],
            signif_str,
            ha="center",
            va="center",
            fontsize=fontsize,
        )


def _format_ax(
    ax: Axes,
    # major_multiple: Optional[float] = None,
    # minor_multiple: Optional[float] = None,
    n_major_ticks: int | None = None,
    n_minor_to_major: int | None = None,
    x_rotation: float = 0.0,
    ylims: tuple[float, float] | None = None,
):
    ax.tick_params(axis="x", rotation=x_rotation)
    # add minor ticks
    ax.minorticks_on()
    # turn it off on x axis
    ax.xaxis.set_tick_params(which="minor", bottom=False)
    # format axis ticks
    ax.tick_params(axis="y", which="major", direction="out", length=7, width=2)
    ax.tick_params(axis="y", which="minor", direction="out", length=3.5, width=2)
    ax.tick_params(axis="x", which="major", direction="out", length=7, width=2)
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # hide x axis label
    ax.set_xlabel("")


def _get_condition_from_cell_ID(agg_output: pd.DataFrame) -> str:
    if len(agg_output) == 0:
        return "Unknown"
    else:
        s_condition: pd.Series = agg_output["cell_ID"].str.split("_").str[0]
        assert (
            s_condition == s_condition.iloc[0]
        ).all(), f"Not all conditions are the same!\n{s_condition}"
        return s_condition.iloc[0]


def plot_processing(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter | None,
    peak_detector: PeakDetector,
    s_stdev: pd.Series | None = None,
):
    """Plot the data processing to check results."""
    cell_ID_list = [c for c in df_input.columns if c != time_colname]
    df_proc, df_peaks = process_data(
        df_input=df_input,
        time_colname="time(m)",
        baseline_fitter=baseline_fitter,
        peak_detector=peak_detector,
        s_stdev=s_stdev,
    )
    df_peaks = df_peaks.set_index("cell_ID")
    for cell_ID in cell_ID_list:
        # plot removed baseline
        fig, axes = plt.subplots(1, 2, figsize=(7, 2))
        axes[0].plot(df_input[time_colname], df_input[cell_ID])
        axes[0].plot(df_proc[time_colname], df_proc[f"{cell_ID}_baseline"])
        fig.suptitle(str(cell_ID))
        if s_stdev is not None:
            _stdev = str(round(s_stdev.loc[cell_ID], 2))
            plot_text = f"noise_stdev={_stdev}"
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
                trace_arr=df_proc[cell_ID],
                x_series=df_proc[time_colname],
                stdev_lower_bound=s_stdev.loc[cell_ID],
            )
        else:
            peak_detector.plot_detection_on_ax(
                ax=axes[1],
                trace_arr=df_proc[cell_ID],
                x_series=df_proc[time_colname],
            )
        fig.show()


def plot_peak_detection(
    df_input: pd.DataFrame,
    time_colname: str,
    baseline_fitter: BaseLineFitter,
    peak_detectors: list[PeakDetector],
    plot_names: list[str],
    stdev: float,
):
    assert df_input.shape[1] == 2
    cell_ID: int = [c for c in df_input.columns if c != time_colname][0]
    # We run the processing but don't yet filter based on min_peak_amplitude!
    _dict_peaks: dict[str, list] = {
        "peak_pos": [],
        time_colname: [],
        "amplitude": [],
        "plot_name": [],
    }
    for detector, plot_name in zip(peak_detectors, plot_names):
        trace_processor = TraceProcessor(
            baseline_fitter=baseline_fitter, peak_detector=detector
        )
        df_proc = df_input[[time_colname]].copy()
        # dict for holding intermediate result over the loop

        output: dict[str, NDArray] = trace_processor.process_trace(
            s_trace=df_input[cell_ID], stdev_lower_bound=stdev
        )
        df_proc[cell_ID] = output["trace_proc"]
        df_proc[f"{cell_ID}_baseline"] = output["baseline"]
        _peak_pos = output["peak_pos"]
        _dict_peaks["peak_pos"] += _peak_pos.tolist()
        _dict_peaks[time_colname] += df_proc[time_colname].iloc[_peak_pos].tolist()
        _dict_peaks["amplitude"] += output["amplitudes"].tolist()
        _dict_peaks["plot_name"] += [plot_name] * len(_peak_pos)
    df_peaks = pd.DataFrame(_dict_peaks).set_index("plot_name")
    for detector, plot_name in zip(peak_detectors, plot_names):
        earliest_peak_idx: int = detector.earliest_peak_idx
        # plot removed baseline
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(df_input[time_colname], df_input[cell_ID])
        axes[0].plot(df_proc[time_colname], df_proc[f"{cell_ID}_baseline"])

        # plot peaks on traces after baseline removal
        axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axes[1].plot(df_proc[time_colname], df_proc[cell_ID])
        axes[1].axvline(x=df_proc[time_colname].iloc[earliest_peak_idx], color="black")
        _stdev = str(round(stdev, 2))
        axes[0].text(
            0.01,
            0.99,
            f"nr {str(cell_ID)}: noise_stdev={_stdev}",
            color="darkred",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=axes[0].transAxes,
        )
        # min_peak_amplitude: float = detector.min_peak_amplitude
        _peak_idx: pd.Series = df_peaks.loc[[plot_name], "peak_pos"]

        axes[1].vlines(
            x=df_proc[time_colname].iloc[_peak_idx],
            ymin=np.zeros(shape=len(_peak_idx)),
            ymax=df_proc[cell_ID].iloc[_peak_idx],
            # colors=_color_list,
            color="red",
            linewidths=1,
        )
        # mark earliest_peak_idx cutoff on the plot

        fig.show()


def calculate_peak_aggregates(df_peaks: pd.DataFrame) -> pd.DataFrame:
    # finding the time index for the maximal peaks is more complex than the rest
    meax_peak_time = df_peaks.loc[
        df_peaks.groupby(["cell_ID"])["amplitude"].idxmax(), ["cell_ID", "time(m)"]
    ].rename({"time(m)": "max_peak_time"}, axis=1)
    df_aggregates = (
        df_peaks.groupby("cell_ID")
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
        .merge(meax_peak_time, on="cell_ID", how="inner")
    )
    return df_aggregates


def format_peak_data(
    df_peaks: pd.DataFrame,
    condition_str: str,
    file_str: str,
) -> pd.DataFrame:
    id_prefix: str = f"{condition_str}_{file_str}"
    df_peaks["peak_ID"] = df_peaks["cell_nr"].apply(lambda x: f"{id_prefix}_{x}")
    # return df_peaks.drop(["cell_nr", "peak_pos"], axis=1)
    return df_peaks.drop(["peak_pos"], axis=1)


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


def two_sample_t_test(sample0, sample1, equal_var: bool) -> dict[str, float]:
    return stats.ttest_ind(a=sample0, b=sample1, equal_var=equal_var).pvalue
