from abc import abstractmethod
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal
from typing import Dict, List, Tuple, Union, Literal
from matplotlib.axes import Axes
from textwrap import dedent


class PeakDetector:
    def __init__(
        self,
        earliest_peak_pos: int = 0,
        min_peak_amplitude: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """A generic class for peak detection.

        It can filter peaks based on position and absolute size, but cannot detect peaks itself.

        Args:
            earliest_peak_pos (int, optional): The earliest row index at which a peak can be detected, all peaks before
                this index get filtered out. Defaults to 0.
            min_peak_amplitude (Tuple[float, float], optional): Filter out all detected peaks under this threshold. A
                tuple of two values, being the threshold at the start and the treshold at the end, with the values
                linearly interploated inbetween. Defaults to (0.0, 0.0).
        """
        self.earliest_peak_pos: int = earliest_peak_pos
        self.min_peak_amplitude: Tuple[float, float] = min_peak_amplitude

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        The detected peaks are filtered, such that no peak is allowed before position
        {self.earliest_peak_pos}, and every peak has to be above a threshold linearly interpolated between
        {self.min_peak_amplitude[0]} at the start and {self.min_peak_amplitude[1]} at the end.\n"""
        return dedent(descr)

    def get_peaks(self, trace_arr: NDArray, **kwargs) -> Dict[str, NDArray]:
        trace_arr_coerced: NDArray = self._coerce_to_1d_array(trace_arr)
        peak_pos: NDArray = self._find_peak_pos(trace_arr=trace_arr_coerced, **kwargs)
        amplitudes: NDArray = self._get_amplitudes(
            trace_arr=trace_arr_coerced, peak_pos=peak_pos
        )
        peak_pos, amplitudes = self._filter_peaks(
            peak_pos=peak_pos, amplitudes=amplitudes, trace_length=len(trace_arr_coerced)
        )
        return {"peak_pos": peak_pos, "amplitudes": amplitudes}

    def plot_filters_on_ax(self, ax: Axes, x_series: pd.Series):
        # mark earliest_peak_pos cutoff on the plot
        ax.axvline(x=x_series.iloc[self.earliest_peak_pos], color="black", alpha=0.6)
        amplitude_thresholds: NDArray = np.linspace(
            self.min_peak_amplitude[0],
            self.min_peak_amplitude[1],
            len(x_series) - self.earliest_peak_pos,
        )
        ax.plot(
            x_series.iloc[self.earliest_peak_pos :],
            amplitude_thresholds,
            color="black",
            ls="--",
            alpha=0.6,
        )

    def plot_detection_on_ax(
        self, ax: Axes, trace_arr: NDArray, x_series: pd.Series, **kwargs
    ):
        trace_arr_corced: NDArray = self._coerce_to_1d_array(trace_arr)
        peak_pos_all: NDArray = self._find_peak_pos(
            trace_arr=trace_arr_corced, **kwargs
        )
        amplitudes_all: NDArray = self._get_amplitudes(
            trace_arr=trace_arr_corced, peak_pos=peak_pos_all
        )
        peak_pos_filt, amplitudes_filt = self._filter_peaks(
            peak_pos=peak_pos_all,
            amplitudes=amplitudes_all,
            trace_length=len(trace_arr_corced),
        )
        _color_list = ["red" if i in peak_pos_filt else "gray" for i in peak_pos_all]
        # plot baseline
        ax.axhline(y=0, color="tab:orange", linestyle="--", linewidth=1, alpha=1.0)
        # plot trace
        ax.plot(x_series, trace_arr, color="tab:blue", alpha=1.0)
        # this only allows a single color! use scatter for dynamic color
        # ax.plot(
        #     x_series.iloc[peak_pos_filt],
        #     amplitudes_filt,
        #     "o",
        #     markersize=2,
        #     c="red"
        # )
        ax.vlines(
            x=x_series.iloc[peak_pos_all],
            ymin=np.zeros(shape=len(peak_pos_all)),
            ymax=amplitudes_all,
            colors=_color_list,
            linewidths=1,
        )
        return

    def _get_amplitudes(self, trace_arr: NDArray, peak_pos: NDArray) -> NDArray:
        assert isinstance(trace_arr, np.ndarray)
        return trace_arr[peak_pos]

    def _filter_peaks(
        self, peak_pos: NDArray, amplitudes: NDArray, trace_length: int
    ) -> Tuple[NDArray, NDArray]:
        """Filter the detected peaks based.

        Args:
            peak_pos (NDArray): The positions at which peaks were detected.
            amplitudes (NDArray): The amplitudes for the detected peaks.
            trace_length (int): The total length of the traces on which the peaks were detected
                (including the part before earliest_peak_pos)

        Returns:
            Tuple[NDArray, NDArray]: The filtered positions at which peaks were detected and
                the corresponding amplitudes.
        """
        # if there are no peaks just return the empty vectors
        if len(peak_pos) == 0:
            return peak_pos, amplitudes
        # apply filter based on time cutoff
        idx_filter: NDArray = self._is_peak_late_enough(peak_pos=peak_pos)
        peak_pos = peak_pos[idx_filter]
        amplitudes = amplitudes[idx_filter]
        # if there are no peaks left just return the empty vectors
        if len(peak_pos) == 0:
            return peak_pos, amplitudes
        # apply filter based on minimum aplitude required at a given position
        ampl_filter: NDArray = self._is_peak_large_enough(
            amplitudes=amplitudes, peak_pos=peak_pos, trace_length=trace_length
        )
        return peak_pos[ampl_filter], amplitudes[ampl_filter]

    def _is_peak_late_enough(self, peak_pos: NDArray) -> NDArray:
        """Generate boolean vector indicating whether a given index can be a peak based on the position.

        Args:
            peak_pos (NDArray): The positions at which peaks were detected.

        Returns:
            NDArray: boolean vector
        """
        return peak_pos >= self.earliest_peak_pos

    def _is_peak_large_enough(
        self, amplitudes: NDArray, peak_pos: NDArray, trace_length: int
    ) -> NDArray:
        """Generate boolean vector indicating whether a given amplitude can be a peak based on the amplitude.

        Args:
            amplitudes (NDArray): The amplitudes for the detected peaks.
            peak_pos (NDArray): The indices for the detected peaks.
            trace_length (int): The total length of the traces on which the peaks were detected
                (including the part before earliest_peak_pos)
        Returns:
            NDArray: boolean vector
        """
        assert len(amplitudes) > 0
        # the linear treshold starts at the earliest_peak_pos
        amplitude_thresholds_at_idx: NDArray = np.linspace(
            self.min_peak_amplitude[0],
            self.min_peak_amplitude[1],
            trace_length - self.earliest_peak_pos,
        )
        assert peak_pos[0] >= self.earliest_peak_pos
        return (
            amplitudes >= amplitude_thresholds_at_idx[peak_pos - self.earliest_peak_pos]
        )

    def _coerce_to_1d_array(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray
        if isinstance(s_trace, pd.Series):
            arr_trace = s_trace.to_numpy()
        elif isinstance(s_trace, np.ndarray):
            arr_trace = s_trace
            if len(arr_trace.shape) == 1:
                pass
            elif len(arr_trace.shape) == 2:
                arr_trace = arr_trace[:, 0]
            else:
                raise ValueError(
                    f"Input must be vector, but has shape {arr_trace.shape}"
                )
        else:
            raise ValueError(
                f"Input must be pandas.Series or numpy.ndarray, but is {type(s_trace)}"
            )
        return arr_trace

    @abstractmethod
    def _find_peak_pos(self, trace_arr: NDArray, **kwargs) -> NDArray:
        pass


class ScipyPeakDetector(PeakDetector):
    def __init__(
        self,
        wlen: int,
        distance: int = 1,
        earliest_peak_pos: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            earliest_peak_pos=earliest_peak_pos,
            min_peak_amplitude=min_peak_amplitude,
            **kwargs,
        )
        self.wlen: int = wlen
        self.distance: int = distance

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        The peaks are detected using scipy.signal.find_peaks() using arguments wlen={self.wlen},
        distance={self.distance} and an exogenous value for prominence.\n"""
        descr = dedent(descr) + super().description
        return descr

    def _find_peak_pos(self, trace_arr: NDArray, prominence: float) -> NDArray:
        return signal.find_peaks(
            trace_arr, prominence=prominence, wlen=self.wlen, distance=self.distance
        )[0]


class NoiseSDScipyPeakDetector(ScipyPeakDetector):
    def __init__(
        self,
        prominence_in_stdev: float,
        wlen: int,
        distance: int = 1,
        earliest_peak_pos: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            earliest_peak_pos=earliest_peak_pos,
            min_peak_amplitude=min_peak_amplitude,
            wlen=wlen,
            distance=distance,
            **kwargs,
        )
        self.prominence_in_stdev: float = prominence_in_stdev

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        The value for prominence is determined as {self.prominence_in_stdev} times
        the noise level of the trace, which is an exogenous feature.\n"""
        descr = dedent(descr) + super().description
        return descr

    def get_peaks(
        self, trace_arr: NDArray, stdev_lower_bound: float
    ) -> Dict[str, NDArray]:
        """Find the peaks and their amplitudes.

        Args:
            trace_arr (NDArray): _description_
            stdev_lower_bound (float): _description_

        Returns:
            Dict[str, NDArray]: _description_
        """
        prominence: float = self.prominence_in_stdev * stdev_lower_bound
        return super().get_peaks(trace_arr=trace_arr, prominence=prominence)

    def plot_detection_on_ax(
        self,
        ax: Axes,
        trace_arr: NDArray,
        x_series: pd.Series,
        stdev_lower_bound: float,
    ):
        prominence: float = self.prominence_in_stdev * stdev_lower_bound
        super().plot_detection_on_ax(
            ax=ax, trace_arr=trace_arr, x_series=x_series, prominence=prominence
        )
        plot_text = (
            f"prominence={self.prominence_in_stdev}*{str(round(stdev_lower_bound, 2))}"
        )
        ax.text(
            0.01,
            0.99,
            plot_text,
            color="darkred",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        return


class RobustNoiseSDScipyPeakDetector(NoiseSDScipyPeakDetector):
    def __init__(
        self,
        prominence_in_stdev: float,
        wlen: int,
        distance: int = 1,
        earliest_peak_pos: int = 0,
        min_peak_amplitude: float = 0.0,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.35,
        which_stdev: Literal["min", "max", "mean", "estimated", "exogenous"] = "min",
        **kwargs,
    ) -> None:
        super().__init__(
            earliest_peak_pos=earliest_peak_pos,
            min_peak_amplitude=min_peak_amplitude,
            wlen=wlen,
            distance=distance,
            prominence_in_stdev=prominence_in_stdev,
            **kwargs,
        )
        assert which_stdev in ["min", "max", "mean", "estimated", "exogenous"]
        self.which_stdev = which_stdev
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        assert self.lower_quantile < self.upper_quantile

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """

        descr: str = f"""
        An estimate of the noise level is calculated using the standard deviation of the data between the quantiles
        {self.lower_quantile} and {self.upper_quantile}. The used noise value is """
        if self.which_stdev == "min":
            descr += "the minimum of the estimated value and the exogenous standard deviation value."
        elif self.which_stdev == "max":
            descr += "the maximum of the estimated value and the exogenous standard deviation value."
        elif self.which_stdev == "mean":
            descr += "the average of the estimated value and the exogenous standard deviation value."
        elif self.which_stdev == "estimated":
            descr += "the estimated value."
        elif self.which_stdev == "exogenous":
            descr += "the exogenous value."
        else:
            raise NotImplementedError()
        descr = dedent(descr + "\n") + super().description
        return descr

    def determine_stdev_lower_bound(self, stdev_default: float, stdev_estim: float) -> float:
        """_summary_

        Args:
            stdev_default (float): _description_
            stdev_estim (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_
        """
        stdev_lower_bound: float
        if self.which_stdev == "min":
            stdev_lower_bound = min(stdev_default, stdev_estim)
        elif self.which_stdev == "max":
            stdev_lower_bound = max(stdev_default, stdev_estim)
        elif self.which_stdev == "mean":
            stdev_lower_bound = np.mean(stdev_default, stdev_estim)
        elif self.which_stdev == "estimated":
            stdev_lower_bound = stdev_estim
        elif self.which_stdev == "exogenous":
            stdev_lower_bound = stdev_default
        else:
            raise NotImplementedError()
        return stdev_lower_bound

    def get_peaks(
        self, trace_arr: NDArray, stdev_lower_bound: float
    ) -> Dict[str, NDArray]:
        """Find the peaks and their amplitudes.

        Args:
            trace_arr (NDArray): _description_
            stdev_lower_bound (float): _description_

        Returns:
            Dict[str, NDArray]: _description_
        """
        stdev_estim: float = self._estimate_noise_stdev(trace_arr)
        return super().get_peaks(
            trace_arr=trace_arr,
            stdev_lower_bound=self.determine_stdev_lower_bound(
                stdev_default=stdev_lower_bound,
                stdev_estim=stdev_estim
            )
        )

    def plot_detection_on_ax(
        self,
        ax: Axes,
        trace_arr: NDArray,
        x_series: pd.Series,
        stdev_lower_bound: float,
    ):
        stdev_estim: float = self._estimate_noise_stdev(trace_arr)
        stdev_lower_bound: float = self.determine_stdev_lower_bound(
            stdev_default=stdev_lower_bound,
            stdev_estim=stdev_estim
        )
        super().plot_detection_on_ax(
            ax=ax,
            trace_arr=trace_arr,
            x_series=x_series,
            stdev_lower_bound=stdev_lower_bound,
        )
        return

    def _estimate_noise_stdev(self, trace_arr: NDArray) -> float:
        # calculate prominence threshold in robust sd units (based on interquartile distance)
        # our reference distribution will be a lognormal instead of a normal
        lower, upper = np.quantile(trace_arr[self.earliest_peak_pos :], [self.lower_quantile, self.upper_quantile])
        noise_band = np.logical_and(
            trace_arr[self.earliest_peak_pos :] >= lower,
            trace_arr[self.earliest_peak_pos :] <= upper,
        )
        return np.std(trace_arr[self.earliest_peak_pos :][noise_band])


class TamasPeakDetector(PeakDetector):
    def __init__(
        self, earliest_peak_pos: int = 0, min_peak_amplitude: float = 0.0, **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_pos=earliest_peak_pos,
            min_peak_amplitude=min_peak_amplitude,
            **kwargs,
        )

    def _find_peak_pos(self, trace_arr: NDArray, threshold: float) -> NDArray:
        """_summary_

        1. identify all points above the threshold
        2. group the these points into continous segments
        3. find the maximum in each group
        """
        _above_threshold: NDArray = trace_arr > threshold
        # a segment boundary is where 0 -> 1 or 1 -> 0 transition happens
        segment_boundaries: NDArray = (
            np.where(np.diff(_above_threshold.astype("int")) != 0)[0] + 1
        )
        # split the values into consecutive segments, for now this includes the below threshold parts too
        segments: List[NDArray] = np.split(trace_arr, segment_boundaries)
        # keep track of absolute position of each value on the segments
        segment_abs_idx: List[NDArray] = np.split(
            np.arange(0, len(trace_arr)), segment_boundaries
        )
        # find the postion of the max in each segment
        segments_max_idx: List[int] = [x.argmax() for x in segments]
        # convert to absolute position (and array format)
        segment_max_abs_idx: NDArray = np.array(
            [x[i] for x, i in zip(segment_abs_idx, segments_max_idx)]
        )
        # get the max values by segment
        segments_max_vals: NDArray = np.take(trace_arr, segment_max_abs_idx)
        # filter maxima above the threshold
        _above_threshold: NDArray = segments_max_vals > threshold
        return segment_max_abs_idx[_above_threshold]


class SingleMaxPeakDetector(PeakDetector):
    def _find_peak_pos(self, trace_arr: NDArray) -> NDArray:
        row_idx_of_max: int = (
            np.argmax(trace_arr[self.earliest_peak_pos :]) + self.earliest_peak_pos
        )
        assert isinstance(row_idx_of_max, np.int64), type(row_idx_of_max)
        return np.array([row_idx_of_max])

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        Only a single peak is detected, which is the largest point in the segment.\n"""
        descr = dedent(descr) + super().description
        return descr
