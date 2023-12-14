from abc import abstractmethod
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal
from typing import Dict, List, Tuple
from matplotlib.axes import Axes


class PeakDetector:
    def __init__(
        self, earliest_peak_idx: int = 0, min_peak_amplitude: Tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """_summary_

        Args:
            earliest_peak_idx (int, optional): The earliest row index at which a peak can be detected, all peaks
                before this index get filtered out. Defaults to 0.
            min_peak_amplitude (Tuple[float, float], optional): Filter out all detected peaks under this threshold.
                A tuple of two values, being the threshold at the start and the treshold at the end, with the values
                linearly interploated inbetween. Defaults to (0.0, 0.0).
        """
        self.earliest_peak_idx: int = earliest_peak_idx
        self.min_peak_amplitude: Tuple[float, float] = min_peak_amplitude

    def _filter_peaks(
        self, peak_row_idx: NDArray, amplitudes: NDArray, trace_length: int
    ) -> Tuple[NDArray, NDArray]:
        """Filter the detected peaks based.

        Args:
            peak_row_idx (NDArray): The positions at which peaks were detected.
            amplitudes (NDArray): The amplitudes for the detected peaks.
            trace_length (int): The total length of the traces on which the peaks were detected
                (including the part before earliest_peak_idx)

        Returns:
            Tuple[NDArray, NDArray]: The filtered positions at which peaks were detected and
                the corresponding amplitudes.
        """
        # if there are no peaks just return the empty vectors
        if len(peak_row_idx) == 0:
            return peak_row_idx, amplitudes
        # apply filter based on time cutoff
        idx_filter: NDArray = self._is_peak_late_enough(
            peak_row_idx=peak_row_idx
        )
        peak_row_idx = peak_row_idx[idx_filter]
        amplitudes = amplitudes[idx_filter]
        # if there are no peaks left just return the empty vectors
        if len(peak_row_idx) == 0:
            return peak_row_idx, amplitudes
        # apply filter based on minimum aplitude required at a given position
        ampl_filter: NDArray = self._is_peak_large_enough(
            amplitudes=amplitudes, peak_row_idx=peak_row_idx, trace_length=trace_length
        )
        return peak_row_idx[ampl_filter], amplitudes[ampl_filter]

    def _is_peak_late_enough(
        self, peak_row_idx: NDArray
    ) -> NDArray:
        """Generate boolean vector indicating whether a given index can be a peak based on the position.

        Args:
            peak_row_idx (NDArray): The positions at which peaks were detected.

        Returns:
            NDArray: boolean vector
        """
        return peak_row_idx >= self.earliest_peak_idx

    def _is_peak_large_enough(
        self, amplitudes: NDArray, peak_row_idx: NDArray, trace_length: int
    ) -> NDArray:
        """Generate boolean vector indicating whether a given amplitude can be a peak based on the amplitude.

        Args:
            amplitudes (NDArray): The amplitudes for the detected peaks.
            peak_row_idx (NDArray): The indices for the detected peaks.
            trace_length (int): The total length of the traces on which the peaks were detected
                (including the part before earliest_peak_idx)
        Returns:
            NDArray: boolean vector
        """
        assert len(amplitudes) > 0
        # the linear treshold starts at the earliest_peak_idx
        amplitude_thresholds_at_idx: NDArray = np.linspace(
            self.min_peak_amplitude[0],
            self.min_peak_amplitude[1],
            trace_length - self.earliest_peak_idx
        )
        assert peak_row_idx[0] >= self.earliest_peak_idx
        return amplitudes >= amplitude_thresholds_at_idx[peak_row_idx - self.earliest_peak_idx]
    
    def plot_filters_on_ax(self, ax: Axes, x_series: pd.Series):
        # mark earliest_peak_idx cutoff on the plot
        ax.axvline(x=x_series.iloc[self.earliest_peak_idx], color="black", alpha=0.6)
        amplitude_thresholds: NDArray = np.linspace(
            self.min_peak_amplitude[0], self.min_peak_amplitude[1], len(x_series) - self.earliest_peak_idx
        )
        ax.plot(x_series.iloc[self.earliest_peak_idx:], amplitude_thresholds, color="black", ls="--", alpha=0.6)
        
    @abstractmethod
    def plot_detection_on_ax(self, ax: Axes, trace_arr: NDArray, x_series: pd.Series, **kwargs):
        pass

    @abstractmethod
    def get_peaks(self, trace_arr: NDArray, **kwgars) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def _find_peak_idx(self, trace_arr: NDArray) -> NDArray:
        pass


class ScipyPeakDetector(PeakDetector):
    def __init__(
        self,
        wlen: int,
        distance: int = 1,
        earliest_peak_idx: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_idx=earliest_peak_idx,
            min_peak_amplitude=min_peak_amplitude,
            **kwargs
        )
        self.wlen: int = wlen
        self.distance: int = distance

    def get_peaks(
        self, trace_arr: NDArray, prominence: float
    ) -> Dict[str, NDArray]:
        peak_row_idx: NDArray = self._find_peak_row_idx(
            trace_arr=trace_arr, prominence=prominence
        )
        amplitudes: NDArray = self._get_amplitudes(
            trace_arr=trace_arr, peak_row_idx=peak_row_idx
        )
        peak_row_idx, amplitudes = self._filter_peaks(
            peak_row_idx=peak_row_idx, amplitudes=amplitudes, trace_length=len(trace_arr)
        )
        return {"peak_row_idx": peak_row_idx, "amplitudes": amplitudes}

    def _get_amplitudes(self, trace_arr: NDArray, peak_row_idx: NDArray) -> NDArray:
        return trace_arr[peak_row_idx]

    def _find_peak_row_idx(self, trace_arr: NDArray, prominence: float) -> NDArray:
        return signal.find_peaks(
            trace_arr, prominence=prominence, wlen=self.wlen, distance=self.distance
        )[0]
        
    def plot_detection_on_ax(self, ax: Axes, trace_arr: NDArray, x_series: pd.Series, prominence: float):
        peak_row_idx_all: NDArray = self._find_peak_row_idx(
            trace_arr=trace_arr, prominence=prominence
        )
        amplitudes_all: NDArray = self._get_amplitudes(
            trace_arr=trace_arr, peak_row_idx=peak_row_idx_all
        )
        peak_row_idx_filt, amplitudes_filt = self._filter_peaks(
            peak_row_idx=peak_row_idx_all, amplitudes=amplitudes_all, trace_length=len(trace_arr)
        )
        _color_list = [
            "red" if i in peak_row_idx_filt else "gray" for i in peak_row_idx_all
        ]
        # this only allows a single color! use scatter for dynamic color
        # ax.plot(
        #     x_series.iloc[peak_row_idx_filt],
        #     amplitudes_filt,
        #     "o",
        #     markersize=2,
        #     c="red"
        # )
        ax.vlines(
            x=x_series.iloc[peak_row_idx_all],
            ymin=np.zeros(shape=len(peak_row_idx_all)),
            ymax=amplitudes_all,
            colors=_color_list,
            linewidths=1,
        )
        return


class NoiseSDScipyPeakDetector(ScipyPeakDetector):
    def __init__(
        self,
        prominence_in_stdev: float,
        wlen: int,
        distance: int = 1,
        earliest_peak_idx: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_idx=earliest_peak_idx,
            min_peak_amplitude=min_peak_amplitude,
            wlen=wlen,
            distance=distance,
            **kwargs
        )
        self.prominence_in_stdev: float = prominence_in_stdev

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
        return super().get_peaks(
            trace_arr=trace_arr, prominence=prominence
        )

    def plot_detection_on_ax(self, ax: Axes, trace_arr: NDArray, x_series: pd.Series, stdev_lower_bound: float):
        prominence: float = self.prominence_in_stdev * stdev_lower_bound
        super().plot_detection_on_ax(ax=ax, trace_arr=trace_arr, x_series=x_series, prominence=prominence)
        return


class RobustNoiseSDScipyPeakDetector(NoiseSDScipyPeakDetector):
    def __init__(
        self,
        prominence_in_stdev: float,
        wlen: int,
        distance: int = 1,
        earliest_peak_idx: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_idx=earliest_peak_idx,
            min_peak_amplitude=min_peak_amplitude,
            wlen=wlen,
            distance=distance,
            prominence_in_stdev=prominence_in_stdev,
            **kwargs
        )

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
            trace_arr=trace_arr, stdev_lower_bound=min(stdev_lower_bound, stdev_estim)
        )
        
    def plot_detection_on_ax(self, ax: Axes, trace_arr: NDArray, x_series: pd.Series, stdev_lower_bound: float):
        stdev_estim: float = self._estimate_noise_stdev(trace_arr)
        super().plot_detection_on_ax(
            ax=ax, trace_arr=trace_arr, x_series=x_series, stdev_lower_bound=min(stdev_lower_bound, stdev_estim)
        )
        return

    def _estimate_noise_stdev(self, trace_arr: NDArray) -> float:
        # calculate prominence threshold in robust sd units (based on interquartile distance)
        # our reference distribution will be a lognormal instead of a normal
        q25, q75 = np.quantile(trace_arr[self.earliest_peak_idx :], [0.25, 0.75])
        # for normal distribution: sd = (q75 - q25) / 1.349
        return np.log(q75 / q25) / 1.349


class TamasPeakDetector(PeakDetector):
    def __init__(
        self, earliest_peak_idx: int = 0, min_peak_amplitude: float = 0.0, **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_idx=earliest_peak_idx,
            min_peak_amplitude=min_peak_amplitude,
            **kwargs
        )

    def _find_peak_row_idx(self, trace_arr: NDArray, threshold: float) -> NDArray:
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
