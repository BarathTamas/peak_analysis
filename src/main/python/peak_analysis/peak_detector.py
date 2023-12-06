from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Dict, Tuple


class PeakDetector:
    def __init__(
        self, earliest_peak_idx: int = 0, min_peak_amplitude: float = 0.0
    ) -> None:
        self.earliest_peak_idx: int = earliest_peak_idx
        self.min_peak_amplitude: float = min_peak_amplitude

    def _filter_peaks(
        self, peak_row_idx: NDArray, amplitudes: NDArray
    ) -> Tuple[NDArray, NDArray]:
        # apply filter based on time cutoff
        idx_filter: NDArray = self._is_peak_late_enough(
            peak_row_idx=peak_row_idx, earliest_peak_idx=self.earliest_peak_idx
        )
        # apply filter based on minimum aplitude
        ampl_filter: NDArray = self._is_peak_large_enough(
            amplitudes=amplitudes, min_peak_amplitude=self.min_peak_amplitude
        )
        _final_filter: NDArray = np.logical_and(idx_filter, ampl_filter)
        return peak_row_idx[_final_filter], amplitudes[_final_filter]

    def _is_peak_late_enough(
        self, peak_row_idx: NDArray, earliest_peak_idx: int
    ) -> NDArray:
        return peak_row_idx >= earliest_peak_idx

    def _is_peak_large_enough(
        self, amplitudes: NDArray, min_peak_amplitude: float
    ) -> NDArray:
        return amplitudes >= min_peak_amplitude

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

    def get_peaks(self, trace_arr: NDArray, prominence: float) -> Dict[str, NDArray]:
        """Find the peaks and their amplitudes.

        Args:
            trace_arr (NDArray): _description_
            prominence (float): _description_

        Returns:
            Dict[str, NDArray]: _description_
        """
        return self._get_peaks_with_prominence(
            trace_arr=trace_arr, prominence=prominence
        )

    def _get_peaks_with_prominence(
        self, trace_arr: NDArray, prominence: float
    ) -> Dict[str, NDArray]:
        peak_row_idx: NDArray = self._find_peak_row_idx(
            trace_arr=trace_arr, prominence=prominence
        )
        amplitudes: NDArray = self._get_amplitudes(
            trace_arr=trace_arr, peak_row_idx=peak_row_idx
        )
        peak_row_idx, amplitudes = self._filter_peaks(
            peak_row_idx=peak_row_idx, amplitudes=amplitudes
        )
        return {"peak_row_idx": peak_row_idx, "amplitudes": amplitudes}

    def _get_amplitudes(self, trace_arr: NDArray, peak_row_idx: NDArray) -> NDArray:
        return trace_arr[peak_row_idx]

    def _find_peak_row_idx(self, trace_arr: NDArray, prominence: float) -> NDArray:
        return signal.find_peaks(
            trace_arr, prominence=prominence, wlen=self.wlen, distance=self.distance
        )[0]


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
        return self._get_peaks_with_prominence(
            trace_arr=trace_arr, prominence=prominence
        )


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
        prominence: float = self.prominence_in_stdev * min(
            stdev_lower_bound, stdev_estim
        )
        return self._get_peaks_with_prominence(
            trace_arr=trace_arr, prominence=prominence
        )

    def _estimate_noise_stdev(self, trace_arr: NDArray) -> float:
        # calculate prominence threshold in robust sd units (based on interquartile distance)
        # our reference distribution will be a lognormal instead of a normal
        q25, q75 = np.quantile(trace_arr[self.earliest_peak_idx:], [0.25, 0.75])
        # for normal distribution: sd = (q75 - q25) / 1.349
        return np.log(q75 / q25) / 1.349
