from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Dict, Tuple


class PeakDetector:
    def __init__(self, earliest_peak_idx: int = 0, min_peak_amplitude: float = 0.0) -> None:
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
    def get_peaks(trace_arr: NDArray) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def _find_peak_idx(self, trace_arr: NDArray) -> NDArray:
        pass


class ScipyPeakDetector(PeakDetector):
    def __init__(
        self,
        prominence_in_stdev: float,
        wlen: int,
        earliest_peak_idx: int = 0,
        min_peak_amplitude: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            earliest_peak_idx=earliest_peak_idx,
            min_peak_amplitude=min_peak_amplitude,
            **kwargs
        )
        self.prominence_in_stdev: float = prominence_in_stdev
        self.wlen: int = wlen

    def get_peaks(self, trace_arr: NDArray, stdev: float) -> Dict[str, NDArray]:
        prominence: float = self.prominence_in_stdev * stdev
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
        return signal.find_peaks(trace_arr, prominence=prominence, wlen=self.wlen)[0]
