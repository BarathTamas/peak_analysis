from peak_analysis import BaseLineFitter, PeakDetector
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Tuple, Union


class TraceProcessor():
    def __init__(self, baseline_fitter: BaseLineFitter, peak_detector: PeakDetector) -> None:
        self.baseline_fitter: BaseLineFitter = baseline_fitter
        self.peak_detector: PeakDetector = peak_detector
        
    def process_trace(self, s_trace: pd.Series, stdev: float) -> Dict[str, NDArray]:
        trace_raw: NDArray = self._coerce_to_array(s_trace)
        baseline: NDArray
        trace_proc: NDArray
        trace_proc, baseline = self._process_baseline(trace_raw)
        # peak_row_idx: NDArray
        # amplitudes: NDArray
        peak_dict: Dict[str, NDArray] = self.peak_detector.get_peaks(trace_arr=trace_proc, stdev=stdev)
        # peak_row_idx = peak_dict["peak_row_idx"]
        # amplitudes = peak_dict["amplitudes"]
        return {
            "trace_proc": trace_proc,
            "baseline": baseline,
            **peak_dict
        }
        
    def _process_baseline(self, trace_raw: NDArray) -> Tuple[NDArray, NDArray]:
        baseline: NDArray = self.baseline_fitter.get_baseline(trace_raw)
        trace_proc: NDArray = trace_raw - baseline
        return trace_proc, baseline
        
    def _coerce_to_array(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray
        if isinstance(s_trace, pd.Series):
            arr_trace = s_trace.to_numpy()
        else:
            arr_trace = s_trace
        return arr_trace