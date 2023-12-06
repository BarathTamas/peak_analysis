import pandas as pd
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from typing import Union
from pybaselines import Baseline


class BaseLineFitter:
    def __init__(self) -> None:
        pass

    def _coerce_to_array(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray
        if isinstance(s_trace, pd.Series):
            arr_trace = s_trace.to_numpy()
        else:
            arr_trace = s_trace
        return arr_trace

    @abstractmethod
    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        pass


class PolyBaseLineFitter(BaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.poly_order: int = poly_order


class ModPolyBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_array(s_trace)
        baseline = Baseline().modpoly(arr_trace, poly_order=self.poly_order)[0]
        return baseline


class IModPolyBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_array(s_trace)
        baseline = Baseline().imodpoly(arr_trace, poly_order=self.poly_order)[0]
        return baseline


class ModPolyCustomBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, optional_segment_length: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)
        self.optional_segment_length = optional_segment_length

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        # shortened alias
        _i_cutoff: int = self.optional_segment_length
        # if array is passed, by default it would be passed by reference
        arr_trace: NDArray = self._coerce_to_array(s_trace).copy()
        # do initial fit
        baseline: NDArray = Baseline().modpoly(arr_trace, poly_order=self.poly_order)[0]
        _residuals: NDArray = arr_trace - baseline
        # If all of the residuals from the first half of the period are positive after the baseline removal
        # it basically implies the starting segment is useless for estimating a baseline.
        if (_residuals[: (_i_cutoff // 2)] <= 0.0).sum() < 1:
            # if np.std(_residuals[:_i_cutoff]) > np.std(_residuals[_i_cutoff:]):
            arr_trace[:_i_cutoff] = np.nan
            baseline = np.full_like(arr_trace, np.nan)
            baseline[_i_cutoff:] = Baseline().modpoly(
                arr_trace[_i_cutoff:], poly_order=self.poly_order
            )[0]
        else:
            pass
        return baseline
