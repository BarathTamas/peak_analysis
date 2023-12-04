import pandas as pd
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from typing import Union
from pybaselines import Baseline


class BaseLineFitter:
    def __init__(self) -> None:
        self.baseline_fitter = Baseline()

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
        baseline = self.baseline_fitter.modpoly(
            arr_trace, poly_order=self.poly_order
        )[0]
        return baseline


class IModPolyBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_array(s_trace)
        baseline = self.baseline_fitter.imodpoly(
            arr_trace, poly_order=self.poly_order
        )[0]
        return baseline
