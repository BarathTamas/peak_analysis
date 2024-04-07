from abc import abstractmethod
from textwrap import dedent
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaselines import Baseline


class BaseLineFitter:
    def __init__(self) -> None:
        pass

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
                raise ValueError(f"Input must be vector, but has shape {arr_trace.shape}")
        else:
            raise ValueError(f"Input must be pandas.Series or numpy.ndarray, but is {type(s_trace)}")
        return arr_trace

    @abstractmethod
    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        pass


class PolyBaseLineFitter(BaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.poly_order: int = poly_order
   
    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        The baseline is fitted as a degree {self.poly_order} polynomial curve.\n"""
        return dedent(descr)


class ModPolyBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)

    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = """
        The baseline is fitted using the Modpoly algorithm.\n"""
        descr = dedent(descr) + super().description
        return descr

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_1d_array(s_trace)
        baseline = Baseline().modpoly(arr_trace, poly_order=self.poly_order)[0]
        return baseline


class IModPolyBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)
    
    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = """
        The baseline is fitted using the IModpoly algorithm.\n"""
        descr = dedent(descr) + super().description
        return descr

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_1d_array(s_trace)
        baseline = Baseline().imodpoly(arr_trace, poly_order=self.poly_order)[0]
        return baseline


class ModPolyCustomBaseLineFitter(PolyBaseLineFitter):
    def __init__(self, poly_order: int, optional_segment_length: int, **kwargs) -> None:
        super().__init__(poly_order=poly_order, **kwargs)
        self.optional_segment_length = optional_segment_length
    
    @property
    def description(self) -> str:
        """A verbose description of what the class does, useful for generating reports with the outputs.

        The descriptions will be concatenated along the inheritance chain.

        Returns:
            str: The description of what the class does to the input.
        """
        descr: str = f"""
        The baseline is fitted using the Modpoly algorithm on either the full trace or ignoring the first
        {self.optional_segment_length} points. The first optional segment is only used if its first half
        manages to 'touch' the baseline. If the first half is completely 'hanging' in the air, it is ignored.\n"""
        descr = dedent(descr) + super().description
        return descr

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        # shortened alias
        _i_cutoff: int = self.optional_segment_length
        # if array is passed, by default it would be passed by reference
        arr_trace: NDArray = self._coerce_to_1d_array(s_trace).copy()
        # do initial fit
        try:
            baseline: NDArray = Baseline().modpoly(arr_trace, poly_order=self.poly_order)[0]
        except ValueError as error:
            print(f"Invalid values in trace with shape {arr_trace.shape}:\n{arr_trace}")
            raise error
        _residuals: NDArray = arr_trace - baseline
        # If all of the residuals from the first half of the period are positive after the baseline removal
        # it basically implies the starting segment is useless for estimating a baseline.
        # print(_residuals[: (_i_cutoff // 2)])
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


class FirstNBaseLineFitter(BaseLineFitter):
    def __init__(self, agg_func: Callable, first_n: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agg_func: Callable = agg_func
        self.first_n: int = first_n

    def get_baseline(self, s_trace: Union[pd.Series, NDArray]) -> NDArray:
        arr_trace: NDArray = self._coerce_to_1d_array(s_trace)
        baseline: float = self.agg_func(arr_trace[:self.first_n])
        arr_baseline: NDArray = np.full_like(arr_trace, baseline)
        return arr_baseline
