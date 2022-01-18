from dataclasses import dataclass
from operator import attrgetter
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.prior.main import Prior
from anml.prior.utils import filter_priors
from anml.variable.main import Variable
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.linalg import block_diag


@dataclass
class SmoothFunction:

    name: str
    fun: Callable
    ifun: Callable
    dfun: Callable
    d2fun: Callable


class Parameter:

    variables = property(attrgetter("_variables"))
    transform = property(attrgetter("_transform"))
    offset = property(attrgetter("_offset"))
    priors = property(attrgetter("_priors"))

    def __init__(self,
                 variables: List[Variable],
                 transform: Optional[SmoothFunction] = None,
                 offset: Optional[Union[str, Component]] = None,
                 priors: Optional[List[Prior]] = None):
        self.variables = variables
        self.transform = transform
        self.offset = offset
        self.priors = priors

    @variables.setter
    def variables(self, variables: List[Variable]):
        variables = list(variables)
        if not all(isinstance(variable, Variable) for variable in variables):
            raise TypeError("Parameter input variables must be a list of "
                            "instances of Variable.")
        self._variables = variables

    @transform.setter
    def transform(self, transform: Optional[SmoothFunction]):
        if transform is not None and not isinstance(transform, SmoothFunction):
            raise TypeError("Parameter input transform must be an instance "
                            "of SmoothFunction or None.")
        self._transform = transform

    @offset.setter
    def offset(self, offset: Optional[Union[str, Component]]):
        if not isinstance(offset, (str, Component)):
            raise TypeError("Parameter input offset has to be a string or "
                            "an instance of Component.")
        if isinstance(offset, str):
            offset = Component(offset, validators=[NoNans()])
        self._offset = offset

    @priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, Prior) for prior in priors):
            raise TypeError("Parameter input priors must be a list of "
                            "instances of Prior.")
        self._priors = priors

    @property
    def size(self) -> int:
        return sum([variable.size for variable in self.variables])

    def attach(self, df: DataFrame):
        for variable in self.variables:
            variable.attach(df)

    def get_design_mat(self, df: DataFrame):
        return np.hstack([variable.get_design_mat(df)
                          for variable in self.variables])

    def get_direct_prior_params(self, prior_type: str) -> NDArray:
        return np.hstack([variable.get_direct_prior_params(prior_type)
                          for variable in self.variables])

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        params, mat = tuple(zip(*[variable.get_linear_prior_params(prior_type)
                                  for variable in self.variables]))
        params = np.hstack(params)
        mat = block_diag(mat)

        linear_priors = filter_priors(prior_type, with_mat=True)
        extra_params = np.hstack([prior.params for prior in linear_priors])
        extra_mat = np.vstack([prior.mat for prior in linear_priors])

        return np.hstack([params, extra_params]), np.vstack([mat, extra_mat])

    def get_params(self,
                   coefs: NDArray,
                   df: Optional[DataFrame] = None) -> NDArray:
        pass

    def get_dparams(self,
                    coefs: NDArray,
                    df: Optional[DataFrame] = None) -> NDArray:
        pass

    def get_d2params(self,
                     coefs: NDArray,
                     df: Optional[DataFrame] = None) -> NDArray:
        pass
