from operator import attrgetter
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.prior.main import Prior
from anml.prior.utils import filter_priors, get_prior_type
from numpy.typing import NDArray
from pandas import DataFrame


class Variable:

    component = property(attrgetter("_component"))
    priors = property(attrgetter("_priors"))
    _prior_types: Tuple[Type, ...] = (Prior,)

    def __init__(self,
                 component: Union[str, Component],
                 priors: Optional[List[Prior]] = None):
        self.component = component
        self.priors = priors

    @component.setter
    def component(self, component: Union[str, Component]):
        if not isinstance(component, (str, Component)):
            raise TypeError("Variable input component has to be a string or "
                            "an instance of Component.")
        if isinstance(component, str):
            component = Component(component, validators=[NoNans()])
        self._component = component

    @priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, self._prior_types) for prior in priors):
            raise TypeError("Variable input priors must be a list of "
                            "instances of Prior.")
        self._priors = priors

    @property
    def size(self) -> Optional[int]:
        return 1

    def attach(self, df: DataFrame):
        self.component.attach(df)

    def get_design_mat(self, df: DataFrame) -> NDArray:
        self.attach(df)
        return self.component.value[:, np.newaxis]

    def get_direct_prior_params(self, prior_type: str) -> NDArray:
        direct_priors = filter_priors(self.priors, prior_type, with_mat=False)
        prior_type = get_prior_type(prior_type)
        if len(direct_priors) == 0:
            return np.repeat(prior_type.default_params, self.size, axis=1)
        if len(direct_priors) >= 2:
            raise ValueError("Variable can only have one direct prior.")
        return direct_priors[0].params

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        linear_priors = filter_priors(self.priors, prior_type, with_mat=True)
        if len(linear_priors) == 0:
            return np.empty((0, self.size)), np.empty((2, 0))
        mat = np.vstack([prior.mat for prior in linear_priors])
        params = np.hstack([prior.params for prior in linear_priors])
        return mat, params

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self.component}, priors={self.priors})"
