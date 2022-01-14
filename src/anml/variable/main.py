from operator import attrgetter
from typing import List, Optional, Tuple, Union

import anml
import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.prior.main import Prior
from numpy.typing import NDArray
from pandas import DataFrame


class Variable:

    component = property(attrgetter("_component"))
    priors = property(attrgetter("_priors"))

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
        if not all(isinstance(prior, Prior) for prior in priors):
            raise TypeError("Variable input priors must be a list of "
                            "instances of Prior.")
        if not all(prior.shape[1] == self.size for prior in priors):
            raise ValueError("Variable input priors shape must align with the "
                             "size of the variable.")
        self._priors = priors

    @property
    def size(self) -> Optional[int]:
        return 1

    def get_design_mat(self, df: DataFrame) -> NDArray:
        self.component.attach(df)
        return self.component.value[:, np.newaxis]

    def filter_priors(self,
                      prior_type: str,
                      with_mat: Optional[bool] = None) -> List[Prior]:
        prior_class = getattr(anml.prior.main, prior_type)

        def condition(prior, prior_class=prior_class, with_mat=with_mat):
            is_prior_class_instance = isinstance(prior, prior_class)
            if with_mat is None:
                return is_prior_class_instance
            if with_mat:
                return prior.mat is not None and is_prior_class_instance
            return prior.mat is None and is_prior_class_instance

        return list(filter(condition, self.priors))

    def get_direct_prior_params(self, prior_type: str) -> NDArray:
        direct_priors = self.filter_priors(prior_type, with_mat=False)
        prior_class = getattr(anml.prior.main, prior_type)
        if len(direct_priors) == 0:
            return np.repeat(prior_class.default_params, self.size, axis=1)
        if len(direct_priors) >= 2:
            raise ValueError(f"Variable can only have one direct {prior_type} "
                             "prior.")
        return direct_priors[0].params

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        linear_priors = self.filter_priors(prior_type, with_mat=True)
        if len(linear_priors) == 0:
            return np.empty((0, self.size)), np.empty((2, 0))
        mat = np.vstack([prior.mat for prior in linear_priors])
        params = np.hstack([prior.params for prior in linear_priors])
        return mat, params
