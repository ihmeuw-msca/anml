from operator import attrgetter
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.getter.prior import SplinePriorGetter
from anml.getter.spline import SplineGetter
from anml.prior.main import Prior
from anml.prior.utils import filter_priors, get_prior_type
from numpy.typing import NDArray
from pandas import DataFrame
from xspline import XSpline


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
        if not all(prior.shape[1] == self.size for prior in priors):
            raise ValueError("Variable input priors shape must align with the "
                             "size of the variable.")
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
        prior_type = get_prior_type(prior_type)
        direct_priors = filter_priors(prior_type, with_mat=False)
        if len(direct_priors) == 0:
            return np.repeat(prior_type.default_params, self.size, axis=1)
        if len(direct_priors) >= 2:
            raise ValueError("Variable can only have one direct prior.")
        return direct_priors[0].params

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        linear_priors = self.filter_priors(prior_type, with_mat=True)
        if len(linear_priors) == 0:
            return np.empty((0, self.size)), np.empty((2, 0))
        mat = np.vstack([prior.mat for prior in linear_priors])
        params = np.hstack([prior.params for prior in linear_priors])
        return mat, params

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self.component}, priors={self.priors})"


SplineVariablePrior: Type = Union[Prior, SplinePriorGetter]


class SplineVariable(Variable):

    spline = property(attrgetter("_spline"))
    _prior_types: Tuple[Type, ...] = SplineVariablePrior.__args__

    def __init__(self,
                 component: Union[str, Component],
                 spline: Union[XSpline, SplineGetter],
                 priors: Optional[List[SplineVariablePrior]] = None):
        super().__init__(component, priors)
        self.spline = spline

    @spline.setter
    def spline(self, spline: Union[XSpline, SplineGetter]):
        if not isinstance(spline, (XSpline, SplineGetter)):
            raise TypeError("Spline variable input spline must be an instance "
                            "of XSpline or SplineGetter.")
        self._spline = spline

    @property
    def size(self) -> int:
        return self.spline.num_spline_bases

    def attach(self, df: DataFrame):
        self.component.attach(df)
        if isinstance(self.spline, SplineGetter):
            self.spline = self.spline.get_spline(self.component.value)
        for i in range(len(self.priors)):
            if isinstance(self.priors[i], SplinePriorGetter):
                self.priors[i] = self.priors[i].get_prior(self.spline)

    def get_design_mat(self, df: DataFrame) -> NDArray:
        self.attach(df)
        return self.spline.design_mat(self.component.value, l_extra=True, r_extra=True)
