from operator import attrgetter
from typing import List, Optional, Tuple, Union

import anml
import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.prior.getter import SplinePriorGetter
from anml.prior.main import Prior
from numpy.typing import NDArray
from pandas import DataFrame
from xspline import XSpline


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

    def attach(self, df: DataFrame):
        self.component.attach(df)

    def get_design_mat(self, df: DataFrame) -> NDArray:
        self.attach(df)
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self.component}, priors={self.priors})"


class SplineGetter:

    def __init__(self,
                 knots: NDArray,
                 degree: int = 3,
                 l_linear: bool = False,
                 r_linear: bool = False,
                 include_first_basis: bool = False,
                 knots_type: str = "abs"):
        self.knots = knots
        self.degree = degree
        self.l_linear = l_linear
        self.r_linear = r_linear
        self.include_first_basis = include_first_basis
        self.knots_type = knots_type

    @property
    def num_spline_bases(self) -> int:
        """Number of the spline bases.

        """
        inner_knots = self.knots[int(self.l_linear):
                                 len(self.knots) - int(self.r_linear)]
        return len(inner_knots) - 2 + self.degree + int(self.include_first_basis)

    def get_spline(self, data: NDArray) -> XSpline:
        if self.knots_type == "abs":
            knots = self.knots
        else:
            if self.knots_type == "rel_domain":
                lb, ub = data.min(), data.max()
                knots = lb + self.knots*(ub - lb)
            else:
                knots = np.quantile(data, self.knots)

        return XSpline(knots,
                       self.degree,
                       l_linear=self.l_linear,
                       r_linear=self.r_linear,
                       include_first_basis=self.include_first_basis)


class SplineVariable(Variable):

    spline = property(attrgetter("_spline"))

    def __init__(self,
                 component: Union[str, Component],
                 spline: Union[XSpline, SplineGetter],
                 priors: Optional[List[Union[Prior, SplinePriorGetter]]] = None):
        super().__init__(component, priors)
        self.spline = spline

    @Variable.priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, (Prior, SplinePriorGetter)) for prior in priors):
            raise TypeError("Variable input priors must be a list of "
                            "instances of Prior.")
        if not all(prior.shape[1] == self.size for prior in priors):
            raise ValueError("Variable input priors shape must align with the "
                             "size of the variable.")
        self._priors = priors

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
