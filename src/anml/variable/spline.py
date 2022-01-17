from operator import attrgetter
from typing import List, Optional, Tuple, Type, Union

from anml.data.component import Component
from anml.getter.prior import SplinePriorGetter
from anml.getter.spline import SplineGetter
from anml.prior.main import Prior
from anml.variable.main import Variable
from numpy.typing import NDArray
from pandas import DataFrame
from xspline import XSpline

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
