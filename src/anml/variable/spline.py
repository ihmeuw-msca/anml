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
"""Allowed prior type for spline variable.

"""


class SplineVariable(Variable):
    """Variable class that contains information of variable, including name,
    priors and spline.

    Parameters
    ----------
    component
        You can pass in the name of the variable corresponding to the column
        name in the data frame. It will be automatically converted into an
        instance of :class:`Component` with :class:`NoNans` as the validator.
        Alternatively, you can also pass in an instance of :class:`Component`,
        with your own set of validators.
    spline
        Given spline for the variable. You can pass in an instance of
        :class:`XSpline` or :class:`SplineGetter`. If input is an instance of
        :class:`SplineGetter`, when use attach data it will automatically
        envolve into an instance of :class:`XSpline`.
    priors
        A list of priors corresponding to the variable. The prior in the list
        can be either an instance of :class:`Prior` or
        :class:`SplinePriorGetter`. When attach data, the instance of
        :class:`SplinePriorGetter` will envolve into an instance of
        :class:`Prior`.

    """

    spline = property(attrgetter("_spline"))
    """Given spline for the variable.

    Raises
    ------
    TypeError
        Raised if input spline is not an instance of :class:`XSpline` nor
        :class:`SplineGetter`.

    """
    _prior_types: Tuple[Type, ...] = SplineVariablePrior.__args__

    def __init__(
        self,
        component: Union[str, Component],
        spline: Union[XSpline, SplineGetter],
        priors: Optional[List[SplineVariablePrior]] = None,
    ):
        super().__init__(component, priors)
        self.spline = spline

    @spline.setter
    def spline(self, spline: Union[XSpline, SplineGetter]):
        if not isinstance(spline, (XSpline, SplineGetter)):
            raise TypeError(
                "Spline variable input spline must be an instance "
                "of XSpline or SplineGetter."
            )
        self._spline = spline

    @property
    def size(self) -> int:
        """Number of the spline bases."""
        if isinstance(self.spline, XSpline):
            knots = self.spline.knots
            degree = self.spline.degree
            ldegree = self.spline.ldegree or 0
            rdegree = self.spline.rdegree or 0
            inner_knots = knots[ldegree : len(knots) - rdegree]
            return len(inner_knots) - 1 + degree

        elif isinstance(self.spline, SplineGetter):
            return self.spline.num_spline_bases

        else:
            raise TypeError("Unknown spline type")

    def attach(self, df: DataFrame):
        """Attach the data to variable. It will attach data to the component.
        And create spline and priors if necessary.

        Parameters
        ----------
        df
            The data frame contains the corresponding data column.

        """
        self.component.attach(df)
        if isinstance(self.spline, SplineGetter):
            self.spline = self.spline.get_spline(self.component.value)
        for i in range(len(self.priors)):
            if isinstance(self.priors[i], SplinePriorGetter):
                self.priors[i] = self.priors[i].get_prior(self.spline)

    def get_design_mat(self, df: DataFrame) -> NDArray:
        self.attach(df)
        return self.spline.get_design_mat(
            self.component.value,
        )
