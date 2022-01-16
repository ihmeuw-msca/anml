import numpy as np
from numpy.typing import NDArray
from xspline import XSpline


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
