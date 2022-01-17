from operator import attrgetter

import numpy as np
from numpy.typing import NDArray
from xspline import XSpline


class SplineGetter:
    """Spline getter for :class:`XSpline` instance. Given the settings of the 
    spline, when attach the data it can infer the knots position, construct and 
    return an instance of :class:`XSpline`.

    Parameters
    ----------
    knots
        Knots placement of the spline. Depends on `knots_type` this will be
        used differently.
    degree
        Degree of the spline. Default to be 3.
    l_linear
        If `True`, spline will use left linear tail. Default to be `False`.
    r_linear
        If `True`, spline will use right linear tail. Default to be `False`.
    include_first_basis
        If `True`, spline will include the first basis of the spline. Default
        to be `True`.
    knots_type : {'abs', 'rel_domain', 'rel_freq'}
        Type of the spline knots. Can only be choosen from three options,
        `'abs'`, `'rel_domian'` and `'rel_freq'`. When it is `'abs'`
        which standards for absolute, the knots will be used as it is. When it
        is `rel_domain` which standards for relative domain, the knots
        requires to be between 0 and 1, and will be interpreted as the
        proportion of the domain. And when it is `rel_freq` which standards
        for relative frequency, it will be interpreted as the frequency of the
        data and required to be between 0 and 1.

    """

    knots_type = property(attrgetter("_knots_type"))
    """Type of the spline knots.

    Raises
    ------
    ValueError
        Raised when the input knots type are not one of 'abs', 'rel_domain' or
        'rel_freq'.
 
    """

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

    @knots_type.setter
    def knots_type(self, knots_type: str):
        if knots_type not in ["abs", "rel_domain", "rel_freq"]:
            raise ValueError("Knots type must be one of 'abs', 'rel_domain' or 'rel_freq'.")
        self._knots_type = knots_type

    @property
    def num_spline_bases(self) -> int:
        """Number of the spline bases.

        """
        inner_knots = self.knots[int(self.l_linear):
                                 len(self.knots) - int(self.r_linear)]
        return len(inner_knots) - 2 + self.degree + int(self.include_first_basis)

    def get_spline(self, data: NDArray) -> XSpline:
        """Get spline instance given data array.

        Parameters
        ----------
        data
            Given data array to infer the knots placement.

        Returns
        -------
        XSpline
            A spline instance.
        """
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
