import numpy as np
from operator import attrgetter
from numpy.typing import NDArray
from xspline import XSpline
from typing import Optional


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
    ldegree
        Left extrapolation polynomial degree.
    rdegree
        Right extrapolation polynomial degree.
    knots_type : {'abs', 'rel_domain', 'rel_freq'}
        Type of the spline knots. Can only be choosen from three options,
        `'abs'`, `'rel_domain'` and `'rel_freq'`. When it is `'abs'`
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

    def __init__(
        self,
        knots: NDArray,
        degree: int = 3,
        ldegree: Optional[int] = None,
        rdegree: Optional[int] = None,
        knots_type: str = "abs",
    ):
        self.knots = knots
        self.degree = degree
        self.ldegree = min(ldegree if ldegree is not None else 0, len(knots) - 2)
        self.rdegree = min(rdegree if rdegree is not None else 0, len(knots) - 2)
        self.knots_type = knots_type

    @knots_type.setter
    def knots_type(self, knots_type: str):
        if knots_type not in ["abs", "rel_domain", "rel_freq"]:
            raise ValueError(
                "Knots type must be one of 'abs', 'rel_domain' or 'rel_freq'."
            )
        self._knots_type = knots_type

    @property
    def num_spline_bases(self) -> int:
        """Number of the spline bases."""
        ldegree = self.ldegree or 0
        rdegree = self.rdegree or 0

        inner_knots = self.knots[ldegree : len(self.knots) - rdegree]

        return len(inner_knots) - 1 + self.degree

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
                knots = lb + self.knots * (ub - lb)
            else:
                knots = np.quantile(data, self.knots)

        return XSpline(
            knots,
            self.degree,
            ldegree=self.ldegree,
            rdegree=self.rdegree,
        )
