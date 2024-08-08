from operator import attrgetter
from typing import Tuple

import numpy as np
from anml.prior.main import Prior
from xspline import XSpline


class SplinePriorGetter:
    """Prior getter for spline variable. :class:`SplinePriorGetter` takes in
    a prior without the attribute `mat`. And when we call the `get_prior`
    function with a spline as the argument, it will generate and assign the
    the design matrix to the prior.

    Parameters
    ----------
    prior
        Prior instance without attribute mat.
    size
        Size of the spline prior. Default is `100`. It determines the number
        of sample points in the specified domain.
    order
        Order of the spline derivative. Default is `0`.
    domain
        Lower and upper bounds for domain. Default is `(0.0, 1.0)`.
    domain_type : {'rel', 'abs'}
        Type of the domain. Default is `'rel'`. It can only be `'abs'` or
        `'rel'`. When it is `'abs'`, lower and upper bounds are interpreted
        as the absolute position of the domain. When it is `'rel'`, lower and
        upper bounds are treated as the percentage of the domain.

    Examples
    --------
    Here are some common spline priors.

    .. code-block:: python

        import numpy as np
        from xspline import XSpline

        from anml.prior.main import UniformPrior
        from anml.prior.getter import SplinePriorGetter

        # increasing prior
        prior_getter = SplinePriorGetter(UniformPrior(lb=0.0, ub=np.inf), order=1)
        # decreasing prior
        prior_getter = SplinePriorGetter(UniformPrior(lb=-np.inf, ub=0.0), order=1)
        # convex prior
        prior_getter = SplinePriorGetter(UniformPrior(lb=0.0, ub=np.inf), order=2)
        # concave prior
        prior_getter = SplinePriorGetter(UniformPrior(lb=-np.inf, ub=0.0), order=2)

        # use the get_prior function create prior
        spline = XSpline(knots=np.linspace(0.0, 2.0, 5), degree=3)
        prior = prior_getter.get_prior(spline)

    """

    prior = property(attrgetter("_prior"))
    """Prior instance without attribute mat.

    Raises
    ------
    TypeError
        Raised when prior is not an instance of :class:`Prior`.
    ValueError
        Raised when prior mat exists.

    """
    size = property(attrgetter("_size"))
    """Size of the spline prior.

    Raises
    ------
    ValueError
        Raised when input size is not positive.

    """
    order = property(attrgetter("_order"))
    """Order of the spline derivative.

    Raises
    ------
    ValueError
        Raised when input order is negative.

    """
    domain = property(attrgetter("_domain"))
    """Lower and upper bounds for domain.

    Raises
    ------
    ValueError
        Raised when length of the input domain is less or geater than 2.
    ValueError
        Raised when domain lower bound is greater than upper bound.

    """
    domain_type = property(attrgetter("_domain_type"))
    """Type of the domain.

    Raises
    ------
    ValueError
        Raised when the input is not one of 'rel' or 'abs'.

    """

    def __init__(
        self,
        prior: Prior,
        size: int = 100,
        order: int = 0,
        domain: Tuple[float, float] = (0.0, 1.0),
        domain_type: str = "rel",
    ):
        self.prior = prior
        self.size = size
        self.order = order
        self.domain = domain
        self.domain_type = domain_type

    @prior.setter
    def prior(self, prior: Prior):
        if not isinstance(prior, Prior):
            raise TypeError("Prior must be an instance of Prior.")
        if prior.mat is not None:
            raise ValueError("Prior mat exists, cannot assign other values.")
        self._prior = prior

    @size.setter
    def size(self, size: int):
        size = int(size)
        if size <= 0:
            raise ValueError("Size must be positive.")
        self._size = size

    @order.setter
    def order(self, order: int):
        order = int(order)
        if order < 0:
            raise ValueError("Order must be non-negative.")
        self._order = order

    @domain.setter
    def domain(self, domain: Tuple[float, float]):
        domain = tuple(domain)
        if len(domain) != 2:
            raise ValueError(
                "Domain must contains two numbers for lower and " "upper bound."
            )
        domain_lb, domain_ub = domain
        if domain_lb > domain_ub:
            raise ValueError("Domain lb must be less than or equal to ub.")
        self._domain = domain

    @domain_type.setter
    def domain_type(self, domain_type: str):
        if domain_type not in ["rel", "abs"]:
            raise ValueError("Domain type must be choicen from 'rel' or 'abs'.")
        self._domain_type = domain_type

    def get_prior(self, spline: XSpline) -> Prior:
        """Generate and assign mat to the prior with information from the input
        spline.

        Parameters
        ----------
        spline
            Given spline instance.

        Returns
        -------
        Prior
            Prior that has information on `mat`.

        Raises
        ------
        TypeError
            Raised when input spline is not an instance of :class:`XSpline`.

        """
        if not isinstance(spline, XSpline):
            raise TypeError("Spline must be an instance of XSpline.")
        knots_lb, knots_ub = spline.knots[0], spline.knots[-1]
        domain_lb, domain_ub = self.domain
        if self.domain_type == "rel":
            domain_lb = knots_lb + (knots_ub - knots_lb) * domain_lb
            domain_ub = knots_lb + (knots_ub - knots_lb) * domain_ub
        points = np.linspace(domain_lb, domain_ub, self.size)
        self.prior.mat = spline.get_design_mat(
            points,
            order=self.order,
        )
        return self.prior
