from operator import attrgetter
from typing import Tuple, Type

import numpy as np
from anml.prior.prior import Prior
from xspline import XSpline


class SplinePriorGetter:

    prior = property(attrgetter("_prior"))
    size = property(attrgetter("_size"))
    order = property(attrgetter("_order"))
    domain = property(attrgetter("_domain"))
    domain_type = property(attrgetter("_domain_type"))

    def __init__(self,
                 prior: Prior,
                 size: int = 100,
                 order: int = 0,
                 domain: Tuple[float, float] = (0.0, 1.0),
                 domain_type: str = "rel"):
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
            raise ValueError("Domain must contains two numbers for lower and "
                             "upper bound.")
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
        if not isinstance(spline, XSpline):
            raise TypeError("Spline must be an instance of XSpline.")
        knots_lb, knots_ub = spline.knots[0], spline.knots[-1]
        domain_lb, domain_ub = self.domain
        if self.domain_type == "rel":
            domain_lb = knots_lb + (knots_ub - knots_lb)*domain_lb
            domain_ub = knots_lb + (knots_ub - knots_lb)*domain_ub
        points = np.linspace(domain_lb, domain_ub, self.size)
        self.prior.mat = spline.design_dmat(points, order=self.order,
                                            l_extra=True, r_extra=True)
        return self.prior
