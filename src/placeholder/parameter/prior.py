"""
===================
Prior Distributions
===================

Gives prior specifications that can be attached to fixed
and/or random effect components of a variable and may be used
in the solver optimization.

The default prior only includes upper and lower bounds for box constraints and defaults
to :python:`[-np.inf, np.inf]`. Alternative priors include
:class:`~placeholder.parameter.prior.GaussianPrior`.
"""

import numpy as np
from dataclasses import dataclass

from placeholder.exceptions import PlaceholderError


class PriorError(PlaceholderError):
    pass


def _check_consistency(x, y):
    if isinstance(x, List) or isinstance(y, List):
        if not (isinstance(x, List) or isinstance(y, List)):
            raise PriorError(f"{x.__name__} and {y.__name__} are not of the same type.")
        if not len(x) == len(y):
            raise PriorError(f"{x.__name__} and {y.__name__} are not of the same length.")


@dataclass
class Prior:

    lower_bound: Union[float, List[float]] = -np.inf
    upper_bound: Union[float, List[float]] = np.inf

    def __post_init__(self):
        _check_consistency(self.lower_bound, self.upper_bound)

    def error_value(self, val):
        raise NotImplementedError


class GaussianPriorError(PriorError):
    pass


@dataclass
class GaussianPrior(Prior):

    mean: Union[float, List[float]] = 0.
    std: Union[float, List[float]] = 1.

    def __post_init__(self):
        _check_consistency(self.mean, self.std)

        if isinstance(self.std, float):
            std_check = [self.std]
        else:
            std_check = self.std
        if any(np.array(std_check) < 0.):
            raise GaussianPriorError("Cannot have negative standard deviation.")

    def error_value(self, val):
        return 0.5 * np.sum(
            (val - self.mean) ** 2 / (self.std ** 2)
        )
