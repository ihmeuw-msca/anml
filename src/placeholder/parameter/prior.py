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

In order to get the error that should be added to the objective function
value, each prior is associated with an :class:`~placeholder.parameter.oracle.Oracle`.
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass, field

from placeholder.parameter.oracle import Oracle, GaussianOracle
from placeholder.exceptions import PlaceholderError
from placeholder.utils import _check_list_consistency


class PriorError(PlaceholderError):
    pass


@dataclass
class Prior:

    lower_bound: Union[float, List[float]] = -np.inf
    upper_bound: Union[float, List[float]] = np.inf

    oracle: Oracle = field(init=False)

    def __post_init__(self):
        _check_list_consistency(self.lower_bound, self.upper_bound, PriorError)

    def error_value(self, val):
        raise NotImplementedError


class GaussianPriorError(PriorError):
    pass


@dataclass
class GaussianPrior(Prior):

    mean: Union[float, List[float]] = 0.
    std: Union[float, List[float]] = 1.

    def __post_init__(self):
        _check_list_consistency(self.mean, self.std, PriorError)

        self.oracle = GaussianOracle(mean=self.mean, std=self.std)

        if isinstance(self.std, float):
            std_check = [self.std]
        else:
            std_check = self.std
        if any(np.array(std_check) < 0.):
            raise GaussianPriorError("Cannot have negative standard deviation.")

    def error_val(self, vals):
        return self.oracle.get_objective(vals=vals)
