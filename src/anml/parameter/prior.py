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
value, each prior is associated with an :class:`~placeholder.parameter.likelihood.Likelihood`.
"""

import numpy as np

from typing import List, Union
from dataclasses import dataclass, field

from anml.parameter.likelihood import Likelihood, GaussianLikelihood
from anml.utils import _check_list_consistency
from anml.exceptions import ANMLError


class PriorError(ANMLError):
    pass


@dataclass
class Prior:

    lower_bound: Union[float, List[float]] = -np.inf
    upper_bound: Union[float, List[float]] = np.inf

    _likelihood: Likelihood = field(init=False)

    def __post_init__(self):
        _check_list_consistency(self.lower_bound, self.upper_bound, PriorError)
        if isinstance(self.lower_bound, list):
            self.x_dim = len(self.lower_bound)
        else:
            self.x_dim = 1 
        self._likelihood = Likelihood()

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

        self._likelihood = GaussianLikelihood(mean=self.mean, std=self.std)

        if isinstance(self.std, float):
            std_check = [self.std]
        else:
            std_check = self.std
        if any(np.array(std_check) < 0.):
            raise GaussianPriorError("Cannot have negative standard deviation.")

    def neg_log_likelihood(self, vals):
        return self._likelihood.get_neg_log_likelihood(vals=vals)
