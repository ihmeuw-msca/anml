"""
===================
Prior Distributions
===================

Gives prior specifications that can be attached to fixed
and/or random effect components of a variable and may be used
in the solver optimization.

The default prior only includes upper and lower bounds for box constraints and defaults
to `[-np.inf, np.inf]`. Alternative priors include
:class:`~anml.parameter.prior.GaussianPrior`.

In order to get the error that should be added to the objective function
value, each prior is associated with an :class:`~anml.parameter.likelihood.Likelihood`.
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

    lower_bound: List[float] = field(default_factory=lambda: [-np.inf])
    upper_bound: List[float] = field(default_factory=lambda: [np.inf])

    _likelihood: Likelihood = field(init=False)

    def __post_init__(self):
        _check_list_consistency(self.lower_bound, self.upper_bound, PriorError)
        self._additional_checks()
        self._set_likelihood()
        self.x_dim = len(self.lower_bound)

    def _additional_checks(self):
        pass

    def _set_likelihood(self):
        self._likelihood = Likelihood()

    def error_value(self, val):
        return 0.0

    def grad(self, val):
        return 0.0


class GaussianPriorError(PriorError):
    pass


@dataclass
class GaussianPrior(Prior):

    mean: List[float] = field(default_factory=lambda: [0.])
    std: List[float] = field(default_factory=lambda: [1.])

    def __post_init__(self):
        Prior.__post_init__(self)

    def _additional_checks(self):
        _check_list_consistency(self.mean, self.std, PriorError)

    def _set_likelihood(self):
        self._likelihood = GaussianLikelihood(mean=self.mean, std=self.std)

    def error_value(self, vals):
        return self._likelihood.get_neg_log_likelihood(vals=vals)

    def grad(self, vals):
        return self._likelihood.get_grad(vals=vals)
