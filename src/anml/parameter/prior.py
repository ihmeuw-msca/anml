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
        if isinstance(self.lower_bound, float):
            self.lower_bound = [self.lower_bound]
            self.upper_bound = [self.upper_bound]
        self._additional_checks()
        self._set_likelihood()
        self.x_dim = len(self.lower_bound)

    def _additional_checks(self):
        pass

    def _set_likelihood(self):
        self._likelihood = Likelihood()

    def error_value(self, val):
        raise NotImplementedError


class GaussianPriorError(PriorError):
    pass


@dataclass
class GaussianPrior(Prior):

    mean: Union[float, List[float]] = 0.
    std: Union[float, List[float]] = 1.

    def _additional_checks(self):
        _check_list_consistency(self.mean, self.std, PriorError)
        if isinstance(self.mean, float):
            self.mean = np.array([self.mean])
            self.std = np.array([self.std])
        else:
            self.mean = np.asarray(self.mean)
            self.std = np.asarray(self.std)
        if any(np.array(self.std) < 0.):
            raise GaussianPriorError("Cannot have negative standard deviation.")

    def _set_likelihood(self):
        self._likelihood = GaussianLikelihood(mean=self.mean, std=self.std)

    def error_value(self, vals):
        return self._likelihood.get_neg_log_likelihood(vals=vals)
