"""
================================
Statistical Distribution Oracles
================================

The :class:`~anml.parameter.oracle.Likelihood` keeps track of all information
related to probability distributions that can be used for priors or data distributions.
"""

from typing import List, Union, Optional
import numpy as np
import scipy

from anml.exceptions import ANMLError


class LikelihoodError(ANMLError):
    """Raised when there is a mismatch in a distribution's parameters."""
    pass


class Likelihood:
    def __init__(self):
        self.parameters = None

    def _set_parameters(self, parameters: Union[List[float], List[np.ndarray]]):
        self.parameters = parameters

    @staticmethod
    def _likelihood(vals: Union[float, np.ndarray], parameters: Union[List[float], List[np.ndarray]]):
        raise NotImplementedError

    @staticmethod
    def _neg_log_likelihood(vals: Union[float, np.ndarray], parameters: Union[List[float], List[np.ndarray]]):
        raise NotImplementedError

    def get_neg_log_likelihood(self, vals: Union[float, np.ndarray],
                               parameters: Optional[Union[List[float], List[np.ndarray]]] = None):
        """Gets the objective value based on parameters of the distribution and current values
        in :python:`vals`.

        Parameters
        ----------
        vals
            location at which to evaluate the oracle
        parameters
            parameters of the oracle, optional. If not passed it will use the parameters attached
            to self, otherwise return an error if there are no parameters.

        Returns
        -------
        float representing the objective value evaluated based on vals and parameters.

        """
        if parameters is None:
            if self.parameters is None:
                raise LikelihoodError("Need to set parameters before calling the objective function.")
            parameters = self.parameters

        return self._neg_log_likelihood(vals=vals, parameters=parameters)


class GaussianLikelihood(Likelihood):
    def __init__(self, mean: List[float]=None, std: List[float]=None):

        super().__init__()
        if mean is None:
            self.mean = [0.0]
        if std is None:
            self.std = [1.0]

        if any(self.std) <= 0:
            raise LikelihoodError("Cannot have negative variance for Gaussian Likelihood.")

        self.parameters = [mean, std]
        self.dist = 

    @staticmethod
    def _likelihood(vals, parameters):
        return scipy.stats.multivariate_normal(mean=parameters[0], cov=np.diag(parameters[1]**2)).pdf(vals)

    @staticmethod
    def _neg_log_likelihood(vals, parameters):
        return -scipy.stats.multivariate_normal(mean=parameters[0], cov=np.diag(parameters[1]**2)).logpdf(vals)
