"""
================================
Statistical Distribution Oracles
================================

The :class:`~placeholder.parameter.oracle.Oracle` keeps track of all information
related to probability distributions that can be used for priors or data distributions.
"""

from typing import List, Union, Optional
import numpy as np

from placeholder.exceptions import PlaceholderError


class OracleError(PlaceholderError):
    """Raised when there is a mismatch in a distribution's parameters."""
    pass


class Oracle:
    def __init__(self):
        self.parameters = None

    def _set_parameters(self, parameters: Union[List[float], List[np.ndarray]]):
        self.parameters = parameters

    def _objective_val(self, vals: Union[float, np.ndarray], parameters: Union[List[float], List[np.ndarray]]):
        raise NotImplementedError

    def get_objective(self, vals: Union[float, np.ndarray],
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
                raise OracleError("Need to set parameters before calling the objective function.")
            parameters = self.parameters

        return self._objective_val(vals=vals, parameters=parameters)


class GaussianOracle(Oracle):
    def __init__(self, mean: Union[float, List[float]] = 0.,
                 std: Union[float, List[float]] = 1.):

        super().__init__()

        if std <= 0:
            raise OracleError("Cannot have negative variance for Gaussian Oracle.")

        self.parameters = [mean, std]

    def _objective_val(self, vals, parameters):
        return 0.5 * (vals - parameters[0]) ** 2 / (parameters[1] ** 2)
