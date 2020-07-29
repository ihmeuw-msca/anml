"""
================
Model Interface 
================

An interface for models.
"""

import numpy as np

from anml.data.data import Data


class Model:
    """Interface for models.
    """

    def __init__(self):
        pass

    def objective(self, x: np.ndarray, data: Data):
        """Objective function for a model.

        Parameters
        ----------
        x : np.ndarray
            input vector
        data : Data
            a :class`~anml.data.data.Data` object

        Raises
        ------
        NotImplementedError
            not implemented in this interface.
        """
        raise NotImplementedError()

    def gradient(self, x: np.ndarray, data: Data):
        """Gradient of objective function computed using complex step method. 
        Can be overwritten in inherited classes.

        Parameters
        ----------
        x : np.ndarray
            inpute vector
        data : Data
            a :class`~anml.data.data.Data` object.

        Returns
        -------
        np.ndarray
            gradient vector
        """
        finfo = np.finfo(float)
        step = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c, data).imag/step
            x_c[i] -= step*1j

        return grad

    def forward(self, x: np.ndarray, *args, **kwargs):
        """Compute an output based on the generating mechanism defined by the model. 

        Parameters
        ----------
        x : np.ndarray
            input vector

        Raises
        ------
        NotImplementedError
            not implemented in interface
        """
        # different from predict() in solver in the sense that both variable and data value can vary.
        # in predict() variable value is at taken to be the optimal.
        raise NotImplementedError()
