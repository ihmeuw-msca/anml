"""
================
Model Interface 
================

An interface for models.
"""
from abc import ABC
from typing import Optional

import numpy as np
from anml.data.prototype import DataPrototype


class Model:
    """Interface for models.
    """

    def __init__(self):
        pass

    def objective(self, x: np.ndarray, data: DataPrototype):
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

    def gradient(self, x: np.ndarray, data: DataPrototype):
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
        step = 1e-16
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


class TrimmingCompatibleModel(Model, ABC):

    def __init__(self):
        super().__init__()

    def _gradient(self, x: np.ndarray, data: DataPrototype) -> np.ndarray:
        """
        This returns the gradient function by data point. So it's an array
        :param x:
        :param data:
        :return:
        """

    def gradient(self, x: np.ndarray, data: DataPrototype) -> np.ndarray:
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

    def _objective(self, x: np.ndarray, data: DataPrototype) -> np.ndarray:
        """
        This returns the objective function by data point.

        :param x:
        :param data:
        :return:
        """

    def objective(self, x: np.ndarray, data: DataPrototype, w: Optional[np.ndarray] = None) -> float:
        """Objective function for a model. This objective function

        Parameters
        ----------
        x : np.ndarray
            input vector
        data : Data
            a :class`~anml.data.data.Data` object
        w : An optional weights vector

        Raises
        ------
        NotImplementedError
            not implemented in this interface.
        """
        raise NotImplementedError()
