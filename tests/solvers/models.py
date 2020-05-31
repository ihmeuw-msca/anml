import pytest
import numpy as np 
from scipy.optimize import rosen, rosen_der

from placeholder.models.interface import Model


class Rosenbrock(Model):

    def __init__(self, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.bounds = np.array([[-2.0, 2.0]] * n_dim)
        self.x_init = np.array([-1.0] * n_dim)

    @staticmethod
    def objective(x, data=None):
        return rosen(x)

    @staticmethod
    def gradient(x, data=None):
        return rosen_der(x)
