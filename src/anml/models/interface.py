import numpy as np

from anml.data.data import Data


class Model:
    """Interface for models.
    """

    def __init__(self):
        self.bounds = None
        self.constraints = None

    def objective(self, x: np.ndarray, data: Data):
        raise NotImplementedError()

    def gradient(self, x: np.ndarray, data: Data):
        finfo = np.finfo(float)
        step = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c, data).imag/step
            x_c[i] -= step*1j

        return grad

    def predict(self, **kwargs):
        raise NotImplementedError()
