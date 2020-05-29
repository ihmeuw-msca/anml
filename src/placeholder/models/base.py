import numpy as np


class Model:

    def __init__(self):
        self.data = None 
    
    def objective(self, x, data):
        raise NotImplementedError()

    def gradient(self, x, data):
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