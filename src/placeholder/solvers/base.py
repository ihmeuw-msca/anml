import numpy as np
import scipy.optimize as sciopt

from placeholder.solvers.interface import Solver

class ScipyOpt(Solver):
    """A concrete class of `Solver` that use `scipy` optimize 
    to fit the model using the L-BFGS-B method.
    """

    def fit(self, data, x_init=None, options=None):
        if x_init is None:
            if self.model.x_init is not None:
                x_init = self.model.x_init 
            else:
                raise ValueError('No initial value for x.')
        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data),
            x0=x_init,
            jac=lambda x: self.model.gradient(x, data),
            bounds=self._model.bounds,
            options=options if options is not None else self.options,
        )
        self.success = result.success
        self.x_opt = result.x
        self.fun_val_opt = result.fun
        self.status = result.message

    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)
