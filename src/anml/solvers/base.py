from typing import Dict, Any, Optional
import numpy as np
import scipy.optimize as sciopt

from anml.data.data import Data
from anml.solvers.interface import Solver


class ScipyOpt(Solver):
    """A concrete class of `Solver` that use `scipy` optimize 
    to fit the model using the L-BFGS-B method.
    """

    def fit(self, data: Data, x_init: np.ndarray = None, options: Optional[Dict[str, Any]] = None):
        self.assert_model_defined()
        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data),
            x0=x_init,
            jac=lambda x: self.model.gradient(x, data),
            bounds=self._model.bounds,
            method=options['method'] if 'method' in options else None,
            options=options,
        )
        self.success = result.success
        self.x_opt = result.x
        self.fun_val_opt = result.fun
        self.status = result.message

    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)
