from typing import Dict, Any, Optional
import numpy as np
import scipy.optimize as sciopt

from anml.data.data import Data
from anml.solvers.interface import Solver


class ScipyOpt(Solver):
    """A concrete class of `Solver` that use `scipy` optimize 
    to fit the model using the L-BFGS-B method.
    """

    def fit(self, x_init: np.ndarray, data: Optional[Data] = None, options: Optional[Dict[str, Any]] = None):
        self.assert_model_defined()
        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data),
            x0=x_init,
            jac=lambda x: self.model.gradient(x, data),
            bounds=self.model.bounds,
            method=options['method'] if 'method' in options else None,
            options=options['solver_options'],
            constraints=self.model.constraints,
        )
        self.success = result.success
        self.x_opt = result.x
        self.fun_val_opt = result.fun
        self.status = result.message

    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)


class ClosedFormSolver(Solver):

    def fit(self, x_init: np.ndarray = None, data: Optional[Data] = None, options: Dict[str, Any] = None):
        if hasattr(self.model, 'closed_form_soln'):
            self.success = True 
            self.x_opt = self.model.closed_form_soln(data)
            self.fun_val_opt = self.model.objective(self.x_opt, data)
        else:
            raise TypeError('Model does not have attribute closed_form_soln')
    
    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)
               
