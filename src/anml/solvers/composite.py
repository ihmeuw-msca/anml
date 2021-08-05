from typing import Callable, Optional, Dict, Any
"""
=================
Composite Solvers
=================

Composite solvers for optimization. Composition or decorator of solvers.
"""

import numpy as np
from scipy.optimize import bisect

from anml.data.data import Data
from anml.solvers.interface import Solver, CompositeSolver
from anml.models.interface import TrimmingCompatibleModel


class MultipleInitializations(CompositeSolver):
    """Solver with multiple initialization
    """

    def __init__(self, sample_fun: Callable, solver: Optional[Solver] = None):
        super().__init__()
        self.sample_fun = sample_fun

    def fit(self, x_init: Optional[np.ndarray] = None, data: Optional[Data] = None, options: Optional[Dict[str, Any]] = None):
        self.assert_solvers_defined()
        if len(self.solvers) > 1:
            raise RuntimeError('Only implemented for single solver.')
        xs_init = self.sample_fun(x_init)
        fun_vals = []
        xs_opt = []
        for x in xs_init:
            self.solvers[0].fit(data=data, x_init=x, options=options)
            fun_vals.append(self.solvers[0].fun_val_opt)
            xs_opt.append(self.solvers[0].x_opt)

        self.x_opt = xs_opt[np.argmin(fun_vals)]
        self.fun_val_opt = np.min(fun_vals)

    def predict(self, **kwargs):
        return self.solvers[0].predict(self.x_opt, **kwargs)


class TrimmingSolver(CompositeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def c_simplex(self, w, h):
        a = np.min(w) - 1.0
        b = np.max(w) - 0.0

        f = lambda x: np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - h

        x = bisect(f, a, b)

        return np.maximum(np.minimum(w - x, 1.0), 0.0)

    def fit(self, x_init: np.ndarray, data: Data, n: int,
            options: Optional[Dict[str, Any]] = None,
            pct_trimming: float = 0.0,
            step_size: float = 1.0, max_iter: int = 100, tol: float = 1e-6):
        self.assert_solvers_defined()
        if len(self.solvers) > 1:
            raise RuntimeError("Only implemented for a single solver.")
        if not isinstance(self.solvers[0].model, TrimmingCompatibleModel):
            raise RuntimeError("The model you're trying to use is not compatible with trimming.")

        # Get the initial fit
        np.random.seed(10)
        self.solvers[0].fit(data=data, x_init=x_init, options=options)
        x_init = self.solvers[0].x_opt

        i = 0

        if pct_trimming > 0:

            # Initialize the weights and trimming model
            w_init = np.repeat(pct_trimming, n)
            h = (1 - pct_trimming) * n

            err = tol + 1.0

            w = w_init
            x = x_init

            while (err > tol) & (i < max_iter):
                print("Trimming iteration: ", i, "/", max_iter, "-- error: ", round(err, 5), end='\r')
                # Get the objective function
                _obj = self.solvers[0].model._objective(x=x, data=data)
                w_new = self.c_simplex(w - step_size*_obj, h=h)

                # get current error
                err = np.linalg.norm(w_new - w)/step_size

                # Update the weights
                w = w_new
                self.solvers[0].model.w = w
                self.solvers[0].fit(data=data, x_init=x, options=options)
                x = self.solvers[0].x_opt
                i += 1

    def predict(self, **kwargs):
        return self.solvers[0].predict(self.x_opt, **kwargs)
