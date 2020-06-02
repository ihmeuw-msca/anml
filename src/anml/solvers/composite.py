from typing import Callable, Optional, Dict, Any
import numpy as np

from anml.solvers.interface import Solver, CompositeSolver


class MultipleInitializations(CompositeSolver):

    def __init__(self, sample_fun: Callable, solver: Optional[Solver] = None):
        super().__init__()
        self.sample_fun = sample_fun

    def fit(self, data, x_init: Optional[np.ndarray] = None, options: Optional[Dict[str, Any]] = None):
        self.assert_solvers_defined()
        if len(self.solvers) > 1:
            raise RuntimeError('Only implemented for single solver.')
        xs_init = self.sample_fun(x_init)
        fun_vals = []
        xs_opt = []
        for x in xs_init:
            self.solvers[0].fit(data, x, options=options)
            fun_vals.append(self.solvers[0].fun_val_opt)
            xs_opt.append(self.solvers[0].x_opt)

        self.x_opt = xs_opt[np.argmin(fun_vals)]
        self.fun_val_opt = np.min(fun_vals)

    def predict(self, **kwargs):
        return self.solvers[0].predict(self.x_opt, **kwargs)
