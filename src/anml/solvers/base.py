"""
============
Base Solvers
============

Basic solvers for optimization. Takes a model directly. 
"""

import ipopt
from typing import Dict, Any, Optional
import numpy as np
import scipy.optimize as sciopt
from scipy.optimize import LinearConstraint, Bounds

from anml.data.data import Data
from anml.solvers.interface import Solver
from anml.solvers.utils import has_bounds, has_constraints


class ScipyOpt(Solver):
    """A concrete class of `Solver` that use `scipy` optimize.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.success = None
        self.status = None
        self.hess_inv = None

    def fit(self, x_init: np.ndarray, data: Optional[Data] = None, options: Optional[Dict[str, Any]] = None):
        self.assert_model_defined()

        if has_bounds(self.model):
            bounds = Bounds(self.model.lb, self.model.ub)
        else:
            bounds = None

        if has_constraints(self.model):
            constraints = LinearConstraint(self.model.C, self.model.c_lb, self.model.c_ub)
        else:
            constraints = None

        if 'method' in options:
            method = options['method']
        elif constraints is not None:
            method = 'trust-constr'
        else:
            method = None

        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data),
            x0=x_init,
            jac=lambda x: self.model.gradient(x, data),
            bounds=bounds,
            method=method,
            options=options['solver_options'],
            constraints=constraints,
        )
        self.success = result.success
        self.x_opt = result.x
        self.fun_val_opt = result.fun
        self.status = result.message
        self.hess_inv = result.hess_inv


class _IPOPTProblem:
    """Define a IPOPT problem class.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data  

    def objective(self, x):
        return self.model.objective(x, self.data)

    def gradient(self, x):
        return self.model.gradient(x, self.data)

    def constraints(self, x):
        return np.dot(self.model.C, x)

    def jacobian(self, x):
        return self.model.C 


class IPOPTSolver(Solver):
    """Solver using IPOPT
    """

    def fit(self, x_init: np.ndarray, data: Optional[Data] = None, options: Optional[Dict[str, Any]] = None):
        problem_obj = _IPOPTProblem(self.model, data)
        if has_bounds(self.model) and has_constraints(self.model):
            problem = ipopt.problem(
                n=len(x_init),
                m=len(self.model.C),
                problem_obj=problem_obj,
                lb=self.model.lb,
                ub=self.model.ub,
                cl=self.model.c_lb,
                cu=self.model.c_ub,
            )
        elif has_bounds(self.model):
            problem_obj.constraints = None 
            problem_obj.jacobian = None
            problem = ipopt.problem(
                n=len(x_init),
                m=0,
                problem_obj=problem_obj,
                lb=self.model.lb,
                ub=self.model.ub,
            ) 
        elif has_constraints(self.model):
            problem = ipopt.problem(
                n=len(x_init),
                m=len(self.model.C),
                problem_obj=problem_obj,
                cl=self.model.c_lb,
                cu=self.model.c_ub,
            )
        else:
            problem_obj.constraints = None
            problem_obj.jacobian = None
            problem = ipopt.problem(
                n=len(x_init),
                m=0,
                problem_obj=problem_obj
            )
        for name, val in options['solver_options'].items():
            problem.addOption(name, options['solver_options'][val])
        self.x_opt, self.info = problem.solve(x_init)
        self.fun_val_opt = problem_obj.objective(self.x_opt)


class ClosedFormSolver(Solver):
    """Solver using closed formed solution defined in corresponding model.
    """

    def fit(self, x_init: np.ndarray = None, data: Optional[Data] = None, options: Dict[str, Any] = None):
        if hasattr(self.model, 'closed_form_soln'):
            self.success = True 
            self.x_opt = self.model.closed_form_soln(data)
            self.fun_val_opt = self.model.objective(self.x_opt, data)
        else:
            raise TypeError('Model does not have attribute closed_form_soln')
