"""
=================
Solvers Interface 
=================
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np 

from anml.data.data import Data
from anml.models.interface import Model
from anml.exceptions import ANMLError


class ModelNotDefinedError(ANMLError):
    pass


class SolverNotDefinedError(ANMLError):
    pass


class Solver:

    def __init__(self, model_instance: Optional[Model] = None):
        self._model = model_instance
        self.x_opt = None
        self.fun_val_opt = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_instance: Model):
        self._model = model_instance

    def assert_model_defined(self):
        if self._model is None:
            raise ModelNotDefinedError()

    def fit(self, x_init: Optional[np.ndarray] = None, data: Optional[Data] = None, options: Optional[Dict[str, Any]] = None, **kwargs):
        raise NotImplementedError()

    def predict(self, **kwargs):
        return self.model.forward(self.x_opt, **kwargs)


class CompositeSolver(Solver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(model_instance=None)
        if solvers_list is not None:
            self._solvers = solvers_list
        else:
            self._solvers = []

    @property
    def solvers(self):
        return self._solvers

    @solvers.setter
    def solvers(self, solvers_list: List[Solver]):
        self._solvers = solvers_list

    def add_solver(self, solver: Solver):
        self._solvers.append(solver)

    @property
    def model(self):
        models = []
        self.assert_solvers_defined()
        for solver in self._solvers:
            models.append(solver.model)
        return models

    @model.setter
    def model(self, model_instances: Union[Model, List[Model]]):
        self.assert_solvers_defined()
        if isinstance(model_instances, list):
            if len(model_instances) != len(self._solvers):
                raise ValueError(
                    'When passing in multiple models its length should equal to the number of solvers passed in.'
                )
            for model, solver in zip(model_instances, self._solvers):
                solver.model = model
        else:
            for solver in self._solvers:
                solver.model = model_instances

    def assert_model_defined(self):
        self.assert_solvers_defined()
        for solver in self._solvers:
            solver.assert_model_defined()

    def assert_solvers_defined(self):
        if len(self._solvers) == 0:
            raise SolverNotDefinedError()
