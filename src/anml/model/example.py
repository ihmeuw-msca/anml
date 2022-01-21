from typing import List

import numpy as np
from anml.data.example import DataExample
from anml.model.prototype import ModelPrototype
from anml.parameter.main import Parameter
from anml.variable.main import Variable
from numpy.typing import NDArray
from pandas import DataFrame


class ModelExample(ModelPrototype):

    def __init__(self,
                 data: DataExample,
                 variables: List[Variable],
                 df: DataFrame):
        parameter = Parameter(variables)
        super().__init__(data, [parameter], df)

    def objective(self, x: NDArray) -> float:
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        value = 0.5*sum((r / self.data.obs_se.value)**2)
        value += self.parameters[0].prior_objective(x)
        return value

    def gradient(self, x: NDArray) -> NDArray:
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        d1 = self.parameters[0].get_params(x, order=1)
        value = -(d1.T / self.data.obs_se**2).dot(r)
        value += self.parameters[0].prior_gradient(x)
        return value

    def hessian(self, x: NDArray) -> NDArray:
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        d1 = self.parameters[0].get_params(x, order=1)
        d2 = self.parameters[0].get_params(x, order=2)
        value = (d1.T / self.data.obs_se.value**2).dot(d1)
        value += np.tensordot(-r / self.data.obs_se**2, d2, axis=1)
        value += self.parameters[0].prior_hessian(x)
        return value

    def jacobian2(self, x: NDArray) -> NDArray:
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        d1 = self.parameters[0].get_params(x, order=1)
        jacobian = d1.T * (-r / self.data.obs_se**2)
        value = jacobian.T.dot(jacobian)
        value += self.parameters[0].prior_hessian(x)
        return value
