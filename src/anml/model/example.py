from typing import List

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
        m = self.parameters[0].get_params(x, order=1)
        value = -(m.T / self.data.obs_se.value**2).dot(r)
        value += self.parameters[0].prior_gradient(x)
        return value

    def hessian(self, x: NDArray) -> NDArray:
        m = self.parameters[0].get_params(x, order=1)
        value = (m.T / self.data.obs_se.value**2).dot(m)
        value += self.parameters[0].prior_hessian(x)
        return value

    def jacobian2(self, x: NDArray) -> NDArray:
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        m = self.parameters[0].get_params(x, order=1)
        jacobian = -(m.T / self.data.obs_se.value**2) * r
        value = jacobian.dot(jacobian.T)
        value += self.parameters[0].prior_hessian(x)
        return value
