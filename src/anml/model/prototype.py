from abc import ABC, abstractmethod
from operator import attrgetter
from typing import List, Optional

import numpy as np
from anml.data.prototype import DataPrototype
from anml.parameter.main import Parameter
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize


class ModelPrototype(ABC):

    data = property(attrgetter("_data"))
    parameters = property(attrgetter("_parameters"))

    def __init__(self,
                 data: DataPrototype,
                 parameters: List[Parameter],
                 df: DataFrame):
        self.data = data
        self.parameters = parameters
        self.attach(df)

        self.result = {}

    @data.setter
    def data(self, data: DataPrototype):
        if not isinstance(data, DataPrototype):
            raise TypeError("ModelPrototype input data must be an instance of "
                            "DataPrototype.")
        self._data = data

    @parameters.setter
    def parameters(self, parameters: List[Parameter]):
        parameters = list(parameters)
        if not all(isinstance(parameter, Parameter)
                   for parameter in parameters):
            raise TypeError("ModelPrototype input parameters must be a list of "
                            "instances of Parameter.")
        self._parameters = parameters
        self._sizes = [parameter.size for parameter in self._parameters]

    def attach(self, df: DataFrame):
        self.data.attach(df)
        for parameter in self.parameters:
            parameter.attach(df)

    @abstractmethod
    def objective(self, x: NDArray) -> float:
        pass

    @abstractmethod
    def gradient(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def hessian(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def jacobian2(self, x: NDArray) -> NDArray:
        pass

    def _splitx(self, x: NDArray) -> List[NDArray]:
        return np.split(x, np.cumsum(self._sizes)[:-1])

    def _get_initial_x(self) -> NDArray:
        return np.zeros(sum(self._sizes))

    def _get_vcov(self, x: NDArray) -> NDArray:
        inv_hessian = np.linalg.pinv(self.hessian(x))
        jacobian2 = self.jacobian2(x)
        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

    def fit(self, x0: Optional[NDArray] = None, **options):
        if x0 is None:
            x0 = self._get_initial_x()
        # bounds
        bounds = np.hstack([parameter.prior_dict["direct"]["UniformPrior"].params
                            for parameter in self.parameters]).T
        # linear constraint
        params = np.hstack([parameter.prior_dict["linear"]["UniformPrior"].params
                            for parameter in self.parameters])
        mat = block_diag(*[parameter.prior_dict["linear"]["UniformPrior"].mat
                           for parameter in self.parameters])
        linear_constraints = []
        if params.size > 0:
            linear_constraints = [LinearConstraint(mat, params[0], params[1])]

        info = minimize(self.objective, x0,
                        method="trust-constr",
                        jac=self.gradient,
                        hess=self.hessian,
                        constraints=linear_constraints,
                        bounds=bounds,
                        options=options)

        # store result
        self.result["x"] = info.x
        self.result["vcov"] = self._get_vcov(self.result["x"])
        self.result["info"] = info

    def predict(self,
                x: Optional[NDArray] = None,
                df: Optional[DataFrame] = None) -> NDArray:
        if x is None:
            x = self.result.get("x", None)
        if x is None:
            raise ValueError("Please fit the model first or provide x.")
        xs = self._splitx(x)
        return [parameter.get_params(xs[i], df, order=0)
                for i, parameter in enumerate(self.parameters)]
