from operator import attrgetter
from typing import List

from anml.data.prototype import DataPrototype
from anml.parameter.main import Parameter
from numpy.typing import NDArray
from pandas import DataFrame


class ModelPrototype:

    data = property(attrgetter("_data"))
    parameters = property(attrgetter("_parameters"))

    def __init__(self,
                 data: DataPrototype,
                 parameters: List[Parameter],
                 df: DataFrame):
        self.data = data
        self.parameters = parameters
        self.attach(df)

        self.opt_result = {}

    @data.setter
    def data(self, data: DataPrototype):
        if not isinstance(data, DataPrototype):
            raise TypeError("ModelPrototype input data must be an instance of "
                            "DataPrototype.")

    @parameters.setter
    def parameters(self, parameters: List[Parameter]):
        parameters = list(parameters)
        if not all(isinstance(parameter, Parameter)
                   for parameter in parameters):
            raise TypeError("ModelPrototype input parameters must be a list of "
                            "instances of Parameter.")

    def attach(self, df: DataFrame):
        self.data.attach(df)
        for parameter in self.parameters:
            parameter.attach(df)

    def objective(self, x: NDArray) -> float:
        raise NotImplementedError()

    def gradient(self, x: NDArray) -> NDArray:
        raise NotImplementedError()

    def hessian(self, x: NDArray) -> NDArray:
        raise NotImplementedError()

    def fit(self, x0: NDArray, **options):
        raise NotImplementedError()

    def predict(self, df: DataFrame, x: NDArray, prefix: str = "") -> DataFrame:
        df = df.copy()
        for i, parameter in self.parameters:
            df[f"{prefix}parameter_{i}"] = parameter.get_params(x, df, order=0)
        return df
