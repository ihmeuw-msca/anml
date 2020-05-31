from dataclasses import fields, field, InitVar
from pydantic.dataclasses import dataclass
from typing import List, Callable
import numpy as np

@dataclass
class Variable:
    """A class that stores information about a variable.

    Attributes
    ----------
    covariate
        string, name of the covariate for this variable. 
    var_link_fun
        callable, link function for this variable. 
    fe_init
        float, initial value to be used in optimization for fixed effect. 
    re_init
        float, initial value to be used in optimization for random effect.
    re_zero_sum_std:
        optional, float, standard deviation of zero sum prior for random effects.
    fe_gprior:
        optional, a list of two floats (e.g., [mean, std]), Gaussian prior for fixed effect.
    re_gprior:
        optional, a list of two floats (e.g., [mean, std]), Gaussian prior for random effect.
    fe_bounds:
        optional, a list of two floats (e.g., [lower bound, upper bound]), box constraint for fixed effect. 
    re_bounds:
        optional, a list of two floats (e.g., [lower bound, upper bound]), box constraint for random effect. 
    
    """
    covariate: str
    var_link_fun: Callable
    fe_init: float
    re_init: float
    re_zero_sum_std: float = field(default=np.inf)
    fe_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])
    re_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])
    fe_bounds: List[float] = field(default_factory=lambda: [-np.inf, np.inf])
    re_bounds: List[float] = field(default_factory=lambda: [-np.inf, np.inf])

    def __post_init__(self):
        assert isinstance(self.covariate, str)
        assert len(self.fe_gprior) == 2
        assert len(self.re_gprior) == 2
        assert len(self.fe_bounds) == 2
        assert len(self.re_bounds) == 2
        assert self.fe_gprior[1] > 0.0
        assert self.re_gprior[1] > 0.0


@dataclass
class Parameter:

    param_name: str
    link_fun: Callable
    variables: InitVar[List[Variable]]

    num_fe: int = field(init=False)
    covariate: List[str] = field(init=False)
    var_link_fun: List[Callable] = field(init=False)
    fe_init: List[float] = field(init=False)
    re_init: List[float] = field(init=False)
    re_zero_sum_std: List[float] = field(init=False)
    fe_gprior: List[List[float]] = field(init=False)
    re_gprior: List[List[float]] = field(init=False)
    fe_bounds: List[List[float]] = field(init=False)
    re_bounds: List[List[float]] = field(init=False)

    def __post_init__(self, variables):
        assert isinstance(variables, list)
        assert len(variables) > 0
        assert isinstance(variables[0], Variable)
        self.num_fe = len(variables)
        for k, v in consolidate(Variable, variables).items():
            self.__setattr__(k, v)


@dataclass
class ParameterFunction:

    param_function_name: str
    param_function: Callable
    param_function_fe_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])

    def __post_init__(self):
        assert isinstance(self.param_function_name, str)
        assert len(self.param_function_fe_gprior) == 2
        assert self.param_function_fe_gprior[1] > 0.0


@dataclass
class ParameterSet:

    parameters: InitVar[List[Parameter]]
    parameter_functions: InitVar[List[ParameterFunction]] = None

    param_name: List[str] = field(init=False)
    num_fe: int = field(init=False)
    link_fun: List[Callable] = field(init=False)
    covariate: List[List[str]] = field(init=False)
    var_link_fun: List[List[Callable]] = field(init=False)
    fe_init: List[List[float]] = field(init=False)
    re_init: List[List[float]] = field(init=False)
    re_zero_sum_std: List[List[float]] = field(init=False)
    fe_gprior: List[List[List[float]]] = field(init=False)
    re_gprior: List[List[List[float]]] = field(init=False)
    fe_bounds: List[List[List[float]]] = field(init=False)
    re_bounds: List[List[List[float]]] = field(init=False)

    param_function_name: List[str] = field(init=False)
    param_function: List[Callable] = field(init=False)
    param_function_fe_gprior: List[List[float]] = field(init=False)

    def __post_init__(self, parameters, parameter_functions):

        for k, v in consolidate(Parameter, parameters, exclude=['num_fe']).items():
            self.__setattr__(k, v)

        for k, v in consolidate(ParameterFunction, parameter_functions).items():
            self.__setattr__(k, v)

        if len(set(self.param_name)) != len(self.param_name):
            raise RuntimeError("Cannot have duplicate parameters in a set.")
        if len(set(self.param_function_name)) != len(self.param_function_name):
            raise RuntimeError("Cannot have duplicate parameter functions in a set.")

        self.num_fe = 0
        for param in parameters:
            self.num_fe += param.num_fe

    def get_param_index(self, param_name):
        try:
            param_index = self.param_name.index(param_name)
        except ValueError:
            raise RuntimeError(f"No {param_name} parameter in this parameter set.")
        return param_index

    def get_param_function_index(self, param_function_name):
        try:
            param_function_index = self.param_function_name.index(param_function_name)
        except ValueError:
            raise RuntimeError(f"No {param_function_name} parameter in this parameter set.")
        return param_function_index

    def delete_random_effects(self):
        param_set = self.clone()
        bounds = np.array(self.re_bounds)
        bounds[:] = 0.
        param_set.re_bounds = bounds.tolist()
        return param_set


def consolidate(cls, instance_list, exclude=None):
    if exclude is None:
        exclude = []
    consolidated = {}
    for f in fields(cls):
        if f.name not in exclude:
            if instance_list is not None:
                consolidated[f.name] = [instance.__getattribute__(f.name) for instance in instance_list]
            else:
                consolidated[f.name] = list()
    return consolidated
    