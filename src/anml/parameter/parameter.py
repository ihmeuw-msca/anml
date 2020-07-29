"""
==========
Parameters
==========
"""

from dataclasses import field, dataclass
from typing import List, Callable

import pandas as pd

from anml.exceptions import ANMLError
from anml.parameter.prior import Prior
from anml.parameter.variables import Variable, ParameterBlock


class ParameterError(ANMLError):
    pass


class ParameterSetError(ANMLError):
    pass


@dataclass
class Parameter(ParameterBlock):
    """A class for parameters. 
    
    Parameters
    ----------
    param_name: str
        name of the parameter
    link_fun: callable
        link function for the parameter
    variables: List[:class:`~anml.parameter.variables.Variable`]
        a list of variables
    
    Attributes
    ----------
    All attributes from :class:`~anml.parameter.parameter.Variable`s in `variables`
    are carried over but are put into a list.

    """
    param_name: str
    variables: List[Variable]
    link_fun: Callable = lambda x: x

    def __post_init__(self):
        assert isinstance(self.variables, list)
        assert len(self.variables) > 0
        assert all(isinstance(variable, Variable) for variable in self.variables)
        self.num_fe = 0
        self.num_re_var = 0
        for variable in self.variables:
            self.num_fe += variable.num_fe
            self.num_re_var += variable.num_re_var

    def _validate_df(self, df: pd.DataFrame):
        for variable in self.variables:
            variable._validate_df(df)

    @property
    def num_re(self):
        n = 0
        for variable in self.variables:
            n += variable.num_re
        return n


@dataclass
class ParameterFunction:
    """A class for function on parameters.

    Parameters
    ----------
    param_function_name: str
        name of the parameter function
    param_function: callable
        parameter function
    param_function_fe_prior: List[float]
        a list of two floats specifying mean and std for Gaussian prior on the function.
    """

    param_function_name: str
    param_function: Callable
    param_function_fe_prior: Prior = Prior()

    def __post_init__(self):
        assert isinstance(self.param_function_name, str)


@dataclass
class ParameterSet(ParameterBlock):
    """A class for a set of parameters.

    Parameters
    ----------
    parameters: List[:class:`~anml.parameter.parameter.Parameter`]
        a list of parameters.
    parameter_functions: List[:class:`~anml.parameter.parameter.ParameterFunction`]
        a list of parameter functions.
    """

    parameters: List[Parameter]
    parameter_functions: List[ParameterFunction] = None

    param_name: List[str] = field(init=False)
    variables: List[Variable] = field(init=False)

    def __post_init__(self):
        assert isinstance(self.parameters, list)
        assert len(self.parameters) > 0
        assert all(isinstance(parameter, Parameter) for parameter in self.parameters)
        
        self.param_name = [param.param_name for param in self.parameters]
        if len(set(self.param_name)) < len(self.param_name):
            raise ParameterSetError("Cannot have duplicate parameters in a set.")
        
        if self.parameter_functions is not None:
            self.param_function_name = [param_func.param_function_name for param_func in self.parameter_functions]
            if len(set(self.param_function_name)) < len(self.param_function_name):
                raise ParameterSetError("Cannot have duplicate parameter functions in a set.")

        self.num_fe = 0
        self.num_re_var = 0
        for param in self.parameters:
            self.num_fe += param.num_fe
            self.num_re_var += param.num_re_var

        self.variables = list()
        for parameter in self.parameters:
            for variable in parameter.variables:
                self.variables.append(variable)

        self.reset()

    @property 
    def num_re(self):
        self._num_re = 0
        for param in self.parameters:
            self._num_re += param.num_re
        return self._num_re 

    def _validate_df(self, df: pd.DataFrame):
        for param in self.parameters:
            param._validate_df(df)

    def get_param_index(self, param_name: str):
        """A function that returns index of a given parameter.

        Parameters
        ----------
        param_name : str
            name of the parameter

        Returns
        -------
        int
            index of the parameter

        Raises
        ------
        RuntimeError
            parameter not found in the parameter set.
        """
        try:
            param_index = self.param_name.index(param_name)
        except ValueError:
            raise ParameterSetError(f"No {param_name} parameter in this parameter set.")
        return param_index

    def get_param_function_index(self, param_function_name: str) -> int:
        """A function that returns index of a given parameter function. 

        Parameters
        ----------
        param_function_name : str
            name of the parameter function

        Returns
        -------
        int
            index of the parameter function

        Raises
        ------
        RuntimeError
            parameter function not found in the parameter set.
        """
        try:
            param_function_index = self.param_function_name.index(param_function_name)
        except ValueError:
            raise ParameterSetError(f"No {param_function_name} parameter function in this parameter set.")
        return param_function_index
