"""
========================
Variables and Parameters
========================

Variables are the most granular object for constructing a model specification.
At the simplest level, a variable is just :class:`~anml.parameter.variable.Intercept`,
which is a column of ones (indicating that it does not change based on the data row, except through
an optional random effect).

Each Variable has a method :func:`~anml.parameter.variable.design_matrix`
that gets the design matrix for that single covariate. Usually, this will just return the same
array of covariate values that is passed, but in the case of a :class:`~anml.parameter.variable.Spline`
it will return a larger design matrix representing the spline basis.

Parameters are made up of variables, e.g. the "mean" is a function of one or more variables.
ParameterSets are sets of parameters that are related to each other, e.g. to parametrize the same
distribution like mean and variance of the normal distribution. ParameterSets can also have
functional priors.
"""

from dataclasses import field, dataclass
from typing import List, Callable
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.linalg import block_diag

from anml.parameter.prior import Prior
from anml.exceptions import ANMLError
from anml.parameter.variables import Variable


class ParameterError(ANMLError):
    pass


class ParameterSetError(ANMLError):
    pass


@dataclass
class Parameter:
    """A class for parameters. 
    
    Parameters
    ----------
    param_name: str
        name of the parameter
    link_fun: callable
        link function for the parameter
    variables: List[:class:`~anml.parameter.parameter.Variable`]
        a list of variables
    
    Attributes
    ----------
    All attributes from :class:`~anml.parameter.parameter.Variable`s in `variables`
    are carried over but are put into a list.

    num_fe: int
        total number of effects (variables) for the parameter

    """
    param_name: str
    variables: List[Variable]
    link_fun: Callable = lambda x: x

    num_fe: int = field(init=False)
    num_re_var: int = field(init=False)

    def __post_init__(self):
        assert isinstance(self.variables, list)
        assert len(self.variables) > 0
        assert all(isinstance(variable, Variable) for variable in self.variables)
        self.num_fe = 0
        self.num_re_var = 0
        for variable in self.variables:
            self.num_fe += variable.num_fe
            self.num_re_var += variable.num_re_var

    def _validate_df(self, df):
        for variable in self.variables:
            variable._validate_df(df)


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
class ParameterSet:
    """A class for a set of parameters.

    Parameters
    ----------
    parameters: List[:class:`~anml.parameter.parameter.Parameter`]
        a list of parameters.
    parameter_functions: List[:class:`~anml.parameter.parameter.ParameterFunction`]
        a list of parameter functions.

    Attributes
    ----------
    All attributes from :class:`~anml.parameter.parameter.Parameter`s in `parameters`
    are carried over and put into a list of lists.

    num_fe: int
        total number of effects (variables) for the parameter set.
    """

    parameters: List[Parameter]
    parameter_functions: List[ParameterFunction] = None

    param_name: List[str] = field(init=False)
    
    num_fe: int = field(init=False)
    num_re_var: int = field(init=False)

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
        