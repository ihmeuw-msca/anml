"""
========================
Variables and Parameters
========================

Variables are the most granular object for constructing a model specification.
At the simplest level, a variable is just :class:`~anml.parameter.variable.Intercept`,
which is a column of ones (indicating that it does not change based on the data row, except through
an optional random effect).

Each Variable has a method :func:`~anml.parameter.variable.design_mat`
that gets the design matrix for that single covariate. Usually, this will just return the same
array of covariate values that is passed, but in the case of a :class:`~anml.parameter.variable.Spline`
it will return a larger design matrix representing the spline basis.

Parameters are made up of variables, e.g. the "mean" is a function of one or more variables.
ParameterSets are sets of parameters that are related to each other, e.g. to parametrize the same
distribution like mean and variance of the normal distribution. ParameterSets can also have
functional priors.
"""

from dataclasses import fields, field
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict
import numpy as np
import pandas as pd
from copy import deepcopy

from xspline import XSpline

from anml.parameter.prior import Prior
from anml.exceptions import ANMLError

PROTECTED_NAMES = ['intercept']


class VariableError(ANMLError):
    pass


class ParameterError(ANMLError):
    pass


class ParameterSetError(ANMLError):
    pass


@dataclass
class Variable:
    """A class that stores information about a variable.

    Parameters
    ----------
    covariate: str
        name of the covariate for this variable. 
    var_link_fun: callable
        link function for this variable. 
    fe_init: float
        initial value to be used in optimization for fixed effect. 
    re_init: float
        initial value to be used in optimization for random effect.
    re_zero_sum_std: float, optional
        standard deviation of zero sum prior for random effects.
    fe_prior: Prior, optional
        a prior of class :class:`~anml.parameter.prior.Prior`
    re_prior: Prior, optional
        a prior of class :class:`~anml.parameter.prior.Prior`

    Attributes
    ----------
    All parameters become attributes after validation.
    
    """
    var_link_fun: Callable = lambda x: x
    covariate: str = None

    fe_init: float = 0.
    re_init: float = 0.

    fe_prior: Prior = Prior()
    re_prior: Prior = Prior()

    re_zero_sum_std: float = field(default=np.inf)

    def __post_init__(self):
        if self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")

    def design_mat(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the design matrix based on a covariate x.

        Parameters
        ----------
        df
            pandas DataFrame of covariate values (one dimensional)

        Returns
        -------
        2-dimensional reshaped version of :python:`x`

        """
        x = df[self.covariate].values
        return np.asarray(x).reshape((len(x), 1))


@dataclass
class Intercept(Variable):
    """An intercept variable.
    """
    def __post_init__(self):
        self.covariate = 'intercept'

    def design_mat(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones((len(df), 1))


@dataclass
class Spline(Variable):
    """A spline variable.
    """

    knots_type: str = 'frequency'
    knots_num: int = 3
    degree: int = 3
    l_linear: bool = False
    r_linear: bool = False

    def __post_init__(self):
        if self.knots_type not in ['frequency', 'domain']:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")

    def design_mat(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.covariate].values

        spline_knots = np.linspace(0, 1, self.knots_num)
        if self.knots_type == 'frequency':
            knots = np.quantile(x, spline_knots)
        elif self.knots_type == 'domain':
            knots = x.min() + spline_knots * (x.max() - x.min())
        else:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")

        xs = XSpline(
            knots=knots,
            degree=self.degree,
            l_linear=self.l_linear,
            r_linear=self.r_linear
        )
        return xs.design_mat(x)[:, 1:]


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
    link_fun: Callable
    variables: List[Variable]

    num_fe: int = field(init=False)
    covariate: List[str] = field(init=False)
    var_link_fun: List[Callable] = field(init=False)

    fe_init: List[float] = field(init=False)
    re_init: List[float] = field(init=False)

    fe_prior: List[List[Prior]] = field(init=False)
    re_prior: List[List[Prior]] = field(init=False)

    re_zero_sum_std: List[float] = field(init=False)

    def __post_init__(self):
        assert isinstance(self.variables, list)
        assert len(self.variables) > 0
        assert isinstance(self.variables[0], Variable)
        self.num_fe = len(self.variables)
        for k, v in consolidate(Variable, self.variables).items():
            self.__setattr__(k, v)


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
    link_fun: List[Callable] = field(init=False)
    covariate: List[List[str]] = field(init=False)
    var_link_fun: List[List[Callable]] = field(init=False)

    fe_init: List[List[float]] = field(init=False)
    re_init: List[List[float]] = field(init=False)

    fe_prior: List[List[List[Prior]]] = field(init=False)
    re_prior: List[List[List[Prior]]] = field(init=False)

    re_zero_sum_std: List[List[float]] = field(init=False)

    param_function_name: List[str] = field(init=False)
    param_function: List[Callable] = field(init=False)
    param_function_fe_prior: List[Prior] = field(init=False)

    def __post_init__(self):

        for k, v in consolidate(Parameter, self.parameters, exclude=['num_fe']).items():
            self.__setattr__(k, v)

        for k, v in consolidate(ParameterFunction, self.parameter_functions).items():
            self.__setattr__(k, v)

        if len(set(self.param_name)) != len(self.param_name):
            raise RuntimeError("Cannot have duplicate parameters in a set.")
        if len(set(self.param_function_name)) != len(self.param_function_name):
            raise RuntimeError("Cannot have duplicate parameter functions in a set.")

        self.num_fe = 0
        for param in self.parameters:
            self.num_fe += param.num_fe

    @property
    def _flat_covariates(self):
        return [item for sublist in self.covariate for item in sublist]

    def _validate_df(self, df: pd.DataFrame):
        for covariate in self._flat_covariates:
            if covariate not in PROTECTED_NAMES and covariate not in df.columns:
                raise ParameterSetError(f"Covariate {covariate} is missing from the data frame.")

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
            raise RuntimeError(f"No {param_name} parameter in this parameter set.")
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
            raise RuntimeError(f"No {param_function_name} parameter function in this parameter set.")
        return param_function_index

    def delete_random_effects(self):
        """A function that deletes random effects for all parameters in the parameter set.

        Returns
        -------
        :class:`~anml.parameter.parameter.ParameterSet`
            a parameter set with no random effects on parameters.
        """
        param_set = deepcopy(self)
        bounds = np.array(self.re_bounds)
        bounds[:] = 0.
        param_set.re_bounds = bounds.tolist()
        return param_set


def consolidate(cls, instance_list,
                exclude: Optional[List[str]] = None) -> Dict[str, List]:
    """A function that given a list of objects of the same type, 
    collect their values corresponding to the same attribute and put into a list.

    Parameters
    ----------
    cls : Object
        the class of the objects in :python:`instance_list`
    instance_list : List[Object]
        a list of objects of the same type
    exclude : List[str], optional
        attributes that do not wish to be collected and consolidated, by default None

    Returns
    -------
    Dict[str, List]
        a dictionary where key is the name of an attribute and value a list of attribute values collected
        from the objects.
    """
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
