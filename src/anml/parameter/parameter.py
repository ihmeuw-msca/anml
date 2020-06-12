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
from dataclasses import dataclass, is_dataclass
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
    add_re: bool, optional
        whether to add random effects to this variable

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
    add_re: bool = False
    col_group: str = None

    re_zero_sum_std: float = field(default=np.inf)

    num_fe: int = field(init=False)

    def __post_init__(self):
        if self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")

        if self.add_re and self.col_group is None:
            raise ValueError('When add_re is True, a group column must be provided.')

        self.num_fe = 1


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
        if self.covariate is None:
            raise VariableError("No covariate has been set.")
        x = df[self.covariate].values
        return np.asarray(x).reshape((len(x), 1))

    def get_constraint_matrix(self):
        return np.array([[1.0]]), self.fe_prior.lower_bound, self.fe_prior.upper_bound


@dataclass
class Intercept(Variable):
    """An intercept variable.
    """
    def __post_init__(self):
        self.covariate = 'intercept'
        self.num_fe = 1

    def design_mat(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones((len(df), 1))


@dataclass 
class SplineLinearConstr:
    """Constraints on spline derivatives. The general form is
    lb <= Ax <= ub
    where x is in some interval domain `x_domain`, and A can be 0th, 1st or 2nd order derivative matrix 
    of the splines evaluated at some discretization points.
    A is not known at the initialization of this object, but will have dimension
    `grid_size` by `number of spline basis`.
    `lb` and `ub` are vectors of multiples of ones.

    This type of constraints can be used to impose monotonicity and convexity constraints.
    For instance, for splines defined on `[0, 5]`, one can specify monotonically decreasing on `[0,1]` with
    `constr = SplineLinearConstr(x_domain=[0, 1], y_bounds=[-np.inf, 0.0], order=1)`, 
    monotonically increasing on `[4, 5]` with 
    `constr = SplineLinearConstr(x_domain=[4, 5], y_bounds=[0.0, np.inf], order=1)`,
    and overall convexity with
    `constr = SplineLinearConstr(x_domain=[0, 5], y_bounds=[0.0, np.inf], order=2)`.

    Parameters
    ----------
    order: int
        order of the derivative
    y_bounds: List[float, float]
        bounds for y = Ax
    x_domain: List[float, float], optional
        domain for x, default to be -inf to inf 
    grid_size: int, optional
        size of grid

    Raises
    ------
    ValueError
        domain for x is not valid.
    ValueError
        bounds for y = Ax is not valid.
    ValueError
        invalid derivative order
    ValueError
        invalid grid size
    """
    order: int
    y_bounds: List[float]
    x_domain: List[float] = field(default_factory=lambda: [-np.inf, np.inf])
    grid_size: int = None

    def __post_init__(self):
        if self.x_domain[0] >= self.x_domain[1]:
            raise ValueError('Domain must have positive length.')
        if self.y_bounds[0] > self.y_bounds[1]:
            raise ValueError('Lower bound cannot be greater than upper bound.')
        if self.order < 0:
            raise ValueError('Order of derivative must be nonnegative.')
        if self.grid_size is not None and self.grid_size < 1:
            raise ValueError('Grid size must be at least 1.')


@dataclass
class Spline(Variable):
    """Spline variable.

    Parameters
    ----------
    knots_type : str
        type of knots. can only be 'frequency' or 'domain'
    knots_num: int 
        number of knots 
    degree: int
        degree of spines
    l_linear: bool
        whether left tail is linear
    r_linear: bool
        whether right tail is linear
    include_intercept: bool
        whether to include intercept in design matrix 
    derivative_constr: List[`~anml.parameter.parameter.SplineLinearConstr`]
        constraints on derivatives 
    constr_grid_size_global: int, optional
        number of points to use when building constraint matrix. used only when `grid_size` for 
        individual `~anml.parameter.parameter.SplineLinearConstr` is not available

    Raises
    ------
    VariableError
        unknown knot type
    VariableError
        no covariate has been set
    """

    knots_type: str = 'frequency'
    knots_num: int = 3
    degree: int = 3
    l_linear: bool = False
    r_linear: bool = False
    include_intercept: bool = False
    derivative_constr: List[SplineLinearConstr] = None
    constr_grid_size_global: int = None

    def __post_init__(self):
        self.add_re = False
        if self.knots_type not in ['frequency', 'domain']:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")
        self.num_fe = self.knots_num - self.l_linear - self.r_linear + self.degree - 1
        if not self.include_intercept:
            self.num_fe -= 1

        self.spline = None

    def create_spline(self, df: pd.DataFrame):
        if self.covariate is None:
            raise VariableError("No covariate has been set.")
        self.x = df[self.covariate].values

        spline_knots = np.linspace(0, 1, self.knots_num)
        if self.knots_type == 'frequency':
            knots = np.quantile(self.x, spline_knots)
        elif self.knots_type == 'domain':
            knots = self.x.min() + spline_knots * (self.x.max() - self.x.min())
        else:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")

        self.spline = XSpline(
            knots=knots,
            degree=self.degree,
            l_linear=self.l_linear,
            r_linear=self.r_linear
        )

    def design_mat(self, df: pd.DataFrame) -> np.ndarray:
        if self.spline is None:
            self.create_spline(df)
        
        if self.include_intercept:
            return self.spline.design_mat(self.x)
        else:
            return self.spline.design_mat(self.x)[:, 1:]

    def get_constraint_matrix(self) -> List[np.ndarray]:
        """build constrain matrix and bounds for
        `constr_lb` <= `constr_matrix` <= `constr_ub`.

        Returns
        -------
        List[np.ndarray]
            constraint matrix, lower bounds and upper bounds.
        """
    
        lb = max(self.fe_prior.lower_bound, min(self.x))
        ub = min(self.fe_prior.upper_bound, max(self.x))

        constr_matrices = []
        constr_lbs = []
        constr_ubs = []
        for constr in self.derivative_constr:
            if constr.x_domain[0] >= ub or constr.x_domain[1] <= lb:
                raise ValueError(f'Domain of constraint does not overlap with domain of spline. lb = {lb}, ub = {ub}.')
            if constr.grid_size is None and self.constr_grid_size_global is None:
                raise ValueError('Either global or individual constraint grid size needs to be specified.')
            
            if constr.grid_size is not None:
                points = np.linspace(max(lb, constr.x_domain[0]), min(ub, constr.x_domain[1]), constr.grid_size)
            else:
                points_all = np.linspace(lb, ub, self.constr_grid_size_global)
                is_in_domain = constr.x_domain[0] <= points_all <= constr.x_domain[1]
                points = points_all[is_in_domain]
            n_points = len(points)
            constr_matrices.append(self.spline.design_dmat(points, constr.order))
            constr_lbs.append([constr.y_bounds[0]] * n_points)
            constr_ubs.append([constr.y_bounds[1]] * n_points)
        
        constr_matrix = np.vstack(constr_matrices)
        constr_lb = np.hstack(constr_lbs)
        constr_ub = np.hstack(constr_ubs)
        if self.include_intercept:
            return constr_matrix, constr_lb, constr_ub 
        else:
            return constr_matrix[:, 1:], constr_lb, constr_ub


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
    covariate: List[str] = field(init=False)
    var_link_fun: List[Callable] = field(init=False)

    fe_init: List[float] = field(init=False)
    re_init: List[float] = field(init=False)

    fe_prior: List[Prior] = field(init=False)
    re_prior: List[Prior] = field(init=False)

    re_zero_sum_std: List[float] = field(init=False)

    def __post_init__(self):
        assert isinstance(self.variables, list)
        assert len(self.variables) > 0
        assert isinstance(self.variables[0], Variable)
        self.num_fe = 0
        for var in self.variables:
            self.num_fe += var.num_fe
        for k, v in consolidate(Variable, self.variables, exclude=['num_fe']).items():
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

    fe_prior: List[List[Prior]] = field(init=False)
    re_prior: List[List[Prior]] = field(init=False)

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
        for param in param_set.parameters:
            for var in param.variables:
                var.re_prior.lower_bound = 0.
                var.re_prior.upper_bound = 0.
                var.__post_init__()
            param.__post_init__()
        param_set.__post_init__()
        return param_set


def consolidate(cls, instance_list,
                exclude: Optional[List[str]] = None) -> Dict[str, List]:
    """A function that given a list of objects of the same type, 
    collect their values corresponding to the same attribute and put into a list.

    Parameters
    ----------
    cls : dataclass
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
    assert is_dataclass(cls)
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
