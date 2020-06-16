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

from dataclasses import fields, field, dataclass, is_dataclass
from typing import List, Callable, Optional, Dict, Union
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from scipy.linalg import block_diag

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
    fe_prior: Prior, optional
        a prior of class :class:`~anml.parameter.prior.Prior`
    add_re: bool, optional
        whether to add random effects to this variable

    Attributes
    ----------
    All parameters become attributes after validation.
    
    """
    covariate: str = None
    var_link_fun: Callable = lambda x: x

    fe_prior: Prior = Prior()
    fe_init: float = 0.0
    
    add_re: bool = False
    col_group: str = None
    re_var_prior: Prior = Prior()
    re_var_init: float = 1.0

    num_fe: int = field(init=False)
    num_re_var: int = field(init=False)

    def __post_init__(self):
        if self.covariate is not None and self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")

        if self.add_re and self.col_group is None:
            raise ValueError('When add_re is True, a group column must be provided.')

        self.num_fe = self._count_num_fe()
        if self.add_re:
            self.num_re_var = self.num_fe
        else:
            self.num_re_var = 0

    def _check_protected_names(self):
        if self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")


    def _count_num_fe(self):
        return 1

    def _validate_df(self, df):
        if self.covariate is None:
            raise VariableError("No covariate has been set.")
        if self.covariate not in df.columns:
            raise VariableError(f"Covariate {self.covariate} is missing from the data frame.")
        if self.add_re and self.col_group not in df:
            raise VariableError(f"Group {self.col_group} is missing from the data frame.")

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
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

    def design_matrix(self, df):
        self._validate_df(df)
        return self._design_matrix(df)

    def constraint_matrix(self):
        return np.array([[1.0]]), self.fe_prior.lower_bound, self.fe_prior.upper_bound

    def constraint_matrix_re_var(self):
        if self.add_re:
            return np.array([[1.0]]), self.re_var_prior.lower_bound, self.re_var_prior.upper_bound
        else:
            return None, None, None


@dataclass
class Intercept(Variable):
    """An intercept variable.
    """
    covariate: str = field(init=False)
    
    def __post_init__(self):
        Variable.__post_init__(self)
        self.covariate = 'intercept'

    def _validate_df(self, df):
        pass

    def design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones((df.shape[0], 1))


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
    fe_init: Union[float, List[float]] = 0.0
    fe_prior: Union[Prior, List[Prior]] = Prior()
    add_re: bool = field(init=False)
    knots_type: str = 'frequency'
    knots_num: int = 3
    degree: int = 3
    l_linear: bool = False
    r_linear: bool = False
    include_intercept: bool = False
    derivative_constr: List[SplineLinearConstr] = None
    constr_grid_size_global: int = None

    def __post_init__(self):
        if self.knots_type not in ['frequency', 'domain']:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")
        self.spline = None
        self.add_re = False
        Variable.__post_init__(self)
        if isinstance(self.fe_init, float):
            self.fe_init = [self.fe_init] * self.num_fe
        if isinstance(self.fe_prior, Prior):
            self.fe_prior = [self.fe_prior] * self.num_fe

    def _count_num_fe(self):
        return self.knots_num - self.l_linear - self.r_linear + self.degree - 1 - int(not self.include_intercept)

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

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        self.create_spline(df)
        if self.include_intercept:
            return self.spline.design_mat(self.x)
        else:
            return self.spline.design_mat(self.x)[:, 1:]

    def constraint_matrix(self) -> List[np.ndarray]:
        """build constrain matrix and bounds for
        `constr_lb` <= `constr_matrix` <= `constr_ub`.

        Returns
        -------
        List[np.ndarray]
            constraint matrix, lower bounds and upper bounds.
        """
        
        lb, ub = min(self.x), max(self.x)
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
            variable._validate_df()


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

    def encode_groups(self, col_group, df: pd.DataFrame):
        group_assign = df[col_group].to_numpy()
        groups = np.unique(group_assign)
        group_id_dict = {grp: i for i, grp in enumerate(groups)}
        return group_id_dict

    def process(self, df):
        design_mat_blocks = []
        constr_mat_blocks = []
        re_mat_groups = defaultdict(dict)
        re_var_constr_groups = defaultdict(list)
        constr_lbs = []
        constr_ubs = []
        constr_lbs_re_var_groups = defaultdict(list)
        constr_ubs_re_var_groups = defaultdict(list)
        self.fe_variables_names = []

        fe_priors = []
        re_var_priors_group = defaultdict(list)
            
        for parameter in self.parameters:
            for variable in parameter.variables:
                # remembering name of variable -- so that we know what each column in X corresponds to
                var_name = parameter.param_name + '_' + variable.covariate
                self.fe_variables_names.append(var_name)

                # getting design matrix corresponding to the variable
                design_mat = variable.design_matrix(df=df)
                design_mat_blocks.append(design_mat)
                
                # getting constraint matrix and bounds for betas (fixed effects, coefficients)
                mat, lb, ub = variable.constraint_matrix()
                constr_mat_blocks.append(mat)
                constr_lbs.append(lb)
                constr_ubs.append(ub)

                # append priors functions
                if isinstance(variable.fe_prior, list):
                    fe_priors.extend(variable.fe_prior)
                else:
                    fe_priors.append(variable.fe_prior)

                # if variable has random effects, collect matrix/bounds according to col_group
                if variable.add_re:
                    # adding matrix to Z
                    re_mat_groups[variable.col_group][var_name] = design_mat
                    
                    # adding constraints and bounds for gammas (random effects variance)
                    mat_re, lb_re, ub_re = variable.constraint_matrix_re_var()
                    re_var_constr_groups[variable.col_group].append(mat_re)
                    constr_lbs_re_var_groups[variable.col_group].append(lb_re)
                    constr_ubs_re_var_groups[variable.col_group].append(ub_re)

                    re_var_priors_group[variable.col_group].append(variable.re_var_prior)

        self.design_matrix = np.hstack(design_mat_blocks)
        constr_matrix = block_diag(*constr_mat_blocks)
        constr_lower_bounds = np.hstack(constr_lbs)
        constr_upper_bounds = np.hstack(constr_ubs)

        # checking dimensions match -- design matrix and constr matrix should have same # of columns == num_fe
        # -- bounds should have same dimension as # of rows of constr matrix
        assert self.design_matrix.shape[1] == constr_matrix.shape[1] == self.num_fe
        assert len(constr_lower_bounds) == len(constr_upper_bounds) == constr_matrix.shape[0]

        if len(re_mat_groups) > 0:
            re_mat_blocks = []
            re_var_constr_blocks = []
            re_var_constr_lbs = []
            re_var_constr_ubs = []
            self.re_variables_names = []
            re_var_priors = []
            for col_group, dct in re_mat_groups.items():
                # converting categoricals to ordinals
                grp_lookup = self.encode_groups(col_group, df)
                grp_assign = [grp_lookup[g] for g in df[col_group]]
                n_group = len(grp_lookup)

                # remebering re variable names
                self.re_variables_names.extend(list(dct.keys()))
                
                # stacking matrices from variables corresponding to the same col_group
                mat = np.hstack(list(dct.values()))
                n_coefs = mat.shape[1]
                # building re matrix for a particular col_group
                re_mat = np.zeros((mat.shape[0], n_coefs * n_group))
                for i, row in enumerate(mat):
                    grp = grp_assign[i]
                    re_mat[i, grp * n_coefs: (grp + 1) * n_coefs] = row 
                re_mat_blocks.append(re_mat)

                # building gamma constraint matrix and bounds
                re_var_constr_blocks.extend(re_var_constr_groups[col_group])
                re_var_constr_lbs.extend(constr_lbs_re_var_groups[col_group])
                re_var_constr_ubs.extend(constr_ubs_re_var_groups[col_group])

                re_var_priors.extend(re_var_priors_group[col_group])

            self.re_matrix = np.hstack(re_mat_blocks)
            if len(re_var_constr_blocks) > 1:
                re_var_constr_matrix = block_diag(re_var_constr_blocks)
                re_var_constr_lower_bounds = np.hstack(re_var_constr_lbs)
                re_var_constr_upper_bounds = np.hstack(re_var_constr_ubs)
            else:
                re_var_constr_matrix = re_var_constr_blocks[0]
                re_var_constr_lower_bounds = np.asarray(re_var_constr_lbs[0])
                re_var_constr_upper_bounds = np.asarray(re_var_constr_ubs[0])

            assert re_var_constr_matrix.shape[1] == self.num_re_var
            assert len(re_var_constr_lower_bounds) == re_var_constr_matrix.shape[1] == len(re_var_constr_upper_bounds)

            self.constr_matrix_full = block_diag(constr_matrix, re_var_constr_matrix)
            self.constr_lower_bounds_full = np.hstack((constr_lower_bounds, re_var_constr_lower_bounds))
            self.constr_upper_bounds_full = np.hstack((constr_upper_bounds, re_var_constr_upper_bounds))

            self.prior_fun = self.collect_priors(fe_priors + re_var_priors)

    def collect_priors(self, priors):
        def prior_fun(x):
            assert len(x) == self.num_fe + self.num_re_var
            s = 0
            val = 0.0
            for prior in priors:
                x_dim = prior.x_dim
                val += prior.error_value(x[s: s + x_dim])
                s += x_dim
            return val 
        return prior_fun

