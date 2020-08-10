"""
=========
Variables
=========

Variables are the most granular object for constructing a model specification.
At the simplest level, a variable is just :class:`~anml.parameter.variables.Intercept`,
which is a column of ones (indicating that it does not change based on the data row, except through
an optional random effect).

Each Variable has a collection of methods (e.g., :func:`~anml.parameter.variable.build_design_matrix_fe`)
that gets the design matrices, constraint matrices and bounds for that single covariate. 
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from anml.exceptions import ANMLError
from anml.parameter.prior import Prior
from anml.parameter.utils import encode_groups, build_re_matrix

PROTECTED_NAMES = ['intercept']


class VariableError(ANMLError):
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
        for fixed effects coefficient 
    add_re: bool, optional
        whether to add random effects to this variable
    col_group: str, optional
        name for group column
    re_var_prior: Prior, optional
        a prior of class :class:`~anml.parameter.prior.Prior` 
        for random effect variance
    re_prior: Prior, optional
        a prior of class :class:`~anml.parameter.prior.Prior` 
        for random effects.
    
    """
    covariate: str = None
    var_link_fun: Callable = lambda x: x

    fe_prior: Prior = Prior()
    
    add_re: bool = False
    col_group: str = None
    re_var_prior: Prior = Prior()
    re_prior: Prior = Prior(lower_bound=[0.0])

    design_matrix_fe: Optional[np.ndarray] = field(init=False)
    design_matrix_re: Optional[np.ndarray] = field(init=False)

    def __post_init__(self):
        if self.covariate is not None and self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")

        self.design_matrix_fe = None
        self.design_matrix_re = None

        if self.add_re and self.col_group is None:
            raise ValueError('When add_re is True, a group column must be provided.')

        self.num_fe = self._count_num_fe()
        if self.add_re:
            self.num_re_var = self.num_fe
        else:
            self.num_re_var = 0

        if self.fe_prior and self.fe_prior.x_dim != self.num_fe:
            raise ValueError(f'Dimension of fe_prior = {self.fe_prior.x_dim} should match num_fe = {self.num_fe}.')
        if self.add_re and self.re_var_prior and self.re_var_prior.x_dim != self.num_re_var:
            raise ValueError(f'Dimension of re_var_prior = {self.re_var_prior.x_dim} should match num_re_var = {self.num_re_var}.')

        self.reset()

    def reset(self):
        # erase everything related to input df
        # (i.e. not intrinsic to variable)
        self.group_lookup = None 
        self.n_groups = None
        self.num_re = 0

    def _check_protected_names(self):
        if self.covariate in PROTECTED_NAMES:
            raise VariableError("Choose a different covariate name that is"
                                f"not in {PROTECTED_NAMES}.")

    def _count_num_fe(self):
        return 1

    def _validate_df(self, df: pd.DataFrame):
        if self.covariate is None:
            raise VariableError("No covariate has been set.")
        if self.covariate not in df.columns:
            raise VariableError(f"Covariate {self.covariate} is missing from the data frame.")
        if self.add_re and self.col_group not in df:
            raise VariableError(f"Group {self.col_group} is missing from the data frame.")

    def encode_groups(self, df: pd.DataFrame):
        """Convert a categorical column into ordinal numbers.

        Parameters
        ----------
        df : pd.DataFrame
            input dataframe

        Returns
        -------
        List[int]
            a list of ints indicating category of each datapoint.

        Raises
        ------
        ValueError
            Only one group in the entire input dataframe.
        """
        group_assign_cat = df[self.col_group].to_numpy()
        self.group_lookup = encode_groups(group_assign_cat)
        self.n_groups = len(self.group_lookup)
        if self.n_groups < 2:
            raise ValueError(f'Only one group in {self.col_group}.')
        self.num_re = self.n_groups * self.num_fe 
        return [self.group_lookup[g] for g in group_assign_cat]

    def _design_matrix(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
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

    def build_design_matrix_fe(self, df: pd.DataFrame, **kwargs):
        """Build design matrix corresponding to fixed effects.

        Parameters
        ----------
        df : pd.DataFrame
            input dataframe
        """
        self._validate_df(df)
        self.design_matrix_fe = self._design_matrix(df, **kwargs)

    def build_design_matrix_re(self, df: pd.DataFrame):
        """Build design matrix corresponding to random effects covariances.

        Parameters
        ----------
        df : pd.DataFrame
            input dataframe
        """
        assert self.add_re, 'No random effects for this variable.'
        if self.design_matrix_fe is None:
            self.build_design_matrix_fe(df)
        group_assign = self.encode_groups(df)
        self.design_matrix_re = build_re_matrix(self.design_matrix_fe, group_assign, self.n_groups)

    def build_bounds_fe(self):
        """Build bounds for fixed effects
        """
        self.lb_fe = self.fe_prior.lower_bound
        self.ub_fe = self.fe_prior.upper_bound

    def build_constraint_matrix_fe(self):
        """Build constraint matrix for fixed effects
        """
        # if using None or [], need to have extra control flow or dimension matching when combining variables
        self.constr_matrix_fe = np.zeros((1, self.num_fe)) 
        self.constr_lb_fe = [0.0]
        self.constr_ub_fe = [0.0]

    def build_bounds_re_var(self):
        """Build bounds for random effects covariance.
        """
        assert self.add_re, 'No random effects for this variable'
        self.lb_re_var = np.maximum(0.0, self.re_var_prior.lower_bound)
        self.ub_re_var = self.re_var_prior.upper_bound

    def build_constraint_matrix_re_var(self):
        """Build constraint matrix for random effects covariance.
        """
        assert self.add_re, 'No random effects for this variable'
        self.constr_matrix_re_var = np.zeros((1, self.num_re_var))
        self.constr_lb_re_var = [0.0]
        self.constr_ub_re_var = [0.0]

    def build_bounds_re(self):
        """Build bounds for random effects.
        """
        assert self.add_re and self.num_re > 0, 'No random effects for this variable or grouping is not defined yet.'
        self.lb_re = self.re_prior.lower_bound * self.num_re
        self.ub_re = self.re_prior.upper_bound * self.num_re

    def build_constraint_matrix_re(self):
        """Build constraint matrix for random effects
        """
        assert self.add_re and self.num_re > 0, 'No random effects for this variable or grouping is not defined yet.'
        self.constr_matrix_re = np.zeros((1, self.num_re))
        self.constr_lb_re = [0.0]
        self.constr_ub_re = [0.0]


@dataclass
class Intercept(Variable):
    """An intercept variable.
    """
    covariate: str = field(init=False)
    
    def __post_init__(self):
        Variable.__post_init__(self)
        self.covariate = 'intercept'

    def _validate_df(self, df: pd.DataFrame):
        pass

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones((df.shape[0], 1))


@dataclass
class ParameterBlock:

    num_fe: int = field(init=False, default=0)
    num_re_var: int = field(init=False, default=0)
    _num_re: int = field(init=False, default=0)

    variables: List[Variable] = field(init=False)

    # Design Matrices
    design_matrix_fe: Optional[np.ndarray] = field(init=False, default=None)
    design_matrix_re: Optional[np.ndarray] = field(init=False, default=None)

    # Constraint Matrices
    constr_matrix_fe: Optional[np.ndarray] = field(init=False, default=None)
    constr_matrix_re_var: Optional[np.ndarray] = field(init=False, default=None)
    constr_matrix_re: Optional[np.ndarray] = field(init=False, default=None)

    # Lower Bounds
    constr_lb_fe: Optional[np.ndarray] = field(init=False, default=None)
    constr_lb_re_var: Optional[np.ndarray] = field(init=False, default=None)
    constr_lb_re: Optional[np.ndarray] = field(init=False, default=None)

    # Upper Bounds
    constr_ub_fe: Optional[np.ndarray] = field(init=False, default=None)
    constr_ub_re_var: Optional[np.ndarray] = field(init=False, default=None)
    constr_ub_re: Optional[np.ndarray] = field(init=False, default=None)

    # Random Effects Additional Specs
    re_priors: Optional[np.ndarray] = field(init=False, default=None)
    re_var_padding: Optional[np.ndarray] = field(init=False, default=None)

    def reset(self):
        self.design_matrix_fe = None
        self.design_matrix_re = None
        self.constr_matrix_fe = None
        self.constr_matrix_re_var = None
        self.constr_matrix_re = None
        self.constr_lb_fe = None
        self.constr_lb_re_var = None
        self.constr_lb_re = None
        self.constr_ub_fe = None
        self.constr_ub_re_var = None
        self.constr_ub_re = None
        self.re_priors = None
        self.re_var_padding = None

    @property
    def num_re(self):
        raise NotImplementedError()


def collect_blocks(
    param_block: ParameterBlock,
    attr_name: str,
    build_func: Optional[str] = None,
    should_include: Optional[Callable] = lambda x: True,
    reset_params: Optional[bool] = False,
    inputs: Optional[pd.DataFrame] = None,
):
    if reset_params:
        param_block.reset()

    blocks = []

    for variable in param_block.variables:
        if should_include(variable):
            if build_func is not None:
                func = getattr(variable, build_func)
                if inputs is not None:
                    func(inputs)
                else:
                    func()
            blocks.append(getattr(variable, attr_name))

    return blocks
