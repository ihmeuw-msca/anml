from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Any, List
import pandas as pd

from anml.parameter.prior import Prior
from anml.exceptions import ANMLError


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

    Attributes
    ----------
    num_fe: int
        number of fixed effects coefficients (betas)
    num_re_var: int
        number of random effects variance (gammas)
    
    """
    covariate: str = None
    var_link_fun: Callable = lambda x: x

    fe_prior: Prior = Prior()
    
    add_re: bool = False
    col_group: str = None
    re_var_prior: Prior = Prior()
    re_prior: Prior = Prior()

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
        self.num_re = None
        self.design_matrix = None 
        self.re_design_matrix = None

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
        group_assign_cat = df[self.col_group].to_numpy()
        self.group_lookup = encode_groups(group_assign_cat)
        self.n_groups = len(self.group_lookup)
        if self.n_groups < 2:
            raise ValueError(f'Only one group in {self.col_group}.')
        self.num_re = self.n_groups * self.num_fe
        return [self.group_lookup[g] for g in group_assign_cat]

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

    def build_design_matrix(self, df: pd.DataFrame, build_re=True):
        self._validate_df(df)
        self.design_matrix = self._design_matrix(df)
        if self.add_re and build_re:
            group_assign = self.encode_groups(df)
            self.design_matrix_re = build_re_matrix(self.design_matrix, group_assign, self.n_groups)

    def constraint_matrix(self):
        return np.identity(self.num_fe), self.fe_prior.lower_bound, self.fe_prior.upper_bound

    def constraint_matrix_re_var(self):
        assert self.add_re, 'No random effects for this variable'
        return np.identity(self.num_re_var), self.re_var_prior.lower_bound, self.re_var_prior.upper_bound

    def constraint_matrix_re(self):
        assert self.add_re and self.num_re, 'No random effects for this variable or grouping is not defined yet.'
        return np.identity(self.num_re), self.re_prior.lower_bound * self.num_re, self.re_prior.upper_bound * self.num_re


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


def encode_groups(group_assign_cat: List[Any]):
    groups = np.unique(group_assign_cat)
    group_id_dict = {grp: i for i, grp in enumerate(groups)}
    return group_id_dict

def build_re_matrix(matrix: np.ndarray, group_assign_ord: List[int], n_groups: int):
    n_coefs = matrix.shape[1]
    re_mat = np.zeros((matrix.shape[0], n_groups * n_coefs))
    for i, row in enumerate(matrix):
        grp = group_assign_ord[i]
        re_mat[i, grp * n_coefs: (grp + 1) * n_coefs] = row 
    return re_mat

