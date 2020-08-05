"""
===============
Spline Variable
===============

A subclass of :class:`anml.parameter.variables.Variable` that handles spline related computations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import pandas as pd
import numpy as np

from xspline import XSpline

from anml.parameter.variables import Variable, VariableError
from anml.parameter.prior import Prior


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
    derivative_constr: List[`~anml.parameter.spline_variable.SplineLinearConstr`]
        constraints on derivatives 
    constr_grid_size_global: int, optional
        number of points to use when building constraint matrix. used only when `grid_size` for 
        individual `~anml.parameter.spline_variable.SplineLinearConstr` is not available

    Raises
    ------
    VariableError
        unknown knot type
    VariableError
        no covariate has been set
    """
    fe_prior: Optional[Prior] = field(init=False)
    add_re: bool = field(init=False)
    knots_type: str = 'frequency'
    knots_num: int = 3
    degree: int = 3
    l_linear: bool = False
    r_linear: bool = False
    include_intercept: bool = False
    derivative_constr: List[SplineLinearConstr] = field(default_factory=lambda: [])
    constr_grid_size_global: int = None

    spline: Optional[XSpline] = field(init=False)
    x: Optional[np.ndarray] = field(init=False)

    constr_matrix_fe: Optional[np.ndarray] = field(init=False)
    constr_lb_fe: Optional[Union[List[float], np.ndarray]] = field(init=False)
    constr_ub_fe: Optional[Union[List[float], np.ndarray]] = field(init=False)

    def __post_init__(self):
        if self.knots_type not in ['frequency', 'domain']:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")
        self.spline = None
        self.add_re = False
        self.fe_prior = None
        Variable.__post_init__(self)
        if self.fe_prior is None:
            self.set_fe_prior(
                Prior(lower_bound=[-np.inf] * self._count_num_fe(),
                      upper_bound=[np.inf] * self._count_num_fe())
            )

    def _count_num_fe(self):
        return self.knots_num - self.l_linear - self.r_linear + self.degree - 1 - int(not self.include_intercept)

    def set_fe_prior(self, prior: Prior):
        if prior.x_dim != self.num_fe:
            raise ValueError(f'Dimension of fe_prior = {prior.x_dim} should match num_fe = {self.num_fe}.')
        self.fe_prior = prior

    def create_spline(self, df: pd.DataFrame):
        if self.covariate is None:
            raise VariableError("No covariate has been set.")
        self.x = df[self.covariate].values

        spline_knots = np.linspace(0, 1, self.knots_num)
        if self.knots_type == 'frequency':
            knots = np.quantile(self.x, spline_knots)
        elif self.knots_type == 'domain':
            knots = np.min(self.x) + spline_knots * (np.max(self.x) - np.min(self.x))
        else:
            raise VariableError(f"Unknown knots_type for Spline {self.knots_type}.")

        self.spline = XSpline(
            knots=knots,
            degree=self.degree,
            l_linear=self.l_linear,
            r_linear=self.r_linear
        )

    def _design_matrix(self, df: pd.DataFrame, create_spline: bool = True) -> np.ndarray:
        if create_spline:
            self.create_spline(df)
        if self.include_intercept:
            return self.spline.design_mat(self.x)
        else:
            return self.spline.design_mat(self.x)[:, 1:]

    def build_constraint_matrix_fe(self):
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
                raise ValueError(
                    f'Domain of constraint = {constr.x_domain} does not'
                    f' overlap with domain of spline. lb = {lb}, ub = {ub}.'
                )
            if constr.grid_size is None and self.constr_grid_size_global is None:
                raise ValueError('Either global or individual constraint grid size needs to be specified.')
            
            if constr.grid_size is not None:
                points = np.linspace(max(lb, constr.x_domain[0]), min(ub, constr.x_domain[1]), constr.grid_size)
            else:
                points_all = np.linspace(lb, ub, self.constr_grid_size_global)
                is_in_domain = constr.x_domain[0] <= points_all <= constr.x_domain[1]
                points = points_all[is_in_domain]
            n_points = len(points)
            if self.include_intercept:
                constr_matrices.append(self.spline.design_dmat(points, constr.order))
            else:
                constr_matrices.append(self.spline.design_dmat(points, constr.order)[:, 1:])
            constr_lbs.append([constr.y_bounds[0]] * n_points)
            constr_ubs.append([constr.y_bounds[1]] * n_points)

        if len(constr_matrices) > 0:
            self.constr_matrix_fe = np.vstack(constr_matrices)
            self.constr_lb_fe = np.hstack(constr_lbs)
            self.constr_ub_fe = np.hstack(constr_ubs)
        else:
            self.constr_matrix_fe = np.zeros((1, self.num_fe)) 
            self.constr_lb_fe = [0.0]
            self.constr_ub_fe = [0.0]