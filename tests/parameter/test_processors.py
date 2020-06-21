import pandas as pd 
import numpy as np
import pytest
import scipy

from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.prior import GaussianPrior, Prior
from anml.parameter.processors import process_all
from anml.parameter.spline_variable import SplineLinearConstr, Spline
from anml.parameter.utils import collect_priors
from anml.parameter.variables import Variable


@pytest.fixture
def df():
    np.random.seed(42)
    return pd.DataFrame({
        'cov1': np.arange(1, 6),
        'cov2': np.random.randn(5) * 2, 
        'group1': ['1', '2', '2', '1', '2'],
        'group2': ['3', '4', '5', '5', '3'],
    })


@pytest.fixture
def variable1():
    return Variable(
        covariate='cov1',
        var_link_fun=lambda x: x,
        fe_prior=GaussianPrior(lower_bound=[-2.0], upper_bound=[3.0]),
        add_re=True,
        col_group='group1',
        re_var_prior=GaussianPrior(lower_bound=[-1.0], upper_bound=[1.0], mean=[1.0], std=[2.0]),
        re_prior=GaussianPrior(lower_bound=[-0.5], upper_bound=[0.5], mean=[0.0], std=[0.5]),
    )

@pytest.fixture
def variable2():
    return Variable(
        covariate='cov1',
        var_link_fun=lambda x: x,
        fe_prior=GaussianPrior(lower_bound=[-2.0], upper_bound=[3.0]),
        add_re=True,
        col_group='group2',
        re_var_prior=GaussianPrior(lower_bound=[-1.0], upper_bound=[1.0], mean=[1.0], std=[2.0]),
        re_prior=GaussianPrior(lower_bound=[-0.5], upper_bound=[0.5], mean=[0.0], std=[0.5]),
    )


@pytest.fixture
def spline_variable():
    constr_mono = SplineLinearConstr(order=1, y_bounds=[0.0, np.inf],x_domain=[-2.0, 2.0], grid_size=5)
    constr_cvx = SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], x_domain=[1.0, 3.0], grid_size=10)    
    spline = Spline(
        covariate='cov2',
        knots_type='domain',
        knots_num=2,
        degree=3,
        l_linear=False,
        r_linear=False,
        derivative_constr=[constr_mono, constr_cvx],
    )
    spline.set_fe_prior(GaussianPrior(mean=[0.0, 1.0, -1.0], std=[1.0, 2.0, 3.0], upper_bound=[10.] * 3, lower_bound=[-10.] * 3))
    return spline

@pytest.fixture
def param_set(variable1, variable2, spline_variable):
    return ParameterSet([Parameter(param_name='foo', variables=[variable1, variable2, spline_variable])])


def test_process_params(param_set, df):
    process_all(param_set, df)
    assert param_set.num_fe == 5
    assert param_set.num_re_var == 2
    assert param_set.num_re == 5

    # ---- check matrices -------
    assert param_set.design_matrix_fe.shape == (5, 5)
    assert param_set.design_matrix_re.shape == (5, 5)
    # check Z for variable1
    assert param_set.design_matrix_re[0, 0] == 1
    assert param_set.design_matrix_re[1, 1] == 2
    assert param_set.design_matrix_re[2, 1] == 3
    assert param_set.design_matrix_re[3, 0] == 4
    assert param_set.design_matrix_re[4, 1] == 5
    # check Z for variable2
    assert param_set.design_matrix_re[0, 2] == 1
    assert param_set.design_matrix_re[1, 3] == 2
    assert param_set.design_matrix_re[2, 4] == 3
    assert param_set.design_matrix_re[3, 4] == 4
    assert param_set.design_matrix_re[4, 2] == 5
    # check constraint matrices
    assert param_set.constr_matrix_fe.shape == (15, 5)
    assert param_set.constr_matrix_re_var.shape == (1, 2)
    assert param_set.constr_matrix_re.shape == (1, 5)

    # ----- check bounds --------
    assert len(param_set.lb_fe) == len(param_set.ub_fe) == param_set.num_fe
    np.testing.assert_allclose(param_set.lb_fe, [-2.0] * 2 + [-10.] * 3)
    np.testing.assert_allclose(param_set.ub_fe, [3.0] * 2 + [10.] * 3)
    assert len(param_set.lb_re_var) == len(param_set.ub_re_var) == param_set.num_re_var
    np.testing.assert_allclose(param_set.lb_re_var, [0.0] * param_set.num_re_var)
    np.testing.assert_allclose(param_set.ub_re_var, [1.0] * param_set.num_re_var)
    assert len(param_set.lb_re) == len(param_set.ub_re) == param_set.num_re
    np.testing.assert_allclose(param_set.lb_re, [-0.5] * param_set.num_re)
    np.testing.assert_allclose(param_set.ub_re, [0.5] * param_set.num_re)

    # ----- check others --------    
    np.testing.assert_allclose(param_set.re_var_padding, np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])) 
    
    x = np.random.rand(5)
    fe_prior_fun = collect_priors(param_set.fe_priors)
    fe_prior_val = (
        -scipy.stats.norm().logpdf(x[0]) -scipy.stats.norm().logpdf(x[1]) 
        - scipy.stats.multivariate_normal(mean=[0.0, 1.0, -1.0], cov=np.diag([1.0, 4.0, 9.0])).logpdf(x[2:param_set.num_fe])
    )
    assert np.abs(fe_prior_fun(x[:param_set.num_fe]) - fe_prior_val) / np.abs(fe_prior_val) < 1e-2
    
    re_var_prior_fun = collect_priors(param_set.re_var_priors)
    re_var_prior_val = -scipy.stats.norm(loc=1.0, scale=2.0).logpdf(x[0]) - scipy.stats.norm(loc=1.0, scale=2.0).logpdf(x[1])
    assert np.abs(re_var_prior_fun(x[:param_set.num_re_var]) - re_var_prior_val) / np.abs(re_var_prior_val) < 1e-2

    re_prior_fun = collect_priors(param_set.re_priors)
    re_prior_val = (
        -scipy.stats.norm(loc=0.0, scale=0.5).logpdf(x[0]) - scipy.stats.norm(loc=0.0, scale=0.5).logpdf(x[1])
        -scipy.stats.norm(loc=0.0, scale=0.5).logpdf(x[2]) - scipy.stats.norm(loc=0.0, scale=0.5).logpdf(x[3])
        -scipy.stats.norm(loc=0.0, scale=0.5).logpdf(x[4])
    )
    assert np.abs(re_prior_fun(x[:param_set.num_re]) - re_prior_val) / np.abs(re_prior_val) < 1e-2


