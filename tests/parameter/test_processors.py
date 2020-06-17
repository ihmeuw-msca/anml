import pandas as pd 
import numpy as np
import pytest
import scipy

from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.prior import GaussianPrior, Prior
from anml.parameter.processors import process_for_marginal
from anml.parameter.spline_variable import SplineLinearConstr, Spline
from anml.parameter.variables import Variable


@pytest.fixture(scope='module')
def df():
    return pd.DataFrame({
        'cov1': np.arange(1, 6),
        'cov2': np.random.randn(5) * 2, 
        'group': ['1', '2', '2', '1', '2'],
    })


@pytest.fixture(scope='module')
def variable():
    return Variable(
        covariate='cov1',
        var_link_fun=lambda x: x,
        fe_prior=GaussianPrior(lower_bound=[-2.0], upper_bound=[3.0]),
        add_re=True,
        col_group='group',
        re_var_prior=GaussianPrior(lower_bound=[-1.0], upper_bound=[1.0], mean=[1.0], std=[2.0]),
        re_prior=Prior(lower_bound=[-10.0], upper_bound=[15.0]),
    )


@pytest.fixture(scope='module')
def spline_variable():
    constr_mono = SplineLinearConstr(order=1, y_bounds=[0.0, np.inf],x_domain=[0.0, 2.0], grid_size=5)
    constr_cvx = SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], grid_size=10)    
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

@pytest.fixture(scope='module')
def param_set(variable, spline_variable):
    return ParameterSet([Parameter(param_name='foo', variables=[variable, spline_variable])])


def test_process_for_marginal(param_set, df):
    process_for_marginal(param_set, df)
    assert param_set.num_fe == 4
    assert param_set.num_re_var == 1

    assert param_set.design_matrix.shape == (5, 4)
    assert param_set.design_matrix_re.shape == (5, 2)
    assert param_set.design_matrix_re[0, 0] == 1
    assert param_set.design_matrix_re[1, 1] == 2
    assert param_set.design_matrix_re[2, 1] == 3
    assert param_set.design_matrix_re[3, 0] == 4
    assert param_set.design_matrix_re[4, 1] == 5

    assert param_set.constr_matrix_full.shape == (20, 5)
    assert np.array_equal(param_set.constr_lower_bounds_full[:4], [-2.0] + [-10.] * 3)
    assert param_set.constr_lower_bounds_full[-1] == -1.0
    assert np.array_equal(param_set.constr_upper_bounds_full[:4], [3.0] + [10.] * 3)
    assert param_set.constr_upper_bounds_full[-1] == 1.0

    x = np.random.rand(5)
    param_set.prior_fun(x) == (
        -scipy.stats.norm().logpdf(x[0]) - scipy.stats.norm(loc=1.0, scale=2.0).logpdf(x[-1])
        - scipy.stats.multivariate_normal(mean=[0.0, 1.0, -1.0], cov=np.diag([1.0, 2.0, 3.0])).logpdf(x[1:-1])
    )