import numpy as np
import pandas as pd 
import pytest

from anml.parameter.spline_variable import Spline, SplineLinearConstr
from anml.parameter.prior import GaussianPrior, Prior


@pytest.fixture(scope='module')
def df():
    return pd.DataFrame({
        'cov1': np.arange(100),
        'cov2': np.random.randn(100) * 2, 
        'group': np.random.choice(5, size=100),
    })


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


class TestSplineVariable:

    def test_spline_variable(self, spline_variable):
        assert spline_variable.num_fe == 3
        assert spline_variable.add_re == False 

    def test_set_fe_prior(self, spline_variable):
        with pytest.raises(ValueError):
            spline_variable.set_fe_prior(Prior())
    
    def test_spline_variable_design_matrix(self, df, spline_variable):
        spline_variable.build_design_matrix_fe(df)
        dmat = spline_variable.design_matrix_fe
        assert dmat.shape == (100, 3)

    def test_spline_variable_bounds(self, spline_variable):
        spline_variable.build_bounds_fe()
        np.testing.assert_allclose(spline_variable.lb_fe, [-10.] * 3)
        np.testing.assert_allclose(spline_variable.ub_fe, [10.] * 3)

    def test_spline_variable_constraints(self, df, spline_variable):
        spline_variable.build_constraint_matrix_fe()
        assert spline_variable.constr_matrix_fe.shape[0] == 15 and spline_variable.constr_matrix_fe.shape[1] == 3
        assert len(spline_variable.constr_lb_fe) == len(spline_variable.constr_ub_fe) == spline_variable.constr_matrix_fe.shape[0]