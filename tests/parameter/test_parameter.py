import pytest
import pandas as pd
import numpy as np
import scipy

from anml.parameter.parameter import Variable, Parameter, Intercept, Spline, SplineLinearConstr
from anml.parameter.parameter import VariableError, ParameterSetError
from anml.parameter.parameter import ParameterSet, ParameterFunction
from anml.parameter.prior import Prior, GaussianPrior


def simple_df():
    return pd.DataFrame({
        'cov1': np.arange(100),
        'cov2': np.random.randn(100) * 2, 
        'group': np.random.choice(5, size=100),
    })


@pytest.fixture(scope='module')
def variable():
    return Variable(
        covariate='cov1',
        var_link_fun=lambda x: x,
        fe_prior=GaussianPrior(lower_bound=[-2.0], upper_bound=[3.0]),
        add_re=True,
        col_group='group',
        re_var_prior=GaussianPrior(lower_bound=[-1.0], upper_bound=[1.0], mean=[1.0], std=[2.0])
    )

@pytest.fixture(scope='module')
def spline_variable():
    constr_mono = SplineLinearConstr(order=1, y_bounds=[0.0, np.inf],x_domain=[0.0, 2.0], grid_size=5)
    constr_cvx = SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], grid_size=10)    
    return Spline(
        covariate='cov2',
        knots_type='domain',
        knots_num=2,
        degree=3,
        l_linear=False,
        r_linear=False,
        derivative_constr=[constr_mono, constr_cvx],
    )


class TestBaseVariable:
    
    def test_variable(self, variable):
        assert isinstance(variable.fe_prior, Prior)
        assert variable.num_fe == 1
        assert variable.num_re_var == 1

    def test_constraints(self, variable):
        _, lb, ub = variable.constraint_matrix()
        assert lb[0] == -2.0
        assert ub[0] == 3.0

    def test_variable_design_matrix(self, variable):
        df = simple_df()
        assert np.array_equal(
            variable.design_matrix(df),
            np.arange(100).reshape((-1, 1)),
        )

    def test_intercept(self):
        df = simple_df()
        with pytest.raises(TypeError):
            i = Intercept(covariate='foo')
        i = Intercept()
        assert i.covariate == 'intercept'
        assert i.num_fe == 1
        assert np.array_equal(
            i.design_matrix(df),
            np.ones((100, 1)),
        )
        _, lb, ub = i.constraint_matrix()
        assert lb[0] == -np.inf
        assert ub[0] == np.inf


class TestSplineVariable:

    def test_spline_variable(self, spline_variable):
        assert spline_variable.num_fe == 3
        assert spline_variable.add_re == False 
        assert len(spline_variable.fe_prior) == 3
        assert len(spline_variable.fe_init) == 3
    
    def test_spline_variable_design_matrix(self, spline_variable):
        df = simple_df()
        dmat = spline_variable.design_matrix(df)
        assert dmat.shape == (100, 3)

    def test_spline_variable_constraints(self, spline_variable):
        df = simple_df()
        spline_variable.create_spline(df)
        matrix, lb, ub = spline_variable.constraint_matrix()
        assert matrix.shape[0] == 15 and matrix.shape[1] == 3
        assert len(lb) == len(ub) == matrix.shape[0]

        assert all(output is None for output in spline_variable.constraint_matrix_re_var())


class TestParameters:

    @pytest.fixture
    def param1(self, variable):
        return Parameter(
            param_name='alpha',
            variables=[variable],
            link_fun=lambda x: x,
        )

    @pytest.fixture
    def param2(self, spline_variable):
        return Parameter(
            param_name='beta',
            variables=[spline_variable],
            link_fun=lambda x: x,
        )

    @pytest.fixture
    def parameter_function(self):
        return ParameterFunction(
            param_function_name='alpha-squared',
            param_function=lambda params: params[0] * params[1],
            param_function_fe_prior=GaussianPrior()
        )

    @pytest.fixture
    def parameter_set(self, param1, param2, parameter_function):
        return ParameterSet(
            parameters=[param1, param2],
            parameter_functions=[parameter_function],
        )

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'observations': np.random.randn(5),
            'obs_std_err': np.random.randn(5),
            'group': ['1', '2', '3', '3', '1'],
            'obs_se': np.random.randn(5),
            'cov1': np.arange(1, 6),
            'cov2': np.random.randn(5)
        })
    
    def test_parameter(self, variable, spline_variable):
        parameter = Parameter(
            param_name='alpha',
            variables=[variable, spline_variable],
            link_fun=lambda x: x,
        )
        assert parameter.num_fe == 4
        assert parameter.num_re_var == 1
        assert callable(parameter.link_fun)

    def test_parameter_set_duplicates(self, param1):
        with pytest.raises(ParameterSetError):
            ParameterSet(
                parameters=[param1, param1]
            )

    def test_parameter_function(self, parameter_function):
        assert parameter_function.param_function_name == 'alpha-squared'
        assert parameter_function.param_function([2, 2]) == 4
        assert isinstance(parameter_function.param_function_fe_prior, GaussianPrior)

    def test_parameter_set(self, parameter_set):
        assert parameter_set.num_fe == 4
        assert parameter_set.parameter_functions[0].param_function([2, 3]) == 6

    def test_parameter_set_index(self, parameter_set):
        assert parameter_set.get_param_index('alpha') == 0
        assert parameter_set.get_param_index('beta') == 1

        with pytest.raises(ParameterSetError):
            parameter_set.get_param_index('gamma')

    def test_parameter_process(self, parameter_set, df):
        parameter_set.process(df)

        assert parameter_set.design_matrix.shape == (5, 4)
        assert parameter_set.re_matrix.shape == (5, 3)
        assert parameter_set.re_matrix[0, 0] == 1
        assert parameter_set.re_matrix[1, 1] == 2
        assert parameter_set.re_matrix[2, 2] == 3
        assert parameter_set.re_matrix[3, 2] == 4
        assert parameter_set.re_matrix[4, 0] == 5

        assert parameter_set.constr_matrix_full.shape == (17, 5)
        assert parameter_set.constr_lower_bounds_full[0] == -2.0
        assert parameter_set.constr_lower_bounds_full[-1] == -1.0
        assert parameter_set.constr_upper_bounds_full[0] == 3.0
        assert parameter_set.constr_upper_bounds_full[-1] == 1.0

        x = np.random.rand(5)
        parameter_set.prior_fun(x) == -scipy.stats.norm().logpdf(x[0]) - scipy.stats.norm(loc=1.0, scale=2.0).logpdf(x[-1])


