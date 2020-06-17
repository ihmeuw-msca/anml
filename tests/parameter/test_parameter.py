import pytest
import pandas as pd
import numpy as np
import scipy

from anml.parameter.variables import Intercept
from anml.parameter.parameter import Parameter, ParameterSetError, ParameterSet, ParameterFunction
from anml.parameter.prior import Prior, GaussianPrior


@pytest.fixture(scope='module')
def df():
    return pd.DataFrame({
        'cov1': np.arange(5),
        'cov2': np.random.randn(5) * 2, 
        'group': np.random.choice(3, size=5),
    })


@pytest.fixture(scope='module')
def param1():
    return Parameter(
        param_name='alpha',
        variables=[Intercept(), Intercept(add_re=True, col_group='group')],
        link_fun=lambda x: x,
    )


@pytest.fixture(scope='module')
def param2():
    return Parameter(
        param_name='beta',
        variables=[Intercept()],
        link_fun=lambda x: x,
    )


@pytest.fixture(scope='module')
def parameter_function():
    return ParameterFunction(
        param_function_name='alpha-squared',
        param_function=lambda params: params[0] * params[1],
        param_function_fe_prior=GaussianPrior()
    )

@pytest.fixture(scope='module')
def parameter_set(param1, param2, parameter_function):
    return ParameterSet(
        parameters=[param1, param2],
        parameter_functions=[parameter_function],
    )


class TestParameters:
    
    def test_parameter(self, param1):
        assert param1.num_fe == 2
        assert param1.num_re_var == 1
        assert callable(param1.link_fun)

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
        assert parameter_set.num_fe == 3
        assert parameter_set.parameter_functions[0].param_function([2, 3]) == 6

    def test_parameter_set_index(self, parameter_set):
        assert parameter_set.get_param_index('alpha') == 0
        assert parameter_set.get_param_index('beta') == 1

        with pytest.raises(ParameterSetError):
            parameter_set.get_param_index('gamma')

    # def test_parameter_process(self, parameter_set, df):
    #     parameter_set.process(df)

    #     assert parameter_set.design_matrix.shape == (5, 4)
    #     assert parameter_set.re_matrix.shape == (5, 3)
    #     assert parameter_set.re_matrix[0, 0] == 1
    #     assert parameter_set.re_matrix[1, 1] == 2
    #     assert parameter_set.re_matrix[2, 2] == 3
    #     assert parameter_set.re_matrix[3, 2] == 4
    #     assert parameter_set.re_matrix[4, 0] == 5

    #     assert parameter_set.constr_matrix_full.shape == (17, 5)
    #     assert parameter_set.constr_lower_bounds_full[0] == -2.0
    #     assert parameter_set.constr_lower_bounds_full[-1] == -1.0
    #     assert parameter_set.constr_upper_bounds_full[0] == 3.0
    #     assert parameter_set.constr_upper_bounds_full[-1] == 1.0

    #     x = np.random.rand(5)
    #     parameter_set.prior_fun(x) == (
    #         -scipy.stats.norm().logpdf(x[0]) - scipy.stats.norm(loc=1.0, scale=2.0).logpdf(x[-1])
    #         - scipy.stats.multivariate_normal(mean=[0.0, 1.0, -1.0], cov=np.diag([1.0, 2.0, 3.0])).logpdf(x[1:-1])
    #     )


