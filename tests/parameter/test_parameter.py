import pytest
import pandas as pd
import numpy as np

from anml.parameter.parameter import Variable, Parameter, Intercept, Spline, SplineLinearConstr
from anml.parameter.parameter import ParameterSet, ParameterFunction
from anml.parameter.prior import Prior, GaussianPrior


@pytest.fixture
def variable():
    return Variable(
        covariate='covariate1',
        var_link_fun=lambda x: x,
        fe_init=0.,
        re_init=0.
    )


def test_variable(variable):
    assert variable.fe_init == 0.
    assert variable.re_init == 0.

    assert isinstance(variable.fe_prior, Prior)
    assert isinstance(variable.re_prior, Prior)


def test_variable_design_mat(variable):
    assert np.array_equal(
        variable.design_mat(pd.DataFrame({'covariate1': np.arange(5)})),
        np.arange(5).reshape((5, 1))
    )


def test_intercept():
    i = Intercept()
    assert i.covariate == 'intercept'
    assert i.num_fe == 1
    assert np.array_equal(
        i.design_mat(pd.DataFrame({'foo': np.arange(5)})),
        np.ones((5, 1))
    )


def test_spline_variable():
    array = np.random.randn(100)
    df = pd.DataFrame({'foo': array})
    spline = Spline(
        covariate='foo',
        knots_type='domain',
        knots_num=2,
        degree=3,
        l_linear=False,
        r_linear=False,
        include_intercept=True,
    )
    assert spline.num_fe == 4
    dmat = spline.design_mat(df)
    assert dmat.shape == (100, 4)


def test_spline_variable_constraints():
    array = np.random.randn(100) * 2
    df = pd.DataFrame({'foo': array})
    constr_mono = SplineLinearConstr(order=1, y_bounds=[0.0, np.inf],x_domain=[0.0, 2.0], grid_size=5)
    constr_cvx = SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], grid_size=10)
    spline = Spline(
        covariate='foo',
        knots_type='domain',
        knots_num=2,
        degree=3,
        l_linear=False,
        r_linear=False,
        derivative_constr=[constr_mono, constr_cvx],
    )
    spline.create_spline(df)
    matrix, lb, ub = spline.get_constraint_matrix()
    assert matrix.shape[0] == 15 and matrix.shape[1] == 3
    assert len(lb) == len(ub) == matrix.shape[0]


def test_parameter():
    var1 = Variable(
        covariate='covariate1', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    var2 = Variable(
        covariate='covariate2', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    parameter = Parameter(
        param_name='alpha',
        variables=[var1, var2],
        link_fun=lambda x: x,
    )
    assert parameter.num_fe == 2
    assert callable(parameter.link_fun)

    assert len(parameter.fe_init) == 2
    assert len(parameter.re_init) == 2
    assert len(parameter.fe_prior) == 2
    assert len(parameter.re_prior) == 2
    assert len(parameter.fe_prior) == 2
    assert len(parameter.re_prior) == 2


@pytest.fixture
def param1():
    var1 = Variable(
        covariate='covariate1', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    parameter1 = Parameter(
        param_name='alpha',
        variables=[var1],
        link_fun=lambda x: x,
    )
    return parameter1


@pytest.fixture
def param2():
    var2 = Variable(
        covariate='covariate2', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    parameter2 = Parameter(
        param_name='beta',
        variables=[var2],
        link_fun=lambda x: x,
    )
    return parameter2


def test_parameter_set_duplicates(param1):
    with pytest.raises(RuntimeError):
        ParameterSet(
            parameters=[param1, param1]
        )


def test_delete_random_effects():
    prior = Prior(lower_bound=[1.], upper_bound=[2.])
    var = Intercept(fe_prior=prior, re_prior=prior)
    param = Parameter(param_name='param', variables=[var])
    param_set = ParameterSet([param])
    for param in param_set.parameters:
        for var in param.variables:
            assert var.re_prior.lower_bound == [1.]
            assert var.re_prior.upper_bound == [2.]
    assert param_set.re_prior[0][0].lower_bound == [1.]
    assert param_set.re_prior[0][0].upper_bound == [2.]
    new_set = param_set.delete_random_effects()
    for param in new_set.parameters:
        for var in param.variables:
            assert var.re_prior.lower_bound == [0.]
            assert var.re_prior.upper_bound == [0.]
    assert new_set.re_prior[0][0].lower_bound == [0.]
    assert new_set.re_prior[0][0].upper_bound == [0.]

 
@pytest.fixture
def parameter_function():
    return ParameterFunction(
        param_function_name='alpha-squared',
        param_function=lambda params: params[0] * params[1],
        param_function_fe_prior=GaussianPrior()
    )


def test_parameter_function(parameter_function):
    assert parameter_function.param_function_name == 'alpha-squared'
    assert parameter_function.param_function([2, 2]) == 4
    assert isinstance(parameter_function.param_function_fe_prior, GaussianPrior)


@pytest.fixture
def parameter_set(param1, param2, parameter_function):
    return ParameterSet(
        parameters=[param1, param2],
        parameter_functions=[parameter_function],
    )


def test_parameter_set(parameter_set):
    assert parameter_set.num_fe == 2
    assert callable(parameter_set.param_function[0])
    assert len(parameter_set.param_function_fe_prior) == 1
    assert isinstance(parameter_set.param_function_fe_prior[0], Prior)
    assert parameter_set.param_function[0]([2, 3]) == 6


def test_parameter_set_index(parameter_set):
    assert parameter_set.get_param_index('alpha') == 0
    assert parameter_set.get_param_index('beta') == 1

    with pytest.raises(RuntimeError):
        parameter_set.get_param_index('gamma')
