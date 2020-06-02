import numpy as np
import pytest

from placeholder.parameter.parameter import Variable, Parameter, ParameterSet, ParameterFunction
from placeholder.parameter.prior import Prior, GaussianPrior


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
