import pytest
import numpy as np 

from anml.solvers.interface import ModelNotDefinedError
from anml.solvers.base import ScipyOpt
from models import Rosenbrock


@pytest.fixture
def rb():
    return Rosenbrock()


def test_scipyopt(rb):
    solver = ScipyOpt()
    with pytest.raises(ModelNotDefinedError):
        solver.assert_model_defined()
    solver.model = rb
    solver.fit(x_init=[-1.0, -1.0], options=dict(solver_options=dict(maxiter=50)))
    assert np.abs(solver.fun_val_opt) < 1e-5
