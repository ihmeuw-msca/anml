import pytest
import numpy as np 

from placeholder.solvers.interface import ModelNotDefinedError
from placeholder.solvers.base import ScipyOpt
from models import Rosenbrock


@pytest.fixture
def rb():
    return Rosenbrock()


def test_scipyopt(rb):
    solver = ScipyOpt()
    with pytest.raises(ModelNotDefinedError):
        solver.assert_model_defined()
    solver.model = rb
    solver.fit(data=None, options=dict(method='TNC', maxiter=50))
    assert np.abs(solver.fun_val_opt) < 1e-5
