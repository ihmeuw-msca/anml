import pytest
import numpy as np 

from placeholder.solvers.interface import Solver
from placeholder.solvers.base import ScipyOpt
from models import Rosenbrock

@pytest.fixture
def rb():
    return Rosenbrock()


def test_scipyopt(rb):
    solver = ScipyOpt(rb)
    solver.fit(data=None, options=dict(method='TNC', maxiter=50))
    assert np.abs(solver.fun_val_opt) < 1e-5

