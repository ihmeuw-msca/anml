import pytest
import numpy as np

from placeholder.solvers.composite import MultipleInitializations
from placeholder.solvers.base import ScipyOpt
from models import Rosenbrock


@pytest.fixture
def rb():
    return Rosenbrock()
    

def test_multi_init(rb):
    num_init = 3
    xs_init = np.random.uniform(
        low=[b[0] for b in rb.bounds],
        high=[b[1] for b in rb.bounds],
        size=(num_init, rb.n_dim),
    )
    sample_fun = lambda x: xs_init
    solver = MultipleInitializations(sample_fun)
    solver.solvers = [ScipyOpt()]
    solver.model = rb
    solver.fit(data=None, options=dict(method='TNC', maxiter=10))

    for x in xs_init:
        assert rb.objective(x) >= solver.fun_val_opt