import pytest
import numpy as np

from anml.solvers.interface import ModelNotDefinedError, SolverNotDefinedError
from anml.solvers.composite import MultipleInitializations
from anml.solvers.base import ScipyOpt
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
    solver = MultipleInitializations(sample_fun=lambda x=None: xs_init)
    with pytest.raises(SolverNotDefinedError):
        solver.assert_solvers_defined()
    solver.solvers = [ScipyOpt()]
    with pytest.raises(ModelNotDefinedError):
        solver.assert_model_defined()
    solver.model = rb
    assert isinstance(solver.solvers[0].model, Rosenbrock)
    # assert isinstance(solver.model[0], Rosenbrock)
    solver.fit(options=dict(solver_options=dict(maxiter=10)))

    for x_init in xs_init:
        assert rb.objective(x_init) >= solver.fun_val_opt
