from placeholder.solvers.base import Solver, CompositeSolver


def test_solver():
    Solver()


def test_composite_solver():
    solver = Solver()
    CompositeSolver(solvers=[solver])
