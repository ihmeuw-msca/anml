
class ModelNotDefinedError(Exception):
    pass


class SolverNotDefinedError(Exception):
    pass


class Solver:

    def __init__(self, model_instance=None):
        self._model = model_instance
        self.success = None
        self.x_opt = None
        self.fun_val_opt = None
        self.success = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_instance):
        self._model = model_instance

    def assert_models_defined(self):
        if self._model is not None:
            return True
        else:
            raise ModelNotDefinedError()

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()

    def predict(self, **kwargs):
        raise NotImplementedError()


class CompositeSolver(Solver):

    def __init__(self, solvers_list=None):
        super().__init__(model_instance=None)
        if solvers_list is not None:
            self._solvers = solvers_list
        else:
            self._solvers = []

    @property
    def solvers(self):
        return self._solvers

    @solvers.setter
    def solvers(self, solvers_list):
        self._solvers = solvers_list

    def add_solver(self, solver):
        self._solvers.extend(solver)

    @property
    def model(self):
        models = []
        if self.assert_solvers_defined() is True:
            for solver in self._solvers:
                models.append(solver.model)
        return models

    @model.setter
    def model(self, model_instances):
        if isinstance(model_instances, list):
            assert len(model_instances) == len(self._solvers)
            if self.assert_solvers_defined() is True:
                for model, solver in zip(model_instances, self._solvers):
                    solver.model = model
        else:
            if self.assert_solvers_defined() is True:
                for solver in self._solvers:
                    solver.model = model_instances

    def assert_models_defined(self):
        if self.assert_solvers_defined() is True:
            for solver in self._solvers:
                solver.assert_models_defined()

    def assert_solvers_defined(self):
        if len(self._solvers) > 0:
            return True
        else:
            raise SolverNotDefinedError()

    def predict(self, **kwargs):
        raise NotImplementedError()
