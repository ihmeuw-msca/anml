
class ModelsNotDefinedError(Exception):
    pass


class SolversNotDefinedError(Exception):
    pass


class Solver:

    def __init__(self, model_instances=None):
        if model_instances is not None:
            self._models = model_instances
        else:
            self._models = []
        self.success = None
        self.x_opt = None
        self.fun_val_opt = None
        self.options = None

    def set_model_instances(self, model_instances):
        self._models = model_instances

    def detach_model_instances(self):
        self._models = []

    def assert_models_defined(self):
        if len(self._models) > 0:
            return True
        else:
            raise ModelsNotDefinedError()

    def set_options(self, options):
        self.options = options

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()

    def predict(self, **kwargs):
        raise NotImplementedError()


class CompositeSolver:

    def __init__(self, solvers=None):
        super().__init__(model_instances=None)
        if solvers is not None:
            self._solvers = solvers
        else:
            self._solvers = []

    def add_solvers(self, solvers):
        self._solvers.extend(solvers)

    def set_solver(self, solvers):
        self._solvers = solvers

    def set_options(self, options: dict):
        if self.assert_solvers_defined():
            for solver in self._solvers:
                solver.set_options(options)

    def set_model_instances(self, model_instances):
        if self.assert_solvers_defined() is True:
            for solver in self._solvers:
                solver.set_model_instances(model_instances)

    def detach_model_instances(self):
        if self.assert_solvers_defined() is True:
            for solver in self._solvers:
                solver.detach_model_instances()

    def assert_models_defined(self):
        if self.assert_solvers_defined() is True:
            for solver in self._solvers:
                solver.assert_models_defined()

    def assert_solvers_defined(self):
        if len(self._solvers) > 0:
            return True
        else:
            raise SolversNotDefinedError()

    def predict(self, **kwargs):
        raise NotImplementedError()