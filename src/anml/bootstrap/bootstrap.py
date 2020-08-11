"""
===============================
Bootstrap Module
===============================

This module will allow you to specify a solver
and then do a non-parametric bootstrap for that solver (i.e. data bootstrap).
"""

from copy import copy


class Bootstrap:
    def __init__(self, solver, model):
        """
        Bootstrap module for a model and a solver.

        Parameters
        ----------
        solver
            The solver to use for each bootstrap replicate
        model
            The model to use for each bootstrap replicate
        """

        self.solver = solver
        self.model = model

        self.parameters = None

    def _process(self, **kwargs):
        """
        Some process that samples from the data
        frame and then processes the data and fits
        the model on the processed data.

        This is purposefully vague because the sampling
        process for a non-parametric bootstrap may vary
        by application, e.g. random sampling v. stratified
        sampling, etc.

        To be overwritten in a subclass.

        Parameters
        ----------
        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError()

    def _boot(self, **kwargs):
        self._process(**kwargs)
        return copy(self.solver.x_opt)

    def run_bootstraps(self, n_bootstraps: int, verbose: bool = True, **kwargs):
        parameters = list()
        for i in range(n_bootstraps):
            if verbose:
                print(f"On bootstrap {i}", end="\r")
            parameters.append(self._boot(**kwargs))
        self.parameters = parameters
