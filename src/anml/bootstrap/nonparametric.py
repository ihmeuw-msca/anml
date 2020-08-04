"""
===============================
Non-Parametric Bootstrap Module
===============================

This module will allow you to specify a solver
and then do a non-parametric bootstrap for that solver (i.e. data bootstrap).
"""

import numpy as np


class NPBootstrap:
    def __init__(self, solver, model):

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
        return self.solver.x_opt

    def run_bootstraps(self, n_bootstraps: int, verbose: bool = True, **kwargs):
        parameters = list()
        for i in range(n_bootstraps):
            if verbose:
                print(f"On bootstrap {i}", "\r")
            parameters.append(self._boot(**kwargs))
        self.parameters = np.vstack(parameters)
