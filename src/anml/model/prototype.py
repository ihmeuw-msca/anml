from __future__ import annotations

from abc import ABC, abstractmethod
from operator import attrgetter
from typing import List, Optional

import numpy as np
from anml.data.prototype import DataPrototype
from anml.parameter.main import Parameter
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize


class ModelPrototype(ABC):
    """An abstract class that contains data and parameters information. The
    class provide interface to the optimization problem that comes from
    minimizing the negative log-likelihood, including the objective, gradient
    and hessian function. And the class also provide interface for the users,
    including fit and predict. Developers can inherit this class and overwrite
    the optimization interface for their use. For an example please check
    :class:`anml.model.example.ModelExample`.

    Parameters
    ----------
    data
        Given data object.
    parameters
        Given list of instances of :class:`Parameter`.

    """

    data = property(attrgetter("_data"))
    """Given data object.

    Raises
    ------
    TypeError
        Raised when the input data is not an instance of :class:`DataPrototype`.
    
    """
    parameters = property(attrgetter("_parameters"))
    """Given list of instances of :class:`Parameter`.

    Raises
    ------
    TypeError
        Raised when the input parameters are not a list of instances of
        :class:`Parameter`.
    
    """

    def __init__(self,
                 data: DataPrototype,
                 parameters: List[Parameter],
                 df: DataFrame):
        self.data = data
        self.parameters = parameters
        self.attach(df)

        self.result = {}

    @data.setter
    def data(self, data: DataPrototype):
        if not isinstance(data, DataPrototype):
            raise TypeError("ModelPrototype input data must be an instance of "
                            "DataPrototype.")
        self._data = data

    @parameters.setter
    def parameters(self, parameters: List[Parameter]):
        parameters = list(parameters)
        if not all(isinstance(parameter, Parameter)
                   for parameter in parameters):
            raise TypeError("ModelPrototype input parameters must be a list of "
                            "instances of Parameter.")
        self._parameters = parameters
        self._sizes = [parameter.size for parameter in self._parameters]

    def attach(self, df: DataFrame):
        """Attach data frame to model attributes including data and parameters.

        Parameters
        ----------
        df
            Given data frame.

        """
        self.data.attach(df)
        for parameter in self.parameters:
            parameter.attach(df)

    @abstractmethod
    def objective(self, x: NDArray) -> float:
        """Objective function for negative log-likelihood.

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        float
            Objective value of the negative likelihood.

        """
        pass

    @abstractmethod
    def gradient(self, x: NDArray) -> NDArray:
        """Gradient function for negative log-likelihood.

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Gradient of the negative likelihood.

        """
        pass

    @abstractmethod
    def hessian(self, x: NDArray) -> NDArray:
        """Hessian function for negative log-likelihood.

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Hessian of the negative likelihood.

        """
        pass

    @abstractmethod
    def jacobian2(self, x: NDArray) -> NDArray:
        """Jacobian square function for negative log-likelihood. This function
        is used for sandwich estimation for posterior variance-covariation
        matrix.

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Jacobian square of the negative likelihood.

        """
        pass

    def _splitx(self, x: NDArray) -> List[NDArray]:
        return np.split(x, np.cumsum(self._sizes)[:-1])

    def _get_initial_x(self) -> NDArray:
        return np.zeros(sum(self._sizes))

    def _get_vcov(self, x: NDArray) -> NDArray:
        inv_hessian = np.linalg.pinv(self.hessian(x))
        jacobian2 = self.jacobian2(x)
        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

    def fit(self, x0: Optional[NDArray] = None, **options):
        """Fit the model. The current implementation uses Scipy optimize solver.
        More specifically we are using the
        `trust-region solver <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_.
        In the future the choice of solver might become an addtional
        option of the fit function. After fitting the model the result will
        be saved in class attribute `result`, which contains `x` as the optimal
        coefficients, `vcov` as the posterior variance-covariance matrix and
        `info` for more solver results details.

        Parameters
        ----------
        x0
            The initial guess for the coefficients. Default is `None`. If
            `x0=None`, the protected function `_get_initial_x` will be called
            to get the initialization of the coefficients.

        """
        if x0 is None:
            x0 = self._get_initial_x()
        # bounds
        bounds = np.hstack([
            parameter.prior_dict["direct"]["UniformPrior"].params
            for parameter in self.parameters
        ]).T
        # linear constraint
        params = np.hstack([
            parameter.prior_dict["linear"]["UniformPrior"].params
            for parameter in self.parameters
        ])
        mat = block_diag(*[
            parameter.prior_dict["linear"]["UniformPrior"].mat
            for parameter in self.parameters
        ])
        linear_constraints = []
        if params.size > 0:
            linear_constraints = [LinearConstraint(mat, params[0], params[1])]

        info = minimize(self.objective, x0,
                        method="trust-constr",
                        jac=self.gradient,
                        hess=self.hessian,
                        constraints=linear_constraints,
                        bounds=bounds,
                        options=options)

        # store result
        self.result["x"] = info.x
        self.result["vcov"] = self._get_vcov(self.result["x"])
        self.result["info"] = info

    def predict(self,
                x: Optional[NDArray] = None,
                df: Optional[DataFrame] = None) -> List[NDArray]:
        """Predict the parameters using the coefficients and the data. User can
        choose to provide the coefficients or use optimal solution. And use can
        choose to provide new prediction data frame or use the training data.

        Parameters
        ----------
        x
            Provided coefficients. Default is `None`. If `x=None`, the function
            will use the optimal coefficients stored in `result`. If the
            optimal coefficients does not exist, an error will be raised.
        df
            Provided prediction data frame. Default is `None`. If `df=None`, the
            function will predict for the training data.

        Returns
        -------
        List[NDArray]
            The function will returns prediction for parameters. It is a list
            of arrays. The order of the list is consistent with the order of the
            parameters list.

        Raises
        ------
        ValueError
            Raised when input `x=None`, and the model has not been fitted.

        """
        if x is None:
            x = self.result.get("x", None)
        if x is None:
            raise ValueError("Please fit the model first or provide x.")
        xs = self._splitx(x)
        return [parameter.get_params(xs[i], df, order=0)
                for i, parameter in enumerate(self.parameters)]
