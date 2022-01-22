from __future__ import annotations

from typing import List

from anml.data.example import DataExample
from anml.model.prototype import ModelPrototype
from anml.parameter.main import Parameter
from anml.variable.main import Variable
from numpy.typing import NDArray
from pandas import DataFrame


class ModelExample(ModelPrototype):
    """An example model class for simple least-square problem. This class
    provides implementations of the optimization problems including the
    objective, gradient and hessian function. And provide a Jacobian matrix
    computation for the posterior variance-covariance matrix. For least-square
    problem we only have one parameter that can contain multiple variables. 

    Parameters
    ----------
    data
        Given data object. Here we use instance from
        :class:`anml.data.example.DataExample`.
    variables
        Given list of variables to parameterize the mean in the linear
        regression problem.
    df
        Given data frame contains the observations and covariates.

    """

    def __init__(self,
                 data: DataExample,
                 variables: List[Variable],
                 df: DataFrame):
        parameter = Parameter(variables)
        super().__init__(data, [parameter], df)

    def objective(self, x: NDArray) -> float:
        """Objective function of the least square negative likelihood.

        .. math:: \\frac{1}{2} (y - Ax)^\\top \\Sigma^{-1} (y - Ax) 

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        float
            Objective value of the negative likelihood.

        """
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        value = 0.5*sum((r / self.data.obs_se.value)**2)
        value += self.parameters[0].prior_objective(x)
        return value

    def gradient(self, x: NDArray) -> NDArray:
        """Gradient function of the least square negative likelihood.

        .. math:: A^\\top \\Sigma^{-1} (Ax - y)

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Gradient of the negative likelihood.

        """
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        m = self.parameters[0].get_params(x, order=1)
        value = -(m.T / self.data.obs_se.value**2).dot(r)
        value += self.parameters[0].prior_gradient(x)
        return value

    def hessian(self, x: NDArray) -> NDArray:
        """Hessian function of the least square negative likelihood.

        .. math:: A^\\top \\Sigma^{-1} A

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Hessian of the negative likelihood.

        """
        m = self.parameters[0].get_params(x, order=1)
        value = (m.T / self.data.obs_se.value**2).dot(m)
        value += self.parameters[0].prior_hessian(x)
        return value

    def jacobian2(self, x: NDArray) -> NDArray:
        """Jacobian square function of the least square negative likelihood.

        .. math:: A^\\top \\Sigma^{-1} \\mathrm{Diag}((y - Ax)^2) \\Sigma^{-1} A

        Parameters
        ----------
        x
            Input coefficients.

        Returns
        -------
        NDArray
            Jacobian square of the negative likelihood.

        """
        r = self.data.obs.value - self.parameters[0].get_params(x, order=0)
        m = self.parameters[0].get_params(x, order=1)
        jacobian = -(m.T / self.data.obs_se.value**2) * r
        value = jacobian.dot(jacobian.T)
        value += self.parameters[0].prior_hessian(x)
        return value
