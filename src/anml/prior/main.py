from __future__ import annotations

from operator import attrgetter
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Prior:
    """Prior information for the variables. It is used for constructing the
    likelihood and solve the optimization problem.

    Parameters
    ----------
    params
        Distribution parameters.
    mat
        Matrix that map the variable to the prior space. Default is `None`.
        When `mat=None`, it will treat as if `mat` is the identity matrix, in
        another word, the prior will be directly applied to the variable.

    """

    default_params: Optional[NDArray] = None
    """Default parameters. This should be distribution specific.

    """
    params = property(attrgetter("_params"))
    """Distribution parameters.

    """
    mat = property(attrgetter("_mat"))
    """Matrix that map the variable to the prior space.

    Raises
    ------
    ValueError
        Raised when matrix is an empty array.
    ValueError
        Raised when parameter is not broadcastable and the first dimension of
        the matrix doesn't match the second dimension of parameters. Both of
        them describe the number of priors.
 
    """

    def __init__(self, params: List[ArrayLike], mat: Optional[ArrayLike] = None):
        self.params = params
        self.mat = mat

    @params.setter
    def params(self, params: List[ArrayLike]):
        if all(np.asarray(param).size == 0 for param in params):
            self._params = np.empty((len(params), 0))
        else:
            self._params = np.column_stack(list(np.broadcast(*params)))

    @mat.setter
    def mat(self, mat: Optional[ArrayLike]):
        if mat is not None:
            mat = np.asarray(mat)
            if self.params.shape[1] == 1:
                self._params = np.repeat(self._params, mat.shape[0], axis=1)
            if self.params.shape[1] != mat.shape[0]:
                raise ValueError("Prior mat and params shape don't match.")
        self._mat = mat

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the prior, with first dimension as the number of priors and
        second dimension the size of the variable.

        """
        if self.mat is None:
            return (self.params.shape[1], self.params.shape[1])
        return self.mat.shape

    def objective(self, x: NDArray) -> float:
        """Objective function for the log likelihood of the prior.

        Parameters
        ----------
        x
            Given variable as a vector.

        Returns
        -------
        float
            Objective value.

        """
        raise NotImplementedError

    def gradient(self, x: NDArray) -> NDArray:
        """Gradient function for the log likelihood of the prior.

        Parameters
        ----------
        x
            Given variable as a vector.

        Returns
        -------
        NDArray
            Gradient of the objective function.

        """
        raise NotImplementedError

    def hessian(self, x: NDArray) -> NDArray:
        """Hessian function for the log likelihood of the prior.

        Parameters
        ----------
        x
            Given variable as a vector.

        Returns
        -------
        NDArray
            Hessian of the objective function.

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        params = self.params.__repr__()
        mat = self.mat.__repr__()
        return f"{type(self).__name__}(params={params}, mat={mat})"


class GaussianPrior(Prior):
    """Gaussian prior.

    Parameters
    ----------
    mean
        Mean of the Gaussian distribution.
    sd
        Standard deviation of the Gaussian distribution.
    mat
        Matrix that map the variable to the prior space. Default is `None`.
        When `mat=None`, it will treat as if `mat` is the identity matrix, in
        another word, the prior will be directly applied to the variable.

    Raises
    ------
    ValueError
        Raised when standard deviations are not all positive.

    Examples
    --------
    The following three ways of defining a :class:`GaussianPrior` are
    equivalent.

    .. code-block:: python

        from anml.prior.main import GaussianPrior

        prior = GaussianPrior(mean=0.0, sd=[0.1, 0.1])
        prior = GaussianPrior(mean=[0.0, 0.0], sd=0.1)
        prior = GaussianPrior(mean=[0.0, 0.0], sd=[0.1, 0.1])

    """

    default_params: Optional[NDArray] = np.array([[0.0], [np.inf]])
    """Gaussian prior default params, with mean zero and standard deviation inf.

    """

    def __init__(self, mean: ArrayLike, sd: ArrayLike, mat: Optional[ArrayLike] = None):
        super().__init__([mean, sd], mat=mat)
        if not (self.params[1] > 0.0).all():
            raise ValueError("Gaussian prior standard deviations must be " "positive.")
        self.mean = self.params[0]
        self.sd = self.params[1]

    def objective(self, x: NDArray) -> float:
        if self.mat is None:
            return 0.5 * np.sum(((x - self.mean) / self.sd) ** 2)
        if self.mat.size == 0:
            return 0.0
        return 0.5 * np.sum(((self.mat.dot(x) - self.mean) / self.sd) ** 2)

    def gradient(self, x: NDArray) -> NDArray:
        if self.mat is None:
            return (x - self.mean) / self.sd**2
        if self.mat.size == 0:
            return np.zeros(x.size)
        return (self.mat.T / self.sd**2).dot(self.mat.dot(x) - self.mean)

    def hessian(self, x: NDArray) -> NDArray:
        if self.mat is None:
            return np.diag(1 / self.sd**2)
        if self.mat.size == 0:
            return np.zeros((x.size, x.size))
        return (self.mat.T / self.sd**2).dot(self.mat)


class UniformPrior(Prior):
    """Uniform prior.

    Parameters
    ----------
    lb
        Lower bounds of the Uniform distribution.
    ub
        Upper bounds of the Uniform distribution.
    mat
        Matrix that map the variable to the prior space. Default is `None`.
        When `mat=None`, it will treat as if `mat` is the identity matrix, in
        another word, the prior will be directly applied to the variable.

    Raises
    ------
    ValueError
        Raised when lower bounds are not all smaller than the upper bounds.

    Examples
    --------
    The following three ways of defining a :class:`UniformPrior` are
    equivalent.

    .. code-block:: python

        from anml.prior.main import UniformPrior

        prior = UniformPrior(lb=0.0, ub=[1.0, 1.0])
        prior = UniformPrior(lb=[0.0, 0.0], ub=1.0)
        prior = UniformPrior(lb=[0.0, 0.0], ub=[1.0, 1.0])

    """

    default_params: Optional[NDArray] = np.array([[-np.inf], [np.inf]])
    """Uniform prior default params, with -inf as the lower bound and inf as the
     upper bound.

    """

    def __init__(self, lb: ArrayLike, ub: ArrayLike, mat: Optional[ArrayLike] = None):
        super().__init__([lb, ub], mat=mat)
        if not (self.params[0] <= self.params[1]).all():
            raise ValueError(
                "Uniform prior lower bounds have to be less than "
                "or equal to the upper bounds."
            )
        self.lb = self.params[0]
        self.ub = self.params[1]
