from operator import attrgetter
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Prior:

    params = property(attrgetter("_params"))
    mat = property(attrgetter("_mat"))

    def __init__(self,
                 params: List[ArrayLike],
                 mat: Optional[ArrayLike] = None):
        self.params = params
        self.mat = mat

    @params.setter
    def params(self, params: List[ArrayLike]):
        self._params = np.column_stack(list(np.broadcast(*params)))

    @mat.setter
    def mat(self, mat: Optional[ArrayLike]):
        if mat is not None:
            mat = np.asarray(mat)
            if mat.size == 0:
                raise ValueError("Prior mat cannot be empty.")
            if self.params.shape[1] == 1:
                self._params = np.repeat(self._params, mat.shape[0], axis=1)
            if self.params.shape[1] != mat.shape[0]:
                raise ValueError("Prior mat and params shape don't match.")
        self._mat = mat

    @property
    def shape(self) -> Tuple[int, int]:
        if self.mat is None:
            return (self.params.shape[1], self.params.shape[1])
        return self.mat.shape

    def objective(self, x: NDArray) -> float:
        raise NotImplementedError

    def gradient(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    def hessian(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    def __repr__(self) -> str:
        params = self.params.__repr__()
        mat = self.mat.__repr__()
        return f"{type(self).__name__}(params={params}, mat={mat})"


class GaussianPrior(Prior):

    def __init__(self,
                 mean: ArrayLike,
                 sd: ArrayLike,
                 mat: Optional[ArrayLike] = None):
        super().__init__([mean, sd], mat=mat)
        if not (self.params[1] > 0.0).all():
            raise ValueError("Gaussian prior standard deviations must be "
                             "positive.")
        self.mean = self.params[0]
        self.sd = self.params[1]

    def objective(self, x: NDArray) -> float:
        if self.mat is None:
            return 0.5*np.sum(((x - self.mean) / self.sd)**2)
        return 0.5*np.sum(((self.mat.dot(x) - self.mean) / self.sd)**2)

    def gradient(self, x: NDArray) -> NDArray:
        if self.mat is None:
            return (x - self.mean) / self.sd**2
        return (self.mat.T / self.sd**2).dot(self.mat.dot(x) - self.mean)

    def hessian(self, x: NDArray) -> NDArray:
        if self.mat is None:
            return np.diag(1 / self.sd**2)
        return (self.mat.T / self.sd**2).dot(self.mat)


class UniformPrior(Prior):

    def __init__(self,
                 lb: ArrayLike,
                 ub: ArrayLike,
                 mat: Optional[ArrayLike] = None):
        super().__init__([lb, ub], mat=mat)
        if not (self.params[0] <= self.params[1]).all():
            raise ValueError("Uniform prior lower bounds have to be less than "
                             "or equal to the upper bounds.")
        self.lb = self.params[0]
        self.ub = self.params[1]
