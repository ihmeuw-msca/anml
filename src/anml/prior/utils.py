from typing import List, Optional, Tuple, Type

import anml
import numpy as np
from anml.prior.main import Prior
from numpy.typing import NDArray


def get_prior_type(prior_type: str) -> Type:
    return getattr(anml.prior.main, prior_type)


def filter_priors(priors: List[Prior],
                  prior_type: str,
                  with_mat: Optional[bool] = None) -> List[Prior]:
    prior_type = get_prior_type(prior_type)

    def condition(prior, prior_type=prior_type, with_mat=with_mat):
        is_prior_instance = isinstance(prior, prior_type)
        if with_mat is None:
            return is_prior_instance
        if with_mat:
            return prior.mat is not None and is_prior_instance
        return prior.mat is None and is_prior_instance

    return list(filter(condition, priors))


def combine_priors_params(priors: List[Prior]) -> Tuple[NDArray, Optional[NDArray]]:
    if not all(isinstance(prior, Prior) for prior in priors):
        raise TypeError("All prior in priors must be an instance of Prior.")
    if len(priors) == 0:
        raise ValueError("Priors must be a non-empty list of priors.")
    if len(priors) == 1:
        return priors[0].params, priors[0].mat
    params = np.hstack([prior.params for prior in priors])
    mat = np.vstack([
        prior.mat if prior.mat is not None else np.identity(prior.shape[1])
        for prior in priors
    ])
    return params, mat
