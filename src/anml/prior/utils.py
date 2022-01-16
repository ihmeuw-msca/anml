from typing import List, Optional, Type

import anml
from anml.prior.main import Prior


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
