from typing import List, Optional, Type

import anml
from anml.prior.main import Prior


def get_prior_type(prior_type: str) -> Type:
    """Get prior type from the prior type name.

    Parameters
    ----------
    prior_type
        Name of the prior type (class).

    Returns
    -------
    Type
        The class corresponding to the given prior type name.

    Examples
    --------
    >>> prior_type = get_prior_type("GaussianPrior")
    >>> prior_type
    <class 'anml.prior.main.GaussianPrior'>

    """
    return getattr(anml.prior.main, prior_type)


def filter_priors(
    priors: List[Prior], prior_type: str, with_mat: Optional[bool] = None
) -> List[Prior]:
    """Filter priors from a list of priors by their type and do they contain
    linear map or not.

    Parameters
    ----------
    priors
        Given list of priors. Note that it is user's responsibility to check if
        all elements in the list are instances of Prior.
    prior_type
        Given prior type name.
    with_mat
        If the filtered priors are all contain a linear map. Default to `None`.
        If `with_mat=None`, the final list will include priors that both
        contain or not contain the linear map.

    Returns
    -------
    List[Prior]
        Filtered priors.

    """
    prior_type = get_prior_type(prior_type)

    def condition(prior, prior_type=prior_type, with_mat=with_mat):
        is_prior_instance = isinstance(prior, prior_type)
        if with_mat is None:
            return is_prior_instance
        if with_mat:
            return prior.mat is not None and is_prior_instance
        return prior.mat is None and is_prior_instance

    return list(filter(condition, priors))
