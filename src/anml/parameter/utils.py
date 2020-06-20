import numpy as np
import pandas as pd
from typing import List, Any, Optional, Callable
from scipy.linalg import block_diag

from anml.parameter.parameter import Prior


def encode_groups(group_assign_cat: List[Any]):
    groups = np.unique(group_assign_cat)
    group_id_dict = {grp: i for i, grp in enumerate(groups)}
    return group_id_dict


def build_re_matrix(matrix: np.ndarray, group_assign_ord: List[int], n_groups: int):
    n_coefs = matrix.shape[1]
    re_mat = np.zeros((matrix.shape[0], n_groups * n_coefs))
    for i, row in enumerate(matrix):
        grp = group_assign_ord[i]
        re_mat[i, grp * n_coefs: (grp + 1) * n_coefs] = row 
    return re_mat


def collect_priors(priors: List[Prior]):
    def prior_fun(x):
        s = 0
        val = 0.0
        for prior in priors:
            x_dim = prior.x_dim
            val += prior.error_value(x[s: s + x_dim])
            s += x_dim
        return val 
    return prior_fun


def collect_blocks(
    param_set, 
    attr_name: str, 
    build_func: Optional[str] = None, 
    should_include: Optional[Callable] = lambda x: True,
    reset_params: Optional[bool] = False,
    inputs: Optional[pd.DataFrame] = None,
):
    if reset_params:
        param_set.reset()
    
    blocks = []
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            if should_include(variable):
                if build_func is not None:
                    func = getattr(variable, build_func)
                    if inputs is not None:
                        func(inputs)
                    else:
                        func()
                blocks.append(getattr(variable, attr_name))
    
    return blocks

