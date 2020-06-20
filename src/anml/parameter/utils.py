import numpy as np
import pandas as pd
from typing import List, Any, Optional, Callable
from scipy.linalg import block_diag

from anml.parameter.prior import Prior


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


def combine_constraints(constr_matrix: np.ndarray, constr_lb: np.ndarray, constr_ub: np.ndarray):
    mat, lb, ub = block_diag(*constr_matrix), np.hstack(constr_lb), np.hstack(constr_ub)
    valid_rows_id = []
    for i in range(mat.shape[0]):
        if np.count_nonzero(mat[i, :]) > 0:
            valid_rows_id.append(i)
    if len(valid_rows_id) > 0:
        return mat[valid_rows_id, :], lb[valid_rows_id], ub[valid_rows_id]
    else:
        return np.zeros((1, mat.shape[1])), [0.0], [0.0]

