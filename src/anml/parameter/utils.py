from typing import List, Any, Tuple

import numpy as np
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


def build_linear_constraint(constraints: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    if len(constraints) == 1:
        mats, lbs, ubs = constraints[0]
    mats, lbs, ubs = zip(*constraints)
    C, c_lb, c_ub = combine_constraints(mats, lbs, ubs)
    if np.count_nonzero(C) == 0:
        return None, None, None
    else:
        return C, c_lb, c_ub
