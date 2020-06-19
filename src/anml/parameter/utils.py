import numpy as np
from typing import List, Any
from scipy.linalg import block_diag


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

