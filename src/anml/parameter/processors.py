from collections import defaultdict
import numpy as np
import pandas as pd 
from scipy.linalg import block_diag
from typing import Callable, Optional

from anml.parameter.parameter import ParameterSet
from anml.parameter.variables import Variable
from anml.parameter.utils import build_re_matrix


def collect_blocks(
    param_set: ParameterSet, 
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


def process_for_marginal(param_set, df):
    param_set.reset()
    
    # collecting all kinds of blocks needed
    design_mat_blocks = collect_blocks(param_set, 'design_matrix', 'build_design_matrix', inputs=df)
    design_mat_re_blocks = collect_blocks(param_set, 'design_matrix_re', should_include=lambda v: v.add_re == True)
    constr_mat_blocks = collect_blocks(param_set, 'constr_matrix', 'build_constraint_matrix')
    constr_mat_re_var_blocks = collect_blocks(param_set, 'constr_matrix_re_var', 'build_constraint_matrix_re_var', should_include=lambda v: v.add_re == True)
    constr_lbs = collect_blocks(param_set, 'constr_lb')
    constr_ubs = collect_blocks(param_set, 'constr_ub')
    constr_lbs_re_var = collect_blocks(param_set, 'constr_lb_re_var', should_include=lambda v: v.add_re == True)
    constr_ubs_re_var = collect_blocks(param_set, 'constr_ub_re_var', should_include=lambda v: v.add_re == True)
    fe_priors = collect_blocks(param_set, 'fe_prior')
    re_var_priors = collect_blocks(param_set, 're_var_prior', should_include=lambda v: v.add_re == True)

    fe_variables_names = []
    re_var_variables_names = []
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            # remembering name of variable -- so that we know what each column in X corresponds to
            var_name = parameter.param_name + '_' + variable.covariate
            fe_variables_names.append(var_name)
            if variable.add_re:
                re_var_variables_names.append(var_name + '_gamma')

    # combining blocks
    param_set.design_matrix = np.hstack(design_mat_blocks)
    constr_matrix = block_diag(*constr_mat_blocks)
    constr_lower_bounds = np.hstack(constr_lbs)
    constr_upper_bounds = np.hstack(constr_ubs)

    # checking dimensions match -- design matrix and constr matrix should have same # of columns == num_fe
    # -- bounds should have same dimension as # of rows of constr matrix
    assert param_set.design_matrix.shape[1] == constr_matrix.shape[1] == param_set.num_fe
    assert len(constr_lower_bounds) == len(constr_upper_bounds) == constr_matrix.shape[0]

    param_set.design_matrix_re = np.hstack(design_mat_re_blocks)
    if len(constr_mat_re_var_blocks) > 1:
        constr_matrix_re_var = block_diag(*constr_mat_re_var_blocks)
        re_var_diag = []
        for block in design_mat_re_blocks:
            re_var_diag.append(np.ones((block.shape[1], 1)))
        param_set.re_var_diag = block_diag(*re_var_diag)
    else:
        constr_matrix_re_var = constr_mat_re_var_blocks[0]
        param_set.re_var_diag = np.identity(design_mat_re_blocks[0].shape[1])
    
    assert param_set.design_matrix_re.shape[1] == param_set.re_var_diag.shape[0]
    constr_lower_bounds_re_var = np.hstack(constr_lbs_re_var)
    constr_upper_bounds_re_var = np.hstack(constr_ubs_re_var)

    assert constr_matrix_re_var.shape[1] == param_set.num_re_var
    assert len(constr_lower_bounds_re_var) == constr_matrix_re_var.shape[0] == len(constr_upper_bounds_re_var)

    param_set.constr_matrix_full = block_diag(constr_matrix, constr_matrix_re_var)
    param_set.constr_lower_bounds_full = np.hstack((constr_lower_bounds, constr_lower_bounds_re_var))
    param_set.constr_upper_bounds_full = np.hstack((constr_upper_bounds, constr_upper_bounds_re_var))

    param_set.prior_fun = collect_priors(fe_priors + re_var_priors)
    param_set.variable_names = fe_variables_names + re_var_variables_names


def collect_priors(priors):
    def prior_fun(x):
        s = 0
        val = 0.0
        for prior in priors:
            x_dim = prior.x_dim
            val += prior.error_value(x[s: s + x_dim])
            s += x_dim
        return val 
    return prior_fun