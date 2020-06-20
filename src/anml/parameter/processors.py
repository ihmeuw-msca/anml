from collections import defaultdict
import numpy as np
import pandas as pd 
from scipy.linalg import block_diag
from typing import Callable, Optional

from anml.parameter.parameter import ParameterSet
from anml.parameter.variables import Variable
from anml.parameter.spline_variable import Spline
from anml.parameter.utils import build_re_matrix, collect_priors, collect_blocks, combine_constraints


def _collect_commons(param_set, df):
    param_set.reset()
    has_re = lambda v: v.add_re == True
    
    # ---- collecting all kinds of blocks needed ----
    # design matrices ... 
    design_mat_blocks = collect_blocks(param_set, 'design_matrix', 'build_design_matrix', inputs=df)
    if param_set.num_re_var > 0:
        design_mat_re_blocks = collect_blocks(param_set, 'design_matrix_re', 'build_design_matrix_re', should_include=has_re, inputs=df)
        grouping = collect_blocks(param_set, 'n_groups', should_include=has_re)
    else:
        design_matrix_re = None
        grouping = None

    # bounds ...
    lbs_fe = collect_blocks(param_set, 'lb_fe', 'build_bounds_fe')
    ubs_fe = collect_blocks(param_set, 'ub_fe')
    
    # constraints ...
    constr_mat_blocks = collect_blocks(param_set, 'constr_matrix', 'build_constraint_matrix_fe')
    constr_lbs = collect_blocks(param_set, 'constr_lb')
    constr_ubs = collect_blocks(param_set, 'constr_ub')

    #priors ...
    fe_priors = collect_blocks(param_set, 'fe_prior')

    # build matrix that pads gammas to have length Z.shape[1]
    if param_set.num_re_var > 0:
        re_var_padding = np.repeat(np.identity(len(grouping)), grouping, axis=0)
    else:
        re_var_padding = None

    # ---- combining blocks -----
    design_matrix = np.hstack(design_mat_blocks)
    if param_set.num_re_var > 0:
        design_matrix_re = np.hstack(design_mat_re_blocks)
    constr_matrix, constr_lower_bounds, constr_upper_bounds = combine_constraints(constr_mat_blocks, constr_lbs, constr_ubs)
    # check dimensions ...
    assert design_matrix.shape[1] == param_set.num_fe
    assert constr_matrix.shape[1] == param_set.num_fe
    assert len(constr_lower_bounds) == len(constr_upper_bounds) == constr_matrix.shape[0] 
    if param_set.num_re_var > 0:
        assert design_matrix_re.shape[1] == re_var_padding.shape[0] == param_set.num_re

    return design_matrix, design_matrix_re, re_var_padding, lbs_fe, ubs_fe, constr_matrix, constr_lower_bounds, constr_upper_bounds, fe_priors, grouping


def process_for_marginal(param_set, df):
    # ----- collecting all blocks ------
    (
        param_set.design_matrix, 
        param_set.design_matrix_re, 
        param_set.re_var_padding,
        lbs_fe, 
        ubs_fe, 
        constr_matrix, 
        constr_lower_bounds, 
        constr_upper_bounds, 
        fe_priors,
        _,
    ) = _collect_commons(param_set, df)
    
    has_re = lambda v: v.add_re == True
    lbs_re_var = collect_blocks(param_set, 'lb_re_var', 'build_bounds_re_var', should_include=has_re)
    ubs_re_var = collect_blocks(param_set, 'ub_re_var', should_include=has_re)

    constr_mat_re_var_blocks = collect_blocks(param_set, 'constr_matrix_re_var', 'build_constraint_matrix_re_var', should_include=has_re)
    constr_lbs_re_var = collect_blocks(param_set, 'constr_lb_re_var', should_include=has_re)
    constr_ubs_re_var = collect_blocks(param_set, 'constr_ub_re_var', should_include=has_re)

    re_var_priors = collect_blocks(param_set, 're_var_prior', should_include=lambda v: v.add_re == True)  
    

    # ------ combining blocks ------
    # bounds ...
    param_set.lower_bounds_full = np.hstack(list(lbs_fe) + list(lbs_re_var))
    param_set.upper_bounds_full = np.hstack(list(ubs_fe) + list(ubs_re_var))
    
    # constraints ...
    constr_matrix_re_var, constr_lower_bounds_re_var, constr_upper_bounds_re_var = combine_constraints(
        constr_mat_re_var_blocks, 
        constr_lbs_re_var, 
        constr_ubs_re_var,
    )
    constr_matrix_full, constr_lower_bounds_full, constr_upper_bounds_full = combine_constraints(
        [constr_matrix, constr_matrix_re_var],
        [constr_lower_bounds, constr_lower_bounds_re_var],
        [constr_upper_bounds, constr_upper_bounds_re_var],
    )

    if np.count_nonzero(constr_matrix_full) == 0:
        param_set.constr_matrix_full, param_set.constr_lower_bounds_full, param_set.constr_upper_bounds_full = None, None, None 
    else:
        param_set.constr_matrix_full = constr_matrix_full
        param_set.constr_lower_bounds_full = constr_lower_bounds_full 
        param_set.constr_upper_bounds_full = constr_upper_bounds_full
    
    assert constr_matrix_re_var.shape[1] == param_set.num_re_var
    assert len(constr_lower_bounds_re_var) == constr_matrix_re_var.shape[0] == len(constr_upper_bounds_re_var)

    # priors ...
    param_set.prior_fun = collect_priors(fe_priors + re_var_priors)

    # names ...
    fe_variables_names = []
    re_var_variables_names = []
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            # remembering name of variable -- so that we know what each column in X corresponds to
            var_name = parameter.param_name + '_' + variable.covariate
            for i in range(variable.num_fe):
                fe_variables_names.append(var_name + '_' + str(i))
            if variable.add_re:
                re_var_variables_names.append(var_name + '_gamma' + str(i))
    param_set.variable_names = fe_variables_names + re_var_variables_names


def process_for_maximal(param_set, df):
    # ----- collecting all blocks ------
    (
        param_set.design_matrix, 
        param_set.design_matrix_re, 
        param_set.re_var_padding,
        lbs_fe, 
        ubs_fe, 
        constr_matrix, 
        constr_lower_bounds, 
        constr_upper_bounds, 
        fe_priors,
        grouping,
    ) = _collect_commons(param_set, df)
    
    has_re = lambda v: v.add_re == True
    lbs_re = collect_blocks(param_set, 'lb_re', 'build_bounds_re', should_include=has_re)
    ubs_re = collect_blocks(param_set, 'ub_re', should_include=has_re)

    constr_mat_re_blocks = collect_blocks(param_set, 'constr_matrix_re', 'build_constraint_matrix_re', should_include=has_re)
    constr_lbs_re = collect_blocks(param_set, 'constr_lb_re', should_include=has_re)
    constr_ubs_re = collect_blocks(param_set, 'constr_ub_re', should_include=has_re)
    
    param_set.re_priors = collect_blocks(param_set, 're_prior', should_include=has_re)

    # bounds ...
    param_set.lower_bounds_full = np.hstack(list(lbs_fe) + list(lbs_re))
    param_set.upper_bounds_full = np.hstack(list(ubs_fe) + list(ubs_re))
    
    # constraints ...
    constr_matrix_re, constr_lower_bounds_re, constr_upper_bounds_re = combine_constraints(
        constr_mat_re_blocks, 
        constr_lbs_re, 
        constr_ubs_re,
    )
    constr_matrix_full, constr_lower_bounds_full, constr_upper_bounds_full = combine_constraints(
        [constr_matrix, constr_matrix_re],
        [constr_lower_bounds, constr_lower_bounds_re],
        [constr_upper_bounds, constr_upper_bounds_re],
    )

    if np.count_nonzero(constr_matrix_full) == 0:
        param_set.constr_matrix_full, param_set.constr_lower_bounds_full, param_set.constr_upper_bounds_full = None, None, None 
    else:
        param_set.constr_matrix_full = constr_matrix_full
        param_set.constr_lower_bounds_full = constr_lower_bounds_full 
        param_set.constr_upper_bounds_full = constr_upper_bounds_full

    assert constr_matrix_re.shape[1] == param_set.num_re
    assert len(constr_lower_bounds_re) == constr_matrix_re.shape[0] == len(constr_upper_bounds_re)

    priors_all = fe_priors
    for n_group, prior in zip(grouping, param_set.re_priors):
        priors_all.extend([prior] * n_group)
    param_set.prior_fun = collect_priors(priors_all)

    # variable names ...
    fe_variables_names = []
    re_variables_names = []
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            # remembering name of variable -- so that we know what each column in X corresponds to
            var_name = parameter.param_name + '_' + variable.covariate
            for i in range(variable.num_fe):
                fe_variables_names.append(var_name + '_' + str(i))
            if variable.add_re:
                for i in range(variable.num_re):
                    re_variables_names.append(var_name + '_u' + str(i))
    param_set.variable_names = fe_variables_names + re_variables_names


def process_for_no_re(param_set, df):
    # ----- collecting all blocks ------
    (
        param_set.design_matrix, 
        param_set.design_matrix_re, 
        param_set.re_var_padding,
        lbs_fe, 
        ubs_fe, 
        param_set.constr_matrix_full, 
        param_set.constr_lower_bounds_full, 
        param_set.constr_upper_bounds_full, 
        fe_priors,
        _,
    ) = _collect_commons(param_set, df)

    param_set.lower_bounds_full = np.hstack(list(lbs_fe))
    param_set.upper_bounds_full = np.hstack(list(ubs_fe))
    param_set.prior_fun = collect_priors(fe_priors)

    fe_variables_names = []
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            # remembering name of variable -- so that we know what each column in X corresponds to
            var_name = parameter.param_name + '_' + variable.covariate
            for i in range(variable.num_fe):
                fe_variables_names.append(var_name + '_' + str(i))

    param_set.variable_names = fe_variables_names
