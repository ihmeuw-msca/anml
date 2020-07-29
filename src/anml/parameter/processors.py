import numpy as np
import pandas as pd

from anml.parameter.utils import combine_constraints
from anml.parameter.variables import ParameterBlock, collect_blocks


def process_all(param_block: ParameterBlock, df: pd.DataFrame):
    process_for_betas(param_block, df)
    if param_block.num_re_var > 0:
        process_for_gammas(param_block, df)
        process_for_us(param_block, df)


def process_for_betas(param_block: ParameterBlock, df: pd.DataFrame, reset=True):
    if reset:
        param_block.reset()
    # collecting blocks related to betas ...
    design_mat_blocks = collect_blocks(param_block, 'design_matrix_fe', 'build_design_matrix_fe', inputs=df)
    lbs_fe = collect_blocks(param_block, 'lb_fe', 'build_bounds_fe')
    ubs_fe = collect_blocks(param_block, 'ub_fe')
    constr_mat_fe_blocks = collect_blocks(param_block, 'constr_matrix_fe', 'build_constraint_matrix_fe')
    constr_lbs_fe = collect_blocks(param_block, 'constr_lb_fe')
    constr_ubs_fe = collect_blocks(param_block, 'constr_ub_fe')
    param_block.fe_priors = collect_blocks(param_block, 'fe_prior')
    # combining blocks ...
    param_block.design_matrix_fe = np.hstack(design_mat_blocks)
    param_block.lb_fe = np.hstack(lbs_fe)
    param_block.ub_fe = np.hstack(ubs_fe)
    param_block.constr_matrix_fe, param_block.constr_lb_fe, param_block.constr_ub_fe = combine_constraints(
        constr_mat_fe_blocks, constr_lbs_fe, constr_ubs_fe
    )

    assert param_block.design_matrix_fe.shape[1] == param_block.num_fe
    assert len(param_block.lb_fe) == len(param_block.ub_fe)
    assert param_block.constr_matrix_fe.shape[1] == param_block.num_fe
    assert len(param_block.constr_lb_fe) == len(param_block.constr_ub_fe) == param_block.constr_matrix_fe.shape[0]


def process_for_gammas(param_block: ParameterBlock, df: pd.DataFrame, reset=False):
    if reset:
        param_block.reset()
    has_re = lambda v: v.add_re == True
    # collecting blocks related to betas ...
    if param_block.design_matrix_re is None:
        design_mat_re_blocks = collect_blocks(
            param_block, 'design_matrix_re', 'build_design_matrix_re', inputs=df, should_include=has_re
        )
    else:
        design_mat_re_blocks = None
    lbs_re_var = collect_blocks(param_block, 'lb_re_var', 'build_bounds_re_var', should_include=has_re)
    ubs_re_var = collect_blocks(param_block, 'ub_re_var', should_include=has_re)
    constr_mat_re_var_blocks = collect_blocks(
        param_block, 'constr_matrix_re_var', 'build_constraint_matrix_re_var', should_include=has_re
    )
    constr_lbs_re_var = collect_blocks(param_block, 'constr_lb_re_var', should_include=has_re)
    constr_ubs_re_var = collect_blocks(param_block, 'constr_ub_re_var', should_include=has_re)
    param_block.re_var_priors = collect_blocks(param_block, 're_var_prior', should_include=has_re)
    grouping = collect_blocks(param_block, 'n_groups', should_include=has_re)
    
    # combining blocks ...
    if param_block.design_matrix_re is None:
        param_block.design_matrix_re = np.hstack(design_mat_re_blocks)
    param_block.lb_re_var = np.hstack(lbs_re_var)
    param_block.ub_re_var = np.hstack(ubs_re_var)
    param_block.constr_matrix_re_var, param_block.constr_lb_re_var, param_block.constr_ub_re_var = combine_constraints(
        constr_mat_re_var_blocks, 
        constr_lbs_re_var, 
        constr_ubs_re_var,
    )
    param_block.re_var_padding = np.repeat(np.identity(len(grouping)), grouping, axis=0)

    assert param_block.design_matrix_re.shape[1] == param_block.re_var_padding.shape[0] == param_block.num_re
    assert param_block.constr_matrix_re_var.shape[1] == param_block.num_re_var
    assert len(param_block.constr_lb_re_var) == \
           param_block.constr_matrix_re_var.shape[0] == \
           len(param_block.constr_ub_re_var)


def process_for_us(param_block: ParameterBlock, df: pd.DataFrame, reset=False):
    if reset:
        param_block.reset()
    has_re = lambda v: v.add_re == True
    # collecting blocks related to betas ...
    if param_block.design_matrix_re is None:
        design_mat_re_blocks = collect_blocks(
            param_block, 'design_matrix_re', 'build_design_matrix_re', inputs=df, should_include=has_re
        )
    else:
        design_mat_re_blocks = None
    lbs_re = collect_blocks(param_block, 'lb_re', 'build_bounds_re', should_include=has_re)
    ubs_re = collect_blocks(param_block, 'ub_re', should_include=has_re)
    constr_mat_re_blocks = collect_blocks(param_block, 'constr_matrix_re', 'build_constraint_matrix_re', should_include=has_re)
    constr_lbs_re = collect_blocks(param_block, 'constr_lb_re', should_include=has_re)
    constr_ubs_re = collect_blocks(param_block, 'constr_ub_re', should_include=has_re)
    re_priors = collect_blocks(param_block, 're_prior', should_include=has_re)
    grouping = collect_blocks(param_block, 'n_groups', should_include=has_re)
    
    # combining blocks ...
    if param_block.design_matrix_re is None:
        param_block.design_matrix_re = np.hstack(design_mat_re_blocks)
    param_block.lb_re = np.hstack(lbs_re)
    param_block.ub_re = np.hstack(ubs_re)
    param_block.constr_matrix_re, param_block.constr_lb_re, param_block.constr_ub_re = combine_constraints(
        constr_mat_re_blocks, 
        constr_lbs_re, 
        constr_ubs_re,
    )
    param_block.re_var_padding = np.repeat(np.identity(len(grouping)), grouping, axis=0)
    param_block.re_priors = []
    for n_group, prior in zip(grouping, re_priors):
        param_block.re_priors.extend([prior] * n_group)
    
    assert param_block.design_matrix_re.shape[1] == param_block.re_var_padding.shape[0] == param_block.num_re
    assert param_block.constr_matrix_re.shape[1] == param_block.num_re
    assert len(param_block.constr_lb_re) == param_block.constr_matrix_re.shape[0] == len(param_block.constr_ub_re)







