from collections import defaultdict
import numpy as np
import pandas as pd 
from scipy.linalg import block_diag

from anml.parameter.parameter import ParameterSet


def process_for_marginal(param_set: ParameterSet, df: pd.DataFrame):
    param_set.reset()
    design_mat_blocks = []
    design_mat_re_blocks = []
    
    constr_mat_blocks = []
    constr_mat_re_var_blocks = []
    constr_lbs = []
    constr_ubs = []
    constr_lbs_re_var = []
    constr_ubs_re_var = []
    
    fe_variables_names = []
    re_var_variables_names = []
    fe_priors = []
    re_var_priors = []
        
    for parameter in param_set.parameters:
        for variable in parameter.variables:
            # remembering name of variable -- so that we know what each column in X corresponds to
            var_name = parameter.param_name + '_' + variable.covariate
            fe_variables_names.append(var_name)
            if variable.add_re:
                re_var_variables_names.append(var_name + '_gamma')

            # getting design matrix corresponding to the variable
            variable.build_design_matrix(df)
            design_mat_blocks.append(variable.design_matrix)
            if variable.add_re:
                design_mat_re_blocks.append(variable.design_matrix_re)
            
            # getting constraint matrix and bounds
            constr_mat, constr_lb, constr_ub = variable.constraint_matrix()
            constr_mat_blocks.append(constr_mat)
            constr_lbs.append(constr_lb)
            constr_ubs.append(constr_ub)
            if variable.add_re:
                constr_mat_re_var, constr_lb_re_var, constr_ub_re_var = variable.constraint_matrix_re_var()
                constr_mat_re_var_blocks.append(constr_mat_re_var)
                constr_lbs_re_var.append(constr_lb_re_var)
                constr_ubs_re_var.append(constr_ub_re_var)

            # append priors functions
            fe_priors.append(variable.fe_prior)
            if variable.add_re:
                re_var_priors.append(variable.re_var_prior)

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
        constr_matrix_re_var = block_diag(constr_mat_re_var_blocks)
    else:
        constr_matrix_re_var = constr_mat_re_var_blocks[0]
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