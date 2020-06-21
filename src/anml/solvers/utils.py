
def has_bounds(model):
    return hasattr(model, 'lb') and hasattr(model, 'ub') and model.lb is not None and model.ub is not None

def has_constraints(model):
    return (
        hasattr(model, 'C') and hasattr(model, 'c_lb') and hasattr(model, 'c_ub') and 
        model.C is not None and model.c_lb is not None and model.c_ub is not None
    )