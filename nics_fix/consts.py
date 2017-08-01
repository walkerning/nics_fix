class DataTypes(object):
    WEIGHT = "weight" # include weights and biases
    ACTIVATION = "activation"
    OTHER = "other"
    
    all = {WEIGHT, ACTIVATION, OTHER}

class FixedKeys(object):
    # Some collection key for tensorflow collections
    FIXED_WEIGHT_DATA = "nf_fix_weight_data"
    FIXED_ACTIVATION_DATA = "nf_fix_activation_data"

    FIXED_WEIGHT_DATA_SCALE = "nf_fix_weight_data_scale"
    FIXED_ACTIVATION_DATA_SCALE = "nf_fix_activation_data_scale"

    FIXED_WEIGHT_GRAD = "nf_fix_weight_grad"
    FIXED_ACTIVATION_GRAD = "nf_fix_activation_grad"

    FIXED_WEIGHT_GRAD_SCALE = "nf_fix_weight_grad_scale"
    FIXED_ACTIVATION_GRAD_SCALE = "nf_fix_activation_grad_scale"

def _get_fixed_key(data_type, grad=False, scale=False):
    data_or_grad = "GRAD" if grad else "DATA"
    emp_or_scale = "_SCALE" if scale else ""
    return getattr(FixedKeys,
                   "FIXED_{}_{}{}".format(data_type.upper(), data_or_grad, emp_or_scale),
                   None)
