
def apply_target_transformation(target_series):
    """Apply optimal target transformation based on EDA analysis"""
    import numpy as np
    
    if 'boxcox' == 'log1p':
        return np.log1p(target_series)
    elif 'boxcox' == 'sqrt':
        return np.sqrt(target_series)
    elif 'boxcox' == 'boxcox':
        from scipy.stats import boxcox
        # Use stored lambda parameter
        lambda_param = -0.07693211157738546
        if lambda_param is not None:
            return boxcox(target_series + 1, lmbda=lambda_param)
        else:
            transformed_data, _ = boxcox(target_series + 1)
            return transformed_data
    else:
        return target_series

def inverse_target_transformation(transformed_series):
    """Apply inverse transformation to get back to original scale"""
    import numpy as np
    
    if 'boxcox' == 'log1p':
        return np.expm1(transformed_series)
    elif 'boxcox' == 'sqrt':
        return np.square(transformed_series)
    elif 'boxcox' == 'boxcox':
        from scipy.special import inv_boxcox
        lambda_param = -0.07693211157738546
        if lambda_param is not None:
            return inv_boxcox(transformed_series, lambda_param) - 1
        else:
            # Approximate inverse for Box-Cox when lambda unknown
            return np.expm1(transformed_series)
    else:
        return transformed_series
        