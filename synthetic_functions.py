import numpy as np

def get_1d_functions():
    """Returns 5 diverse 1D functions with training gaps."""
    functions = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "cos_trend": {
            "func": lambda x: np.cos(x) + x / 10,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "poly": {
            "func": lambda x: x**2 / 50,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "damped_osc": {
            "func": lambda x: np.exp(-x / 5) * np.sin(2 * x),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "log_mod": {
            "func": lambda x: np.log(x + 1) * np.sin(x),
            "gap": (3.5, 6.5),
            "range": (0.1, 10),
        },
    }
    return functions

def get_2d_functions():
    """Returns 5 diverse 2D functions with training gaps."""
    functions = {
        "sin_cos": {
            "func": lambda x1, x2: np.sin(x1) * np.cos(x2),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic": {
            "func": lambda x1, x2: (x1**2 + x2**2) / 100,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_mod": {
            "func": lambda x1, x2: np.sin(x1 + x2) + 0.1 * x1 * x2,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "gaussian": {
            "func": lambda x1, x2: np.exp(-(x1**2 + x2**2) / 10),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "abs_sin": {
            "func": lambda x1, x2: np.abs(x1 - x2) + np.sin(x1 * x2),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_3d_functions():
    """Returns 5 diverse 3D functions with training gaps."""
    functions = {
        "sin_cos_sin": {
            "func": lambda x1, x2, x3: np.sin(x1) * np.cos(x2) * np.sin(x3),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_3d": {
            "func": lambda x1, x2, x3: (x1**2 + x2**2 + x3**2) / 150,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_3d": {
            "func": lambda x1, x2, x3: np.sin(x1 + x2 + x3) + 0.1 * x1 * x2 * x3,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "gaussian_3d": {
            "func": lambda x1, x2, x3: np.exp(-(x1**2 + x2**2 + x3**2) / 15),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "sin_exp_cos": {
            "func": lambda x1, x2, x3: np.sin(x1) * np.exp(-x2 / 5) * np.cos(x3),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_4d_functions():
    """Returns 3 diverse 4D functions with training gaps."""
    functions = {
        "sin_cos_4d": {
            "func": lambda x1, x2, x3, x4: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_4d": {
            "func": lambda x1, x2, x3, x4: (x1**2 + x2**2 + x3**2 + x4**2) / 200,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_4d": {
            "func": lambda x1, x2, x3, x4: np.sin(x1 + x2 + x3 + x4) + 0.05 * x1 * x2 * x3 * x4,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_5d_functions():
    """Returns 3 diverse 5D functions with training gaps."""
    functions = {
        "sin_cos_5d": {
            "func": lambda x1, x2, x3, x4, x5: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_5d": {
            "func": lambda x1, x2, x3, x4, x5: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2) / 250,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "gaussian_5d": {
            "func": lambda x1, x2, x3, x4, x5: np.exp(-(x1**2 + x2**2 + x3**2 + x4**2 + x5**2) / 20),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
    }
    return functions

def get_6d_functions():
    """Returns 3 diverse 6D functions with training gaps."""
    functions = {
        "sin_cos_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2) / 300,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "friedman_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: 10 * np.sin(np.pi * x1 * x2 / 100) + 20 * (x3 / 10 - 0.5)**2 + 10 * x4 / 10 + 5 * x5 / 10 + x6 / 10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_7d_functions():
    """Returns 3 diverse 7D functions with training gaps."""
    functions = {
        "sin_cos_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2) / 350,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "interaction_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: (x1*x2 + x2*x3 + x3*x4 + x4*x5 + x5*x6 + x6*x7) / 50,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_8d_functions():
    """Returns 3 diverse 8D functions with training gaps."""
    functions = {
        "sin_cos_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2) / 400,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "exp_sum_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: np.exp(-( (x1-5)**2 + (x2-5)**2 + (x3-5)**2 + (x4-5)**2 + (x5-5)**2 + (x6-5)**2 + (x7-5)**2 + (x8-5)**2 ) / 80),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_9d_functions():
    """Returns 3 diverse 9D functions with training gaps."""
    functions = {
        "sin_cos_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2) / 450,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "multi_modal_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: (np.cos(2*x1) + np.cos(2*x2) + np.cos(2*x3) + np.cos(2*x4) + np.cos(2*x5) + np.cos(2*x6) + np.cos(2*x7) + np.cos(2*x8) + np.cos(2*x9)) + (x1+x2+x3+x4+x5+x6+x7+x8+x9)/10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_10d_functions():
    """Returns 3 diverse 10D functions with training gaps."""
    functions = {
        "sin_cos_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2 + x10**2) / 500,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "friedman_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: 10 * np.sin(np.pi * x1 * x2 / 100) + 20 * (x3 / 10 - 0.5)**2 + 10 * x4 / 10 + 5 * x5 / 10 + (x6+x7+x8+x9+x10)/10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions
