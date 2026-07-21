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

def get_11d_functions():
    """Returns 1 diverse 11D function with training gaps."""
    functions = {
        "sin_cos_11d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10) * np.sin(x11),
            "gap": (4, 6),
            "range": (0, 10),
        }
    }
    return functions

def get_12d_functions():
    """Returns 1 diverse 12D function with training gaps."""
    functions = {
        "sin_cos_12d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10) * np.sin(x11) * np.cos(x12),
            "gap": (4, 6),
            "range": (0, 10),
        }
    }
    return functions

def get_13d_functions():
    """Returns 1 diverse 13D function with training gaps."""
    functions = {
        "sin_cos_13d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10) * np.sin(x11) * np.cos(x12) * np.sin(x13),
            "gap": (4, 6),
            "range": (0, 10),
        }
    }
    return functions

def get_14d_functions():
    """Returns 1 diverse 14D function with training gaps."""
    functions = {
        "sin_cos_14d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10) * np.sin(x11) * np.cos(x12) * np.sin(x13) * np.cos(x14),
            "gap": (4, 6),
            "range": (0, 10),
        }
    }
    return functions

def get_15d_functions():
    """Returns 1 diverse 15D function with training gaps."""
    functions = {
        "sin_cos_15d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10) * np.sin(x11) * np.cos(x12) * np.sin(x13) * np.cos(x14) * np.sin(x15),
            "gap": (4, 6),
            "range": (0, 10),
        }
    }
    return functions


def branin_func(x1, x2):
    """
    Branin-Hoo 2D benchmark function.
    Inputs in [0, 10] mapped to x1_real in [-5, 10] and x2_real in [0, 15].
    Global minimum value f(x*) = 0.397887.
    """
    x1_real = x1 - 5.0
    x2_real = 1.5 * x2
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2_real - b * x1_real**2 + c * x1_real - r)**2 + s * (1.0 - t) * np.cos(x1_real) + s


def hartmann3_func(x1, x2, x3):
    """
    Hartmann 3D benchmark function.
    Inputs x_j in [0, 1]. Global minimum value f(x*) = -3.86278.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [3815, 7636, 3165]
    ])
    
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    x3_arr = np.asarray(x3)
    
    X = np.stack([x1_arr, x2_arr, x3_arr], axis=-1)
    res = np.zeros(X.shape[:-1], dtype=np.float64)
    for i in range(4):
        diff = X - P[i]
        res += alpha[i] * np.exp(-np.sum(A[i] * diff**2, axis=-1))
    return -res


def hartmann6_func(x1, x2, x3, x4, x5, x6):
    """
    Hartmann 6D benchmark function.
    Inputs x_j in [0, 1]. Global minimum value f(x*) = -3.32237.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    x1_arr = np.asarray(x1)
    x2_arr = np.asarray(x2)
    x3_arr = np.asarray(x3)
    x4_arr = np.asarray(x4)
    x5_arr = np.asarray(x5)
    x6_arr = np.asarray(x6)
    
    X = np.stack([x1_arr, x2_arr, x3_arr, x4_arr, x5_arr, x6_arr], axis=-1)
    res = np.zeros(X.shape[:-1], dtype=np.float64)
    for i in range(4):
        diff = X - P[i]
        res += alpha[i] * np.exp(-np.sum(A[i] * diff**2, axis=-1))
    return -res


def get_branin_hartmann_functions():
    """
    Returns dedicated dictionary containing classic Branin (2D), Hartmann-3D,
    and Hartmann-6D benchmark functions for BO and UQ evaluation.
    """
    functions = {
        "branin": {
            "func": branin_func,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "hartmann3": {
            "func": hartmann3_func,
            "gap": (0.4, 0.6),
            "range": (0, 1),
        },
        "hartmann6": {
            "func": hartmann6_func,
            "gap": (0.4, 0.6),
            "range": (0, 1),
        }
    }
    return functions
