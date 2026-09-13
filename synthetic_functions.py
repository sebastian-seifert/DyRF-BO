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

def ackley_func(*args, a=20.0, b=0.2, c=2*np.pi):
    """Vectorized Ackley function for arbitrary dimension d = len(args)."""
    x = np.stack(args, axis=-1)
    d = x.shape[-1]
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e

def rosenbrock_func(*args, a=1.0, b=100.0):
    """Vectorized Rosenbrock function for arbitrary dimension d = len(args)."""
    x = np.stack(args, axis=-1)
    return np.sum(b * (x[..., 1:] - x[..., :-1]**2)**2 + (a - x[..., :-1])**2, axis=-1)

def hartmann_6d_func(x1, x2, x3, x4, x5, x6):
    """Vectorized Hartmann 6D benchmark function."""
    x = np.stack([x1, x2, x3, x4, x5, x6], axis=-1)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    y = np.zeros(x.shape[:-1])
    for i in range(4):
        diff = x - P[i]
        y -= alpha[i] * np.exp(-np.sum(A[i] * (diff**2), axis=-1))
    return y

def get_2d_functions():
    """Returns 7 diverse 2D functions with training gaps."""
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
        "ackley_2d": {
            "func": lambda x1, x2: ackley_func(x1, x2),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "rosenbrock_2d": {
            "func": lambda x1, x2: rosenbrock_func(x1, x2),
            "gap": (0.5, 1.5),
            "range": (-2.0, 2.0),
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
    """Returns 5 diverse 4D functions with training gaps."""
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
        "ackley_4d": {
            "func": lambda x1, x2, x3, x4: ackley_func(x1, x2, x3, x4),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "rosenbrock_4d": {
            "func": lambda x1, x2, x3, x4: rosenbrock_func(x1, x2, x3, x4),
            "gap": (0.5, 1.5),
            "range": (-2.0, 2.0),
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
    """Returns 4 diverse 6D functions with training gaps."""
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
        "gaussian_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: np.exp(-(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2) / 25),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "friedman_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: 10 * np.sin(np.pi * x1 * x2 / 100) + 20 * (x3 / 10 - 0.5)**2 + 10 * x4 / 10 + 5 * x5 / 10 + x6 / 10,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "hartmann_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: hartmann_6d_func(x1, x2, x3, x4, x5, x6),
            "gap": (0.45, 0.75),
            "range": (0.0, 1.0),
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
        "gaussian_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: np.exp(-(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2 + x10**2) / 40),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
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
            "bounds": (0, 10),
            "dim": 2,
        },
        "hartmann3": {
            "func": hartmann3_func,
            "gap": (0.4, 0.6),
            "range": (0, 1),
            "bounds": (0, 1),
            "dim": 3,
        },
        "hartmann6": {
            "func": hartmann6_func,
            "gap": (0.4, 0.6),
            "range": (0, 1),
            "bounds": (0, 1),
            "dim": 6,
        }
    }
    return functions


def get_all_normal_functions():
    """
    Returns dictionary of all 41 normal synthetic benchmark functions across 1D-15D.
    Excludes the 10 special named functions.
    Every function entry contains 'func', 'gap', 'range', 'bounds', and 'dim'.
    """
    normal_funcs = {}

    # 1D (5)
    for name, cfg in get_1d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 1
        normal_funcs[name] = entry

    # 2D (5 normal)
    f2 = get_2d_functions()
    for name in ["sin_cos", "quadratic", "sin_sum_mod", "gaussian", "abs_sin"]:
        entry = dict(f2[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 2
        normal_funcs[name] = entry

    # 3D (5)
    for name, cfg in get_3d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 3
        normal_funcs[name] = entry

    # 4D (3 normal)
    f4 = get_4d_functions()
    for name in ["sin_cos_4d", "quadratic_4d", "sin_sum_4d"]:
        entry = dict(f4[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 4
        normal_funcs[name] = entry

    # 5D (3)
    for name, cfg in get_5d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 5
        normal_funcs[name] = entry

    # 6D (3 normal)
    f6 = get_6d_functions()
    for name in ["sin_cos_6d", "quadratic_6d", "gaussian_6d"]:
        entry = dict(f6[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 6
        normal_funcs[name] = entry

    # 7D (3)
    for name, cfg in get_7d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 7
        normal_funcs[name] = entry

    # 8D (3)
    for name, cfg in get_8d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 8
        normal_funcs[name] = entry

    # 9D (3)
    for name, cfg in get_9d_functions().items():
        entry = dict(cfg)
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 9
        normal_funcs[name] = entry

    # 10D (3 normal)
    f10 = get_10d_functions()
    for name in ["sin_cos_10d", "quadratic_10d", "gaussian_10d"]:
        entry = dict(f10[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 10
        normal_funcs[name] = entry

    # 11D - 15D (1 each)
    dim_getters = [
        (11, get_11d_functions),
        (12, get_12d_functions),
        (13, get_13d_functions),
        (14, get_14d_functions),
        (15, get_15d_functions),
    ]
    for dim, getter in dim_getters:
        for name, cfg in getter().items():
            entry = dict(cfg)
            entry["bounds"] = entry.get("bounds", entry["range"])
            entry["dim"] = dim
            normal_funcs[name] = entry

    return normal_funcs


def get_special_functions():
    """
    Returns dictionary of the 10 special named benchmark functions:
    ackley_2d, rosenbrock_2d, ackley_4d, rosenbrock_4d, friedman_6d, hartmann_6d,
    friedman_10d, branin, hartmann3, hartmann6.
    Every function entry contains 'func', 'gap', 'range', 'bounds', and 'dim'.
    """
    special_funcs = {}

    f2 = get_2d_functions()
    for name in ["ackley_2d", "rosenbrock_2d"]:
        entry = dict(f2[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 2
        special_funcs[name] = entry

    f4 = get_4d_functions()
    for name in ["ackley_4d", "rosenbrock_4d"]:
        entry = dict(f4[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 4
        special_funcs[name] = entry

    f6 = get_6d_functions()
    for name in ["friedman_6d", "hartmann_6d"]:
        entry = dict(f6[name])
        entry["bounds"] = entry.get("bounds", entry["range"])
        entry["dim"] = 6
        special_funcs[name] = entry

    f10 = get_10d_functions()
    entry = dict(f10["friedman_10d"])
    entry["bounds"] = entry.get("bounds", entry["range"])
    entry["dim"] = 10
    special_funcs["friedman_10d"] = entry

    bh = get_branin_hartmann_functions()
    b_entry = dict(bh["branin"])
    b_entry["bounds"] = b_entry.get("bounds", b_entry["range"])
    b_entry["dim"] = 2
    special_funcs["branin"] = b_entry

    h3_entry = dict(bh["hartmann3"])
    h3_entry["bounds"] = h3_entry.get("bounds", h3_entry["range"])
    h3_entry["dim"] = 3
    special_funcs["hartmann3"] = h3_entry

    h6_entry = dict(bh["hartmann6"])
    h6_entry["bounds"] = h6_entry.get("bounds", h6_entry["range"])
    h6_entry["dim"] = 6
    special_funcs["hartmann6"] = h6_entry

    return special_funcs


def get_all_epistemic_ood_benchmark_functions():
    """
    Returns combined master dictionary of all 51 benchmark functions (41 normal + 10 special).
    """
    return {**get_all_normal_functions(), **get_special_functions()}

