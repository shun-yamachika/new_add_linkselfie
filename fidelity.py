
# fidelity.py
# Keep fidelity/importance generators small and explicit.
import random

def generate_fidelity_list_random(path_num: int, alpha: float = 0.95, beta: float = 0.85, variance: float = 0.04):
    """
    Generate `path_num` fidelities.
    One "good" link around alpha, the rest around beta, clipped to [0.5, 1.0].
    """
    vals = []
    for i in range(path_num):
        mu = alpha if i == 0 else beta
        # simple Gaussian around mu, but clipped
        v = random.gauss(mu, variance**0.5)
        v = max(0.5, min(1.0, v))
        vals.append(v)
    # shuffle so the "good" link is not always index 0
    random.shuffle(vals)
    return vals

def generate_importance_list_random(n: int, low: float = 0.5, high: float = 2.0):
    """Return a list of n importances I_n ~ Uniform[low, high]."""
    return [random.uniform(low, high) for _ in range(n)]
