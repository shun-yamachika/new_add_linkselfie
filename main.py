# main.py — Run experiments: budget, random-gap, fixed-gap, and #pairs
import os
import random

import numpy as np

from evaluation import plot_accuracy_vs_budget
from evaluationgap import (
    plot_accuracy_vs_gap,          # random jittered (alpha - beta = gap)
    plot_accuracy_vs_gap_fixgap,   # fixed arithmetic sequence (max -> step -gap)
)
from evaluationpair import plot_accuracy_vs_pairs


# =====================
# Simple configuration
# =====================
# Toggle which experiments to run
RUN_BUDGET     = True
RUN_GAP_RANDOM = True   # existing: alpha - beta = gap (randomized fidelities)
RUN_GAP_FIX    = True   # NEW: fixed arithmetic sequence fidelities
RUN_PAIRS      = True

# Global seed + common settings
SEED        = 12
NOISE_MODEL = "Depolar"
BOUNCES     = (1, 2, 3, 4)
REPEAT      = 10
SCHEDULERS  = ['LNaive', 'Greedy']

# Importance settings
IMPORTANCE_MODE    = "fixed"      # "fixed" or "uniform"
IMPORTANCE_UNIFORM = (0.0, 1.0)   # only used if IMPORTANCE_MODE == "uniform"

# -----------------
# 1) Budget sweep
# -----------------
BUDGET_LIST    = [3000, 4000, 5000, 6000, 7000, 8000]
NODE_PATH_LIST = [5, 5, 5]                 # candidate-link counts per destination pair
IMPORTANCE_LIST = [1.0, 1.0, 1.0]          # used when IMPORTANCE_MODE == "fixed"

# --------------
# 2) Gap sweeps
# --------------
# (a) Random (alpha - beta = gap) version
GAP_LIST_RANDOM = [0.005, 0.01, 0.02, 0.03]
ALPHA_BASE      = 0.95
VARIANCE        = 0.10
C_GAP_TOTAL     = 5000  # total budget per gap point

# (b) Fixed arithmetic-sequence version
GAP_LIST_FIX = [0.005, 0.01, 0.02, 0.03]
FIDELITY_MAX = 1.0      # sequence starts at this max and steps down by 'gap'

# --------------------
# 3) #Pairs (N) sweep
# --------------------
PAIRS_LIST      = [1, 2, 3, 4, 5, 6]       # number of destination pairs
PATHS_PER_PAIR  = 5                         # candidate links per pair
C_PAIRS_TOTAL   = 6000                      # total budget per N


def set_random_seed(seed: int = 12):
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import netsquid as ns
        ns.set_random_state(seed)
    except Exception:
        pass


def main():
    set_random_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    # (1) Budget vs Accuracy
    if RUN_BUDGET:
        plot_accuracy_vs_budget(
            budget_list=BUDGET_LIST,
            scheduler_names=SCHEDULERS,
            noise_model=NOISE_MODEL,
            node_path_list=NODE_PATH_LIST,
            importance_list=IMPORTANCE_LIST,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

    # (2a) Gap vs Accuracy (randomized: alpha - beta = gap)
    if RUN_GAP_RANDOM:
        plot_accuracy_vs_gap(
            gap_list=GAP_LIST_RANDOM,
            scheduler_names=SCHEDULERS,
            noise_model=NOISE_MODEL,
            node_path_list=NODE_PATH_LIST,
            importance_list=IMPORTANCE_LIST,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            alpha_base=ALPHA_BASE,
            variance=VARIANCE,
            C_total_override=C_GAP_TOTAL,
            verbose=True,
        )

    # (2b) Gap vs Accuracy (fixed arithmetic sequence)
    if RUN_GAP_FIX:
        plot_accuracy_vs_gap_fixgap(
            gap_list=GAP_LIST_FIX,
            scheduler_names=SCHEDULERS,
            noise_model=NOISE_MODEL,
            node_path_list=NODE_PATH_LIST,
            importance_list=IMPORTANCE_LIST,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,                 # used only if IMPORTANCE_MODE == "uniform"
            fidelity_max=FIDELITY_MAX, # sequence head value
            C_total_override=C_GAP_TOTAL,
            verbose=True,
        )

    # (3) #Pairs vs Accuracy
    if RUN_PAIRS:
        plot_accuracy_vs_pairs(
            pairs_list=PAIRS_LIST,
            paths_per_pair=PATHS_PER_PAIR,
            C_total=C_PAIRS_TOTAL,
            scheduler_names=SCHEDULERS,
            noise_model=NOISE_MODEL,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )


if __name__ == "__main__":
    main()
