# main.py — Run experiments: budget, random-gap, fixed-gap, and #pairs
import os
import random
import numpy as np

from evaluation import plot_accuracy_vs_budget, plot_value_vs_used
from evaluationgap import plot_accuracy_vs_gap, plot_accuracy_vs_gap_fixgap
from evaluationpair import plot_accuracy_vs_pairs

# =====================
# Simple configuration
# =====================
# Toggle which experiments to run
RUN_BUDGET     = False   
RUN_GAP_RANDOM = False   
RUN_GAP_FIX    = True
RUN_PAIRS      = False

# Global seed + common settings
SEED        = 13
NOISE_MODEL = "Depolar"
BOUNCES     = (1, 2, 3, 4)
REPEAT      = 5
SCHEDULERS  = ["LNaive","Greedy"]

# Importance settings
# NOTE: "uniform" のときは *_IMPORTANCES は使われず、各リピートで U[a,b] から再サンプルされます
IMPORTANCE_MODE    = "uniform"          # "fixed" or "uniform"
IMPORTANCE_UNIFORM = (0.0, 1.0)         # used only if IMPORTANCE_MODE == "uniform"

# -----------------
# 1) Budget sweep
# -----------------
BUDGET_LIST         = [4000]
BUDGET_NODE_PATHS   = [4,4,4,4,4,4,4,4,4,4]         
BUDGET_IMPORTANCES  = [1,1,1,1,1,1,1,1,1,0.1]   # Budget専用: IMPORTANCE_MODE == "fixed" のときのみ使用

# --------------
# 2) Gap sweeps
# --------------
# (a) Random (alpha - beta = gap) version
GAP_LIST_RANDOM        = [0.025, 0.05, 0.075, 0.10, 0.125, 0.150]
ALPHA_BASE             = 0.95
VARIANCE               = 0.025
C_GAP_TOTAL            = 10000           # total budget per gap point
GAP_RANDOM_NODE_PATHS  = [4,4,4,4,4]  
GAP_RANDOM_IMPORTANCES = [0.3, 0.6, 0.9, 0.6, 0.3]  # fixed時のみ使用（長さ=ペア数に合わせる）

# (b) Fixed arithmetic-sequence version
GAP_LIST_FIX        = [0.01, 0.02,0.05,0.1]
FIDELITY_MAX        = 1.0              # sequence starts at this max and steps down by 'gap'
GAP_FIX_NODE_PATHS  = [4,4,4,4]  
GAP_FIX_IMPORTANCES = [0.3, 0.6, 0.9, 0.3]    # fixed時のみ使用

# --------------------
# 3) #Pairs (N) sweep
# --------------------
PAIRS_LIST      = [3, 4, 5, 6, 7, 8]   # number of destination pairs
PATHS_PER_PAIR  = 4                    # candidate links per pair
C_PAIRS_TOTAL   = 10000                # total budget per N

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
            node_path_list=BUDGET_NODE_PATHS,
            importance_list=BUDGET_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )
        # 価値関数プロット（必要ならコメント解除）
        plot_value_vs_used(
            budget_list=BUDGET_LIST,
            scheduler_names=SCHEDULERS,
            noise_model=NOISE_MODEL,
            node_path_list=BUDGET_NODE_PATHS,
            importance_list=BUDGET_IMPORTANCES,
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
            node_path_list=GAP_RANDOM_NODE_PATHS,
            importance_list=GAP_RANDOM_IMPORTANCES,
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
            node_path_list=GAP_FIX_NODE_PATHS,
            importance_list=GAP_FIX_IMPORTANCES,
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
