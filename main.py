# main.py — Run experiments across multiple noise models
import os
import random
import numpy as np

from evaluation import (
    plot_accuracy_vs_budget,
    plot_value_vs_used,
    plot_value_vs_budget,
)
from evaluationgap import (
    plot_accuracy_vs_gap,
    plot_accuracy_vs_gap_fixgap,
    plot_value_vs_gap,
    plot_value_vs_gap_fixgap,
)
from evaluationpair import (
    plot_accuracy_vs_pairs,
    plot_value_vs_pairs,
)

# =====================
# Simple configuration
# =====================
# Toggle which experiments to run
RUN_BUDGET     = False
RUN_GAP_RANDOM = False
RUN_GAP_FIX    = False
RUN_PAIRS      = True

# Global seed + common settings
SEED          = 13
NOISE_MODELS  = ["Depolar", "Dephase", "AmplitudeDamping", "BitFlip"]
BOUNCES       = (1, 2, 3, 4)
REPEAT        = 5
SCHEDULERS    = ["LNaive", "Groups", "Greedy", "WNaive"]

# Importance settings
# NOTE: "uniform" のときは *_IMPORTANCES は使われず、各リピートで U[a,b] から再サンプルされます
IMPORTANCE_MODE    = "fixed"          # "fixed" or "uniform"
IMPORTANCE_UNIFORM = (0.0, 1.0)       # used only if IMPORTANCE_MODE == "uniform"

# -----------------
# 1) Budget sweep
# -----------------
BUDGET_LIST         = [500, 1000, 1500, 2000, 2500, 3000]
BUDGET_NODE_PATHS   = [4,4,4,4]
BUDGET_IMPORTANCES  = [0.2,0.4,0.6,0.8]   # IMPORTANCE_MODE == "fixed" のときのみ使用

# --------------
# 2) Gap sweeps
# --------------
# (a) Random (alpha - beta = gap) version
GAP_LIST_RANDOM        = [0.05,0.10,0.15,0.20]
ALPHA_BASE             = 0.95
VARIANCE               = 0.025
C_GAP_TOTAL            = 3000           # total budget per gap point
GAP_RANDOM_NODE_PATHS  = [4, 4, 4, 4, 4]
GAP_RANDOM_IMPORTANCES = [0.1,0.3,0.5,0.7,0.9]  # fixed時のみ使用（長さ=ペア数に合わせる）

# (b) Fixed arithmetic-sequence version
GAP_LIST_FIX        = [0.01, 0.05, 0.01, 0.15, 0.20]
FIDELITY_MAX        = 1.0              # sequence starts at this max and steps down by 'gap'
GAP_FIX_NODE_PATHS  = [4, 4, 4, 4, 4]
GAP_FIX_IMPORTANCES = [0.1,0.3,0.5,0.7,0.9]    # fixed時のみ使用

# --------------------
# 3) #Pairs (N) sweep
# --------------------
PAIRS_LIST      = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]   # number of destination pairs
PATHS_PER_PAIR  = 4                    # candidate links per pair
C_PAIRS_TOTAL   = 3000                # total budget per N

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

def run_for_noise_model(noise_model: str):
    """1つのノイズモデルについて全実験を実行"""
    # (1) Budget vs Accuracy/Value
    if RUN_BUDGET:
        # タイポ修正済み: importance_list
        plot_accuracy_vs_budget(
            budget_list=BUDGET_LIST,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            node_path_list=BUDGET_NODE_PATHS,
            importance_list=BUDGET_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

        plot_value_vs_used(
            budget_list=BUDGET_LIST,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            node_path_list=BUDGET_NODE_PATHS,
            importance_list=BUDGET_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

        plot_value_vs_budget(
            budget_list=BUDGET_LIST,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            node_path_list=BUDGET_NODE_PATHS,
            importance_list=BUDGET_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

    # (2a) Gap vs Accuracy/Value (randomized)
    if RUN_GAP_RANDOM:
        plot_accuracy_vs_gap(
            gap_list=GAP_LIST_RANDOM,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
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

        plot_value_vs_gap(
            gap_list=GAP_LIST_RANDOM,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
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

    # (2b) Gap vs Accuracy/Value (fixed arithmetic sequence)
    if RUN_GAP_FIX:
        plot_accuracy_vs_gap_fixgap(
            gap_list=GAP_LIST_FIX,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            node_path_list=GAP_FIX_NODE_PATHS,
            importance_list=GAP_FIX_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,                 # uniform時のみ意味あり
            fidelity_max=FIDELITY_MAX, # 等差列の先頭値
            C_total_override=C_GAP_TOTAL,
            verbose=True,
        )

        plot_value_vs_gap_fixgap(
            gap_list=GAP_LIST_FIX,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            node_path_list=GAP_FIX_NODE_PATHS,
            importance_list=GAP_FIX_IMPORTANCES,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,                 # uniform時のみ意味あり
            fidelity_max=FIDELITY_MAX, # 等差列の先頭値
            C_total_override=C_GAP_TOTAL,
            verbose=True,
        )

    # (3) #Pairs vs Accuracy/Value
    if RUN_PAIRS:
        plot_accuracy_vs_pairs(
            pairs_list=PAIRS_LIST,
            paths_per_pair=PATHS_PER_PAIR,
            C_total=C_PAIRS_TOTAL,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

        plot_value_vs_pairs(
            pairs_list=PAIRS_LIST,
            paths_per_pair=PATHS_PER_PAIR,
            C_total=C_PAIRS_TOTAL,
            scheduler_names=SCHEDULERS,
            noise_model=noise_model,
            bounces=BOUNCES,
            repeat=REPEAT,
            importance_mode=IMPORTANCE_MODE,
            importance_uniform=IMPORTANCE_UNIFORM,
            seed=SEED,
            verbose=True,
        )

def main():
    set_random_seed(SEED)
    os.makedirs("outpdf", exist_ok=True)
    os.makedirs("outpickle", exist_ok=True)
    for nm in NOISE_MODELS:
        print(f"\n===== Run experiments for noise model: {nm} =====", flush=True)
        run_for_noise_model(nm)

if __name__ == "__main__":
    main()
