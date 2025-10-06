# main.py — ほぼ従来どおり。seed だけ CLI から渡せる最小変更版

import os
import random
import argparse
from multiprocessing.pool import Pool

from evaluation import (
    plot_accuracy_vs_budget,
    plot_value_vs_used,
    plot_value_vs_budget_target,
    plot_widthsum_alllinks_vs_budget,
    plot_minwidthsum_perpair_vs_budget,
    plot_widthsum_alllinks_weighted_vs_budget,
    plot_minwidthsum_perpair_weighted_vs_budget,
    plot_importance_discovery_value_vs_budget,
)

# ---- 乱数初期化（標準/NumPy/NetSquid） ----
def set_random_seed(seed: int = 12):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import netsquid as ns
        ns.set_random_state(seed)
    except Exception:
        pass

def main():
    # ★ 変更点は seed だけ（デフォルト12、--seed で上書き可能）
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12, help="random seed")
    args = ap.parse_args()

    set_random_seed(args.seed)

    # 従来どおりの固定設定（必要ならここを好きに書き換えてね）
    noise_model = "Depolar"
    budget_list = [3000, 4000, 5000, 6000, 7000, 8000]
    node_path_list = [5, 5, 5]
    importance_list = [0.3,0.6,0.9]
    importance_mode = "uniform"          # "fix" or "uniform"
    importance_uniform = (0.0, 1.0)    # when importance_mode == "uniform"
    bounces = [1, 2, 3, 4]
    repeat = 25
    delta = 0.01
    scheduler_names = ["Greedy", "LNaive"]

    print("==== Config ====")
    print(f"noise_model={noise_model}")
    print(f"budget_list={budget_list}")
    print(f"node_path_list={node_path_list}")
    print(f"importance_mode={importance_mode}, importance_uniform={importance_uniform}")
    print(f"importance_list={importance_list}")
    print(f"bounces={bounces}, repeat={repeat}")
    print(f"seed={args.seed}")
    print("================\n")

    os.makedirs("outputs", exist_ok=True)

    # 従来通り Pool を使って並列実行
    jobs = []
    with Pool() as p:
        jobs.append(p.apply_async(
            plot_accuracy_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_value_vs_used,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_value_vs_budget_target,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_widthsum_alllinks_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_minwidthsum_perpair_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_widthsum_alllinks_weighted_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_minwidthsum_perpair_weighted_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "importance_mode": importance_mode, "importance_uniform": importance_uniform, "seed": args.seed, "verbose": True}
        ))
        jobs.append(p.apply_async(
            plot_importance_discovery_value_vs_budget,
            args=(budget_list, scheduler_names, noise_model, node_path_list, importance_list, bounces, repeat),
            kwds={"y": 0.10, "delta": delta, "use_f": "Fhat",
                  "importance_mode": importance_mode, "importance_uniform": importance_uniform,
                  "seed": args.seed, "verbose": True}
        ))

        for j in jobs:
            j.get()

if __name__ == "__main__":
    main()
