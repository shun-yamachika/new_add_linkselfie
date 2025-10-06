#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — evaluation.py の各種プロットを一括実行
  Unweighted:
    1) plot_accuracy_vs_budget
    2) plot_value_vs_used
    3) plot_value_vs_budget_target
    4) plot_widthsum_alllinks_vs_budget
    5) plot_minwidthsum_perpair_vs_budget
  Weighted（重要度 I_d を幅に掛ける版）:
    6) plot_widthsum_alllinks_weighted_vs_budget
    7) plot_minwidthsum_perpair_weighted_vs_budget

出力:
  - 生データ pickle -> ./outputs/
  - 図 PDF -> カレントディレクトリ
"""

from multiprocessing.pool import Pool
import os
import random

# 任意：あなたの環境のユーティリティ。無ければフォールバック。
try:
    from utils import set_random_seed
except Exception:
    def set_random_seed(seed: int = 12):
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass

# evaluation 側のプロット関数
from evaluation import (
    # Accuracy / Value 系
    plot_accuracy_vs_budget,
    plot_value_vs_used,
    plot_value_vs_budget_target,

    # 幅（UB-LB）系 - Unweighted
    plot_widthsum_alllinks_vs_budget,
    plot_minwidthsum_perpair_vs_budget,

    # 幅（UB-LB）系 - Weighted (× I_d)
    plot_widthsum_alllinks_weighted_vs_budget,
    plot_minwidthsum_perpair_weighted_vs_budget,
)


def main():
    # ===== 実験パラメータ =====
    set_random_seed(12)
    num_workers      = max(1, (os.cpu_count() or 4) // 2)
    noise_model_list = ["Depolar"]              # 例: ["Depolar", "Dephase"]
    scheduler_names  = ["LNaive", "Greedy"]     # 実装済みスケジューラ名に合わせて
    node_path_list   = [5, 5, 5]                # ペアごとのリンク本数
    importance_list  = [0.3, 0.6, 0.9]          # value系＆weighted幅系で使用
    budget_list      = [3000, 6000, 9000, 12000, 15000, 18000]
    bounces          = (1, 2, 3, 4)             # 測定深さ候補（あなたの定義に従う）
    repeat           = 10                       # 反復回数（精度と時間のトレードオフ）
    delta            = 0.1                      # 幅用の信頼度パラメータ（Hoeffding）

    print("=== Config ===")
    print(f"workers={num_workers}, noise_models={noise_model_list}")
    print(f"schedulers={scheduler_names}")
    print(f"node_path_list={node_path_list}, importance_list={importance_list}")
    print(f"budgets={budget_list}, bounces={bounces}, repeat={repeat}, delta={delta}")
    print("================\n")

    # ===== 実行キュー =====
    p = Pool(processes=num_workers)
    jobs = []

    for noise_model in noise_model_list:
        # --- Accuracy ---
        jobs.append(p.apply_async(
            plot_accuracy_vs_budget,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"verbose": True}
        ))

        # --- Value: x=used（実コスト平均） ---
        jobs.append(p.apply_async(
            plot_value_vs_used,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"verbose": True}
        ))

        # --- Value: x=target（指定予算） ---
        jobs.append(p.apply_async(
            plot_value_vs_budget_target,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"verbose": True}
        ))

        # --- Width (UB-LB) Unweighted: 全リンク総和 ---
        jobs.append(p.apply_async(
            plot_widthsum_alllinks_vs_budget,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "verbose": True}
        ))

        # --- Width (UB-LB) Unweighted: ペア最小幅の総和 ---
        jobs.append(p.apply_async(
            plot_minwidthsum_perpair_vs_budget,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "verbose": True}
        ))

        # --- Width (UB-LB) Weighted: 全リンク I_d·幅 総和 ---
        jobs.append(p.apply_async(
            plot_widthsum_alllinks_weighted_vs_budget,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "verbose": True}
        ))

        # --- Width (UB-LB) Weighted: ペアごとの I_d·最小幅 総和 ---
        jobs.append(p.apply_async(
            plot_minwidthsum_perpair_weighted_vs_budget,
            args=(budget_list, scheduler_names, noise_model,
                  node_path_list, importance_list, bounces, repeat),
            kwds={"delta": delta, "verbose": True}
        ))

    # ===== 実行 & 同期 =====
    p.close()
    p.join()
    for j in jobs:
        j.get()

    print("\nAll jobs finished.")
    print("Pickles -> ./outputs/,  PDF -> カレントディレクトリ に保存されます。")


if __name__ == "__main__":
    main()
