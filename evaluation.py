# evaluation.py — Run shared sweep once; all plots aggregate from cache (reproducible with seed)

import math
import os
import pickle
import time
import shutil
import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# metrics / viz を外出し（UNIX的分離）
from metrics.widths import (
    ci_radius_hoeffding,
    sum_weighted_widths_all_links,
    sum_weighted_min_widths_perpair,
    sum_widths_all_links,
    sum_minwidths_perpair,
)
from viz.plots import mean_ci95, plot_with_ci_band

from network import QuantumNetwork
from schedulers import run_scheduler  # スケジューラ呼び出し

import matplotlib as mpl
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["savefig.bbox"] = "tight"   # すべての savefig に適用

# ---- Matplotlib style（互換性重視: hex色 & 無難な記号類）----
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
mpl.rcParams["font.size"] = 20

default_cycler = (
    cycler(color=["#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#a65628"])
    + cycler(marker=["s", "v", "o", "x", "*", "+"])
    + cycler(linestyle=[":", "--", "-", "-.", "--", ":"])
)
plt.rc("axes", prop_cycle=default_cycler)

# =========================
# Fidelity generators (old API; keep for compatibility)
# =========================
def generate_fidelity_list_avg_gap(path_num):
    result = []
    fidelity_max = 1
    fidelity_min = 0.9
    gap = (fidelity_max - fidelity_min) / path_num
    fidelity = fidelity_max
    for _ in range(path_num):
        result.append(fidelity)
        fidelity -= gap
    assert len(result) == path_num
    return result

def generate_fidelity_list_fix_gap(path_num, gap, fidelity_max=1):
    result = []
    fidelity = fidelity_max
    for _ in range(path_num):
        result.append(fidelity)
        fidelity -= gap
    assert len(result) == path_num
    return result

def generate_fidelity_list_random(path_num, alpha=0.95, beta=0.85, variance=0.1):
    """(非決定版) Generate `path_num` links with a guaranteed top-1 gap."""
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        result = []
        for i in range(path_num):
            mu = mean[i]
            # [0.8, 1.0] の範囲に入るまでサンプリング
            while True:
                r = np.random.normal(mu, variance)
                if 0.8 <= r <= 1.0:
                    break
            result.append(r)
        assert len(result) == path_num
        sorted_res = sorted(result, reverse=True)
        if sorted_res[0] - sorted_res[1] > 0.02:
            return result

# 再現性のため：rng を使ったバージョン
def _generate_fidelity_list_random_rng(rng, path_num, alpha=0.95, beta=0.85, variance=0.1):
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        res = []
        for mu in mean:
            # [0.8, 1.0] の範囲に入るまでサンプリング
            while True:
                r = rng.normal(mu, variance)
                if 0.8 <= r <= 1.0:
                    break
            res.append(float(r))
        sorted_res = sorted(res, reverse=True)
        if sorted_res[0] - sorted_res[1] > 0.02:
            return res

# =========================
# Progress helpers
# =========================
def _start_timer():
    return {"t0": time.time(), "last": time.time()}

def _tick(timer):
    now = time.time()
    dt_total = now - timer["t0"]
    dt_step = now - timer["last"]
    timer["last"] = now
    return dt_total, dt_step

def _log(msg):
    print(msg, flush=True)

# =========================
# Shared sweep (cache) helpers with file lock
# =========================
def _sweep_signature(budget_list, scheduler_names, noise_model,
                     node_path_list, importance_list, bounces, repeat,
                     importance_mode="fixed", importance_uniform=(0.0, 1.0), seed=None):
    payload = {
        "budget_list": list(budget_list),
        "scheduler_names": list(scheduler_names),
        "noise_model": str(noise_model),
        "node_path_list": list(node_path_list),
        "importance_list": list(importance_list) if importance_list is not None else None,
        "importance_mode": str(importance_mode),
        "importance_uniform": list(importance_uniform) if importance_uniform is not None else None,
        "bounces": list(bounces),
        "repeat": int(repeat),
        "seed": int(seed) if seed is not None else None,
        "version": 3,  # 鍵仕様更新（seed/importance含む）
    }
    sig = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return payload, sig

def _shared_sweep_path(noise_model, sig):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"shared_sweep_{noise_model}_{sig}.pickle")

def _run_or_load_shared_sweep(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10,
    importance_mode="fixed", importance_uniform=(0.0, 1.0),
    seed=None,
    verbose=True, print_every=1,
):
    config, sig = _sweep_signature(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list, bounces, repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed
    )
    cache_path = _shared_sweep_path(noise_model, sig)
    lock_path  = cache_path + ".lock"
    STALE_LOCK_SECS = 6 * 60 * 60        # 6時間無更新ならロック回収
    HEARTBEAT_EVERY = 5.0                # 生成側のロック更新間隔（秒）

    rng = np.random.default_rng(seed)    # ★ 乱数生成器（再現性の核）

    # 既存キャッシュがあれば即ロード
    if os.path.exists(cache_path):
        if verbose: _log(f"[shared] Load cached sweep: {os.path.basename(cache_path)}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # --- ロック獲得（初回生成は1プロセスのみ）---
    got_lock = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
            break
        except FileExistsError:
            # 他プロセスが生成中：完成を待つ（タイムアウトなし）
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # スタックロック検出：長時間 mtime 更新がない場合は回収
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = 0
            if age > STALE_LOCK_SECS:
                if verbose: _log("[shared] Stale lock detected. Removing...")
                try: os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue

            # 進捗待ち
            if verbose: _log("[shared] Waiting for cache to be ready...")
            time.sleep(1.0)

    try:
        if verbose: _log(f"[shared] Run sweep and cache to: {os.path.basename(cache_path)}")

        data = {name: {k: [] for k in range(len(budget_list))} for name in scheduler_names}
        last_hb = time.time()

        # === 1リピート=1トポロジを固定し、そのまま全ての budget を評価 ===
        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                _log(f"[shared] Repeat {r+1}/{repeat} (fixed topology)")

            # この repeat 内で使い回す固定トポロジ（rng版）
            fidelity_bank = [_generate_fidelity_list_random_rng(rng, n) for n in node_path_list]

            # importance per repeat (fixed or uniform sample; rng使用)
            if str(importance_mode).lower() == "uniform":
                a, b = map(float, importance_uniform)
                imp_list_r = [float(rng.uniform(a, b)) for _ in node_path_list]
            else:
                imp_list_r = list(importance_list)

            def network_generator(path_num, pair_idx):
                return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

            # 同一トポロジのまま、予算だけを変えて実行
            for k, C_total in enumerate(budget_list):
                if verbose:
                    _log(f"=== [SHARED {noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===")

                # ハートビート（ロックの mtime を更新）
                now = time.time()
                if now - last_hb >= HEARTBEAT_EVERY:
                    try:
                        os.utime(lock_path, None)
                    except FileNotFoundError:
                        pass
                    last_hb = now

                for name in scheduler_names:
                    per_pair_results, total_cost, per_pair_details = run_scheduler(
                        node_path_list=node_path_list, importance_list=imp_list_r,
                        scheduler_name=name,
                        bounces=list(bounces),
                        C_total=int(C_total),
                        network_generator=network_generator,
                        return_details=True,
                    )
                    data[name][k].append({
                        "per_pair_results": per_pair_results,
                        "per_pair_details": per_pair_details,
                        "total_cost": total_cost,
                        "importance_list": imp_list_r
                    })

        payload = {"config": config, "budget_list": list(budget_list), "data": data}

        # アトミック書き込み
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)

        return payload

    finally:
        if got_lock:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

# =========================
# 1) Accuracy: 平均のみ（CIなし）
# =========================
def plot_accuracy_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10,
    importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_accuracy_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"accs": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_results = run["per_pair_results"]

                # --- 修正版ここから ---
                vals = []
                for r in per_pair_results:
                    if isinstance(r, tuple):
                        c = r[0]
                    elif isinstance(r, (int, float, bool)):
                        c = bool(r)
                    else:
                        raise TypeError(f"per_pair_results element has unexpected type: {type(r)} -> {r}")
                    vals.append(1.0 if c else 0.0)
                acc = float(np.mean(vals)) if vals else 0.0
                # --- 修正版ここまで ---

                results[name]["accs"][k].append(acc)

    # plot
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, data in results.items():
        avg_accs = [float(np.mean(v)) if v else 0.0 for v in data["accs"]]
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, avg_accs, linewidth=2.0, label=label)
    ax.set_xlabel("Total Budget (C)")
    ax.set_ylabel("Average Correctness")
    ax.grid(True); ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)
    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 2) Value vs Used（x=実コスト平均）
# =========================
def plot_value_vs_used(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_value_vs_used_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"values": [[] for _ in budget_list], "costs": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                total_cost = int(run["total_cost"])
                # value = Σ_d I_d Σ_l est(d,l) * alloc(d,l)
                value = 0.0
                I_used = run.get("importance_list", importance_list)
                for d, det in enumerate(per_pair_details):
                    alloc = det.get("alloc_by_path", {})
                    est   = det.get("est_fid_by_path", {})
                    inner = sum(float(est.get(l, 0.0)) * int(b) for l, b in alloc.items())
                    I = float(I_used[d]) if d < len(I_used) else 1.0
                    value += I * inner
                results[name]["values"][k].append(float(value))
                results[name]["costs"][k].append(total_cost)

    # plot
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, dat in results.items():
        xs = [float(np.mean(v)) if v else 0.0 for v in dat["costs"]]
        ys = [float(np.mean(v)) if v else 0.0 for v in dat["values"]]
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, ys, linewidth=2.0, marker="o", label=label)
    ax.set_xlabel("Total Measured Cost (used)")
    ax.set_ylabel("Total Value")
    ax.grid(True); ax.legend(title="Scheduler")
    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 3) Value vs Budget target（x=目標予算）
# =========================
def plot_value_vs_budget_target(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_value_vs_budget_target_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"values": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                value = 0.0
                I_used = run.get("importance_list", importance_list)
                for d, det in enumerate(per_pair_details):
                    alloc = det.get("alloc_by_path", {})
                    est   = det.get("est_fid_by_path", {})
                    inner = sum(float(est.get(l, 0.0)) * int(b) for l, b in alloc.items())
                    I = float(I_used[d]) if d < len(I_used) else 1.0
                    value += I * inner
                results[name]["values"][k].append(float(value))

    # plot
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        ys = [float(np.mean(v)) if v else 0.0 for v in dat["values"]]
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, ys, linewidth=2.0, marker="o", label=label)
    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Total Value")
    ax.grid(True); ax.legend(title="Scheduler")

    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 4) 幅（UB-LB）Unweighted: 全リンク総和
# =========================
def plot_widthsum_alllinks_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, delta=0.1,
    importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_widthsum_alllinks_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                v = sum_widths_all_links(per_pair_details, delta=delta)
                results[name]["sums"][k].append(v)

    # plot (mean ± 95%CI)
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        means, halfs = [], []
        for vals in dat["sums"]:
            m, h = mean_ci95(vals); means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, marker="o", label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)
    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Sum of (UB - LB) over all links")
    ax.grid(True); ax.legend(title="Scheduler")
    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 5) 幅（UB-LB）Unweighted: ペア最小幅の総和
# =========================
def plot_minwidthsum_perpair_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, delta=0.1,
    importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_minwidthsum_perpair_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                v = sum_minwidths_perpair(per_pair_details, delta=delta)
                results[name]["sums"][k].append(v)

    # plot (mean ± 95%CI)
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        means, halfs = [], []
        for vals in dat["sums"]:
            m, h = mean_ci95(vals); means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, marker="o", label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)
    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Sum over pairs of min (UB - LB)")
    ax.grid(True); ax.legend(title="Scheduler")

    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 6) 幅（UB-LB）Weighted: 全リンク I_d·幅 総和
# =========================
def plot_widthsum_alllinks_weighted_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, importance_mode="fixed", importance_uniform=(0.0,1.0), delta=0.1, seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_widthsum_alllinks_weighted_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                I_used = run.get("importance_list", importance_list)
                v = sum_weighted_widths_all_links(per_pair_details, I_used, delta=delta)
                results[name]["sums"][k].append(v)

    # plot (mean ± 95%CI)
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        means, halfs = [], []
        for vals in dat["sums"]:
            m, h = mean_ci95(vals); means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, marker="o", label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)
    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Weighted Sum of Widths")
    ax.grid(True); ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)

    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 7) 幅（UB-LB）Weighted: ペアごとの I_d·最小幅 総和
# =========================
def plot_minwidthsum_perpair_weighted_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, importance_mode="fixed", importance_uniform=(0.0,1.0), delta=0.1, seed=None,
    verbose=True, print_every=1,
):
    file_name = f"plot_minwidthsum_perpair_weighted_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run["per_pair_details"]
                I_used = run.get("importance_list", importance_list)
                v = sum_weighted_min_widths_perpair(per_pair_details, I_used, delta=delta)
                results[name]["sums"][k].append(v)

    # plot (mean ± 95%CI)
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        means, halfs = [], []
        for vals in dat["sums"]:
            m, h = mean_ci95(vals); means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, marker="o", label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)
    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Weighted sum over pairs of min")
    ax.grid(True); ax.legend(title="Scheduler")
    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")

# =========================
# 8) Importance-Discovery Value vs Budget
# =========================
def plot_importance_discovery_value_vs_budget(
    budget_list, scheduler_names, noise_model,
    node_path_list, importance_list,
    bounces=(1,2,3,4), repeat=10, importance_mode="fixed", importance_uniform=(0.0,1.0),
    y=0.10,                 # 許容 CI 幅しきい値
    delta=0.1,             # Hoeffding の δ
    use_f="Fhat",            # "LCB" または "Fhat"
    seed=None,
    verbose=True, print_every=1,
):
    """
    縦軸: I(C) = Σ_{n∈S_sel} I_n * F_n
      - ここで S_sel = { n | (UCB_n* - LCB_n*) ≤ y }
      - j_n* = argmax_j \hat f_{n,j}
      - LCB_n* = clip(\hat f_n* - r_n*, 0, 1)
      - UCB_n* = clip(\hat f_n* + r_n*, 0, 1)
      - r_n* = ci_radius_hoeffding(N_n*, delta)
      - F_n = LCB_n* (保守的) または Fhat (= \hat f_n*)

    横軸: 目標予算 C（budget_list）
    """
    file_name = f"plot_importance_discovery_value_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_shared_sweep(
        budget_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed,
        verbose=verbose, print_every=print_every,
    )

    # 収集バッファ（予算ごとに複数リピートの値を保持）
    results = {name: {"Ivals": [[] for _ in budget_list]} for name in scheduler_names}

    for name in scheduler_names:
        for k in range(len(budget_list)):
            for run in payload["data"][name][k]:
                per_pair_details = run.get("per_pair_details", [])
                total_I = 0.0

                # 各ペア n ごとに j_n* を選び、CI 幅を計算して発見判定
                for n_idx, det in enumerate(per_pair_details):
                    alloc = det.get("alloc_by_path", {}) or {}
                    est   = det.get("est_fid_by_path", {}) or {}
                    if not est:
                        continue

                    # j_n* = argmax_j \hat f_{n,j}
                    j_star, fhat_star = max(est.items(), key=lambda kv: float(kv[1]))
                    j_star = int(j_star)
                    fhat_star = float(fhat_star)
                    N_star = int(alloc.get(j_star, 0))

                    # r_n* と LCB/UCB / 幅
                    r = ci_radius_hoeffding(N_star, delta)
                    lcb = max(0.0, fhat_star - r)
                    ucb = min(1.0, fhat_star + r)
                    width = ucb - lcb

                    # 発見条件 x_n ≤ y
                    if width <= float(y):
                        I_used = run.get("importance_list", importance_list)
                        I = float(I_used[n_idx]) if n_idx < len(I_used) else 1.0
                        F_n = lcb if str(use_f).upper() == "LCB" else fhat_star
                        total_I += I * float(F_n)

                results[name]["Ivals"][k].append(float(total_I))

    # === plot: mean ± 95% CI ===
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(budget_list)
    for name, dat in results.items():
        means, halfs = [], []
        for vals in dat["Ivals"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, marker="o", label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Discovered pairs score")
    ax.grid(True); ax.legend(title="Scheduler")
    pdf = f"{file_name}.pdf"
    plt.savefig(pdf)
    if shutil.which("pdfcrop"): os.system(f'pdfcrop --margins "8 8 8 8" {pdf} {pdf}')
    _log(f"Saved: {pdf}")
