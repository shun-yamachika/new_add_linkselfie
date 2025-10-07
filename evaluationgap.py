# evaluationgap.py — Gap sweep: x = gap, y = accuracy (mean ± 95% CI)
# Supports:
#   (2a) Random gap mode   : alpha = alpha_base, beta = alpha - gap, then random sampling (utils.fidelity)
#   (2b) Fixed  gap mode   : deterministic arithmetic sequence with gap       (utils.fidelity)
#
# Both modes inject true_fid_by_path with 1-origin keys and normalize est_fid_by_path to 1-origin.

import os
import json
import time
import pickle
import hashlib
import shutil
from typing import List, Sequence, Dict, Any, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from network import QuantumNetwork
from schedulers import run_scheduler
from viz.plots import mean_ci95

from utils.ids import to_idx0, normalize_to_1origin, is_keys_1origin
from utils.fidelity import (
    generate_fidelity_list_fix_gap,
    _generate_fidelity_list_random_rng,
)

# ---- Matplotlib style (align with evaluation.py) ----
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
mpl.rcParams["font.size"] = 20

_default_cycler = (
    cycler(color=["#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#a65628"])
    + cycler(marker=["s", "v", "o", "x", "*", "+"])
    + cycler(linestyle=[":", "--", "-", "-.", "--", ":"])
)
plt.rc("axes", prop_cycle=_default_cycler)


# -----------------------------
# Cache helpers (gap sweep)
# -----------------------------
def _gap_sweep_signature(gap_list: Sequence[float], scheduler_names: Sequence[str], noise_model: str,
                         node_path_list: Sequence[int], importance_list: Sequence[float],
                         bounces: Sequence[int], repeat: int,
                         mode: str,   # "random" or "fixed"
                         importance_mode: str = "fixed", importance_uniform: Tuple[float, float] = (0.0, 1.0),
                         seed: int = None, alpha_base: float = 0.95, variance: float = 0.10,
                         C_total: int = 5000) -> Tuple[Dict[str, Any], str]:
    payload = {
        "gap_list": list(map(float, gap_list)),
        "scheduler_names": list(scheduler_names),
        "noise_model": str(noise_model),
        "node_path_list": list(map(int, node_path_list)),
        "importance_list": list(importance_list) if importance_list is not None else None,
        "importance_mode": str(importance_mode),
        "importance_uniform": list(importance_uniform) if importance_uniform is not None else None,
        "bounces": list(map(int, bounces)),
        "repeat": int(repeat),
        "seed": int(seed) if seed is not None else None,
        "mode": str(mode),  # "random" / "fixed"
        "alpha_base": float(alpha_base),
        "variance": float(variance),
        "C_total": int(C_total),
        "version": 4,  # schema: 1-origin injection & normalized est keys; fidelity_bank per gap stored
    }
    sig = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return payload, sig


def _shared_gap_path(noise_model: str, sig: str) -> str:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"shared_gap_{noise_model}_{sig}.pickle")


def _run_or_load_shared_gap_sweep(
    gap_list: Sequence[float], scheduler_names: Sequence[str], noise_model: str,
    node_path_list: Sequence[int], importance_list: Sequence[float],
    bounces=(1, 2, 3, 4), repeat: int = 10,
    importance_mode: str = "fixed", importance_uniform: Tuple[float, float] = (0.0, 1.0),
    seed: int = None, alpha_base: float = 0.95, variance: float = 0.10,
    C_total: int = 5000, mode: str = "random",
    verbose: bool = True, print_every: int = 1,
) -> Dict[str, Any]:
    """
    For each gap in gap_list, run `repeat` times. For each (gap, repeat) we create ONE fidelity_bank
    and reuse it for:
      - network generation (per pair)
      - true_fid_by_path injection
    so that there is no re-sampling mismatch.
    """
    config, sig = _gap_sweep_signature(
        gap_list, scheduler_names, noise_model,
        node_path_list, importance_list, bounces, repeat,
        mode=mode,
        importance_mode=importance_mode, importance_uniform=importance_uniform,
        seed=seed, alpha_base=alpha_base, variance=variance, C_total=C_total
    )
    cache_path = _shared_gap_path(noise_model, sig)
    lock_path = cache_path + ".lock"
    STALE_LOCK_SECS = 6 * 60 * 60
    HEARTBEAT_EVERY = 5.0

    rng = np.random.default_rng(seed)

    # Fast path: cached
    if os.path.exists(cache_path):
        if verbose:
            print(f"[gap-shared] Load cached: {os.path.basename(cache_path)}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Lock acquisition (single writer)
    got_lock = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
            break
        except FileExistsError:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = 0
            if age > STALE_LOCK_SECS:
                try: os.remove(lock_path)
                except FileNotFoundError: pass
                continue
            if verbose:
                print("[gap-shared] Waiting for cache to be ready...", flush=True)
            time.sleep(1.0)

    try:
        if verbose:
            print(f"[gap-shared] Run gap sweep and cache to: {os.path.basename(cache_path)}", flush=True)

        data = {name: {k: [] for k in range(len(gap_list))} for name in scheduler_names}
        last_hb = time.time()

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                print(f"[gap-shared] Repeat {r+1}/{repeat}", flush=True)

            # Importance per repeat
            if str(importance_mode).lower() == "uniform":
                a, b = map(float, importance_uniform)
                imp_list_r = [float(rng.uniform(a, b)) for _ in node_path_list]
            else:
                imp_list_r = list(importance_list)

            # Sweep gaps
            for k, gap in enumerate(gap_list):
                if verbose:
                    print(f"=== [GAP {noise_model}] gap={gap} ({k+1}/{len(gap_list)}), mode={mode} ===", flush=True)

                # Heartbeat
                now = time.time()
                if now - last_hb >= HEARTBEAT_EVERY:
                    try: os.utime(lock_path, None)
                    except FileNotFoundError: pass
                    last_hb = now

                # (重要) gap×repeat ごとに fidelity_bank を先に作って保存 → 再利用
                fidelity_bank: List[List[float]] = []
                for pair_idx, path_num in enumerate(node_path_list):
                    if mode == "fixed":
                        # 等差列: fidelity_max から gap ずつ下げる
                        fids = generate_fidelity_list_fix_gap(
                            path_num=int(path_num), gap=float(gap), fidelity_max=1.0
                        )
                    else:
                        # ランダム: alpha=alpha_base, beta=alpha_base-gap
                        alpha = float(alpha_base)
                        beta = float(alpha_base) - float(gap)
                        fids = _generate_fidelity_list_random_rng(
                            rng=rng, path_num=int(path_num),
                            alpha=alpha, beta=beta, variance=float(variance)
                        )
                    fidelity_bank.append(fids)

                # network generator uses the saved bank
                def network_generator(path_num: int, pair_idx: int):
                    return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

                for name in scheduler_names:
                    per_pair_results, total_cost, per_pair_details = run_scheduler(
                        node_path_list=node_path_list, importance_list=imp_list_r,
                        scheduler_name=name,
                        bounces=list(bounces),
                        C_total=int(C_total),
                        network_generator=network_generator,
                        return_details=True,
                    )

                    # Inject truth (1..L) and normalize estimated map (to 1..L)
                    for d, det in enumerate(per_pair_details):
                        L = int(node_path_list[d])
                        est_map = det.get("est_fid_by_path", {})

                        if est_map:
                            est_map_norm = normalize_to_1origin({int(k): float(v) for k, v in est_map.items()}, L)
                        else:
                            est_map_norm = {}

                        # true map from the saved fidelity_bank (no re-sampling)
                        true_list = fidelity_bank[d]  # 0-origin
                        true_map = {pid: float(true_list[to_idx0(pid)]) for pid in range(1, L + 1)}

                        if est_map_norm and not is_keys_1origin(est_map_norm.keys(), L):
                            raise RuntimeError(f"[inject] est_fid_by_path keys not 1..{L} (pair={d})")

                        det["est_fid_by_path"]  = est_map_norm
                        det["true_fid_by_path"] = true_map

                    data[name][k].append({
                        "per_pair_results": per_pair_results,
                        "per_pair_details": per_pair_details,
                        "total_cost": total_cost,
                        "importance_list": imp_list_r,
                        "gap": float(gap),
                        "C_total": int(C_total),
                        "alpha_base": float(alpha_base),
                        "variance": float(variance),
                        "mode": str(mode),
                        "node_path_list": list(map(int, node_path_list)),
                    })

        payload = {
            "config": config,
            "gap_list": list(map(float, gap_list)),
            "data": data,
        }

        # atomic write
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)

        return payload

    finally:
        if got_lock:
            try: os.remove(lock_path)
            except FileNotFoundError: pass


# -----------------------------
# Public APIs
# -----------------------------
def plot_accuracy_vs_gap(
    gap_list: Sequence[float], scheduler_names: Sequence[str], noise_model: str,
    node_path_list: Sequence[int], importance_list: Sequence[float],
    bounces=(1, 2, 3, 4), repeat: int = 10,
    importance_mode: str = "fixed", importance_uniform: Tuple[float, float] = (0.0, 1.0),
    seed: int = None, alpha_base: float = 0.95, variance: float = 0.10,
    C_total_override: int = None,
    verbose: bool = True, print_every: int = 1,
) -> str:
    """
    (2a) Gap vs Accuracy — Random mode (utils.fidelity)
    """
    file_name = f"plot_accuracy_vs_gap_random_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    C_total = int(C_total_override) if C_total_override is not None else 5000

    payload = _run_or_load_shared_gap_sweep(
        gap_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform,
        seed=seed, alpha_base=alpha_base, variance=variance,
        C_total=C_total, mode="random",
        verbose=verbose, print_every=print_every,
    )

    # Collect accuracy arrays per gap
    results = {name: {"accs": [[] for _ in gap_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(gap_list)):
            for run in payload["data"][name][k]:
                per_pair_results = run["per_pair_results"]
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
                results[name]["accs"][k].append(acc)

    # Plot
    plt.rc("axes", prop_cycle=_default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(map(float, gap_list))

    for name, data in results.items():
        means, halfs = [], []
        for vals in data["accs"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)

        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Gap (alpha - beta)")
    ax.set_ylabel("Average Correctness (mean ± 95% CI)")
    ax.grid(True); ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)

    pdf = os.path.join(outdir, f"{file_name}.pdf")
    plt.savefig(pdf)
    if shutil.which("pdfcrop"):
        os.system(f'pdfcrop --margins "8 8 8 8" "{pdf}" "{pdf}"')
    print(f"Saved: {pdf}", flush=True)
    return pdf


def plot_accuracy_vs_gap_fixgap(
    gap_list: Sequence[float], scheduler_names: Sequence[str], noise_model: str,
    node_path_list: Sequence[int], importance_list: Sequence[float],
    bounces=(1, 2, 3, 4), repeat: int = 10,
    importance_mode: str = "fixed", importance_uniform: Tuple[float, float] = (0.0, 1.0),
    seed: int = None, fidelity_max: float = 1.0,
    C_total_override: int = None,
    verbose: bool = True, print_every: int = 1,
) -> str:
    """
    (2b) Gap vs Accuracy — Fixed arithmetic sequence mode (utils.fidelity)
    """
    # 固定列では rng は使わないが、署名の再現性のため seed を渡しておく
    file_name = f"plot_accuracy_vs_gap_fixed_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    # alpha_base/variance は未使用だが、シグネチャ整合のためデフォルト値を渡す
    C_total = int(C_total_override) if C_total_override is not None else 5000

    payload = _run_or_load_shared_gap_sweep(
        gap_list, scheduler_names, noise_model,
        node_path_list, importance_list,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform,
        seed=seed, alpha_base=0.95, variance=0.10,
        C_total=C_total, mode="fixed",
        verbose=verbose, print_every=print_every,
    )

    # Collect accuracy arrays per gap
    results = {name: {"accs": [[] for _ in gap_list]} for name in scheduler_names}
    for name in scheduler_names:
        for k in range(len(gap_list)):
            for run in payload["data"][name][k]:
                per_pair_results = run["per_pair_results"]
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
                results[name]["accs"][k].append(acc)

    # Plot
    plt.rc("axes", prop_cycle=_default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(map(float, gap_list))

    for name, data in results.items():
        means, halfs = [], []
        for vals in data["accs"]:
            m, h = mean_ci95(vals)
            means.append(m)
            halfs.append(h)
        means = np.asarray(means)
        halfs = np.asarray(halfs)

        label = name.replace("Vanilla NB", "VanillaNB").replace("Succ. Elim. NB", "SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Gap (arithmetic sequence)")
    ax.set_ylabel("Average Correctness (mean ± 95% CI)")
    ax.grid(True)
    ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)

    pdf = os.path.join(outdir, f"{file_name}.pdf")
    plt.savefig(pdf)
    if shutil.which("pdfcrop"):
        os.system(f'pdfcrop --margins "8 8 8 8" "{pdf}" "{pdf}"')
    print(f"Saved: {pdf}", flush=True)
    return pdf
