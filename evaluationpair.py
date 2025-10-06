
# evaluationpair.py — Sweep "number of destination pairs" (x) vs Accuracy (y)
# Designed to align with the existing evaluation.py/main.py pipeline.
#
# Usage example (from your own script or REPL):
#   from evaluationpair import plot_accuracy_vs_pairs
#   plot_accuracy_vs_pairs(
#       pairs_list=[1,2,3,4,5,6],
#       paths_per_pair=5,
#       C_total=6000,
#       scheduler_names=["Greedy", "LNaive"],
#       noise_model="Depolar",
#       bounces=(1,2,3,4),
#       repeat=25,
#       importance_mode="uniform",   # or "fixed"
#       importance_uniform=(0.0,1.0),
#       seed=12,
#       verbose=True
#   )
#
# Produces: outputs/plot_accuracy_vs_pairs_<noise_model>.pdf

import os
import time
import json
import pickle
import hashlib
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# Reuse core components from your existing codebase
from network import QuantumNetwork  # existing class
from schedulers import run_scheduler    # existing dispatcher
from viz.plots import mean_ci95         # existing helper

# ---- Matplotlib style (match evaluation.py) ----
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


# =========================
# Utilities
# =========================
def _log(msg: str):
    print(msg, flush=True)

def _generate_fidelity_list_random_rng(rng: np.random.Generator, path_num: int,
                                       alpha: float = 0.95, beta: float = 0.85, variance: float = 0.1):
    """Generate `path_num` link fidelities in [0.8, 1.0], ensuring a small top-1 gap."""
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        res = []
        for mu in mean:
            while True:
                r = rng.normal(mu, variance)
                if 0.8 <= r <= 1.0:
                    break
            res.append(float(r))
        sorted_res = sorted(res, reverse=True)
        if sorted_res[0] - sorted_res[1] > 0.02:
            return res


# =========================
# Pair-sweep cache helpers
# =========================
def _sweep_signature_pairs(pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
                           bounces, repeat, importance_mode="fixed", importance_uniform=(0.0,1.0), seed=None):
    payload = {
        "pairs_list": list(pairs_list),
        "paths_per_pair": int(paths_per_pair),
        "C_total": int(C_total),
        "scheduler_names": list(scheduler_names),
        "noise_model": str(noise_model),
        "bounces": list(bounces),
        "repeat": int(repeat),
        "importance_mode": str(importance_mode),
        "importance_uniform": list(importance_uniform) if importance_uniform is not None else None,
        "seed": int(seed) if seed is not None else None,
        "version": 1
    }
    sig = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return payload, sig

def _shared_pair_sweep_path(noise_model: str, sig: str):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"pair_sweep_{noise_model}_{sig}.pickle")


def _run_or_load_pair_sweep(
    pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
    bounces=(1,2,3,4), repeat=10,
    importance_mode="fixed", importance_uniform=(0.0,1.0),
    seed=None,
    verbose=True, print_every=1,
):
    config, sig = _sweep_signature_pairs(
        pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
        bounces, repeat, importance_mode=importance_mode, importance_uniform=importance_uniform, seed=seed
    )
    cache_path = _shared_pair_sweep_path(noise_model, sig)
    lock_path = cache_path + ".lock"
    STALE_LOCK_SECS = 6 * 60 * 60
    HEARTBEAT_EVERY = 5.0

    rng = np.random.default_rng(seed)

    # Quick load if exists
    if os.path.exists(cache_path):
        if verbose: _log(f"[pair-sweep] Load cached: {os.path.basename(cache_path)}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Acquire lock (single producer; others wait)
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
                if verbose: _log("[pair-sweep] Stale lock detected. Removing...")
                try: os.remove(lock_path)
                except FileNotFoundError: pass
                continue
            if verbose: _log("[pair-sweep] Waiting for cache to be ready...")
            time.sleep(1.0)

    try:
        if verbose: _log(f"[pair-sweep] Run sweep and cache to: {os.path.basename(cache_path)}")

        data = {name: {k: [] for k in range(len(pairs_list))} for name in scheduler_names}
        last_hb = time.time()

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                _log(f"[pair-sweep] Repeat {r+1}/{repeat}")

            # For each N (number of destination pairs), build one fixed topology per repeat
            for k, N_pairs in enumerate(pairs_list):
                # Heartbeat
                now = time.time()
                if now - last_hb >= HEARTBEAT_EVERY:
                    try: os.utime(lock_path, None)
                    except FileNotFoundError: pass
                    last_hb = now

                node_path_list = [int(paths_per_pair)] * int(N_pairs)

                # Fidelity bank for this N (used consistently across schedulers)
                fidelity_bank = [_generate_fidelity_list_random_rng(rng, paths_per_pair) for _ in node_path_list]

                # Importance list for this N
                if str(importance_mode).lower() == "uniform":
                    a, b = map(float, importance_uniform)
                    importance_list = [float(rng.uniform(a, b)) for _ in node_path_list]
                else:
                    # fixed mode: if user wants exact values, they can tune outside; default all ones
                    importance_list = [1.0 for _ in node_path_list]

                def network_generator(path_num, pair_idx):
                    return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

                for name in scheduler_names:
                    per_pair_results, total_cost, per_pair_details = run_scheduler(
                        node_path_list=node_path_list,
                        importance_list=importance_list,
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
                        "importance_list": importance_list,
                        "node_path_list": node_path_list,
                    })

        payload = {"config": config, "pairs_list": list(pairs_list), "data": data}

        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)

        return payload

    finally:
        if got_lock:
            try: os.remove(lock_path)
            except FileNotFoundError: pass


# =========================
# Plot: Accuracy (mean ± 95% CI) vs #Destination Pairs
# =========================
def plot_accuracy_vs_pairs(
    pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
    bounces=(1,2,3,4), repeat=10,
    importance_mode="fixed", importance_uniform=(0.0,1.0),
    seed=None,
    verbose=True, print_every=1,
):
    """
    pairs_list: list[int]     # x-axis = number of destination pairs (N)
    paths_per_pair: int       # number of candidate links per pair (each L_n = paths_per_pair)
    C_total: int              # total budget for the whole experiment (fixed while N varies)
    scheduler_names: list[str]
    noise_model: str
    bounces: tuple/list[int]  # NB bounce vector
    repeat: int               # repeats per N
    importance_mode: "fixed" or "uniform"
    importance_uniform: (a,b) # when uniform, sample I_n ~ U[a,b]
    seed: int
    """
    file_name = f"plot_accuracy_vs_pairs_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(root_dir, "outputs")
    os.makedirs(outdir, exist_ok=True)

    payload = _run_or_load_pair_sweep(
        pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform,
        seed=seed, verbose=verbose, print_every=print_every
    )

    results = {name: {"accs": [[] for _ in pairs_list]} for name in scheduler_names}

    for name in scheduler_names:
        for k in range(len(pairs_list)):
            for run in payload["data"][name][k]:
                per_pair_results = run["per_pair_results"]

                # Normalize elements to bool → 0/1
                vals = []
                for r in per_pair_results:
                    if isinstance(r, tuple):
                        c = r[0]
                    elif isinstance(r, (int, float, bool)):
                        c = bool(r)
                    else:
                        raise TypeError(f"Unexpected per_pair_results element: {type(r)} -> {r}")
                    vals.append(1.0 if c else 0.0)

                acc = float(np.mean(vals)) if vals else 0.0
                results[name]["accs"][k].append(acc)

    # Plot
    plt.rc("axes", prop_cycle=_default_cycler)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    xs = list(pairs_list)

    for name, data in results.items():
        means, halfs = [], []
        for vals in data["accs"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)

        label = name.replace("Vanilla NB","VanillaNB").replace("Succ. Elim. NB","SuccElimNB")
        ax.plot(xs, means, linewidth=2.0, label=label)
        ax.fill_between(xs, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Number of Destination Pairs (N)")
    ax.set_ylabel("Average Correctness (mean ± 95% CI)")
    ax.grid(True); ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)

    pdf = os.path.join(outdir, f"{file_name}.pdf")
    plt.savefig(pdf)
    if shutil.which("pdfcrop"):
        os.system(f'pdfcrop --margins "8 8 8 8" "{pdf}" "{pdf}"')
    _log(f"Saved: {pdf}")

    return {
        "pdf": pdf,
        "pairs_list": list(pairs_list),
        "config": payload["config"],
    }


if __name__ == "__main__":
    # Minimal CLI for quick testing
    # Example: python evaluationpair.py
    pairs_list = [1, 2, 3, 4, 5, 6]
    paths_per_pair = 5
    C_total = 6000
    scheduler_names = ["Greedy", "LNaive"]
    noise_model = "Depolar"
    bounces = (1,2,3,4)
    repeat = 10
    importance_mode = "uniform"
    importance_uniform = (0.0, 1.0)
    seed = 12

    plot_accuracy_vs_pairs(
        pairs_list, paths_per_pair, C_total, scheduler_names, noise_model,
        bounces=bounces, repeat=repeat,
        importance_mode=importance_mode, importance_uniform=importance_uniform,
        seed=seed, verbose=True
    )
