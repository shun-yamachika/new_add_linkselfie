# evaluation.py
# Run evaluation and plot figures
import math
import os
import pickle
import time
import shutil

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from algorithms import benchmark_using_algorithm  # may be used elsewhere
from network import QuantumNetwork
from schedulers import run_scheduler  # パッケージ化したものを使う

# ---- Matplotlib style (IEEE-ish) ----
plt.rc("font", family="Times New Roman")
plt.rc("font", size=20)
default_cycler = (
    cycler(color=["#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#a65628"])
    + cycler(marker=["s", "v", "o", "x", "*", "+"])
    + cycler(linestyle=[":", "--", "-", "-.", "--", ":"])
)
plt.rc("axes", prop_cycle=default_cycler)


# =========================
# Fidelity generators
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
    """Generate `path_num` links.
    u_1 = alpha, u_i = beta for all i = 2, 3, ..., n.
    Fidelity_i ~ N(u_i, variance), clipped to [0.8, 1].
    Ensure the top-1 gap is large enough (> 0.02) for termination guarantees.
    """
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        result = []
        for i in range(path_num):
            mu = mean[i]
            # Sample a Gaussian random variable and make sure its value is in the valid range
            while True:
                r = np.random.normal(mu, variance)
                # Depolarizing noise and amplitude damping noise models require fidelity >= 0.5
                # Be conservative: require >= 0.8
                if 0.8 <= r <= 1.0:
                    break
            result.append(r)
        assert len(result) == path_num
        sorted_res = sorted(result, reverse=True)
        # To guarantee the termination of algorithms, we require that the gap is large enough
        if sorted_res[0] - sorted_res[1] > 0.02:
            return result


# =========================
# Progress helpers (LinkSelfie風)
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
# Plots
# =========================
def plot_accuracy_vs_budget(
    budget_list,          # e.g., [1000, 2000, 3000, ...] (x-axis)
    scheduler_names,      # e.g., ["LNaive", "Greedy", ...]
    noise_model,          # e.g., "Depolar"
    node_path_list,       # e.g., [5, 5, 5]
    importance_list,      # e.g., [0.4, 0.7, 1.0] (not used here, but kept for interface)
    bounces=(1, 2, 3, 4),
    repeat=10,
    verbose=True,
    print_every=1,
):
    file_name = f"plot_accuracy_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        _log("Pickle data exists, skip simulation and plot the data directly.")
        _log("To rerun, delete the pickle in `outputs`.")
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
            budget_list = payload["budget_list"]
            results = payload["results"]
    else:
        results = {name: {"accs": [[] for _ in budget_list]} for name in scheduler_names}
        for k, C_total in enumerate(budget_list):
            timer = _start_timer()
            if verbose:
                _log(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===")
            for r in range(repeat):
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    _log(f"  [repeat {r+1}/{repeat}] generating topology …")
                # 1リピート = 1トポロジ（全スケジューラで共有）
                fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

                def network_generator(path_num, pair_idx):
                    return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

                for name in scheduler_names:
                    if verbose and ((r + 1) % print_every == 0 or r == 0):
                        _log(f"    - {name}: running …")
                    per_pair_results, _ = run_scheduler(
                        node_path_list=node_path_list,
                        importance_list=importance_list,
                        scheduler_name=name,
                        bounces=list(bounces),
                        C_total=int(C_total),
                        network_generator=network_generator,
                    )
                    acc = (
                        float(np.mean([1.0 if c else 0.0 for (c, _cost, _bf) in per_pair_results]))
                        if per_pair_results
                        else 0.0
                    )
                    results[name]["accs"][k].append(acc)
                    if verbose and ((r + 1) % print_every == 0 or r == 0):
                        _log(f"      -> acc={acc:.3f}")
            if verbose:
                tot, _ = _tick(timer)
                _log(f"=== done Budget={C_total} | elapsed {tot:.1f}s ===")

        with open(file_path, "wb") as f:
            pickle.dump({"budget_list": list(budget_list), "results": results}, f)

    # --- Plot ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        avg_accs = [float(np.mean(v)) if v else 0.0 for v in data["accs"]]
        label = name.replace("Vanilla NB", "VanillaNB").replace("Succ. Elim. NB", "SuccElimNB")
        ax.plot(x, avg_accs, linewidth=2.0, label=label)

    ax.set_xlabel("Total Budget (C)")
    ax.set_ylabel("Average Correctness")
    ax.grid(True)
    ax.legend(title="Scheduler", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    _log(f"Saved: {pdf_name}")


def plot_value_vs_used(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    verbose=True,
    print_every=1,
):
    """x = 実コスト平均（used）で描く版。旧 plot_value_vs_budget と同等の挙動。"""
    file_name = f"plot_value_vs_used_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        name: {"values": [[] for _ in budget_list], "costs": [[] for _ in budget_list]}
        for name in scheduler_names
    }

    for k, C_total in enumerate(budget_list):
        timer = _start_timer()
        if verbose:
            _log(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===")

        # 1リピート = 1トポロジ（全スケジューラで共有）
        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                _log(f"  [repeat {r+1}/{repeat}]")
            for name in scheduler_names:
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    _log(f"    - {name}: running …")
                per_pair_results, total_cost, per_pair_details = run_scheduler(
                    node_path_list=node_path_list,
                    importance_list=importance_list,
                    scheduler_name=name,
                    bounces=list(bounces),
                    C_total=int(C_total),
                    network_generator=network_generator,
                    return_details=True,
                )
                # 価値の合成
                value = 0.0
                for d, details in enumerate(per_pair_details):
                    alloc = details.get("alloc_by_path", {})
                    est = details.get("est_fid_by_path", {})
                    inner = sum(float(est.get(l, 0.0)) * int(b) for l, b in alloc.items())
                    value += float(importance_list[d]) * inner

                results[name]["values"][k].append(float(value))
                results[name]["costs"][k].append(int(total_cost))
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    _log(f"      -> used={total_cost}, value={value:.2f}")

        if verbose:
            tot, _ = _tick(timer)
            _log(f"=== done Budget={C_total} | elapsed {tot:.1f}s ===")

    # --- Plot (x = 実コスト平均) ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    for name, data in results.items():
        xs = [float(np.mean(v)) if v else 0.0 for v in data["costs"]]
        ys = [float(np.mean(v)) if v else 0.0 for v in data["values"]]
        ax.plot(xs, ys, linewidth=2.0, marker="o", label=name)

    ax.set_xlabel("Total Measured Cost (used)")
    ax.set_ylabel("Total Value (Σ I_d Σ f̂_{d,l}·B_{d,l})")
    ax.grid(True)
    ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    _log(f"Saved: {pdf_name}")


def plot_value_vs_budget_target(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    verbose=True,
    print_every=1,
):
    """x = 目標予算（指定した budget_list をそのまま x 軸に）で描く版。"""
    file_name = f"plot_value_vs_budget_target_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        name: {"values": [[] for _ in budget_list], "costs": [[] for _ in budget_list]}
        for name in scheduler_names
    }

    for k, C_total in enumerate(budget_list):
        timer = _start_timer()
        if verbose:
            _log(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===")

        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                _log(f"  [repeat {r+1}/{repeat}]")
            for name in scheduler_names:
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    _log(f"    - {name}: running …")
                per_pair_results, total_cost, per_pair_details = run_scheduler(
                    node_path_list=node_path_list,
                    importance_list=importance_list,
                    scheduler_name=name,
                    bounces=list(bounces),
                    C_total=int(C_total),
                    network_generator=network_generator,
                    return_details=True,
                )
                value = 0.0
                for d, details in enumerate(per_pair_details):
                    alloc = details.get("alloc_by_path", {})
                    est = details.get("est_fid_by_path", {})
                    inner = sum(float(est.get(l, 0.0)) * int(b) for l, b in alloc.items())
                    value += float(importance_list[d]) * inner

                results[name]["values"][k].append(float(value))
                results[name]["costs"][k].append(int(total_cost))
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    _log(f"      -> used={total_cost}, value={value:.2f}")

        if verbose:
            tot, _ = _tick(timer)
            _log(f"=== done Budget={C_total} | elapsed {tot:.1f}s ===")

    # --- Plot (x = 目標予算) ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        ys = [float(np.mean(v)) if v else 0.0 for v in data["values"]]
        ax.plot(x, ys, linewidth=2.0, marker="o", label=name)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Total Value (Σ I_d Σ f̂_{d,l}·B_{d,l})")
    ax.grid(True)
    ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    _log(f"Saved: {pdf_name}")


# =========================
# CI width helpers and plots
# =========================
def _ci_radius_hoeffding(n: int, delta: float = 0.1) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(0.5 * math.log(2.0 / delta) / n)


# =========================
# Width-sum metrics (new)
# =========================

def _sum_widths_all_links(per_pair_details, delta: float = 0.1) -> float:
    """
    すべてのペア・すべてのリンクについて、(UB - LB) を合計。
    est が無いリンクはスキップ（=寄与0）。測定していないリンクは数えません。
    """
    total = 0.0
    for det in per_pair_details:
        alloc = det.get("alloc_by_path", {})  # n = 測定回数
        est   = det.get("est_fid_by_path", {})  # 標本平均
        for pid, m in est.items():
            n = int(alloc.get(pid, 0))
            rad = _ci_radius_hoeffding(n, delta)
            lb = max(0.0, float(m) - rad)
            ub = min(1.0, float(m) + rad)
            total += (ub - lb)
    return float(total)


def _sum_min_widths_per_pair(per_pair_details, delta: float = 0.1) -> float:
    """
    ペアごとにリンクの (UB - LB) を算出し、その「最小値」を取り、全ペアで合計。
    est が空のペアは保守的に 1.0 を加算（“全く分からない”幅として扱う）。
    """
    s = 0.0
    for det in per_pair_details:
        alloc = det.get("alloc_by_path", {})
        est   = det.get("est_fid_by_path", {})
        if not est:
            s += 1.0
            continue
        widths = []
        for pid, m in est.items():
            n = int(alloc.get(pid, 0))
            rad = _ci_radius_hoeffding(n, delta)
            lb = max(0.0, float(m) - rad)
            ub = min(1.0, float(m) + rad)
            widths.append(ub - lb)
        s += (min(widths) if widths else 1.0)
    return float(s)


def plot_widthsum_alllinks_vs_budget(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    delta=0.1,
    verbose=True,
    print_every=1,
):
    """
    y = 全リンク(UB-LB)総和 の平均 ±95%CI を、x = 目標予算 で描画。
    生データは outputs/plot_widthsum_alllinks_vs_budget_*.pickle に保存。
    """
    file_name = f"plot_widthsum_alllinks_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}

    for k, C_total in enumerate(budget_list):
        if verbose:
            print(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===", flush=True)

        # 1リピート=1トポロジ（全スケジューラ共有）
        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                print(f"  [repeat {r+1}/{repeat}]", flush=True)

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
                v = _sum_widths_all_links(per_pair_details, delta=delta)
                results[name]["sums"][k].append(v)
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    print(f"    - {name}: sum_alllinks={v:.4f} (used={total_cost})", flush=True)

    # --- Save raw data (.pickle) ---
    file_path = os.path.join(output_dir, f"{file_name}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump({"budget_list": list(budget_list), "results": results}, f)
    print(f"Saved pickle: {file_path}")

    # --- Plot mean ± 95% CI across repeats ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        means, halfs = [], []
        for vals in data["sums"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        ax.plot(x, means, linewidth=2.0, marker="o", label=name)
        ax.fill_between(x, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Sum of (UB - LB) over all links")
    ax.grid(True)
    ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    print(f"Saved: {pdf_name}")


def plot_minwidthsum_perpair_vs_budget(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    delta=0.1,
    verbose=True,
    print_every=1,
):
    """
    y = ペアごとの (UB-LB) 最小値の合計 の平均 ±95%CI、x = 目標予算。
    生データは outputs/plot_minwidthsum_perpair_vs_budget_*.pickle に保存。
    """
    file_name = f"plot_minwidthsum_perpair_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}

    for k, C_total in enumerate(budget_list):
        if verbose:
            print(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===", flush=True)

        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                print(f"  [repeat {r+1}/{repeat}]", flush=True)

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
                v = _sum_min_widths_per_pair(per_pair_details, delta=delta)
                results[name]["sums"][k].append(v)
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    print(f"    - {name}: sum_min_perpair={v:.4f} (used={total_cost})", flush=True)

    # --- Save raw data (.pickle) ---
    file_path = os.path.join(output_dir, f"{file_name}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump({"budget_list": list(budget_list), "results": results}, f)
    print(f"Saved pickle: {file_path}")

    # --- Plot mean ± 95% CI across repeats ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        means, halfs = [], []
        for vals in data["sums"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        ax.plot(x, means, linewidth=2.0, marker="o", label=name)
        ax.fill_between(x, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Sum over pairs of min (UB - LB)")
    ax.grid(True)
    ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    print(f"Saved: {pdf_name}")



# =========================
# Weighted width-sum metrics (add-on)
# =========================

def _sum_weighted_widths_all_links(per_pair_details, importance_list, delta: float = 0.1) -> float:
    """
    すべてのペア・すべてのリンクの (UB-LB) に、ペア重要度 I_d を掛けて合計。
    importance_list[d] が無ければ I_d=1.0 として扱う。
    """
    total = 0.0
    for d, det in enumerate(per_pair_details):
        I = float(importance_list[d]) if d < len(importance_list) else 1.0
        alloc = det.get("alloc_by_path", {})
        est   = det.get("est_fid_by_path", {})
        for pid, m in est.items():
            n = int(alloc.get(pid, 0))
            rad = _ci_radius_hoeffding(n, delta)
            lb = max(0.0, float(m) - rad)
            ub = min(1.0, float(m) + rad)
            total += I * (ub - lb)
    return float(total)


def _sum_weighted_min_widths_per_pair(per_pair_details, importance_list, delta: float = 0.1) -> float:
    """
    ペア d ごとに min_l (UB-LB) を計算し、I_d を掛けて全ペアで合計。
    est が空のペアは保守的に幅=1.0 として I_d*1.0 を加算。
    """
    s = 0.0
    for d, det in enumerate(per_pair_details):
        I = float(importance_list[d]) if d < len(importance_list) else 1.0
        alloc = det.get("alloc_by_path", {})
        est   = det.get("est_fid_by_path", {})
        if not est:
            s += I * 1.0
            continue
        widths = []
        for pid, m in est.items():
            n = int(alloc.get(pid, 0))
            rad = _ci_radius_hoeffding(n, delta)
            lb = max(0.0, float(m) - rad)
            ub = min(1.0, float(m) + rad)
            widths.append(ub - lb)
        s += I * (min(widths) if widths else 1.0)
    return float(s)


def plot_widthsum_alllinks_weighted_vs_budget(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    delta=0.1,
    verbose=True,
    print_every=1,
):
    """
    y = Σ_d Σ_l I_d·(UB-LB) の平均 ±95%CI、x = 目標予算。
    生データは outputs/plot_widthsum_alllinks_weighted_vs_budget_*.pickle に保存。
    """
    file_name = f"plot_widthsum_alllinks_weighted_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}

    for k, C_total in enumerate(budget_list):
        if verbose:
            print(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===", flush=True)

        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                print(f"  [repeat {r+1}/{repeat}]", flush=True)

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
                v = _sum_weighted_widths_all_links(per_pair_details, importance_list, delta=delta)
                results[name]["sums"][k].append(v)
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    print(f"    - {name}: wsum_alllinks={v:.4f} (used={total_cost})", flush=True)

    # --- Save raw data (.pickle) ---
    file_path = os.path.join(output_dir, f"{file_name}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump({"budget_list": list(budget_list), "results": results}, f)
    print(f"Saved pickle: {file_path}")

    # --- Plot mean ± 95% CI ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        means, halfs = [], []
        for vals in data["sums"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        ax.plot(x, means, linewidth=2.0, marker="o", label=name)
        ax.fill_between(x, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Weighted sum of (UB - LB) over all links (× I_d)")
    ax.grid(True); ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    print(f"Saved: {pdf_name}")


def plot_minwidthsum_perpair_weighted_vs_budget(
    budget_list,
    scheduler_names,
    noise_model,
    node_path_list,
    importance_list,
    bounces=(1, 2, 3, 4),
    repeat=10,
    delta=0.1,
    verbose=True,
    print_every=1,
):
    """
    y = Σ_d I_d·min_l(UB-LB) の平均 ±95%CI、x = 目標予算。
    生データは outputs/plot_minwidthsum_perpair_weighted_vs_budget_*.pickle に保存。
    """
    file_name = f"plot_minwidthsum_perpair_weighted_vs_budget_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results = {name: {"sums": [[] for _ in budget_list]} for name in scheduler_names}

    for k, C_total in enumerate(budget_list):
        if verbose:
            print(f"\n=== [{noise_model}] Budget={C_total} ({k+1}/{len(budget_list)}) ===", flush=True)

        fidelity_bank = [generate_fidelity_list_random(n) for n in node_path_list]

        def network_generator(path_num, pair_idx):
            return QuantumNetwork(path_num, fidelity_bank[pair_idx], noise_model)

        for r in range(repeat):
            if verbose and ((r + 1) % print_every == 0 or r == 0):
                print(f"  [repeat {r+1}/{repeat}]", flush=True)

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
                v = _sum_weighted_min_widths_per_pair(per_pair_details, importance_list, delta=delta)
                results[name]["sums"][k].append(v)
                if verbose and ((r + 1) % print_every == 0 or r == 0):
                    print(f"    - {name}: wsum_min_perpair={v:.4f} (used={total_cost})", flush=True)

    # --- Save raw data (.pickle) ---
    file_path = os.path.join(output_dir, f"{file_name}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump({"budget_list": list(budget_list), "results": results}, f)
    print(f"Saved pickle: {file_path}")

    # --- Plot mean ± 95% CI ---
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    x = list(budget_list)
    for name, data in results.items():
        means, halfs = [], []
        for vals in data["sums"]:
            m, h = mean_ci95(vals)
            means.append(m); halfs.append(h)
        means = np.asarray(means); halfs = np.asarray(halfs)
        ax.plot(x, means, linewidth=2.0, marker="o", label=name)
        ax.fill_between(x, means - halfs, means + halfs, alpha=0.25)

    ax.set_xlabel("Budget (target)")
    ax.set_ylabel("Weighted sum over pairs of min (UB - LB) (× I_d)")
    ax.grid(True); ax.legend(title="Scheduler")
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {pdf_name} {pdf_name}")
    print(f"Saved: {pdf_name}")


# =========================
# 95%CI helpers (repeats 可変対応)
# =========================
# 小 n 用の簡易表（両側95%、df = n-1）
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}


def tcrit_95(n: int) -> float:
    """repeats=n に対する両側95% t臨界値 (df=n-1)。n<2 は 0 を返す。"""
    if n <= 1:
        return 0.0
    df = n - 1
    if df in _T95:
        return _T95[df]
    if df >= 30:
        return 1.96  # 正規近似
    return 2.13  # 小 n 保守値


def mean_ci95(vals):
    """同一 budget 上の値列 vals（可変 n）に対して (mean, halfwidth) を返す。"""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(arr[0]), 0.0
    mean = float(arr.mean())
    s = float(arr.std(ddof=1))
    half = tcrit_95(n) * (s / math.sqrt(n))
    return mean, half


def _plot_with_ci_band(ax, xs, mean, half, *, label, line_kwargs=None, band_kwargs=None):
    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    band_kwargs = {"alpha": 0.25} | ({} if band_kwargs is None else dict(band_kwargs))
    ax.plot(xs, mean, label=label, **line_kwargs)
    ax.fill_between(xs, mean - half, mean + half, **band_kwargs)