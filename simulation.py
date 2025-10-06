# simulation.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import csv, os, math, random

# ===== 既存ネットワークAPIに合わせたアダプタ =====
class Adapter:
    """
    あなたの network.py / nb_protocol.py を変更せずに使うための薄いラッパ。
    - QuantumNetwork(path_num, fidelity_list, noise_model) を自前で構築
    - 単一ペア 'Alice-Bob' に path_id=1..path_num を割当
    - スケジューラが期待する nb_protocol 互換API（sample_path / true_fidelity）を Shim で提供
    """
    def __init__(self, noise_model: str, path_num: int, fidelity_list: List[float], seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        import network as qnet
        # QuantumNetwork を直接構築（network.py を変更しない）
        self.net = qnet.QuantumNetwork(path_num=path_num, fidelity_list=fidelity_list, noise_model=noise_model)
        self.pairs = ["Alice-Bob"]
        self.paths_map = {"Alice-Bob": list(range(1, path_num + 1))}
        # nb_protocol 互換 Shim
        self.nbp = _NBPShim(self.net)

    # ---- ヘルパ ----
    def true_fidelity(self, path_id: Any) -> float:
        return self.nbp.true_fidelity(self.net, path_id)

    def list_pairs(self) -> List[Any]:
        return list(self.pairs)

    def list_paths_of(self, pair_id: Any) -> List[Any]:
        return list(self.paths_map.get(pair_id, []))

    # ---- スケジューラ呼び出し ----
    def run_scheduler(self, scheduler_name: str, budget_target: int,
                      importance: Dict[Any, float]) -> Dict[str, Any]:
        """
        スケジューラに共通IFで実行要求する。
        返り値の想定（辞書）:
          {
            'used_cost_total': int,
            'per_pair_details': [
               {
                 'pair_id': pair_id,
                 'alloc_by_path': {path_id: sample_count, ...},
                 'est_fid_by_path': {path_id: mean_estimate, ...},
                 'best_pred_path': path_id,
               }, ...
            ]
          }
        """
        if scheduler_name == "greedy":
            from schedulers.greedy_scheduler import run as greedy_run
            return greedy_run(self.net, self.pairs, self.paths_map, budget_target, importance, self.nbp)
        elif scheduler_name == "naive":
            from schedulers.lnaive_scheduler import run as naive_run
            return naive_run(self.net, self.pairs, self.paths_map, budget_target, importance, self.nbp)
        elif scheduler_name == "online_nb":
            from schedulers.lonline_nb import run as onb_run
            return onb_run(self.net, self.pairs, self.paths_map, budget_target, importance, self.nbp)
        else:
            raise ValueError(f"unknown scheduler: {scheduler_name}")

# ===== 便利関数 =====
def hoeffding_radius(n: int, delta: float = 0.05) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(0.5 * math.log(2.0 / delta) / n)

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# ===== CSV I/O =====
CSV_HEADER = [
    "run_id", "noise", "scheduler", "budget_target",
    "used_cost_total",
    "pair_id", "path_id",
    "importance",               # I_d
    "samples",                  # B_{d,l}
    "est_mean", "lb", "ub", "width",
    "is_best_true", "is_best_pred"
]

def open_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow(CSV_HEADER)
    return f, w

# ===== メインシミュレーション =====
@dataclass
class ExperimentConfig:
    noise_model: str
    budgets: List[int]
    schedulers: List[str]            # ["greedy", "naive", "online_nb", ...]
    repeats: int
    importance_mode: str = "both"    # "both" / "weighted_only" / "unweighted_only"
    delta_ci: float = 0.05           # 95%CI相当
    out_csv: str = "outputs/raw_simulation_data.csv"
    seed: int | None = None
    # QuantumNetwork 構築用
    path_num: int = 5
    fidelity_list: List[float] | None = None

def _importance_for_pairs(pairs: List[Any], mode: str) -> Dict[str, Dict[Any, float]]:
    res: Dict[str, Dict[Any, float]] = {}
    if mode in ("both", "unweighted_only"):
        res["unweighted"] = {p: 1.0 for p in pairs}
    if mode in ("both", "weighted_only"):
        # 重要度は例として一様乱数（必要なら差替え）
        res["weighted"] = {p: 0.5 + random.random() for p in pairs}
    return res

def run_and_append_csv(cfg: ExperimentConfig) -> str:
    fid = cfg.fidelity_list or _default_fidelities(cfg.path_num)
    adp = Adapter(cfg.noise_model, cfg.path_num, fid, seed=cfg.seed)
    pairs = adp.list_pairs()
    importance_sets = _importance_for_pairs(pairs, cfg.importance_mode)

    f, w = open_csv(cfg.out_csv)
    try:
        run_id = 0
        for _ in range(cfg.repeats):
            run_id += 1
            for budget in cfg.budgets:
                for sched in cfg.schedulers:
                    for imp_tag, I in importance_sets.items():
                        # スケジューラ実行
                        result = adp.run_scheduler(sched, budget, I)

                        used_cost_total = int(result.get("used_cost_total", budget))
                        per_pair_details: List[Dict[str, Any]] = result.get("per_pair_details", [])

                        # 真の最良パス（正答率判定用）
                        true_best_by_pair = {}
                        for pair in pairs:
                            paths = adp.list_paths_of(pair)
                            best = None
                            bestv = -1.0
                            for pid in paths:
                                tf = adp.true_fidelity(pid)
                                if tf > bestv:
                                    bestv, best = tf, pid
                            true_best_by_pair[pair] = best

                        # CSV行を形成
                        for det in per_pair_details:
                            pair_id = det["pair_id"]
                            alloc = det.get("alloc_by_path", {}) or {}
                            est   = det.get("est_fid_by_path", {}) or {}
                            pred  = det.get("best_pred_path")

                            for path_id, samples in alloc.items():
                                m = float(est.get(path_id, 0.5))
                                r = hoeffding_radius(int(samples), cfg.delta_ci)
                                lb = clamp01(m - r)
                                ub = clamp01(m + r)
                                width = ub - lb

                                is_true_best = (true_best_by_pair.get(pair_id) == path_id)
                                is_best_pred = (pred == path_id)

                                w.writerow([
                                    f"{run_id}-{imp_tag}",
                                    cfg.noise_model,
                                    sched,
                                    budget,
                                    used_cost_total,
                                    pair_id,
                                    path_id,
                                    I.get(pair_id, 1.0),
                                    int(samples),
                                    m, lb, ub, width,
                                    int(is_true_best), int(is_best_pred),
                                ])
    finally:
        f.close()
    return cfg.out_csv

# ===== nb_protocol 互換 Shim =====
class _NBPShim:
    """
    スケジューラが期待する nb_protocol 風のAPIを提供:
      - sample_path(net, path_id, n): QuantumNetwork.benchmark_path を呼ぶ
      - true_fidelity(net, path_id): 量子チャネルの ground truth を返す
    """
    def __init__(self, net):
        self.net = net

    def sample_path(self, net, path_id: int, n: int) -> float:
        # 1-bounce を n 回の測定にマップ（nb_protocol.NBProtocolAlice の設計に整合）
        p, _cost = self.net.benchmark_path(path_id, bounces=[1], sample_times={1: int(n)})
        return float(p)

    def true_fidelity(self, net, path_id: int) -> float:
        return float(self.net.quantum_channels[path_id - 1].fidelity)

# ===== デフォルト忠実度の簡易生成（必要なら差替え） =====
def _default_fidelities(path_num: int) -> List[float]:
    alpha, beta, var = 0.93, 0.85, 0.02
    res = [max(0.8, min(0.999, random.gauss(beta, var))) for _ in range(path_num)]
    best_idx = random.randrange(path_num)
    res[best_idx] = max(0.85, min(0.999, random.gauss(alpha, var)))
    return res
