from __future__ import annotations
import numpy as np

def generate_fidelity_list_avg_gap(path_num: int):
    assert path_num >= 1
    fidelity_max = 1.0
    fidelity_min = 0.9
    if path_num == 1:
        return [fidelity_max]
    gap = (fidelity_max - fidelity_min) / (path_num - 1)
    return [fidelity_max - i * gap for i in range(path_num)]

def generate_fidelity_list_fix_gap(path_num: int, gap: float, fidelity_max: float = 1.0):
    assert path_num >= 1
    return [float(fidelity_max) - i * gap for i in range(path_num)]

def generate_fidelity_list_random(path_num: int, alpha: float = 0.90, beta: float = 0.85, variance: float = 0.1):
    """(非決定版) Generate `path_num` links with a guaranteed top-1 gap."""
    assert path_num >= 2
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        result = []
        for mu in mean:
            # [0.8, 1.0] の範囲に入るまでサンプリング
            while True:
                r = np.random.normal(mu, variance)
                if 0.8 <= r <= 1.0:
                    break
            result.append(float(r))
        sorted_res = sorted(result, reverse=True)
        if sorted_res[0] - sorted_res[1] > 0.02:
            return result

# 再現性のため：rng を使ったバージョン（← 正式名はアンダースコア付き）
def _generate_fidelity_list_random_rng(rng: np.random.Generator, path_num: int,
                                       alpha: float = 0.86, beta: float = 0.85, variance: float = 0.1):
    assert path_num >= 2
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
        if sorted_res[0] - sorted_res[1] > 0.005:
            return res

__all__ = [
    "generate_fidelity_list_avg_gap",
    "generate_fidelity_list_fix_gap",
    "generate_fidelity_list_random",
    "_generate_fidelity_list_random_rng",  # ← 公開はアンダースコア版のみ
]
