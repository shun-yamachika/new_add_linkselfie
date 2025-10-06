# metrics/widths.py — width metrics (Py3.8 safe typing)

import math
from typing import List, Dict, Any

def ci_radius_hoeffding(n: int, delta: float = 0.1) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(0.5 * math.log(2.0 / delta) / n)

def _widths_for_details_single_pair(det: Dict[str, Any], delta: float) -> List[float]:
    alloc = det.get("alloc_by_path", {}) or {}
    est   = det.get("est_fid_by_path", {}) or {}
    widths: List[float] = []
    for pid, m in est.items():
        n = int(alloc.get(pid, 0))
        r = ci_radius_hoeffding(n, delta)
        lb = max(0.0, float(m) - r)
        ub = min(1.0, float(m) + r)
        widths.append(ub - lb)
    return widths

def sum_widths_all_links(per_pair_details: List[Dict[str, Any]], delta: float = 0.1) -> float:
    s = 0.0
    for det in per_pair_details:
        for w in _widths_for_details_single_pair(det, delta):
            s += float(w)
    return s

def sum_minwidths_perpair(per_pair_details: List[Dict[str, Any]], delta: float = 0.1) -> float:
    s = 0.0
    for det in per_pair_details:
        widths = _widths_for_details_single_pair(det, delta)
        if widths:
            s += float(min(widths))
        else:
            s += 1.0  # 未推定ペアは1.0を加算（保守的）
    return s

def sum_weighted_widths_all_links(per_pair_details: List[Dict[str, Any]], importance_list: List[float], delta: float = 0.1) -> float:
    s = 0.0
    for d, det in enumerate(per_pair_details):
        I = float(importance_list[d]) if d < len(importance_list) else 1.0
        for w in _widths_for_details_single_pair(det, delta):
            s += I * float(w)
    return s

def sum_weighted_min_widths_perpair(per_pair_details: List[Dict[str, Any]], importance_list: List[float], delta: float = 0.1) -> float:
    s = 0.0
    for d, det in enumerate(per_pair_details):
        I = float(importance_list[d]) if d < len(importance_list) else 1.0
        widths = _widths_for_details_single_pair(det, delta)
        if widths:
            s += I * float(min(widths))
        else:
            s += I * 1.0
    return s
