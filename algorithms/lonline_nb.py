# === ここから追記（lonline_nb.py の末尾などに） ===
import math

def _ns_for_round(s, k, C_const, delta, min_sets):
    """LinkSelFiE 由来の Ns(s,k) を計算（k=len(candidate_set)）。"""
    Ns = math.ceil(C_const * (2 ** (2 * s)) * math.log2(max((2 ** s) * k / delta, 2)))
    return max(Ns, min_sets)

def lonline_init(
    network, path_list, bounces, C_budget,
    *, return_details=False, C_const=0.01, delta=0.1, min_sets=4
):
    """
    広域探索フェーズ（s=1 の 1 ラウンドだけ実行）を行い、state を返す。
    出力:
      correctness, cost, best_path_fidelity, [alloc_by_path, est_fid_by_path,] state
    state には { 's': 1, 'candidate_set': [...], 'estimated_fidelities': {...},
                 'alloc_by_path': {...}, 'est_fid_by_path': {...}, 'bounces': [...]} を含む
    """
    # 1ラウンドぶんだけ消費して、逐次除去して帰る実装
    candidate_set = list(path_list)
    alloc_by_path = {int(p): 0 for p in path_list}
    est_fid_by_path = {}
    estimated_fidelities = {}
    cost = 0
    if not candidate_set or C_budget <= 0:
        base = (False, 0, None)
        if return_details:
            base += (alloc_by_path, est_fid_by_path)
        state = {
            "s": 1, "candidate_set": candidate_set, "estimated_fidelities": estimated_fidelities,
            "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": list(bounces)
        }
        return (*base, state)

    cost_per_sample_unit = sum(bounces) if sum(bounces) > 0 else 1
    s = 1
    Ns = _ns_for_round(s, len(candidate_set), C_const, delta, min_sets)
    sample_cost_one_path = Ns * cost_per_sample_unit
    if sample_cost_one_path > C_budget:
        # そもそも s=1 を 1経路すら回せない → 何もせず state だけ返す
        base = (False, 0, None)
        if return_details:
            base += (alloc_by_path, est_fid_by_path)
        state = {
            "s": 1, "candidate_set": candidate_set, "estimated_fidelities": estimated_fidelities,
            "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": list(bounces)
        }
        return (*base, state)

    sample_times = {h: int(Ns) for h in bounces}
    p_s = {}
    measured = []
    for path in list(candidate_set):
        if cost + sample_cost_one_path > C_budget:
            break
        p, used = network.benchmark_path(path, bounces, sample_times)
        cost += int(used)
        fidelity = p + (1 - p) / 2.0
        estimated_fidelities[path] = fidelity
        p_s[path] = p
        measured.append(path)
        alloc_by_path[int(path)] = alloc_by_path.get(int(path), 0) + int(used)
        est_fid_by_path[int(path)] = float(fidelity)

    if p_s:
        p_max = max(p_s.values())
        new_cand = [path for path in measured if (p_s[path] + 2 ** (-s) > p_max - 2 ** (-s))]
        candidate_set = new_cand or candidate_set

    best_path_fid = None
    if estimated_fidelities:
        best_path = max(estimated_fidelities, key=estimated_fidelities.get)
        best_path_fid = estimated_fidelities[best_path]
        correctness = (best_path == getattr(network, "best_path", None))
    else:
        correctness = False

    state = {
        "s": s, "candidate_set": candidate_set, "estimated_fidelities": estimated_fidelities,
        "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": list(bounces),
        "C_const": C_const, "delta": delta, "min_sets": min_sets
    }
    base = (bool(correctness), int(cost), best_path_fid)
    if return_details:
        base += (alloc_by_path, est_fid_by_path)
    return (*base, state)

def lonline_continue(
    network, C_budget, *, state, return_details=False,
    C_const=None, delta=None, min_sets=None
):
    """
    集中的活用フェーズ。state（s>=1, candidate_set など）を引き継いで s=2 以降を実行。
    予算が次ラウンドの 1経路分すら足りない場合は、コストを消費せず
    insufficient_budget=True を返す。

    出力:
      correctness, cost, best_path_fidelity, [alloc_by_path, est_fid_by_path,]
      new_state, insufficient_budget
    """
    # state から引き継ぎ
    s = int(state.get("s", 1))
    candidate_set = list(state.get("candidate_set", []))
    estimated_fidelities = dict(state.get("estimated_fidelities", {}))
    alloc_by_path = dict(state.get("alloc_by_path", {}))
    est_fid_by_path = dict(state.get("est_fid_by_path", {}))
    bounces = list(state.get("bounces", []))

    C_const = state.get("C_const", 0.01) if C_const is None else C_const
    delta   = state.get("delta", 0.1)    if delta   is None else delta
    min_sets = state.get("min_sets", 4)  if min_sets is None else min_sets

    cost = 0
    if not candidate_set or C_budget <= 0 or len(candidate_set) <= 1:
        # 宛先が確定している or 予算なし
        best_path_fid = None
        if estimated_fidelities:
            best_path = max(estimated_fidelities, key=estimated_fidelities.get)
            best_path_fid = estimated_fidelities[best_path]
            correctness = (best_path == getattr(network, "best_path", None))
        else:
            correctness = False
        base = (bool(correctness), int(cost), best_path_fid)
        if return_details:
            base += (alloc_by_path, est_fid_by_path)
        return (*base, {**state, "s": s}, False)

    cost_per_sample_unit = sum(bounces) if sum(bounces) > 0 else 1

    insufficient_budget = False
    while cost < C_budget and len(candidate_set) > 1:
        s += 1
        Ns = _ns_for_round(s, len(candidate_set), C_const, delta, min_sets)
        sample_cost_one_path = Ns * cost_per_sample_unit

        if cost + sample_cost_one_path > C_budget:
            # この宛先では次ラウンドに入れない → 消費せず終了
            insufficient_budget = True
            s -= 1  # 進めていないので据え置き
            break

        sample_times = {h: int(Ns) for h in bounces}
        p_s = {}
        measured = []

        for path in list(candidate_set):
            if cost + sample_cost_one_path > C_budget:
                break
            p, used = network.benchmark_path(path, bounces, sample_times)
            cost += int(used)
            fidelity = p + (1 - p) / 2.0
            estimated_fidelities[path] = fidelity
            p_s[path] = p
            measured.append(path)
            alloc_by_path[int(path)] = alloc_by_path.get(int(path), 0) + int(used)
            est_fid_by_path[int(path)] = float(fidelity)

        if not p_s:
            break

        p_max = max(p_s.values())
        new_cand = [path for path in measured if (p_s[path] + 2 ** (-s) > p_max - 2 ** (-s))]
        candidate_set = new_cand or candidate_set

    best_path_fid = None
    if estimated_fidelities:
        best_path = max(estimated_fidelities, key=estimated_fidelities.get)
        best_path_fid = estimated_fidelities[best_path]
        correctness = (best_path == getattr(network, "best_path", None))
    else:
        correctness = False

    new_state = {
        "s": s, "candidate_set": candidate_set, "estimated_fidelities": estimated_fidelities,
        "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": bounces,
        "C_const": C_const, "delta": delta, "min_sets": min_sets
    }
    base = (bool(correctness), int(cost), best_path_fid)
    if return_details:
        base += (alloc_by_path, est_fid_by_path)
    return (*base, new_state, insufficient_budget)
# === 追記ここまで ===
