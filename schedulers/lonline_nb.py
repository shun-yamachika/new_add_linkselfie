# === lonline_nb.py: 末尾の追記ブロック（修正版） ===
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
    広域探索フェーズ（s=1 の 1 ラウンドのみ）。候補全リンクに一律 Ns セットを投入できる場合に限り実行。
    出力:
      correctness, cost, best_path_fidelity, [alloc_by_path, est_fid_by_path,] state
    """
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
            "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": list(bounces),
            "C_const": C_const, "delta": delta, "min_sets": min_sets
        }
        return (*base, state)

    c_B = sum(bounces) if sum(bounces) > 0 else 1
    s = 1
    Ns = _ns_for_round(s, len(candidate_set), C_const, delta, min_sets)

    # ★ フェーズ単位の均等投入: 候補全体に Ns セットを入れられるかを事前判定
    round_cost_all = len(candidate_set) * Ns * c_B
    if round_cost_all > C_budget:
        base = (False, 0, None)
        if return_details:
            base += (alloc_by_path, est_fid_by_path)
        state = {
            "s": s, "candidate_set": candidate_set, "estimated_fidelities": estimated_fidelities,
            "alloc_by_path": alloc_by_path, "est_fid_by_path": est_fid_by_path, "bounces": list(bounces),
            "C_const": C_const, "delta": delta, "min_sets": min_sets
        }
        return (*base, state)

    # ここから候補全リンクに一律 Ns セット投入（途中打ち切りなし）
    sample_times = {h: int(Ns) for h in bounces}
    p_s, measured = {}, []
    for path in list(candidate_set):
        p, used = network.benchmark_path(path, bounces, sample_times)
        cost += int(used)  # 実質的に round_cost_all と一致するはず
        fidelity = p + (1 - p) / 2.0
        estimated_fidelities[path] = fidelity
        p_s[path] = p
        measured.append(path)
        alloc_by_path[int(path)] = alloc_by_path.get(int(path), 0) + int(used)
        est_fid_by_path[int(path)] = float(fidelity)

    # 逐次除去（現時点では 2^{-s} 閾値ルール）
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
    集中的活用フェーズ。state を引き継ぎ s=2 以降を「フェーズ単位」で実施。
    1フェーズ分の均等投入コストが入らない場合は何も実行せず insufficient_budget=True を返す。
    出力:
      correctness, cost, best_path_fidelity, [alloc_by_path, est_fid_by_path,] new_state, insufficient_budget
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

    c_B = sum(bounces) if sum(bounces) > 0 else 1
    insufficient_budget = False

    while cost < C_budget and len(candidate_set) > 1:
        s += 1
        Ns = _ns_for_round(s, len(candidate_set), C_const, delta, min_sets)

        # ★ フェーズ単位の均等投入判定（候補全リンクに一律 Ns セット）
        round_cost_all = len(candidate_set) * Ns * c_B
        if cost + round_cost_all > C_budget:
            insufficient_budget = True
            s -= 1  # このフェーズは未実行
            break

        sample_times = {h: int(Ns) for h in bounces}
        p_s, measured = {}, []

        # 候補全リンクに一律 Ns セット投入（途中打ち切りなし）
        for path in list(candidate_set):
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

        # 逐次除去（現時点では 2^{-s} 閾値ルール）
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
# === 修正ここまで ===
def _dry_phase_cost(state, C_budget, C_const=None, delta=None, min_sets=None):
    import math
    s = int(state.get("s", 1))
    k = len(state.get("candidate_set", []))
    bounces = state.get("bounces", [])
    c_B = sum(bounces) if bounces else 1
    C_const = state.get("C_const", 0.01) if C_const is None else C_const
    delta   = state.get("delta", 0.1)    if delta   is None else delta
    min_sets = state.get("min_sets", 4)  if min_sets is None else min_sets
    def Ns(s,k): 
        val = math.ceil(C_const * (2**(2*s)) * math.log2(max((2**s)*k/delta, 2)))
        return max(val, min_sets)
    need_s2 = k * Ns(2,k) * c_B
    need_s3 = k * Ns(3,k) * c_B
    return dict(s=s, k=k, c_B=c_B, need_s2=need_s2, need_s3=need_s3, C_budget=int(C_budget))
