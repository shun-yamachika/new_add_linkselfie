# lnaive_nb.py

def naive_network_benchmarking_with_budget(network, path_list, bounces, C_budget, return_details=False):
    """
    均等配分（L-Naive）版 NB。

    既存の戻り値:
      correctness: bool
      cost: int
      best_path_fidelity: float | None

    return_details=True のとき追加で返す:
      alloc_by_path:   dict[int,int]    {path_id: バウンス総数}
      est_fid_by_path: dict[int,float]  {path_id: 推定忠実度}

    想定:
      - network.benchmark_path(path, bounces, sample_times) -> (p, used_cost)
      - 忠実度変換は既存と同じ: fidelity = p + (1 - p)/2
    """
    fidelity = {}
    cost = 0
    n_paths = len(path_list)
    if n_paths == 0:
        if return_details:
            return False, 0, None, {}, {}
        return False, 0, None

    per_sample_cost = sum(bounces) or 1
    per_path_budget = int(C_budget) // n_paths
    Ns = per_path_budget // per_sample_cost  # 各パスのサンプル数
    if Ns <= 0:
        if return_details:
            return False, 0, None, {}, {}
        return False, 0, None

    # 各 hop に同じ Ns を配る（既存 naive と同じ割当表）
    sample_times = {h: int(Ns) for h in bounces}

    # 追加: 詳細記録用
    alloc_by_path = {int(p): 0 for p in path_list}
    est_fid_by_path = {}

    # 各パスを均等回数でベンチマーク
    for path in path_list:
        p, used = network.benchmark_path(path, bounces, sample_times)
        f = p + (1 - p) / 2.0  # 忠実度変換（既存式）
        fidelity[path] = f
        cost += int(used)

        # 追加: 詳細記録
        alloc_by_path[int(path)] = alloc_by_path.get(int(path), 0) + int(used)
        est_fid_by_path[int(path)] = float(f)

    if not fidelity:
        if return_details:
            return False, int(cost), None, alloc_by_path, est_fid_by_path
        return False, int(cost), None

    best_path = max(fidelity, key=fidelity.get)
    correctness = (best_path == getattr(network, "best_path", None))
    best_path_fidelity = fidelity[best_path]

    if return_details:
        return bool(correctness), int(cost), best_path_fidelity, alloc_by_path, est_fid_by_path
    return bool(correctness), int(cost), best_path_fidelity
