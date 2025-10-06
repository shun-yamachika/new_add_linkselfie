# lnaive_nb.py
def naive_network_benchmarking_with_budget(network, path_list, bounces, C_budget):
    """
    目的:
      総予算 C_budget を各パスへ均等割り当てし、均等サンプリングで NB を実行。
      実行コストは常に予算内（超過しない）。

    出力:
      (correctness, cost, best_path_fidelity)
        correctness … 推定最良パスが真の最良と一致したか
        cost        … 実測で消費した総コスト
        best_path_fidelity … 推定最良パスの推定忠実度（naive変換後）
    """
    fidelity, cost = {}, 0
    n_paths = len(path_list)
    if n_paths == 0:
        return False, 0, None

    per_sample_cost = sum(bounces) or 1
    per_path_budget = int(C_budget) // n_paths
    Ns = per_path_budget // per_sample_cost  # 各パスのサンプル数
    if Ns <= 0:
        return False, 0, None

    # 各 hop に同じ Ns を配る（既存 naive と同じ割当表）
    sample_times = {h: int(Ns) for h in bounces}

    # 各パスを均等回数でベンチマーク
    for path in path_list:
        p, used = network.benchmark_path(path, bounces, sample_times)
        fidelity[path] = p + (1 - p) / 2  # 既存 naive と同じ忠実度変換
        cost += used

    if not fidelity:
        return False, cost, None

    best_path = max(fidelity, key=fidelity.get)
    correctness = (best_path == getattr(network, "best_path", None))
    best_path_fidelity = fidelity[best_path]
    return correctness, cost, best_path_fidelity
