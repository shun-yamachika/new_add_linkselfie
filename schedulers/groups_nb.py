# groups_nb.py
import random  # グループ内から測定リンクをランダム選択

def groups_network_benchmarking_with_budget(network, path_list, bounces, C_budget, groups, return_details=True):
    fidelity = {}
    cost = 0
    # groups バリデーション（1-origin想定）
    ids_set = set(int(p) for p in path_list)
    flat = [int(x) for g in groups for x in g]
    if not flat:
        raise ValueError("groups must not be empty")
    if any((pid not in ids_set) for pid in flat):
        raise ValueError("groups contains invalid path id(s)")
    # 全被覆を強制したいなら以下を有効化（任意）
    # if set(flat) != ids_set:
    #     raise ValueError("groups must cover all path ids exactly")


    
    per_sample_cost = sum(bounces) or 1
    # 等分配は「グループ本数」基準
    n_groups = max(1, len(groups))
    per_group_budget = int(C_budget) // n_groups
    Ns = per_group_budget // per_sample_cost

    if Ns <= 0:
        if return_details:
            return False, 0, None, {}, {}
        return False, 0, None
    

    # 追加: 詳細記録用
    alloc_by_path = {int(p): 0 for p in path_list}
    est_fid_by_path = {}


    # （変更後）
    # 各グループについて Ns 回まわし、毎回ランダムに 1 本を選んで測定
    sample_times_one = {h: 1 for h in bounces}   # ★ 1回分のNBセット
    for grp in groups:
        f_sum = 0.0
        for _ in range(int(Ns)):                 # ★ Ns 回くり返す
            chosen = int(random.choice(grp))
            p, used = network.benchmark_path(chosen, bounces, sample_times_one)
            f = p + (1 - p) / 2.0                # 既存の忠実度変換式
            f_sum += f
            cost += int(used)
            alloc_by_path[chosen] = alloc_by_path.get(chosen, 0) + int(used)

        # ★ グループの推定値は Ns 回の平均を全リンクにコピー
        f_group = f_sum / float(Ns)
        for pid in grp:
            pid = int(pid)
            fidelity[pid] = f_group
            est_fid_by_path[pid] = float(f_group)

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
