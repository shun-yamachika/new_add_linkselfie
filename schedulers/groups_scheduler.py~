# schedulers/lnaive_scheduler.py
from .lnaive_nb import naive_network_benchmarking_with_budget

def lnaive_budget_scheduler(
    node_path_list,      # 例: [2, 2, 2] … 各ペアのパス本数
    importance_list,     # 例: [0.3, 0.5, 0.7] … 長さは node_path_list と同じ（ここでは未使用）
    bounces,             # 例: [1,2,3,4]（重複なし）
    C_total,             # 総予算（切り捨て配分、超過しない）
    network_generator,   # callable: (path_num, pair_idx) -> network
    return_details=False,
):
    num_pairs = len(node_path_list)
    assert num_pairs == len(importance_list), "length mismatch: node_path_list vs importance_list"
    if num_pairs == 0:
        return ([], 0, []) if return_details else ([], 0)

    assert len(bounces) == len(set(bounces)), "bounces must be unique"
    assert all(isinstance(w, int) and w > 0 for w in bounces), "bounces must be positive ints"

    # 均等配分：1ペアあたりの割当
    C_per_pair = int(C_total // max(num_pairs, 1))

    per_pair_results = []
    per_pair_details = []
    total_cost = 0

    for pair_idx, path_num in enumerate(node_path_list):
        if path_num <= 0:
            per_pair_results.append((False, 0, None))
            if return_details:
                per_pair_details.append({"alloc_by_path": {}, "est_fid_by_path": {}})
            continue

        network = network_generator(path_num, pair_idx)
        path_list = list(range(1, path_num + 1))

        if return_details:
            correctness, cost, best_path_fidelity, alloc_by_path, est_fid_by_path = \
                naive_network_benchmarking_with_budget(
                    network, path_list, list(bounces), C_per_pair, return_details=True
                )
            per_pair_details.append({
                "alloc_by_path": {int(k): int(v) for k, v in alloc_by_path.items()},
                "est_fid_by_path": {int(k): float(v) for k, v in est_fid_by_path.items()},
            })
        else:
            correctness, cost, best_path_fidelity = naive_network_benchmarking_with_budget(
                network, path_list, list(bounces), C_per_pair
            )

        per_pair_results.append((bool(correctness), int(cost), best_path_fidelity))
        total_cost += int(cost)

    return (per_pair_results, total_cost, per_pair_details) if return_details \
           else (per_pair_results, total_cost)
