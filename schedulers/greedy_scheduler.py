# schedulers/greedy_scheduler.py
from .lonline_nb import lonline_init, lonline_continue

def greedy_budget_scheduler(
    node_path_list,      # 例: [2, 2, 2] … 各ペアのパス本数
    importance_list,     # 例: [0.3, 0.5, 0.7] … 長さは node_path_list と同じ
    bounces,             # 例: [1,2,3,4]（重複なし）
    C_total,             # 総予算
    network_generator,   # callable: (path_num, pair_idx) -> network
    min_sets_per_link=4, # （互換のため残すが、lonline_init側で min=4 を保証するので未使用でもOK）
    return_details=False,
    # 既定値は現状コードと同じ：C_const=0.01, delta=0.1
    C_const=0.01,
    delta=0.1,
):
    """
    Two-Phase Greedy（広域探索→集中的活用）用の Greedy スケジューラ。

    フェーズ1（広域探索）:
      - 各ペア n で lonline_init(..., s=1 のみ) を呼ぶ。
      - best_path_fid を f_n^(init) として受け取り、価値 V_n = I_n * f_n^(init) を算出。
      - s と candidate_set を含む state_n を保持する（あとで s=2 から再開）。

    フェーズ2（集中的活用）:
      - V_n の降順に各ペアを処理し、lonline_continue(..., state=state_n) で s=2 以降を実行。
      - 予算が次ラウンドに入らないとき、lonline_continue は insufficient_budget=True を返すので、
        そのペアはスキップして次のペアへ。（コストは増えない）
      - 候補が1本に収束したら、そのペアは確定とみなして以後は回さない。
    """
    assert len(node_path_list) == len(importance_list), "node_path_list と importance_list の長さが一致しません。"

    N_pairs = len(node_path_list)
    consumed_total = 0

    # 返却用の器
    per_pair_results = {i: (False, 0, None) for i in range(N_pairs)}  # (correctness, cost, best_path_fid)
    per_pair_details = [
        {"alloc_by_path": {}, "est_fid_by_path": {}} for _ in range(N_pairs)
    ] if return_details else None

    # フェーズ1: 広域探索（各ペアで s=1 を1ラウンド分だけ）
    # ここでは、lonline_init に「残余予算」をそのまま渡しても s=1 しか実行せず、
    # 各リンク1回の s=1 ラウンド分ずつしか消費しない実装になっている。
    networks   = [None] * N_pairs
    states     = [None] * N_pairs
    f_init     = [0.0] * N_pairs
    init_costs = [0] * N_pairs

    for pair_idx, path_num in enumerate(node_path_list):
        if consumed_total >= C_total or path_num <= 0:
            continue

        remaining = int(C_total) - int(consumed_total)
        if remaining <= 0:
            break

        network = network_generator(path_num, pair_idx)
        path_list = list(range(1, path_num + 1))
        networks[pair_idx] = network

        if return_details:
            correctness, cost, best_path_fid, alloc0, est0, state = lonline_init(
                network, path_list, list(bounces), int(remaining),
                return_details=True, C_const=C_const, delta=delta, min_sets=4
            )
            # 詳細をマージ（配分は加算・推定は後勝ち）
            for l, b in alloc0.items():
                per_pair_details[pair_idx]["alloc_by_path"][int(l)] = \
                    per_pair_details[pair_idx]["alloc_by_path"].get(int(l), 0) + int(b)
            per_pair_details[pair_idx]["est_fid_by_path"].update({int(k): float(v) for k, v in est0.items()})
        else:
            correctness, cost, best_path_fid, state = lonline_init(
                network, path_list, list(bounces), int(remaining),
                return_details=False, C_const=C_const, delta=delta, min_sets=4
            )

        # 記録
        init_costs[pair_idx] = int(cost)
        consumed_total      += int(cost)
        states[pair_idx]     = state
        f_init[pair_idx]     = float(best_path_fid) if best_path_fid is not None else 0.0

        # 初期段階の結果も per_pair_results に入れておく（後で加算更新する）
        per_pair_results[pair_idx] = (
            bool(correctness),
            int(cost),
            best_path_fid,
        )

        if consumed_total >= C_total:
            break

    # V_n = I_n * f_n^(init) に基づいて、集中的活用フェーズの順序を決定
    order = sorted(
        range(N_pairs),
        key=lambda i: (importance_list[i] * f_init[i]) if node_path_list[i] > 0 else -1.0,
        reverse=True
    )

    # フェーズ2: 集中的活用（s=2 以降を V_n の高い順で実行）
    for pair_idx in order:
        if consumed_total >= C_total:
            break
        if node_path_list[pair_idx] <= 0:
            continue
        if states[pair_idx] is None:
            continue  # 初期フェーズで何もできなかった

        network   = networks[pair_idx]
        state     = states[pair_idx]

        while consumed_total < C_total:
            remaining = int(C_total) - int(consumed_total)
            if remaining <= 0:
                break

            if return_details:
                correctness, cost, best_path_fid, alloc_more, est_more, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=True
                )
                # もし insufficient=True なら、次ラウンドの最小単位すら入らない → コストは増えない
                if insufficient:
                    break

                # 詳細マージ
                for l, b in alloc_more.items():
                    per_pair_details[pair_idx]["alloc_by_path"][int(l)] = \
                        per_pair_details[pair_idx]["alloc_by_path"].get(int(l), 0) + int(b)
                per_pair_details[pair_idx]["est_fid_by_path"].update({int(k): float(v) for k, v in est_more.items()})
            else:
                correctness, cost, best_path_fid, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=False
                )
                if insufficient:
                    break

            # 成果の反映
            consumed_total += int(cost)
            state           = new_state
            states[pair_idx]= new_state  # 念のため保持

            # 既存結果に加算更新
            prev_correct, prev_cost, _prev_fid = per_pair_results[pair_idx]
            per_pair_results[pair_idx] = (
                bool(correctness),                          # 最新の correctness を採用
                int(prev_cost) + int(cost),                 # コストは加算
                best_path_fid,                              # 最新の fid を採用
            )

            # 候補が1本に絞れていれば、このペアは確定とみなして次へ
            cand = list(new_state.get("candidate_set", []))
            if len(cand) <= 1:
                break

            if consumed_total >= C_total:
                break

    # 返却（互換維持）
    return (per_pair_results, int(consumed_total), per_pair_details) if return_details \
           else (per_pair_results, int(consumed_total))
