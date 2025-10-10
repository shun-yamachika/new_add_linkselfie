# schedulers/greedy_scheduler.py
from .lonline_nb import lonline_init, lonline_continue

def greedy_budget_scheduler(
    node_path_list,      # 例: [2, 2, 2] … 各ペアのパス本数
    importance_list,     # 例: [0.3, 0.5, 0.7] … 長さは node_path_list と同じ
    bounces,             # 例: [1,2,3,4]（重複なし）
    C_total,             # 総予算
    network_generator,   # callable: (path_num, pair_idx) -> network
    min_sets_per_link=4, # 互換用（lonline 側で min=4 を保証）
    return_details=False,
    # 既定値は現状コードと同じ：C_const=0.01, delta=0.1
    C_const=0.01,
    delta=0.1,
):
    """
    Two-Phase Greedy スケジューラ（1-origin対応）
    - 入出力キーは常に 1..L
    """

    # 前処理
    N_pairs = len(node_path_list)
    networks         = [None] * N_pairs
    states           = [None] * N_pairs
    per_pair_results = [(False, 0, None)] * N_pairs
    per_pair_details = [dict(alloc_by_path={}, est_fid_by_path={}) for _ in range(N_pairs)]
    init_costs       = [0] * N_pairs
    f_init           = [0.0] * N_pairs
    consumed_total   = 0

    # -----------------------
    # フェーズ1: 広域探索
    # -----------------------
    for pair_idx, path_num in enumerate(node_path_list):
        if consumed_total >= C_total or path_num <= 0:
             per_pair_results[pair_idx] = (False, 0, None)
             continue


        print(f"[INIT] pair={pair_idx} remain={int(C_total)-int(consumed_total)} "
              f"paths={path_num} bounces={bounces}")

        remaining = int(C_total) - int(consumed_total)
        if remaining <= 0:
            break

        # ★ 1-origin の path_list
        path_list = list(range(1, int(path_num) + 1))
        network = network_generator(int(path_num), pair_idx)
        networks[pair_idx] = network

        if return_details:
            correctness, cost, best_path_fid, alloc0, est0, state = lonline_init(
                network, path_list, list(bounces), int(remaining),
                return_details=True, C_const=C_const, delta=delta, min_sets=4
            )
            for l, b in alloc0.items():
                per_pair_details[pair_idx]["alloc_by_path"][int(l)] = \
                    per_pair_details[pair_idx]["alloc_by_path"].get(int(l), 0) + int(b)
            per_pair_details[pair_idx]["est_fid_by_path"].update(
                {int(k): float(v) for k, v in est0.items()}
            )
            print(f"[INIT<-] pair={pair_idx} cost={int(cost)} best_path_fid={best_path_fid} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")
            _sum_after_init = sum(per_pair_details[pair_idx]["alloc_by_path"].values())
            print(f"[CHECK:init] pair={pair_idx} sum_alloc_by_path={_sum_after_init} "
                  f"(should equal init cost={int(cost)})")
        else:
            correctness, cost, best_path_fid, state = lonline_init(
                network, path_list, list(bounces), int(remaining),
                return_details=False, C_const=C_const, delta=delta, min_sets=4
            )
            print(f"[INIT<-] pair={pair_idx} cost={int(cost)} best_path_fid={best_path_fid} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")

        init_costs[pair_idx] = int(cost)
        consumed_total      += int(cost)
        states[pair_idx]     = state
        f_init[pair_idx]     = float(best_path_fid) if best_path_fid is not None else 0.0
        per_pair_results[pair_idx] = (bool(correctness), int(cost), best_path_fid)

        if consumed_total >= C_total:
            break

    print(f"[CHECK:after-init] sum_init={sum(init_costs)} consumed_total_after_init={consumed_total}")

    # V_n = I_n * f_n^(init)
    def _score(idx):
        imp = importance_list[idx] if importance_list is not None else 1.0
        return float(imp) * float(f_init[idx])

    order = sorted(
        [i for i in range(N_pairs) if (states[i] is not None and node_path_list[i] > 0)],
        key=_score,
        reverse=True
    )
    debug_scores = [(i, _score(i)) for i in range(N_pairs)]
    print(f"[ORDER] by importance*init_fid desc: {sorted(debug_scores, key=lambda x: x[1], reverse=True)}")

    # -----------------------
    # フェーズ2: 集中的活用
    # -----------------------
    for pair_idx in order:
        print(f"[GREEDY] pre-loop pair={pair_idx} consumed_total={consumed_total}")

        if consumed_total >= C_total:
            break
        if states[pair_idx] is None:
            continue

        network   = networks[pair_idx]
        state     = states[pair_idx]

        while consumed_total < C_total:
            remaining = int(C_total) - int(consumed_total)
            if remaining <= 0:
                break

            print(f"[GREEDY] pair={pair_idx} remain={remaining} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")

            if return_details:
                correctness, cost, best_path_fid, alloc_more, est_more, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=True
                )
                print(f"[GREEDY<-] pair={pair_idx} cost={int(cost)} insufficient={bool(insufficient)} "
                      f"best_path_fid={best_path_fid} "
                      f"s'={new_state.get('s') if new_state else None} "
                      f"k'={len(new_state.get('candidate_set', [])) if new_state else None} "
                      f"consumed_total→{consumed_total + int(cost)}")

                for l, b in alloc_more.items():
                    per_pair_details[pair_idx]["alloc_by_path"][int(l)] = \
                        per_pair_details[pair_idx]["alloc_by_path"].get(int(l), 0) + int(b)
                per_pair_details[pair_idx]["est_fid_by_path"].update(
                    {int(k): float(v) for k, v in est_more.items()}
                )
                _sum_after_round = sum(per_pair_details[pair_idx]["alloc_by_path"].values())
                print(f"[CHECK:round] pair={pair_idx} add={int(cost)} sum_alloc_by_path={_sum_after_round}")
            else:
                correctness, cost, best_path_fid, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=False
                )
                print(f"[GREEDY<-] pair={pair_idx} cost={int(cost)} insufficient={bool(insufficient)} "
                      f"best_path_fid={best_path_fid} "
                      f"s'={new_state.get('s') if new_state else None} "
                      f"k'={len(new_state.get('candidate_set', [])) if new_state else None} "
                      f"consumed_total→{consumed_total + int(cost)}")

            consumed_total += int(cost)
            print(f"[GREEDY] post-accum pair={pair_idx} consumed_total={consumed_total}")

            state            = new_state
            states[pair_idx] = new_state

            prev_correctness, prev_cost, prev_best = per_pair_results[pair_idx]
            per_pair_results[pair_idx] = (
                bool(prev_correctness and correctness),
                int(prev_cost) + int(cost),
                best_path_fid,
            )

            if bool(insufficient):
                print(f"[GREEDY] break(insufficient) pair={pair_idx}")
                break

            cand = list(new_state.get("candidate_set", []))
            if len(cand) <= 1:
                print(f"[GREEDY] converged pair={pair_idx} "
                      f"s={new_state.get('s')} k={len(cand)} consumed_total={consumed_total}")
                break

            if consumed_total >= C_total:
                break

    print(f"[CHECK:return] consumed_total={consumed_total}")
    return (per_pair_results, int(consumed_total), per_pair_details) if return_details \
           else (per_pair_results, int(consumed_total))
