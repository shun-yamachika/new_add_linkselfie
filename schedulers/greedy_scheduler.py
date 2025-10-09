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
    Two-Phase Greedy スケジューラ（PRINTデバッグ行のみ追加版）
    - 既存の入出力・挙動は一切変更していません
    - 追加されたログタグ:
        [INIT]   フェーズ1呼び出し前
        [INIT<-] フェーズ1戻り
        [CHECK:init]  フェーズ1直後の per-link 配分合計（return_details=True のとき）
        [ORDER]  フェーズ2の実行順とスコア
        [GREEDY] pre-loop  各ペアのフェーズ2突入時の累積コスト
        [GREEDY] 呼び出し前の残余 / s / k
        [GREEDY<-] lonline_continue 戻り（cost/insufficient等）
        [GREEDY] post-accum  コスト加算後の累積コスト
        [GREEDY] converged 候補が一本化して収束
        [CHECK:after-init] フェーズ1全体終了直後の合計
        [CHECK:return]    return 直前の累積コスト
    """

    # ネットワーク作成など前処理
    N_pairs = len(node_path_list)
    networks         = [None] * N_pairs
    states           = [None] * N_pairs
    per_pair_results = [None] * N_pairs
    per_pair_details = [dict(alloc_by_path={}, est_fid_by_path={}) for _ in range(N_pairs)]
    init_costs       = [0] * N_pairs
    f_init           = [0.0] * N_pairs
    consumed_total   = 0

    # -----------------------
    # フェーズ1: 広域探索
    # -----------------------
    for pair_idx, path_num in enumerate(node_path_list):
        if consumed_total >= C_total or path_num <= 0:
            continue

        # --- DEBUG: フェーズ1 呼び出し前 ---
        print(f"[INIT] pair={pair_idx} remain={int(C_total)-int(consumed_total)} "
              f"paths={path_num} bounces={bounces}")

        remaining = int(C_total) - int(consumed_total)
        if remaining <= 0:
            break

        # 元実装どおりに path_list / network を構築
        path_list = list(range(int(path_num)))
        network = network_generator(int(path_num), pair_idx)
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
            # --- DEBUG: フェーズ1 戻り（詳細あり） ---
            _sum_after_init = sum(per_pair_details[pair_idx]["alloc_by_path"].values())
            print(f"[INIT<-] pair={pair_idx} cost={int(cost)} best_path_fid={best_path_fid} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")
            print(f"[CHECK:init] pair={pair_idx} sum_alloc_by_path={_sum_after_init} "
                  f"(should equal init cost={int(cost)})")
        else:
            correctness, cost, best_path_fid, state = lonline_init(
                network, path_list, list(bounces), int(remaining),
                return_details=False, C_const=C_const, delta=delta, min_sets=4
            )
            # --- DEBUG: フェーズ1 戻り（簡易） ---
            print(f"[INIT<-] pair={pair_idx} cost={int(cost)} best_path_fid={best_path_fid} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")

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

    # --- DEBUG: フェーズ1全体の確認 ---
    print(f"[CHECK:after-init] sum_init={sum(init_costs)} consumed_total_after_init={consumed_total}")

    # V_n = I_n * f_n^(init) に基づいて、集中的活用フェーズの順序を決定
    def _score(idx):
        imp = importance_list[idx] if importance_list is not None else 1.0
        return float(imp) * float(f_init[idx])

    order = sorted(
        [i for i in range(N_pairs) if (states[i] is not None and node_path_list[i] > 0)],
        key=_score,
        reverse=True
    )

    # --- DEBUG: 並べ替え順とスコア ---
    debug_scores = [(i, _score(i)) for i in range(N_pairs)]
    print(f"[ORDER] by importance*init_fid desc: {sorted(debug_scores, key=lambda x: x[1], reverse=True)}")

    # -----------------------
    # フェーズ2: 集中的活用
    # -----------------------
    for pair_idx in order:
        # 各ペアの頭で“いまの”累積消費を出す
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

            # --- DEBUG: フェーズ2 呼び出し前（残余 / s / k） ---
            print(f"[GREEDY] pair={pair_idx} remain={remaining} "
                  f"s={state.get('s') if state else None} "
                  f"k={len(state.get('candidate_set', [])) if state else None}")

            if return_details:
                correctness, cost, best_path_fid, alloc_more, est_more, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=True
                )
                # --- DEBUG: 戻り（詳細あり） ---
                print(f"[GREEDY<-] pair={pair_idx} cost={int(cost)} insufficient={bool(insufficient)} "
                      f"best_path_fid={best_path_fid} "
                      f"s'={new_state.get('s') if new_state else None} "
                      f"k'={len(new_state.get('candidate_set', [])) if new_state else None} "
                      f"consumed_total→{consumed_total + int(cost)}")

                was_insufficient = bool(insufficient)
                # 詳細マージ
                for l, b in alloc_more.items():
                    per_pair_details[pair_idx]["alloc_by_path"][int(l)] = \
                        per_pair_details[pair_idx]["alloc_by_path"].get(int(l), 0) + int(b)
                per_pair_details[pair_idx]["est_fid_by_path"].update({int(k): float(v) for k, v in est_more.items()})
                # --- DEBUG: フェーズ2 加算後の per-link 合計 ---
                _sum_after_round = sum(per_pair_details[pair_idx]["alloc_by_path"].values())
                print(f"[CHECK:round] pair={pair_idx} add={int(cost)} sum_alloc_by_path={_sum_after_round}")
            else:
                correctness, cost, best_path_fid, new_state, insufficient = lonline_continue(
                    network, int(remaining), state=state, return_details=False
                )
                # --- DEBUG: 戻り（簡易 / False 分岐目印） ---
                print(f"[GREEDY<-] pair={pair_idx} cost={int(cost)} insufficient={bool(insufficient)} "
                      f"best_path_fid={best_path_fid} "
                      f"s'={new_state.get('s') if new_state else None} "
                      f"k'={len(new_state.get('candidate_set', [])) if new_state else None} "
                      f"consumed_total→{consumed_total + int(cost)}")
                was_insufficient = bool(insufficient)
            # 成果の反映（元の実装どおり）
            consumed_total += int(cost)
            # --- DEBUG: コスト加算“後”の累積値 ---
            print(f"[GREEDY] post-accum pair={pair_idx} consumed_total={consumed_total}")

            state            = new_state
            states[pair_idx] = new_state

            # per_pair_results の加算更新（元コードのまま）
            prev_correctness, prev_cost, prev_best = per_pair_results[pair_idx]
            per_pair_results[pair_idx] = (
                bool(prev_correctness and correctness),     # 正解フラグは論理積
                int(prev_cost) + int(cost),                 # コストは加算
                best_path_fid,                              # 最新の fid を採用
            )
            if was_insufficient:
                print(f"[GREEDY] break(insufficient) pair={pair_idx}")
                break

            # 候補が1本に絞れていれば、このペアは確定とみなして次へ
            cand = list(new_state.get("candidate_set", []))
            if len(cand) <= 1:
                print(f"[GREEDY] converged pair={pair_idx} "
                      f"s={new_state.get('s')} k={len(cand)} consumed_total={consumed_total}")
                break

            if consumed_total >= C_total:
                break

    # 返却（互換維持）
    print(f"[CHECK:return] consumed_total={consumed_total}")
    return (per_pair_results, int(consumed_total), per_pair_details) if return_details \
           else (per_pair_results, int(consumed_total))
