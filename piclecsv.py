import os, pickle, csv, sys, re

# ここに変換したいファイルを列挙（相対パスOK）
files = [
    "outputs/plot_minwidthsum_perpair_vs_budget_Depolar.pickle",
    "outputs/plot_minwidthsum_perpair_weighted_vs_budget_Depolar.pickle",
    "outputs/plot_widthsum_alllinks_vs_budget_Depolar.pickle",
    "outputs/plot_widthsum_alllinks_weighted_vs_budget_Depolar.pickle",
]

# 指標候補（見つかった順に採用）
PREFERRED_KEYS = [
    "minwidthsum_weighted",
    "minwidthsum",
    "widthsum_alllinks_weighted",
    "widthsum_alllinks",
    "accuracy",  # 念のため
    "value",     # 念のため
    "metric"     # 念のため
]

def pick_metric_key(results):
    """results は {budget: {...}}。どのキーでCSV化するか自動推定"""
    for b, r in results.items():
        if isinstance(r, dict) and r:
            keys = set(r.keys())
            # ノイズになりがちなキーを除外
            keys -= {"per_pair_details", "details", "meta"}
            # 優先候補から探す
            for k in PREFERRED_KEYS:
                if k in keys:
                    return k
            # それでも見つからなければ、数値っぽい最初のキー
            for k in keys:
                v = r.get(k)
                if isinstance(v, (int, float)) or (v is not None and not isinstance(v, (dict, list, tuple, set))):
                    return k
    return None

for path in files:
    if not os.path.exists(path):
        print(f"[WARN] not found: {path}")
        continue

    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] pickle.load failed for {path}: {e}")
            continue

    budgets = obj.get("budget_list", [])
    results = obj.get("results", {})

    if not budgets or not isinstance(results, dict) or not results:
        print(f"[WARN] {path}: budgets or results is empty（サイズが小さい594Bケースかも）")
        continue

    metric_key = pick_metric_key(results)
    if not metric_key:
        print(f"[WARN] {path}: 指標キーが見つからないためスキップ（resultsの中身を要確認）")
        continue

    out_csv = os.path.splitext(path)[0] + ".csv"  # 同名で .csv を outputs/ に出力
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["budget", metric_key])
        for b in budgets:
            v = results.get(b, {}).get(metric_key)
            w.writerow([b, v])

    print(f"[OK] {out_csv} （列: budget,{metric_key}）")
