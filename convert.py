# convert.py
import sys, json, pickle, numpy as np, pandas as pd
from pathlib import Path

def json_safe(o):
    if isinstance(o, (str, int, float, bool)) or o is None: return o
    if isinstance(o, (list, tuple, set)): return [json_safe(x) for x in o]
    if isinstance(o, dict): return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, pd.DataFrame): return [json_safe(r) for r in o.to_dict(orient="records")]
    if isinstance(o, pd.Series): return json_safe(o.to_dict())
    if isinstance(o, np.ndarray): return json_safe(o.tolist())
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return repr(o)

def dict_of_lists_to_records(d):
    lists = {k:v for k,v in d.items() if isinstance(v, (list, tuple, np.ndarray))}
    if not lists: return None
    lens = {len(v) for v in lists.values()}
    if len(lens) != 1: return None
    n = next(iter(lens))
    recs = []
    for i in range(n):
        rec = {}
        for k,v in d.items():
            rec[k] = v[i] if isinstance(v, (list, tuple, np.ndarray)) else v
        recs.append(json_safe(rec))
    return recs

def to_records(obj):
    if isinstance(obj, pd.DataFrame): return obj.to_dict(orient="records")
    if isinstance(obj, list) and (len(obj)==0 or isinstance(obj[0], dict)): return obj
    if isinstance(obj, dict):
        for k in ("data","results","records","runs","experiments"):
            if k in obj and isinstance(obj[k], list): return obj[k]
        recs = dict_of_lists_to_records(obj)
        if recs is not None: return recs
        return [json_safe(obj)]
    return [json_safe(obj)]

def main(src_path, out_dir=None):
    src = Path(src_path)
    if not src.exists(): raise FileNotFoundError(src)
    base_dir = Path(__file__).parent
    out_dir = Path(out_dir) if out_dir else (base_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as f: obj = pickle.load(f)
    safe = json_safe(obj)
    records = to_records(obj)
    stem = src.stem
    out_array  = out_dir / f"{stem}.records.json"
    out_ndjson = out_dir / f"{stem}.records.ndjson"
    out_raw    = out_dir / f"{stem}.raw.json"

    with out_array.open("w", encoding="utf-8") as f: json.dump(json_safe(records), f, ensure_ascii=False, indent=2)
    with out_ndjson.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(json_safe(rec), f, ensure_ascii=False); f.write("\n")
    with out_raw.open("w", encoding="utf-8") as f: json.dump(safe, f, ensure_ascii=False, indent=2)

    print("Wrote:\n- {}\n- {}\n- {}".format(out_array, out_ndjson, out_raw))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <src.pkl> [out_dir]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
