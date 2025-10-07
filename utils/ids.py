# utils/ids.py — path_id(1-origin) と idx(0-origin) の相互変換ユーティリティ

from typing import Dict, Iterable

def to_idx0(path_id_1origin: int) -> int:
    """1-origin の path_id を 0-origin の配列インデックスへ。"""
    return int(path_id_1origin) - 1

def to_id1(idx0: int) -> int:
    """0-origin の配列インデックスを 1-origin の path_id へ。"""
    return int(idx0) + 1

def is_keys_1origin(keys: Iterable[int], L: int) -> bool:
    """キー集合が 1..L を完全に満たすか（欠番なし）"""
    keys = list(keys)
    return bool(keys) and min(keys) == 1 and max(keys) == L and len(keys) == L

def is_keys_0origin(keys: Iterable[int], L: int) -> bool:
    """キー集合が 0..L-1 を完全に満たすか（欠番なし）"""
    keys = list(keys)
    return bool(keys) and min(keys) == 0 and max(keys) == L - 1 and len(keys) == L

def normalize_to_1origin(d: Dict[int, float], L: int) -> Dict[int, float]:
    """
    キーが 0..L-1 なら 1..L に正規化。すでに 1..L ならそのまま。
    想定外は例外。
    """
    if not d:
        # 空の場合は 1..L の枠を用意して NaN など入れるより、呼び出し側で未測定扱いにするのが自然
        return d
    keys = list(d.keys())
    if is_keys_1origin(keys, L):
        return d
    if is_keys_0origin(keys, L):
        return {k + 1: v for k, v in d.items()}
    raise RuntimeError(f"unexpected key scheme for L={L}: keys={sorted(keys)[:6]}...")
