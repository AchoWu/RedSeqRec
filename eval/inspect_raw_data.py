"""
inspect_raw_data.py
====================
流式探查大 jsonl 数据文件的结构。
打印前 N 行的字段、嵌套结构、序列长度、样例值。

用法:
    python inspect_raw_data.py \
        --input /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --n 5
"""
import argparse
import json


def summarize_value(v, depth=0, max_depth=6):
    """递归打印字段类型 / 形状 / 样例。"""
    pad = "  " * depth
    t = type(v).__name__
    if isinstance(v, list):
        info = f"list(len={len(v)})"
        if len(v) > 0 and depth < max_depth:
            print(f"{pad}{info}  [first elem ↓]")
            summarize_value(v[0], depth + 1, max_depth)
            if len(v) > 1:
                print(f"{pad}  [last elem ↓]")
                summarize_value(v[-1], depth + 1, max_depth)
        else:
            print(f"{pad}{info}")
    elif isinstance(v, dict):
        print(f"{pad}dict(keys={list(v.keys())})")
        if depth < max_depth:
            for k in v:
                print(f"{pad}  - {k}:")
                summarize_value(v[k], depth + 2, max_depth)
    else:
        repr_v = repr(v)
        if len(repr_v) > 200:
            repr_v = repr_v[:200] + "..."
        print(f"{pad}{t}: {repr_v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_depth", type=int, default=6,
                    help="how deep to recurse into nested structures")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            print(f"\n========== record #{i} ==========")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [parse error] {e}")
                print(f"  raw[:300]={line[:300]!r}")
                continue
            print(f"  top-level keys = {list(obj.keys())}")
            for k, v in obj.items():
                print(f"  -- field [{k}] --")
                summarize_value(v, depth=2, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
