"""
sanity_check_processed.py
=========================
Light sanity check on the processed jsonl. Streams through, samples a few
records to sanity-check schema, and aggregates a few invariants.

Checks:
  - schema  : every record has the required top-level keys
  - history :
      * history_len matches len(history)
      * history_len <= max_len
      * ts is non-decreasing inside each user
      * valid_count matches count(label=='pos')
  - labels  : aggregate label counts (pos / hardneg / noise)
  - lengths : histogram of history_len and valid_count
  - dedupe  : duplicate qimei36 detection (set-based, may take memory)
"""
import argparse
import json
import os
import time
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--max-lines", type=int, default=-1,
                    help="if >0 stop after scanning this many users")
    ap.add_argument("--print-samples", type=int, default=2,
                    help="how many record samples to pretty-print")
    ap.add_argument("--max-len-allowed", type=int, default=500,
                    help="upper bound expected for history_len")
    ap.add_argument("--check-dup", action="store_true",
                    help="check for duplicate qimei36 (uses memory)")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    fsize = os.path.getsize(args.in_jsonl)
    print(f"[check] in_jsonl  = {args.in_jsonl}", flush=True)
    print(f"[check] file size = {fsize / (1024**3):.2f} GiB", flush=True)

    n_users = 0
    n_invalid_schema = 0
    n_invalid_histlen = 0
    n_overlong = 0
    n_invalid_validcount = 0
    n_unsorted_ts = 0
    label_counter = Counter()
    histlen_counter = Counter()       # bucketed
    validcount_counter = Counter()    # bucketed
    age_counter = Counter()
    gender_counter = Counter()
    seen_users = set() if args.check_dup else None
    n_dups = 0

    histlen_buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100),
                       (100, 200), (200, 300), (300, 500), (500, 10**9)]
    valid_buckets = [(5, 10), (10, 20), (20, 50), (50, 100),
                     (100, 200), (200, 500), (500, 10**9)]

    def bucket_of(n, table):
        for lo, hi in table:
            if lo <= n < hi:
                return f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        return "?"

    t0 = time.time()
    n_printed = 0
    required_keys = {"qimei36", "age", "gender", "valid_count",
                     "history_len", "history"}

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_users += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_invalid_schema += 1
                continue

            keys = set(obj.keys())
            if not required_keys.issubset(keys):
                n_invalid_schema += 1
                continue

            qimei = obj["qimei36"]
            history = obj["history"]
            histlen = obj["history_len"]
            vc = obj["valid_count"]

            # 1) histlen
            if histlen != len(history):
                n_invalid_histlen += 1
            if histlen > args.max_len_allowed:
                n_overlong += 1

            # 2) valid_count
            real_pos = sum(1 for h in history if h.get("label") == "pos")
            if real_pos != vc:
                n_invalid_validcount += 1

            # 3) ts ordering
            last_ts = -1
            for h in history:
                ts = h.get("ts", 0)
                if ts < last_ts:
                    n_unsorted_ts += 1
                    break
                last_ts = ts

            # 4) labels
            for h in history:
                label_counter[h.get("label", "?")] += 1

            # 5) length / valid distributions
            histlen_counter[bucket_of(histlen, histlen_buckets)] += 1
            validcount_counter[bucket_of(vc, valid_buckets)] += 1

            # 6) demographics
            age_counter[obj.get("age", "")] += 1
            gender_counter[obj.get("gender", "")] += 1

            # 7) duplicates
            if seen_users is not None:
                if qimei in seen_users:
                    n_dups += 1
                else:
                    seen_users.add(qimei)

            # samples
            if n_printed < args.print_samples:
                print(f"\n--- sample #{n_printed+1} ---", flush=True)
                print(f"  qimei36     : {qimei}")
                print(f"  age/gender  : {obj.get('age')} / {obj.get('gender')}")
                print(f"  valid_count : {vc}  (real pos={real_pos})")
                print(f"  history_len : {histlen}")
                print(f"  first event : {history[0] if history else None}")
                print(f"  last event  : {history[-1] if history else None}")
                lc = Counter(h.get("label") for h in history)
                print(f"  label_counts: {dict(lc)}")
                n_printed += 1

            if n_users % args.progress_every == 0:
                el = time.time() - t0
                pos = f.tell()
                pct = pos / max(fsize, 1) * 100
                print(f"[check] {pct:5.1f}%  users={n_users:,}  "
                      f"elapsed={el:.0f}s", flush=True)

            if args.max_lines > 0 and n_users >= args.max_lines:
                print(f"[check] stopped at --max-lines={args.max_lines:,}",
                      flush=True)
                break

    el = time.time() - t0
    print(f"\n[check] DONE  total_users={n_users:,}  elapsed={el:.0f}s",
          flush=True)

    print(f"\n=== schema ===")
    print(f"  invalid_schema      : {n_invalid_schema}")
    print(f"  invalid_history_len : {n_invalid_histlen}")
    print(f"  history_len > {args.max_len_allowed}: {n_overlong}")
    print(f"  invalid_valid_count : {n_invalid_validcount}")
    print(f"  unsorted_ts_users   : {n_unsorted_ts}")
    if seen_users is not None:
        print(f"  duplicate_qimei36   : {n_dups}")

    print(f"\n=== labels (event-level) ===")
    total_ev = sum(label_counter.values())
    for lab in ("pos", "hardneg", "noise", "?"):
        c = label_counter.get(lab, 0)
        pct = c / max(total_ev, 1) * 100
        print(f"  {lab:>10s}: {c:>14,d}  ({pct:5.2f}%)")

    print(f"\n=== history_len distribution ===")
    for lo, hi in histlen_buckets:
        key = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        c = histlen_counter.get(key, 0)
        pct = c / max(n_users, 1) * 100
        print(f"  {key:>14s}: {c:>14,d}  ({pct:5.2f}%)")

    print(f"\n=== valid_count distribution ===")
    for lo, hi in valid_buckets:
        key = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        c = validcount_counter.get(key, 0)
        pct = c / max(n_users, 1) * 100
        print(f"  {key:>14s}: {c:>14,d}  ({pct:5.2f}%)")

    print(f"\n=== gender ===")
    for g, c in gender_counter.most_common():
        print(f"  {g!r:>10s}: {c:>14,d}  ({c / max(n_users,1)*100:5.2f}%)")

    print(f"\n=== age top-15 ===")
    # try to coerce age to int and also print buckets
    age_buckets = Counter()
    age_unknown = 0
    for a, c in age_counter.items():
        try:
            ai = int(a)
            if ai < 18:        age_buckets["1-17"] += c
            elif ai < 25:      age_buckets["18-24"] += c
            elif ai < 35:      age_buckets["25-34"] += c
            elif ai < 45:      age_buckets["35-44"] += c
            elif ai < 55:      age_buckets["45-54"] += c
            else:              age_buckets["55+"] += c
        except Exception:
            age_unknown += c
    age_buckets["unknown"] += age_unknown
    for lab in ["1-17", "18-24", "25-34", "35-44", "45-54", "55+", "unknown"]:
        c = age_buckets.get(lab, 0)
        pct = c / max(n_users, 1) * 100
        print(f"  {lab:>10s}: {c:>14,d}  ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
