"""
sample_users.py
================
Read the per-user tsv produced by stat_valid_clicks.py, then sample N users
with stratified sampling so that all length buckets are well represented.
By default we keep only users with valid_count in [10, 200] and aim for
500_000 selected users.

The stratified plan is:
    [10,  20)    target = TOTAL * 0.20    (e.g. 100,000)
    [20,  50)    target = TOTAL * 0.25    (e.g. 125,000)
    [50, 100)    target = TOTAL * 0.20    (e.g. 100,000)
    [100, 150)   target = TOTAL * 0.18    (e.g.  90,000)
    [150, 200]   target = TOTAL * 0.17    (e.g.  85,000)

If a bucket has fewer users than its target, we take all of them and
*redistribute the deficit* across the remaining buckets. So you always
end up with TOTAL users (assuming the global pool is large enough).

Output:
  --out-users : a plain-text file, one qimei36 per line. The order is
                shuffled (so that downstream parallel readers see balanced
                distributions even when they only consume a prefix).
  --out-stat  : a small .json with the actual per-bucket count picked.

Usage
-----
    python sample_users.py \
        --in-tsv     ./valid_click_counts.tsv \
        --out-users  ./selected_users.txt \
        --out-stat   ./selected_users.stat.json \
        --total      500000 \
        --seed       20260525
"""
import argparse
import json
import os
import random
from collections import defaultdict


# Must mirror stat_valid_clicks.LEN_BUCKETS labels for [10..200].
SAMPLE_BUCKETS = ["10-19", "20-49", "50-99", "100-149", "150-199"]
SAMPLE_BUCKET_RATIOS = {
    "10-19":   0.20,
    "20-49":   0.25,
    "50-99":   0.20,
    "100-149": 0.18,
    "150-199": 0.17,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-tsv", required=True)
    ap.add_argument("--out-users", required=True)
    ap.add_argument("--out-stat", required=True)
    ap.add_argument("--total", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20260525)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) load per-bucket pools
    print(f"[sample] loading {args.in_tsv} ...", flush=True)
    bucket2users = defaultdict(list)
    n_total = 0
    n_in_pool = 0
    with open(args.in_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            qi_idx = header.index("qimei36")
            bk_idx = header.index("len_bucket")
        except ValueError:
            print(f"[sample] tsv header malformed: {header}")
            return
        for line in f:
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= bk_idx:
                continue
            bucket = parts[bk_idx]
            if bucket in SAMPLE_BUCKETS:
                bucket2users[bucket].append(parts[qi_idx])
                n_in_pool += 1

    print(f"[sample] users in tsv         = {n_total:,}", flush=True)
    print(f"[sample] users in [10,200)    = {n_in_pool:,}", flush=True)
    for b in SAMPLE_BUCKETS:
        print(f"[sample]   bucket {b:>8s}    : {len(bucket2users[b]):>10,d}",
              flush=True)
    print(f"[sample] target total         = {args.total:,}", flush=True)
    if n_in_pool < args.total:
        print(f"[sample] WARNING: pool ({n_in_pool}) < target ({args.total}); "
              f"will keep all available", flush=True)

    # 2) initial per-bucket targets
    targets = {b: int(round(args.total * SAMPLE_BUCKET_RATIOS[b]))
               for b in SAMPLE_BUCKETS}
    # rounding fixup
    diff = args.total - sum(targets.values())
    if diff != 0:
        targets[SAMPLE_BUCKETS[0]] += diff

    # 3) redistribute deficit if some bucket is short
    avail = {b: len(bucket2users[b]) for b in SAMPLE_BUCKETS}
    final = {b: 0 for b in SAMPLE_BUCKETS}
    deficit = 0
    # first pass: take min(target, avail)
    for b in SAMPLE_BUCKETS:
        take = min(targets[b], avail[b])
        final[b] = take
        deficit += targets[b] - take
    # second pass: distribute deficit to buckets with capacity left
    while deficit > 0:
        progress = False
        # buckets with capacity, sorted by remaining ratio descending
        cands = [b for b in SAMPLE_BUCKETS if avail[b] - final[b] > 0]
        if not cands:
            print(f"[sample] cannot fully fill, residual deficit = {deficit}",
                  flush=True)
            break
        # split deficit proportionally to original ratios among cands
        ratio_sum = sum(SAMPLE_BUCKET_RATIOS[b] for b in cands)
        added_total = 0
        for b in cands:
            give = int(round(deficit * SAMPLE_BUCKET_RATIOS[b] / ratio_sum))
            give = min(give, avail[b] - final[b])
            if give > 0:
                final[b] += give
                added_total += give
                progress = True
        deficit -= added_total
        if not progress:
            break

    # 4) sample within each bucket
    selected = []
    for b in SAMPLE_BUCKETS:
        users = bucket2users[b]
        n = final[b]
        if n >= len(users):
            picked = users
        else:
            picked = rng.sample(users, n)
        selected.extend(picked)
        print(f"[sample] picked  {b:>8s} : {len(picked):>10,d} "
              f"(target={targets[b]:,}, avail={avail[b]:,})",
              flush=True)

    rng.shuffle(selected)
    print(f"[sample] selected total       = {len(selected):,}", flush=True)

    # 5) write outputs
    os.makedirs(os.path.dirname(os.path.abspath(args.out_users)) or ".",
                exist_ok=True)
    with open(args.out_users, "w", encoding="utf-8") as f:
        for u in selected:
            f.write(u + "\n")

    stat = {
        "seed": args.seed,
        "target_total": args.total,
        "actual_total": len(selected),
        "buckets": {
            b: {"target": targets[b], "available": avail[b],
                "picked": final[b], "ratio": SAMPLE_BUCKET_RATIOS[b]}
            for b in SAMPLE_BUCKETS
        },
        "in_tsv": os.path.abspath(args.in_tsv),
    }
    with open(args.out_stat, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    print(f"[sample] wrote {args.out_users}  ({len(selected):,} lines)",
          flush=True)
    print(f"[sample] wrote {args.out_stat}", flush=True)


if __name__ == "__main__":
    main()
