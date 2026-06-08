"""
sample_from_processed.py
========================
Read an already-processed per-user jsonl produced by ``process_all_users.py``,
look at the actual valid_count distribution, then sample N users with
stratified bucket sampling so all length ranges are well represented.

This is fully decoupled from the heavy file processing: you can re-run
this script with different ``--total`` / bucket settings without ever
touching the 253 GiB raw input again. Two passes over the (much smaller)
processed jsonl:

    Pass 1: count users per bucket (build the population stats)
    Pass 2: stream again, online reservoir-sampling within each bucket

Why two passes? Because we want *fully random* sampling within each
bucket (Algorithm R) without keeping the whole jsonl in memory.

Output:
  --out-jsonl  : the sampled training set, one user per line
  --out-stat   : sampling stat json (per-bucket capacity, picked, ratios)

Usage
-----
    python sample_from_processed.py \
        --in-jsonl    data_prep/all_qualified_users.jsonl \
        --out-jsonl   data_prep/train_users.jsonl \
        --out-stat    data_prep/train_users.sample.json \
        --total       500000 \
        --seed        20260525
"""
import argparse
import json
import os
import random
import time
from collections import Counter


# Default bucket plan. You can tweak after seeing the real population
# distribution in pass 1 (the script will warn you if any bucket can't
# be filled and will redistribute the deficit automatically).
LEN_BUCKETS = [
    (5,    10,    "5-9"),
    (10,   20,    "10-19"),
    (20,   50,    "20-49"),
    (50,   100,   "50-99"),
    (100,  150,   "100-149"),
    (150,  200,   "150-199"),
    (200,  10**9, "200+"),     # users whose stored history >= max_len, valid_count
                               # may equal max_len; we still group them here.
]

# What fraction of --total to pick from each bucket. Buckets not listed here
# are NOT sampled. The numbers must sum to <= 1.0; remainder will go to the
# first bucket via rounding fixup. Tune to your taste.
# Default ratios tuned for the actually observed distribution in
# all_qualified_users.jsonl (see process_stat.json kept_buckets):
#   5-9     : 23.20%  (614k pool)
#   10-19   : 20.85%  (552k pool)
#   20-49   : 22.64%  (599k pool)
#   50-99   : 12.43%  (329k pool)
#   100-149 :  5.89%  (156k pool)
#   150-199 :  6.38%  (169k pool)
#   200+    :  8.61%  (228k pool)
# We bias toward medium-length users (richer behavior signal) while
# still covering all buckets. All target counts are well within the
# available population, so deficit redistribution should not trigger.
DEFAULT_BUCKET_RATIOS = {
    "5-9":     0.10,   # 50k    short users for diversity
    "10-19":   0.20,   # 100k
    "20-49":   0.25,   # 125k   primary mass
    "50-99":   0.18,   #  90k
    "100-149": 0.10,   #  50k
    "150-199": 0.09,   #  45k
    "200+":    0.08,   #  40k   long-history power users
}


def get_bucket(n):
    for lo, hi, label in LEN_BUCKETS:
        if lo <= n < hi:
            return label
    return None      # below 10 -> shouldn't happen since processor filters


class ReservoirBucket:
    """Algorithm-R reservoir sampler with full record retention."""

    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.reservoir = []           # list[json line str]
        self.seen = 0

    def offer(self, line):
        self.seen += 1
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(line)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.capacity:
                self.reservoir[j] = line


def _fmt_eta(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-stat", required=True)
    ap.add_argument("--total", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20260525)
    ap.add_argument("--ratios", type=str, default="",
                    help="optional override, e.g. "
                         "'10-19:0.20,20-49:0.30,50-99:0.25,"
                         "100-149:0.15,150-199:0.06,200+:0.04'")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    # Parse ratios
    if args.ratios.strip():
        ratios = {}
        for piece in args.ratios.split(","):
            k, v = piece.split(":")
            ratios[k.strip()] = float(v)
    else:
        ratios = dict(DEFAULT_BUCKET_RATIOS)
    bucket_labels = list(ratios.keys())
    assert all(any(b == lbl for _, _, lbl in LEN_BUCKETS) for b in bucket_labels), \
        f"unknown bucket label in ratios: {bucket_labels}"

    rng = random.Random(args.seed)
    fsize = os.path.getsize(args.in_jsonl)
    print(f"[sample] in_jsonl  = {args.in_jsonl}", flush=True)
    print(f"[sample] file size = {fsize / (1024**3):.2f} GiB", flush=True)
    print(f"[sample] total target = {args.total:,}", flush=True)
    print(f"[sample] bucket ratios = {ratios}", flush=True)

    # ----- PASS 1: count users per bucket -----
    print(f"\n[sample] === PASS 1: counting buckets ===", flush=True)
    t0 = time.time()
    pop = Counter()
    n_lines = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        # readline() in a while-loop instead of `for line in f:` so that
        # f.tell() works (the iterator-form uses an internal buffer that
        # disables tell()).
        while True:
            line = f.readline()
            if not line:
                break
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vc = int(obj.get("valid_count", 0))
            b = get_bucket(vc)
            if b is None:
                continue
            pop[b] += 1
            if n_lines % args.progress_every == 0:
                el = time.time() - t0
                pos = f.tell()
                pct = pos / max(fsize, 1) * 100
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                print(f"[sample] pass1 [{bar}] {pct:5.1f}%  "
                      f"lines={n_lines:,}  elapsed={_fmt_eta(el)}",
                      flush=True)
    el1 = time.time() - t0
    n_total_in = n_lines
    print(f"[sample] pass1 done  total_lines={n_total_in:,}  "
          f"elapsed={_fmt_eta(el1)}", flush=True)
    print(f"[sample] population per bucket:", flush=True)
    for _, _, lbl in LEN_BUCKETS:
        if lbl in ratios:
            n = pop.get(lbl, 0)
            pct = n / max(n_total_in, 1) * 100
            print(f"  {lbl:>10s} : {n:>12,d}  ({pct:.2f}%)", flush=True)

    # ----- compute targets, with deficit redistribution -----
    targets = {b: int(round(args.total * ratios[b])) for b in bucket_labels}
    diff = args.total - sum(targets.values())
    if diff != 0:
        targets[bucket_labels[0]] += diff
    avail = {b: pop.get(b, 0) for b in bucket_labels}
    final_caps = {b: min(targets[b], avail[b]) for b in bucket_labels}
    deficit = sum(targets[b] - final_caps[b] for b in bucket_labels)
    while deficit > 0:
        cands = [b for b in bucket_labels if avail[b] - final_caps[b] > 0]
        if not cands:
            print(f"[sample] WARN  cannot fully fill, residual deficit={deficit}",
                  flush=True)
            break
        ratio_sum = sum(ratios[b] for b in cands)
        added = 0
        for b in cands:
            give = int(round(deficit * ratios[b] / ratio_sum))
            give = min(give, avail[b] - final_caps[b])
            if give > 0:
                final_caps[b] += give
                added += give
        if added == 0:
            for b in cands:
                if avail[b] - final_caps[b] > 0:
                    final_caps[b] += 1
                    added += 1
                    if added >= deficit:
                        break
        deficit -= added

    print(f"\n[sample] per-bucket plan after deficit redistribution:", flush=True)
    for b in bucket_labels:
        print(f"  {b:>10s} : target={targets[b]:>8,d}  avail={avail[b]:>10,d}  "
              f"plan_pick={final_caps[b]:>8,d}", flush=True)
    plan_total = sum(final_caps.values())
    print(f"  --> total plan = {plan_total:,}", flush=True)

    # ----- PASS 2: reservoir-sample each bucket -----
    reservoirs = {b: ReservoirBucket(final_caps[b], rng) for b in bucket_labels}
    print(f"\n[sample] === PASS 2: reservoir sampling ===", flush=True)
    t1 = time.time()
    n_lines2 = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_lines2 += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vc = int(obj.get("valid_count", 0))
            b = get_bucket(vc)
            if b in reservoirs:
                reservoirs[b].offer(line)
            if n_lines2 % args.progress_every == 0:
                el = time.time() - t1
                pos = f.tell()
                pct = pos / max(fsize, 1) * 100
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                got = sum(len(r.reservoir) for r in reservoirs.values())
                print(f"[sample] pass2 [{bar}] {pct:5.1f}%  "
                      f"lines={n_lines2:,}  filled={got:,}/{plan_total:,}  "
                      f"elapsed={_fmt_eta(el)}", flush=True)
    el2 = time.time() - t1
    print(f"[sample] pass2 done  elapsed={_fmt_eta(el2)}", flush=True)

    # ----- write output -----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".",
                exist_ok=True)
    # collect all picked lines and shuffle (so downstream sees mixed buckets)
    all_picked = []
    for b in bucket_labels:
        all_picked.extend(reservoirs[b].reservoir)
    rng.shuffle(all_picked)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for line in all_picked:
            # line still has its trailing newline from input
            if not line.endswith("\n"):
                line += "\n"
            f.write(line)
    print(f"[sample] wrote {len(all_picked):,} users to {args.out_jsonl}",
          flush=True)

    stat = {
        "in_jsonl": os.path.abspath(args.in_jsonl),
        "out_jsonl": os.path.abspath(args.out_jsonl),
        "seed": args.seed,
        "target_total": args.total,
        "actual_total": len(all_picked),
        "ratios": ratios,
        "buckets": {
            b: {
                "available": avail[b],
                "target": targets[b],
                "plan_pick": final_caps[b],
                "actual_pick": len(reservoirs[b].reservoir),
                "ratio": ratios[b],
            }
            for b in bucket_labels
        },
        "pass1_elapsed_sec": el1,
        "pass2_elapsed_sec": el2,
    }
    with open(args.out_stat, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(f"[sample] wrote {args.out_stat}", flush=True)


if __name__ == "__main__":
    main()
