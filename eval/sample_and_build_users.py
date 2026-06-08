"""
sample_and_build_users.py
==========================
One-pass alternative to (stat_valid_clicks.py + sample_users.py).

Stream-scan the big jsonl ONCE. For each user:
  - compute valid_count = #events with wt = max(rd, pd) >= MIN_WT_SEC
  - decide its length bucket
  - run online reservoir sampling per bucket so that, at the end of the
    scan, each bucket holds at most TARGET_PER_BUCKET users picked
    uniformly at random from the entire stream.

This avoids the need to first do "preview statistics on 1M lines" and
then guess bucket ratios -- one full pass directly produces both:

  - selected_users.txt           the 500k users to use for training
  - bucket_distribution.txt      per-bucket scan stats + final sample
  - valid_click_counts.tsv       (optional) per-user counts dump

The reservoir sampling uses the standard "Algorithm R": for the i-th
user that *qualifies* a bucket, accept with probability k/i, where k
is the bucket capacity. Result is an unbiased uniform sample without
needing to know the population size in advance.

Memory: O(sum of bucket capacities * avg_qimei36_len) ~ 50MB for 500k.

Usage
-----
    python sample_and_build_users.py \
        --input /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --out-dir data_prep/ \
        --min-wt-sec 5.0 \
        --total 500000 \
        --seed 20260525
"""
import argparse
import json
import os
import random
import time
from collections import Counter


# Length buckets used for sampling.
# (lo, hi) is half-open, except 200+ which is closed-open infinity.
LEN_BUCKETS = [
    (0,    1,     "0"),
    (1,    5,     "1-4"),
    (5,    10,    "5-9"),
    (10,   20,    "10-19"),
    (20,   50,    "20-49"),
    (50,   100,   "50-99"),
    (100,  150,   "100-149"),
    (150,  200,   "150-199"),
    (200,  10**9, "200+"),
]

# Buckets we actually sample from + their target ratios.
#
# Tuned based on a 100k-line preview which showed bucket population:
#   10-19   :  5.45%
#   20-49   :  5.80%
#   50-99   :  3.24%
#   100-149 :  1.42%
#   150-199 :  0.81%   <-- rarest, was the bottleneck
# The earlier (0.20, 0.25, 0.20, 0.18, 0.17) plan would have made the
# 150-199 bucket the bottleneck (~85k from a ~810k pool, half the
# whole sub-pool) and forced a long full-file scan. We instead bias
# toward the more common medium-length users which hold richer
# behavior signal anyway.
SAMPLE_BUCKETS = ["10-19", "20-49", "50-99", "100-149", "150-199"]
SAMPLE_BUCKET_RATIOS = {
    "10-19":   0.20,   # 100k   (pool ~5.45%  -> abundant)
    "20-49":   0.30,   # 150k   (pool ~5.80%  -> abundant)
    "50-99":   0.25,   # 125k   (pool ~3.24%  -> abundant)
    "100-149": 0.17,   #  85k   (pool ~1.42%  -> sufficient)
    "150-199": 0.08,   #  40k   (pool ~0.81%  -> just enough, ~50% scan)
}


def get_bucket(n):
    for lo, hi, label in LEN_BUCKETS:
        if lo <= n < hi:
            return label
    return "0"


class ReservoirBucket:
    """Algorithm R reservoir sampler for one bucket.

    Accepts each *qualifying* user one at a time. Keeps at most ``capacity``
    qimei36 strings inside ``self.reservoir`` such that they form a uniform
    sample of all users that ever qualified.
    """

    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.reservoir = []          # list[str]
        self.seen = 0                # how many users qualified

    def offer(self, qimei36):
        self.seen += 1
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(qimei36)
        else:
            # Pick a slot in [0, seen). With prob capacity/seen,
            # this slot < capacity, so we replace; else discard.
            j = self.rng.randint(0, self.seen - 1)
            if j < self.capacity:
                self.reservoir[j] = qimei36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-wt-sec", type=float, default=5.0)
    ap.add_argument("--total", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20260525)
    ap.add_argument("--progress-every", type=int, default=200_000,
                    help="print a progress bar every N lines (default 200k)")
    ap.add_argument("--dump-tsv", action="store_true",
                    help="also dump per-user valid counts (large)")
    ap.add_argument("--max-lines", type=int, default=-1,
                    help="if >0, stop after scanning this many lines "
                         "(for debugging, NOT recommended for final run)")
    ap.add_argument("--early-stop", action="store_true",
                    help="OFF by default. If set, stop scanning once every "
                         "bucket is full. Faster but slightly biased toward "
                         "earlier-encountered users. For an unbiased uniform "
                         "sample over the full population, do NOT pass this.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_users = os.path.join(args.out_dir, "selected_users.txt")
    out_dist  = os.path.join(args.out_dir, "bucket_distribution.txt")
    out_stat  = os.path.join(args.out_dir, "selected_users.stat.json")
    out_tsv   = os.path.join(args.out_dir, "valid_click_counts.tsv") if args.dump_tsv else None

    rng = random.Random(args.seed)

    # 1) build per-bucket reservoirs
    bucket_targets = {}
    total_target = 0
    for b in SAMPLE_BUCKETS:
        cap = int(round(args.total * SAMPLE_BUCKET_RATIOS[b]))
        bucket_targets[b] = cap
        total_target += cap
    diff = args.total - total_target
    if diff != 0:
        bucket_targets[SAMPLE_BUCKETS[0]] += diff

    reservoirs = {b: ReservoirBucket(bucket_targets[b], rng)
                  for b in SAMPLE_BUCKETS}

    # 2) scan the big file once
    print(f"[onepass] input        = {args.input}", flush=True)
    print(f"[onepass] out_dir      = {args.out_dir}", flush=True)
    print(f"[onepass] min_wt_sec   = {args.min_wt_sec}", flush=True)
    print(f"[onepass] total target = {args.total:,}", flush=True)
    for b in SAMPLE_BUCKETS:
        print(f"[onepass]   bucket {b:>8s} : "
              f"capacity = {bucket_targets[b]:,}", flush=True)
    fsize = os.path.getsize(args.input)
    print(f"[onepass] file size    = {fsize / (1024**3):.2f} GiB", flush=True)
    if args.max_lines > 0:
        print(f"[onepass] DEBUG MODE  max_lines = {args.max_lines:,}",
              flush=True)
    print(f"[onepass] starting stream ...", flush=True)

    t0 = time.time()
    n_lines = 0
    n_bad = 0
    n_no_user = 0
    bucket_seen = Counter()       # all-bucket counts (incl. <10 and 200+)

    fout_tsv = open(out_tsv, "w", encoding="utf-8") if out_tsv else None
    if fout_tsv:
        fout_tsv.write("qimei36\tvalid_count\ttotal_events\texp_count\tlen_bucket\n")

    def _fmt_eta(seconds):
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            sa = obj.get("seq_all") or {}
            seqs = sa.get("seqs") or {}
            ev = seqs.get("smallvideo_float") or []
            qimei = obj.get("qimei36", "") or sa.get("qimei36", "")
            if not qimei:
                n_no_user += 1
                continue

            total = len(ev)
            valid = 0
            exp_n = 0
            for e in ev:
                rd = e.get("rd", 0) or 0
                pd = e.get("pd", 0) or 0
                wt = rd if rd >= pd else pd
                if wt >= args.min_wt_sec:
                    valid += 1
                if e.get("exp", 0) == 1:
                    exp_n += 1

            bucket = get_bucket(valid)
            bucket_seen[bucket] += 1

            if fout_tsv:
                fout_tsv.write(
                    f"{qimei}\t{valid}\t{total}\t{exp_n}\t{bucket}\n"
                )

            # offer to reservoir if this bucket is one we sample
            if bucket in reservoirs:
                reservoirs[bucket].offer(qimei)

            # optional early stop when every bucket is fully populated
            if args.early_stop:
                if all(len(reservoirs[b].reservoir) >= reservoirs[b].capacity
                       for b in SAMPLE_BUCKETS):
                    print(f"[onepass] all buckets filled at line {n_lines:,}, "
                          f"early-stop enabled, stopping scan.", flush=True)
                    break

            if n_lines % args.progress_every == 0:
                el = time.time() - t0
                rate = n_lines / max(el, 1e-9)
                # byte-level progress = the most reliable indicator
                pos = fin.tell()
                pct = pos / max(fsize, 1) * 100
                eta = (fsize - pos) / max(pos / max(el, 1e-9), 1) if pos > 0 else 0
                got = sum(len(reservoirs[b].reservoir) for b in SAMPLE_BUCKETS)
                # bar visualization: 30 chars wide
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                # per-bucket fill summary, like "10-19:100k/100k 20-49:88k/150k"
                bucket_fill = " ".join(
                    f"{b}:{len(reservoirs[b].reservoir):,}/{reservoirs[b].capacity:,}"
                    for b in SAMPLE_BUCKETS
                )
                print(f"[onepass] [{bar}] {pct:5.1f}%  "
                      f"bytes={pos / (1024**3):.2f}/{fsize / (1024**3):.2f}GiB  "
                      f"lines={n_lines:,}  rate={rate:.0f} l/s  "
                      f"elapsed={_fmt_eta(el)}  ETA={_fmt_eta(eta)}",
                      flush=True)
                print(f"[onepass]   filled {got:,}/{args.total:,}  |  {bucket_fill}",
                      flush=True)

            if args.max_lines > 0 and n_lines >= args.max_lines:
                print(f"[onepass] reached --max-lines={args.max_lines:,}, stop.",
                      flush=True)
                break

    if fout_tsv:
        fout_tsv.close()

    el = time.time() - t0
    total_users_seen = sum(bucket_seen.values())
    print(f"\n[onepass] DONE  lines={n_lines:,}  bad={n_bad}  "
          f"no_user={n_no_user}  elapsed={el:.0f}s", flush=True)

    # 3) collect selected
    selected = []
    for b in SAMPLE_BUCKETS:
        selected.extend(reservoirs[b].reservoir)
    rng.shuffle(selected)

    with open(out_users, "w", encoding="utf-8") as f:
        for u in selected:
            f.write(u + "\n")

    # 4) write distribution report
    lines_out = []
    lines_out.append(f"# sample_and_build_users.py report")
    lines_out.append(f"# input         = {args.input}")
    lines_out.append(f"# out_dir       = {args.out_dir}")
    lines_out.append(f"# min_wt_sec    = {args.min_wt_sec}")
    lines_out.append(f"# seed          = {args.seed}")
    lines_out.append(f"# total target  = {args.total:,}")
    lines_out.append(f"# total scanned = {n_lines:,} lines  "
                     f"({total_users_seen:,} users)")
    lines_out.append(f"# bad / no_user = {n_bad} / {n_no_user}")
    lines_out.append(f"# elapsed       = {el:.0f}s")
    lines_out.append("")
    lines_out.append(
        f"{'bucket':>10s} | {'seen':>12s} | {'pct(of total)':>13s} | "
        f"{'capacity':>10s} | {'picked':>10s}"
    )
    lines_out.append("-" * 70)
    for _, _, label in LEN_BUCKETS:
        seen = bucket_seen.get(label, 0)
        pct = seen / max(total_users_seen, 1) * 100
        cap = bucket_targets.get(label, 0)
        if label in reservoirs:
            picked = len(reservoirs[label].reservoir)
            cap_s = f"{cap:,}"
            pick_s = f"{picked:,}"
        else:
            cap_s = "-"
            pick_s = "-"
        lines_out.append(
            f"{label:>10s} | {seen:>12,d} | {pct:>11.2f} % | "
            f"{cap_s:>10s} | {pick_s:>10s}"
        )
    lines_out.append("")
    lines_out.append(f"# selected total = {len(selected):,}")
    lines_out.append(f"# wrote         {out_users}")
    if out_tsv:
        lines_out.append(f"# also wrote    {out_tsv}")

    report = "\n".join(lines_out)
    print("\n" + report, flush=True)
    with open(out_dist, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 5) machine-readable stat
    stat = {
        "seed": args.seed,
        "min_wt_sec": args.min_wt_sec,
        "target_total": args.total,
        "actual_total": len(selected),
        "scanned_lines": n_lines,
        "scanned_users": total_users_seen,
        "buckets": {
            b: {
                "seen": bucket_seen.get(b, 0),
                "capacity": bucket_targets.get(b, 0),
                "picked": len(reservoirs[b].reservoir),
                "ratio": SAMPLE_BUCKET_RATIOS.get(b, 0.0),
            }
            for b in SAMPLE_BUCKETS
        },
        "input": os.path.abspath(args.input),
    }
    with open(out_stat, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    print(f"[onepass] wrote {out_users}  ({len(selected):,} users)", flush=True)
    print(f"[onepass] wrote {out_dist}", flush=True)
    print(f"[onepass] wrote {out_stat}", flush=True)


if __name__ == "__main__":
    main()
