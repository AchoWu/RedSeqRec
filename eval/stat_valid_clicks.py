"""
stat_valid_clicks.py
====================
Stream-scan the big raw jsonl and compute, for every user, the number of
"valid clicks" inside seq_all.seqs.smallvideo_float, where a valid click is
defined as

    wt = max(rd, pd) >= MIN_WT_SEC          (default 5.0s)

Outputs:
  --out-tsv : one record per user
              qimei36 \t valid_count \t total_smallvideo_events \t exp_count
  --out-hist: a printable histogram + bucket counts (also echoed to stdout)

This is the foundation step before user sampling. It scans the file once
(estimated 30~40 minutes on the 253GB file) and writes a small (a few
hundred MB at most) tsv that downstream scripts can mmap / pandas-load.

Usage
-----
    python stat_valid_clicks.py \
        --input /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --out-tsv  ./valid_click_counts.tsv \
        --out-hist ./valid_click_counts.hist.txt \
        --min-wt-sec 5.0
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

# Length buckets used for sampling. We also print a finer histogram for
# diagnostics, but the *bucket* column in the tsv reflects this list so that
# sample_users.py can read it without recomputing.
LEN_BUCKETS = [
    (0, 1, "0"),
    (1, 5, "1-4"),
    (5, 10, "5-9"),
    (10, 20, "10-19"),
    (20, 50, "20-49"),
    (50, 100, "50-99"),
    (100, 150, "100-149"),
    (150, 200, "150-199"),
    (200, 10**9, "200+"),
]


def get_bucket(n):
    for lo, hi, label in LEN_BUCKETS:
        if lo <= n < hi:
            return label
    return "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-tsv", required=True,
                    help="path to write per-user valid-click counts")
    ap.add_argument("--out-hist", required=True,
                    help="path to write the human-readable histogram")
    ap.add_argument("--min-wt-sec", type=float, default=5.0,
                    help="threshold of wt = max(rd, pd) to count as valid click")
    ap.add_argument("--progress-every", type=int, default=200_000)
    ap.add_argument("--max-lines", type=int, default=-1,
                    help="if >0, stop after scanning this many lines "
                         "(useful for quick preview on a large file)")
    args = ap.parse_args()

    if os.path.exists(args.out_tsv):
        print(f"[warn] {args.out_tsv} already exists, will be overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)) or ".", exist_ok=True)

    print(f"[stat_valid] input        = {args.input}", flush=True)
    print(f"[stat_valid] out-tsv      = {args.out_tsv}", flush=True)
    print(f"[stat_valid] out-hist     = {args.out_hist}", flush=True)
    print(f"[stat_valid] min_wt_sec   = {args.min_wt_sec}", flush=True)
    fsize = os.path.getsize(args.input)
    print(f"[stat_valid] file size    = {fsize / (1024**3):.2f} GiB", flush=True)
    if args.max_lines > 0:
        print(f"[stat_valid] PREVIEW MODE  max_lines = {args.max_lines:,}", flush=True)
    print(f"[stat_valid] starting stream ...", flush=True)

    t0 = time.time()
    n_lines = 0
    n_bad = 0
    n_no_seq = 0
    bucket_counts = Counter()
    valid_count_total = 0
    total_event_total = 0
    exp_event_total = 0

    def _fmt_eta(seconds):
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_tsv, "w", encoding="utf-8") as fout:
        # tsv header
        fout.write("qimei36\tvalid_count\ttotal_events\texp_count\tlen_bucket\n")

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
                n_no_seq += 1
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
            bucket_counts[bucket] += 1
            valid_count_total += valid
            total_event_total += total
            exp_event_total += exp_n

            fout.write(f"{qimei}\t{valid}\t{total}\t{exp_n}\t{bucket}\n")

            if args.max_lines > 0 and n_lines >= args.max_lines:
                print(f"[stat_valid] reached --max-lines={args.max_lines:,}, stop.",
                      flush=True)
                break

            if n_lines % args.progress_every == 0:
                el = time.time() - t0
                rate = n_lines / max(el, 1e-9)
                pos = fin.tell()
                pct = pos / max(fsize, 1) * 100
                eta = (fsize - pos) / max(pos / max(el, 1e-9), 1) if pos > 0 else 0
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                # show running pool count for usable users (valid in [10,200))
                pool_now = (
                    bucket_counts.get("10-19", 0)
                    + bucket_counts.get("20-49", 0)
                    + bucket_counts.get("50-99", 0)
                    + bucket_counts.get("100-149", 0)
                    + bucket_counts.get("150-199", 0)
                )
                print(f"[stat_valid] [{bar}] {pct:5.1f}%  "
                      f"bytes={pos / (1024**3):.2f}/{fsize / (1024**3):.2f}GiB  "
                      f"lines={n_lines:,}  rate={rate:.0f} l/s  "
                      f"elapsed={_fmt_eta(el)}  ETA={_fmt_eta(eta)}",
                      flush=True)
                print(f"[stat_valid]   pool([10,200))={pool_now:,}  "
                      f"valid_total={valid_count_total:,}  "
                      f"events_total={total_event_total:,}",
                      flush=True)

    el = time.time() - t0
    print(f"\n[stat_valid] DONE  lines={n_lines:,}  "
          f"bad={n_bad}  no_seq={n_no_seq}  elapsed={el:.0f}s", flush=True)

    # ----- write histogram report -----
    total_users = sum(bucket_counts.values())
    lines_out = []
    lines_out.append(f"# stat_valid_clicks.py report")
    lines_out.append(f"# input        = {args.input}")
    lines_out.append(f"# out-tsv      = {args.out_tsv}")
    lines_out.append(f"# min_wt_sec   = {args.min_wt_sec}")
    lines_out.append(f"# total_users  = {total_users:,}")
    lines_out.append(f"# bad_lines    = {n_bad}")
    lines_out.append(f"# no_seq_users = {n_no_seq}")
    lines_out.append(f"# total_smallvideo_events = {total_event_total:,}")
    lines_out.append(f"# total_valid_clicks      = {valid_count_total:,}")
    lines_out.append(f"# total_exp_events        = {exp_event_total:,}")
    if total_users > 0:
        lines_out.append(
            f"# avg_valid_per_user      = "
            f"{valid_count_total / total_users:.2f}"
        )
    lines_out.append("")
    lines_out.append("valid_count_bucket | n_users | pct(of total)")
    lines_out.append("-" * 60)
    bar_unit = max(total_users // 50, 1)
    for _, _, label in LEN_BUCKETS:
        n = bucket_counts.get(label, 0)
        pct = n / max(total_users, 1) * 100
        bar = "#" * (n // bar_unit)
        lines_out.append(f"  {label:>10s}  | {n:>10,d} | {pct:5.2f}%  {bar}")

    # users actually usable: valid in [10, 200]
    pool = (
        bucket_counts.get("10-19", 0)
        + bucket_counts.get("20-49", 0)
        + bucket_counts.get("50-99", 0)
        + bucket_counts.get("100-149", 0)
        + bucket_counts.get("150-199", 0)
    )
    lines_out.append("")
    lines_out.append(f"# users with valid_count in [10, 200) = {pool:,}")
    if total_users > 0:
        lines_out.append(
            f"# = {pool / total_users * 100:.2f}% of all users"
        )

    report = "\n".join(lines_out)
    print("\n" + report, flush=True)
    with open(args.out_hist, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[stat_valid] histogram written to {args.out_hist}", flush=True)


if __name__ == "__main__":
    main()
