"""
process_all_users.py
====================
Single-pass full processor. Walks the giant raw jsonl ONCE. For every
user with at least MIN_VALID_CLICKS valid clicks (wt = max(rd, pd) >=
MIN_WT_SEC), emit one line of training-ready jsonl with the full
history list (positives + hard-negatives + noise, all labelled).

This script does NOT do any sampling. It just persists *everything
qualified*. Sampling / bucketing is done by a downstream script
(``sample_from_processed.py``) so we can re-sample to different sizes
or bucket schemes without re-reading the 253 GiB input.

Output schema (one JSON object per line):
    {
      "qimei36":     "...",
      "age":         "40",
      "gender":      "女",
      "valid_count": 87,           # # of pos events in stored history
      "history_len": 142,          # length of stored history (<=max_len)
      "history": [
        {"cid":"...","ts":...,"wt":12.5,"rd":12.5,"pd":0.0,"exp":0,"label":"pos"},
        ...
      ]
    }

Usage
-----
    python process_all_users.py \
        --input /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --out-jsonl  data_prep/all_qualified_users.jsonl \
        --out-stat   data_prep/process_stat.json \
        --min-wt-sec 5.0 \
        --min-valid-clicks 10 \
        --max-len 200 \
        --hard-neg-max-wt 1.0
"""
import argparse
import json
import os
import time
from collections import Counter


# Length buckets only used for the running-progress and final stats; this
# script does not actually filter by bucket, it filters by min-valid-clicks.
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


def get_bucket(n):
    for lo, hi, label in LEN_BUCKETS:
        if lo <= n < hi:
            return label
    return "0"


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
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-jsonl", required=True,
                    help="path to write per-user jsonl (kept users only)")
    ap.add_argument("--out-stat", required=True,
                    help="path to write process stat json")
    ap.add_argument("--min-wt-sec", type=float, default=5.0,
                    help="threshold of wt = max(rd, pd) for a positive click")
    ap.add_argument("--min-valid-clicks", type=int, default=10,
                    help="users with fewer pos clicks than this are dropped")
    ap.add_argument("--max-len", type=int, default=200,
                    help="cap each user's history to the most recent N events")
    ap.add_argument("--hard-neg-max-wt", type=float, default=1.0,
                    help="upper bound of wt to label an exposed event as hardneg")
    ap.add_argument("--progress-every", type=int, default=200_000)
    ap.add_argument("--max-lines", type=int, default=-1,
                    help="if >0 stop after scanning this many lines (debug)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".",
                exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_stat)) or ".",
                exist_ok=True)

    fsize = os.path.getsize(args.input)

    print(f"[process] input             = {args.input}", flush=True)
    print(f"[process] out_jsonl         = {args.out_jsonl}", flush=True)
    print(f"[process] out_stat          = {args.out_stat}", flush=True)
    print(f"[process] min_wt_sec        = {args.min_wt_sec}", flush=True)
    print(f"[process] min_valid_clicks  = {args.min_valid_clicks}", flush=True)
    print(f"[process] max_len           = {args.max_len}", flush=True)
    print(f"[process] hard_neg_max_wt   = {args.hard_neg_max_wt}", flush=True)
    print(f"[process] file size         = {fsize / (1024**3):.2f} GiB", flush=True)
    if args.max_lines > 0:
        print(f"[process] DEBUG MODE  max_lines = {args.max_lines:,}", flush=True)
    print(f"[process] starting stream ...", flush=True)

    t0 = time.time()
    n_lines = 0
    n_bad = 0
    n_no_user = 0
    n_no_seq = 0
    n_kept = 0
    n_dropped_short = 0
    bucket_kept = Counter()              # over kept users only
    pos_events_total = 0
    hardneg_events_total = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        # NOTE: use readline() in a while-loop instead of `for line in fin:`
        # because the latter uses an internal read-ahead buffer that disables
        # fin.tell()  (raises "OSError: telling position disabled by next()
        # call"). readline() preserves accurate byte position for the
        # progress bar.
        while True:
            line = fin.readline()
            if not line:
                break
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
            if not ev:
                n_no_seq += 1
                continue

            # Sort by ts ascending (defensive; usually already sorted)
            ev_sorted = sorted(ev, key=lambda x: x.get("ts", 0))

            # Build labelled history
            history = []
            for e in ev_sorted:
                rd = float(e.get("rd", 0) or 0)
                pd = float(e.get("pd", 0) or 0)
                wt = rd if rd >= pd else pd
                exp = int(e.get("exp", 0) or 0)
                if wt >= args.min_wt_sec:
                    lab = "pos"
                elif exp == 1 and wt < args.hard_neg_max_wt:
                    lab = "hardneg"
                else:
                    lab = "noise"
                history.append({
                    "cid": e.get("cid", ""),
                    "ts": int(e.get("ts", 0) or 0),
                    "wt": round(wt, 3),
                    "rd": round(rd, 3),
                    "pd": round(pd, 3),
                    "exp": exp,
                    "label": lab,
                })

            # Keep only the most recent max_len events
            if len(history) > args.max_len:
                history = history[-args.max_len:]

            valid_count = sum(1 for h in history if h["label"] == "pos")
            if valid_count < args.min_valid_clicks:
                n_dropped_short += 1
            else:
                neg_count = sum(1 for h in history if h["label"] == "hardneg")
                pos_events_total += valid_count
                hardneg_events_total += neg_count

                bucket_kept[get_bucket(valid_count)] += 1
                n_kept += 1

                rec = {
                    "qimei36":     qimei,
                    "age":         sa.get("age", ""),
                    "gender":      sa.get("gender", ""),
                    "valid_count": valid_count,
                    "history_len": len(history),
                    "history":     history,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if n_lines % args.progress_every == 0:
                el = time.time() - t0
                pos = fin.tell()
                pct = pos / max(fsize, 1) * 100
                rate = n_lines / max(el, 1e-9)
                eta = (fsize - pos) / max(pos / max(el, 1e-9), 1) if pos > 0 else 0
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                print(f"[process] [{bar}] {pct:5.1f}%  "
                      f"bytes={pos / (1024**3):.2f}/{fsize / (1024**3):.2f}GiB  "
                      f"lines={n_lines:,}  rate={rate:.0f} l/s  "
                      f"elapsed={_fmt_eta(el)}  ETA={_fmt_eta(eta)}",
                      flush=True)
                # show kept-buckets-so-far
                seg = " ".join(
                    f"{b}:{bucket_kept.get(b, 0):,}"
                    for _, _, b in LEN_BUCKETS if b not in ("0", "1-4", "5-9")
                )
                print(f"[process]   kept={n_kept:,}  dropped_short="
                      f"{n_dropped_short:,}  bad={n_bad}  no_user={n_no_user}  "
                      f"no_seq={n_no_seq}", flush=True)
                print(f"[process]   buckets[kept]: {seg}", flush=True)

            if args.max_lines > 0 and n_lines >= args.max_lines:
                print(f"[process] reached --max-lines={args.max_lines:,}, stop.",
                      flush=True)
                break

    el = time.time() - t0
    print(f"\n[process] DONE  scanned={n_lines:,}  kept={n_kept:,}  "
          f"dropped_short={n_dropped_short:,}  bad={n_bad}  "
          f"no_user={n_no_user}  no_seq={n_no_seq}  elapsed={el:.0f}s",
          flush=True)
    print(f"[process] total positive events  = {pos_events_total:,}", flush=True)
    print(f"[process] total hard-neg events  = {hardneg_events_total:,}", flush=True)
    if n_kept > 0:
        print(f"[process] avg pos / user         = "
              f"{pos_events_total / n_kept:.2f}", flush=True)
        print(f"[process] avg hardneg / user     = "
              f"{hardneg_events_total / n_kept:.2f}", flush=True)

    # -- per-bucket distribution over the *kept* set --
    print(f"\n[process] kept users by bucket:", flush=True)
    bar_unit = max(n_kept // 50, 1)
    for _, _, label in LEN_BUCKETS:
        n = bucket_kept.get(label, 0)
        pct = n / max(n_kept, 1) * 100
        b = "#" * (n // bar_unit)
        print(f"  {label:>10s} | {n:>12,d} | {pct:5.2f}%  {b}", flush=True)

    # -- write machine-readable stat --
    stat = {
        "input": os.path.abspath(args.input),
        "out_jsonl": os.path.abspath(args.out_jsonl),
        "min_wt_sec": args.min_wt_sec,
        "min_valid_clicks": args.min_valid_clicks,
        "max_len": args.max_len,
        "hard_neg_max_wt": args.hard_neg_max_wt,
        "scanned_lines": n_lines,
        "bad_lines": n_bad,
        "no_user": n_no_user,
        "no_seq": n_no_seq,
        "kept_users": n_kept,
        "dropped_short_users": n_dropped_short,
        "total_pos_events": pos_events_total,
        "total_hardneg_events": hardneg_events_total,
        "kept_buckets": {b: bucket_kept.get(b, 0) for _, _, b in LEN_BUCKETS},
        "elapsed_sec": el,
    }
    with open(args.out_stat, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(f"\n[process] wrote {args.out_jsonl}", flush=True)
    print(f"[process] wrote {args.out_stat}", flush=True)


if __name__ == "__main__":
    main()
