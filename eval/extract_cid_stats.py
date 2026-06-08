"""
extract_cid_stats.py
====================
Stream-scan a per-user jsonl (e.g. train_users_500k.jsonl) and aggregate,
for every distinct cid that ever appears in any user's history, three
counters:

    n_pos     : how many times this cid was a positive event
    n_hardneg : how many times exposed-but-not-watched (hard negative)
    n_noise   : how many other appearances (e.g. very short watch)

This is the foundation for two downstream steps:

    1. Build the global negative item pool   (uses n_pos -> popularity)
    2. Determine which embeddings we need    (any cid with total > 0)

We also bucket cids by total frequency so we can decide later how to
truncate the long tail.

Outputs
-------
  --out-tsv : "cid \t n_pos \t n_hardneg \t n_noise \t n_total"
              one line per distinct cid, sorted by n_total descending.
  --out-stat: machine-readable summary (json)

Memory: O(distinct_cid_count). For 50w users * ~150 events/user the upper
bound is 75M event slots; distinct cids likely ~10~30M. Each entry is
~100 bytes in Python, so peak memory ~1~3 GB. Acceptable for one-off
processing; if too tight we'll switch to a sharded counter.

Usage
-----
    python extract_cid_stats.py \
        --in-jsonl  data_prep/train_users_500k.jsonl \
        --out-tsv   data_prep/cid_stats.tsv \
        --out-stat  data_prep/cid_stats.json
"""
import argparse
import json
import os
import time


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
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-stat", required=True)
    ap.add_argument("--progress-every", type=int, default=20_000,
                    help="log progress every N users")
    args = ap.parse_args()

    fsize = os.path.getsize(args.in_jsonl)
    print(f"[cidstat] in_jsonl  = {args.in_jsonl}", flush=True)
    print(f"[cidstat] file size = {fsize / (1024**3):.2f} GiB", flush=True)
    print(f"[cidstat] streaming ...", flush=True)

    # cid -> [n_pos, n_hardneg, n_noise]
    counters = {}
    n_users = 0
    n_events = 0
    n_pos_evt = 0
    n_neg_evt = 0
    n_noise_evt = 0

    t0 = time.time()
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_users += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for h in obj.get("history", []):
                cid = h.get("cid", "")
                if not cid:
                    continue
                lab = h.get("label", "noise")
                if cid not in counters:
                    counters[cid] = [0, 0, 0]
                if lab == "pos":
                    counters[cid][0] += 1
                    n_pos_evt += 1
                elif lab == "hardneg":
                    counters[cid][1] += 1
                    n_neg_evt += 1
                else:
                    counters[cid][2] += 1
                    n_noise_evt += 1
                n_events += 1

            if n_users % args.progress_every == 0:
                el = time.time() - t0
                pos = f.tell()
                pct = pos / max(fsize, 1) * 100
                bar_w = 30
                filled = int(pct / 100 * bar_w)
                bar = "#" * filled + "-" * (bar_w - filled)
                eta = (fsize - pos) / max(pos / max(el, 1e-9), 1) if pos > 0 else 0
                print(f"[cidstat] [{bar}] {pct:5.1f}%  "
                      f"users={n_users:,}  events={n_events:,}  "
                      f"distinct_cids={len(counters):,}  "
                      f"elapsed={_fmt_eta(el)}  ETA={_fmt_eta(eta)}",
                      flush=True)

    el = time.time() - t0
    n_distinct = len(counters)
    print(f"\n[cidstat] DONE  users={n_users:,}  events={n_events:,}  "
          f"distinct_cids={n_distinct:,}  elapsed={_fmt_eta(el)}", flush=True)
    print(f"[cidstat]   pos     events = {n_pos_evt:,}", flush=True)
    print(f"[cidstat]   hardneg events = {n_neg_evt:,}", flush=True)
    print(f"[cidstat]   noise   events = {n_noise_evt:,}", flush=True)

    # ----- frequency-bucket histogram (over distinct cids) -----
    freq_buckets = [
        (1,    1,     "=1"),
        (2,    2,     "=2"),
        (3,    5,     "3-5"),
        (6,    10,    "6-10"),
        (11,   50,    "11-50"),
        (51,   200,   "51-200"),
        (201,  1000,  "201-1k"),
        (1001, 10**9, "1k+"),
    ]
    fb_count = {b[2]: 0 for b in freq_buckets}
    pos_only = 0
    neg_only = 0
    has_pos = 0
    only_noise = 0
    for cid, (np_, nh, no) in counters.items():
        tot = np_ + nh + no
        for lo, hi, lab in freq_buckets:
            if lo <= tot <= hi:
                fb_count[lab] += 1
                break
        if np_ > 0:
            has_pos += 1
            if nh == 0 and no == 0:
                pos_only += 1
        else:
            if nh > 0 and no == 0:
                neg_only += 1
            elif nh == 0 and no > 0:
                only_noise += 1
    print(f"\n[cidstat] cid frequency histogram (over distinct cids):",
          flush=True)
    bar_unit = max(n_distinct // 50, 1)
    for _, _, lab in freq_buckets:
        c = fb_count[lab]
        pct = c / max(n_distinct, 1) * 100
        bar = "#" * (c // bar_unit)
        print(f"  {lab:>8s} | {c:>12,d} | {pct:5.2f}%  {bar}", flush=True)

    print(f"\n[cidstat] cid label coverage:", flush=True)
    print(f"  cids with any pos     : {has_pos:>12,d} "
          f"({has_pos / n_distinct * 100:.2f}%)", flush=True)
    print(f"  cids pos-only         : {pos_only:>12,d}", flush=True)
    print(f"  cids hardneg-only     : {neg_only:>12,d}", flush=True)
    print(f"  cids noise-only       : {only_noise:>12,d}", flush=True)

    # ----- write tsv (sorted by n_total desc) -----
    print(f"\n[cidstat] sorting and writing tsv ...", flush=True)
    items = [
        (cid, np_, nh, no, np_ + nh + no)
        for cid, (np_, nh, no) in counters.items()
    ]
    items.sort(key=lambda x: x[4], reverse=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)) or ".",
                exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as fout:
        fout.write("cid\tn_pos\tn_hardneg\tn_noise\tn_total\n")
        for cid, np_, nh, no, tot in items:
            fout.write(f"{cid}\t{np_}\t{nh}\t{no}\t{tot}\n")
    print(f"[cidstat] wrote {args.out_tsv}  ({n_distinct:,} rows)", flush=True)

    # ----- write summary stat -----
    stat = {
        "in_jsonl": os.path.abspath(args.in_jsonl),
        "out_tsv": os.path.abspath(args.out_tsv),
        "n_users": n_users,
        "n_events": n_events,
        "n_pos_events": n_pos_evt,
        "n_hardneg_events": n_neg_evt,
        "n_noise_events": n_noise_evt,
        "n_distinct_cids": n_distinct,
        "n_cids_with_pos": has_pos,
        "n_cids_pos_only": pos_only,
        "n_cids_hardneg_only": neg_only,
        "n_cids_noise_only": only_noise,
        "freq_buckets": fb_count,
        "elapsed_sec": el,
    }
    with open(args.out_stat, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(f"[cidstat] wrote {args.out_stat}", flush=True)


if __name__ == "__main__":
    main()
