"""
show_long_user.py
=================

Quick validator: scan the big jsonl, find a few users with sizable
seq_all.seqs.smallvideo_float, and report:

  - first event sample (full dict)
  - clk distribution (clk=1 count vs total events)
  - rd>0 / pd>0 / both>0 ratio
  - wt = max(rd, pd) histogram

This is to confirm:
  1. clk == 1 indeed marks real clicks (so we can filter on it)
  2. rd/pd are populated for active users (so wt>=5s filter is meaningful)

Usage:
    python show_long_user.py \
        --input /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --target-users 5 \
        --min-len 200
"""
import argparse
import json
from collections import Counter


def stat_events(events):
    """events: list[ {cid, clk, ctype, exp, pd, rd, ts, ...} ]"""
    n = len(events)
    n_clk = sum(1 for e in events if e.get("clk", 0) == 1)
    n_exp = sum(1 for e in events if e.get("exp", 0) == 1)
    n_rd_pos = sum(1 for e in events if float(e.get("rd", 0) or 0) > 0)
    n_pd_pos = sum(1 for e in events if float(e.get("pd", 0) or 0) > 0)

    wt_buckets = [
        (0, 1e-6, "==0"),
        (1e-6, 3, "0-3s"),
        (3, 5, "3-5s"),
        (5, 10, "5-10s"),
        (10, 30, "10-30s"),
        (30, 60, "30-60s"),
        (60, 180, "60-180s"),
        (180, 1e12, "180s+"),
    ]
    wt_hist = Counter()
    wt_total = 0.0
    n_wt_pos = 0
    for e in events:
        rd = float(e.get("rd", 0) or 0)
        pd = float(e.get("pd", 0) or 0)
        wt = max(rd, pd)
        wt_total += wt
        if wt > 0:
            n_wt_pos += 1
        for lo, hi, label in wt_buckets:
            if lo <= wt < hi:
                wt_hist[label] += 1
                break

    # additionally: clk=1 AND wt>=5s
    n_valid_5s = sum(
        1 for e in events
        if e.get("clk", 0) == 1
        and max(float(e.get("rd", 0) or 0), float(e.get("pd", 0) or 0)) >= 5.0
    )
    n_valid_3s = sum(
        1 for e in events
        if e.get("clk", 0) == 1
        and max(float(e.get("rd", 0) or 0), float(e.get("pd", 0) or 0)) >= 3.0
    )
    # clk=1 alone (no wt filter)
    n_clk_only = n_clk

    print(f"  total events           = {n}")
    print(f"  clk == 1               = {n_clk}  ({n_clk/max(n,1)*100:.1f}%)")
    print(f"  exp == 1               = {n_exp}  ({n_exp/max(n,1)*100:.1f}%)")
    print(f"  rd > 0                 = {n_rd_pos}  ({n_rd_pos/max(n,1)*100:.1f}%)")
    print(f"  pd > 0                 = {n_pd_pos}  ({n_pd_pos/max(n,1)*100:.1f}%)")
    print(f"  wt = max(rd,pd) > 0    = {n_wt_pos}  ({n_wt_pos/max(n,1)*100:.1f}%)")
    print(f"  wt mean                = {wt_total/max(n,1):.2f}s")
    print(f"  wt distribution:")
    for _, _, label in wt_buckets:
        c = wt_hist.get(label, 0)
        print(f"    {label:>10s}: {c:>8d}  ({c/max(n,1)*100:5.1f}%)")
    print(f"  ----- effective click candidates -----")
    print(f"  clk==1                 = {n_clk_only}  ({n_clk_only/max(n,1)*100:.1f}%)")
    print(f"  clk==1 AND wt>=3s      = {n_valid_3s}  ({n_valid_3s/max(n,1)*100:.1f}%)")
    print(f"  clk==1 AND wt>=5s      = {n_valid_5s}  ({n_valid_5s/max(n,1)*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target-users", type=int, default=5,
                    help="how many qualifying users to print")
    ap.add_argument("--min-len", type=int, default=200,
                    help="only print users whose smallvideo_float seq length >= this")
    ap.add_argument("--max-scan", type=int, default=2_000_000,
                    help="stop scanning after this many lines if not enough users found")
    args = ap.parse_args()

    found = 0
    scanned = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sa = obj.get("seq_all") or {}
            seqs = sa.get("seqs") or {}
            ev = seqs.get("smallvideo_float") or []
            if len(ev) < args.min_len:
                if scanned >= args.max_scan:
                    print(f"\n[stop] scanned {scanned:,} lines, only "
                          f"found {found} qualifying users")
                    return
                continue

            found += 1
            print(f"\n========== user #{found}  "
                  f"qimei36={sa.get('qimei36','?')[:16]}...  "
                  f"smallvideo_float_len={len(ev)} ==========")
            print(f"  age={sa.get('age','?')}  gender={sa.get('gender','?')}  "
                  f"total_len={sa.get('total_len','?')}")
            print(f"  first event:")
            for k, v in ev[0].items():
                print(f"    {k}: {v!r}")
            print(f"  ----- stats over {len(ev)} events -----")
            stat_events(ev)

            if found >= args.target_users:
                break

    print(f"\n[done] scanned={scanned:,}  found_users={found}")


if __name__ == "__main__":
    main()
