"""
build_train_data.py
====================
Second pass over the big jsonl. For each *selected* user, extract the
smallvideo_float event list, keep all events (positive / hard-neg / noise),
truncate to the most recent MAX_LEN, and emit a per-user training record.

Why keep ALL events (not only valid clicks)?
  - Positives  : wt = max(rd, pd) >= 5s
  - Hard negs  : exp == 1 AND wt < 1s     (exposed but not consumed)
  - Noise      : the rest                 (may be skipped by the trainer)

Storing all events lets the dataloader at training time freely choose
its supervision policy without re-reading the 253GB raw file.

Output: jsonl, one line per selected user. Each line has the schema:
{
  "qimei36":   "...",
  "age":       "40",
  "gender":    "女",
  "valid_count": 87,
  "len_bucket":  "50-99",
  "history": [
      {"cid": "...", "ts": 1714000000, "wt": 12.5, "rd": 12.5, "pd": 0.0,
       "exp": 0, "label": "pos"},                  # wt >= 5
      {"cid": "...", "ts": 1714000060, "wt":  0.0, "rd":  0.0, "pd": 0.0,
       "exp": 1, "label": "hardneg"},              # exp=1 & wt < 1
      {"cid": "...", "ts": 1714000120, "wt":  1.4, "rd":  1.4, "pd": 0.0,
       "exp": 0, "label": "noise"},                # rest
      ...
  ]
}

The history list is sorted by ts ascending (oldest first), and capped to
the most recent --max-len events.

Usage
-----
    python build_train_data.py \
        --input         /group/40094/ruiwentao/user_sequence_dire_prediction/user_seq_full_v3.filtered_match.jsonl \
        --selected-users ./selected_users.txt \
        --out-jsonl      ./train_users.jsonl \
        --max-len        200 \
        --pos-min-wt     5.0 \
        --hard-neg-max-wt 1.0
"""
import argparse
import json
import os
import time


def label_event(e, pos_min_wt, hard_neg_max_wt):
    rd = float(e.get("rd", 0) or 0)
    pd = float(e.get("pd", 0) or 0)
    wt = rd if rd >= pd else pd
    exp = int(e.get("exp", 0) or 0)
    if wt >= pos_min_wt:
        lab = "pos"
    elif exp == 1 and wt < hard_neg_max_wt:
        lab = "hardneg"
    else:
        lab = "noise"
    return wt, exp, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--selected-users", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-len", type=int, default=200,
                    help="cap each user's history to the most recent N events")
    ap.add_argument("--pos-min-wt", type=float, default=5.0,
                    help="threshold of wt to label an event as positive")
    ap.add_argument("--hard-neg-max-wt", type=float, default=1.0,
                    help="upper bound of wt for hard negatives")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    # 1) load selected qimei36 set
    print(f"[build] loading selected users from {args.selected_users} ...",
          flush=True)
    selected = set()
    with open(args.selected_users, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u:
                selected.add(u)
    print(f"[build] selected_users = {len(selected):,}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".",
                exist_ok=True)

    # 2) stream the big file once, write only selected users
    print(f"[build] streaming {args.input} ...", flush=True)
    fsize = os.path.getsize(args.input)
    print(f"[build] file size = {fsize / (1024**3):.2f} GiB", flush=True)
    t0 = time.time()
    n_lines = 0
    n_hit = 0
    n_no_seq = 0
    n_pos_total = 0
    n_neg_total = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            qimei = obj.get("qimei36", "")
            if not qimei:
                sa0 = obj.get("seq_all") or {}
                qimei = sa0.get("qimei36", "")
            if not qimei or qimei not in selected:
                if n_lines % args.progress_every == 0:
                    el = time.time() - t0
                    print(f"[build] lines={n_lines:,}  hit={n_hit:,}  "
                          f"rate={n_lines / max(el, 1e-9):.0f} l/s  "
                          f"elapsed={el:.0f}s", flush=True)
                continue

            sa = obj.get("seq_all") or {}
            seqs = sa.get("seqs") or {}
            ev = seqs.get("smallvideo_float") or []
            if not ev:
                n_no_seq += 1
                continue

            # sort by ts ascending; some sources may already be sorted but be safe
            ev_sorted = sorted(ev, key=lambda x: x.get("ts", 0))

            # build history: keep ALL events, label them, then truncate to most recent N
            history = []
            for e in ev_sorted:
                wt, exp, lab = label_event(
                    e, args.pos_min_wt, args.hard_neg_max_wt
                )
                history.append({
                    "cid": e.get("cid", ""),
                    "ts": int(e.get("ts", 0) or 0),
                    "wt": round(wt, 3),
                    "rd": round(float(e.get("rd", 0) or 0), 3),
                    "pd": round(float(e.get("pd", 0) or 0), 3),
                    "exp": exp,
                    "label": lab,
                })
            if len(history) > args.max_len:
                history = history[-args.max_len:]

            valid_count = sum(1 for h in history if h["label"] == "pos")
            neg_count = sum(1 for h in history if h["label"] == "hardneg")

            # Skip users whose post-truncation pos count drops below 10 (
            # rare since we already truncated by recency, but safety net).
            if valid_count < 10:
                continue

            n_pos_total += valid_count
            n_neg_total += neg_count

            out_record = {
                "qimei36": qimei,
                "age": sa.get("age", ""),
                "gender": sa.get("gender", ""),
                "valid_count": valid_count,
                "history_len": len(history),
                "history": history,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            n_hit += 1

            if n_lines % args.progress_every == 0:
                el = time.time() - t0
                print(f"[build] lines={n_lines:,}  hit={n_hit:,}  "
                      f"pos_total={n_pos_total:,}  "
                      f"neg_total={n_neg_total:,}  "
                      f"rate={n_lines / max(el, 1e-9):.0f} l/s  "
                      f"elapsed={el:.0f}s", flush=True)

            # early stop if we've already seen all selected users
            if n_hit == len(selected):
                print(f"[build] all selected users matched, early stop "
                      f"at line {n_lines:,}", flush=True)
                break

    el = time.time() - t0
    print(f"\n[build] DONE  scanned={n_lines:,}  hit={n_hit:,}  "
          f"missing={len(selected) - n_hit:,}  "
          f"no_seq={n_no_seq}  elapsed={el:.0f}s", flush=True)
    print(f"[build] total positive events     = {n_pos_total:,}", flush=True)
    print(f"[build] total hard-neg events     = {n_neg_total:,}", flush=True)
    if n_hit > 0:
        print(f"[build] avg pos / user            = "
              f"{n_pos_total / n_hit:.1f}", flush=True)
        print(f"[build] avg hard-neg / user       = "
              f"{n_neg_total / n_hit:.1f}", flush=True)
    print(f"[build] wrote {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
