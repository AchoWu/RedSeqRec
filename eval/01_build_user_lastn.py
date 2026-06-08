"""
Convert ../DataProcess/users_3000.jsonl  ->  user_lastn_my.json (the format
expected by REDRecEvalUserDataset).

Output JSON schema (per user):
{
  "homefeed_noteid_lastn": [...exactly LASTN_LEN entries, left-padded with note_id='-1'...],
  "ads_click_noteid_lastn": [],   # we don't have ads, leave it empty
  "ads_click_noteid_target": [{"note_id": "..."}, ...],   # used by REDRecEvalUserDataset
  "eval_doc_ids": [...]           # plain list of target doc_ids for our custom eval
}

Why we hard-pad here:
- The repo's `user_dataset_collator` does NOT pad note_seqs across users in the
  same batch. `compute_user_embedding` then does `np.array(batch["note_seqs_homefeed"])`,
  which requires equal length per row.
- We therefore pre-pad every user's homefeed_lastn to exactly LASTN_LEN, using
  note_id = '-1' as the pad marker (the model's attention_mask = (lastn != '-1')).
- After `process_click_lastn` -> `[-LASTN_LEN:]`, every user keeps exactly
  LASTN_LEN steps regardless of their original sequence length.

Action signals (is_like / is_collect / ...) are all set to 0 because we only
have click info. With add_item_action_embed/add_hour_embed/add_position_embed
all False in our eval config, these signals are ignored at inference time.
A small valid timestamp is fabricated from `day_offset` so that
`process_action -> datetime.fromtimestamp(...)` does not crash.
"""
import os
import json
import argparse


def make_real_item(doc_id, day_offset, play_time):
    # base epoch 1700000000 ~ 2023-11-14
    ts = 1700000000 + int(day_offset) * 86400
    return {
        "note_id": str(doc_id),
        "timestamp": int(ts),
        "duration": int(play_time) if play_time is not None else 0,
        "is_click": 1,
        "is_click_profile": 0, "is_collect": 0, "is_comment": 0, "is_follow": 0,
        "is_hide": 0, "is_like": 0, "is_nns": 0, "is_pagetime": 0,
        "is_read_comment": 0, "is_share": 0, "is_videoend": 0,
        "bid": 0.0, "page_key": 0, "type": "note",
    }


def make_pad_item():
    """Pad item: note_id='-1' -> attention_mask = 0 -> ignored by user LLM."""
    return {
        "note_id": "-1",
        "timestamp": 1700000000,
        "duration": 0,
        "is_click": 0,
        "is_click_profile": 0, "is_collect": 0, "is_comment": 0, "is_follow": 0,
        "is_hide": 0, "is_like": 0, "is_nns": 0, "is_pagetime": 0,
        "is_read_comment": 0, "is_share": 0, "is_videoend": 0,
        "bid": 0.0, "page_key": 0, "type": "note",
    }


def pad_left_to(seq, target_len):
    """Keep the most recent items; left-pad with pad items if shorter."""
    if len(seq) >= target_len:
        return seq[-target_len:]
    return [make_pad_item() for _ in range(target_len - len(seq))] + list(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl", type=str,
        default="../DataProcess/users_3000.jsonl",
    )
    parser.add_argument(
        "--output_json", type=str,
        default="user_lastn_my.json",
    )
    parser.add_argument(
        "--lastn_len", type=int, default=96,
        help="Must equal data.lastn_max_click_note_num_homefeed in the eval config.",
    )
    args = parser.parse_args()

    out = {}
    n_kept, n_skipped = 0, 0
    seq_lens = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_id = row.get("qimei36")
            train_clicks = row.get("train_clicks", []) or []
            eval_doc_ids = row.get("eval_doc_ids", []) or []
            if not user_id or len(train_clicks) == 0 or len(eval_doc_ids) == 0:
                n_skipped += 1
                continue

            # sort by day_offset ascending so the most recent click is at the end
            train_clicks = sorted(
                train_clicks,
                key=lambda c: (int(c.get("day_offset", 0)), str(c.get("doc_id", ""))),
            )

            real_items = [
                make_real_item(c["doc_id"], c.get("day_offset", 0), c.get("play_time", 0))
                for c in train_clicks if "doc_id" in c
            ]
            seq_lens.append(len(real_items))
            homefeed_lastn = pad_left_to(real_items, args.lastn_len)

            target_list = [{"note_id": str(d)} for d in eval_doc_ids]
            out[str(user_id)] = {
                "homefeed_noteid_lastn": homefeed_lastn,
                "ads_click_noteid_lastn": [],
                "ads_click_noteid_target": target_list,
                "eval_doc_ids": [str(d) for d in eval_doc_ids],
            }
            n_kept += 1

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    if seq_lens:
        seq_lens.sort()
        n = len(seq_lens)
        print("[01_build_user_lastn] real-seq-length stats:",
              f"min={seq_lens[0]} median={seq_lens[n // 2]} "
              f"mean={sum(seq_lens) / n:.1f} p90={seq_lens[int(0.9 * n)]} max={seq_lens[-1]}")
    print(f"[01_build_user_lastn] kept={n_kept} skipped={n_skipped}  "
          f"lastn_len={args.lastn_len}  ->  {args.output_json}")


if __name__ == "__main__":
    main()
