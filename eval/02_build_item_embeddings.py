"""
Build pickled item embeddings for our custom RedSeqRec eval (Plan B).

Inputs
------
- ../DataProcess/doc2rowkey_with_emb.csv
  columns: doc_id, rowkey, embedding (512d, comma-separated string)

Outputs (under eval/emb/<tag_name>/)
------------------------------------
1. lastn_item_embedding_pr.pkl   -> list[ {id, embed (1536d), embed_type='cur_embed_64d'} ]
   The 1536d vector is the original 512d embedding REPEATED 3 TIMES (Plan B).
   This file is consumed by `compute_user_embedding` -> `raw_embeds`, which is
   directly fed as `inputs_embeds` to the User LLM. For Qwen2.5-1.5B,
   item_llm.config.hidden_size = 1536, so 3*512 == 1536 fits exactly.

2. base_item_embedding_pr.pkl    -> list[ {id, embed (64d),  embed_type='basepool_embed_64d'} ]
   The 64d vector is the FIRST 64 dims of the original 512d embedding (Plan B).
   This is the candidate pool against which the user's 64d output is scored.

The file shape and `embed_type` strings are chosen to be drop-in compatible
with the official `generate_lastn_item_embedding.py` / `generate_base_item_embedding.py`
output, so downstream `generate_user_embedding.py` and `eval.py` work unchanged.

Usage
-----
    cd eval
    python 02_build_item_embeddings.py --tag_name my_data
"""
import os
import csv
import sys
import json
import pickle
import argparse


def parse_emb(s):
    """Parse "0.13,-0.50,..." into list[float]. Tolerates surrounding quotes/brackets."""
    s = s.strip()
    if not s:
        return []
    if s[0] in '"\'[':
        s = s.strip('"').strip("'").strip('[]')
    return [float(p) for p in s.split(',') if p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv", type=str,
        default="../DataProcess/doc2rowkey_with_emb.csv",
    )
    parser.add_argument("--tag_name", type=str, default="my_data")
    parser.add_argument(
        "--filter_doc_ids_json", type=str, default="",
        help="optional path to user_lastn_my.json; if given, only keep doc_ids "
             "that appear in any user's lastn or target.",
    )
    parser.add_argument("--lastn_dim", type=int, default=1536,
                        help="must equal item_llm.config.hidden_size (Qwen2.5-1.5B = 1536)")
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=512)
    args = parser.parse_args()

    out_dir = os.path.join("emb", args.tag_name)
    os.makedirs(out_dir, exist_ok=True)

    keep_set = None
    if args.filter_doc_ids_json:
        keep_set = set()
        with open(args.filter_doc_ids_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for _, v in obj.items():
            for n in v.get("homefeed_noteid_lastn", []):
                nid = str(n.get("note_id", ""))
                if nid and nid != "-1":
                    keep_set.add(nid)
            for n in v.get("ads_click_noteid_target", []):
                keep_set.add(str(n["note_id"]))
        print(f"[02_build_item_embeddings] doc-id filter set size: {len(keep_set)}")

    # The csv embedding cell is large; raise the field-size limit just in case.
    csv.field_size_limit(sys.maxsize)

    if args.lastn_dim != 3 * args.input_dim:
        raise ValueError(
            f"lastn_dim ({args.lastn_dim}) must be 3 * input_dim ({args.input_dim}). "
            f"For Qwen2.5-1.5B, hidden_size = 1536 = 3 * 512."
        )

    lastn_records, base_records = [], []
    n_total, n_kept, n_bad = 0, 0, 0
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            doc_id = str(row.get("doc_id", "")).strip()
            emb_str = row.get("embedding", "")
            if not doc_id or not emb_str:
                n_bad += 1
                continue
            if keep_set is not None and doc_id not in keep_set:
                continue
            try:
                emb = parse_emb(emb_str)
            except Exception:
                n_bad += 1
                continue
            if len(emb) != args.input_dim:
                n_bad += 1
                continue

            # Plan B core: 1536d = 512d repeated 3 times; 64d = first 64 dims.
            lastn_emb = emb + emb + emb
            base_emb = emb[: args.base_dim]

            # Round to keep file size manageable, like the official scripts do.
            lastn_emb = [round(x, 6) for x in lastn_emb]
            base_emb = [round(x, 6) for x in base_emb]

            lastn_records.append({"id": doc_id, "embed": lastn_emb,
                                  "embed_type": "cur_embed_64d"})
            base_records.append({"id": doc_id, "embed": base_emb,
                                 "embed_type": "basepool_embed_64d"})
            n_kept += 1
            if n_kept % 20000 == 0:
                print(f"  ...processed {n_kept} docs")

    lastn_path = os.path.join(out_dir, "lastn_item_embedding_pr.pkl")
    base_path = os.path.join(out_dir, "base_item_embedding_pr.pkl")
    with open(lastn_path, "wb") as f:
        pickle.dump(lastn_records, f)
    with open(base_path, "wb") as f:
        pickle.dump(base_records, f)

    print(f"[02_build_item_embeddings] total_rows={n_total}  kept={n_kept}  bad={n_bad}")
    print(f"  lastn -> {lastn_path}  ({len(lastn_records)} items, dim={args.lastn_dim})")
    print(f"  base  -> {base_path}   ({len(base_records)} items, dim={args.base_dim})")


if __name__ == "__main__":
    main()
