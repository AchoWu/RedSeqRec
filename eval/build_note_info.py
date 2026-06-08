"""
Build note_info_my.json for the official Red-Mmu-Rec item tower.

Input
-----
- ../DataProcess/hbase_fields_clean.tsv
    columns: doc_id, rowkey, title, ocr, summary
- (optional) user_lastn_my.json
    used to filter notes down to the ones that actually appear in any user's
    lastn / target. Skipping this step would still work but waste GPU time.

Output
------
- note_info_my.json  (same dir as this script, i.e. eval/)
    list[ { "note_id": str,
             "title":  str,
             "content": str,   # we use the AI summary as `content`
             "ocr":    str,
             "fimg_url": "" } ]

This file is consumed by 03_build_official_item_embeddings.py and matches the
shape expected by REDRec/data/dataset/dataset.py:process_item_for_note_inference.
"""
import os
import csv
import sys
import json
import argparse


def load_keep_ids(user_lastn_path: str):
    if not user_lastn_path or not os.path.isfile(user_lastn_path):
        return None
    with open(user_lastn_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    keep = set()
    for _, v in obj.items():
        for n in v.get("homefeed_noteid_lastn", []):
            nid = str(n.get("note_id", ""))
            if nid and nid != "-1":
                keep.add(nid)
        for n in v.get("ads_click_noteid_target", []):
            nid = str(n.get("note_id", ""))
            if nid and nid != "-1":
                keep.add(nid)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_tsv",
        type=str,
        default="../DataProcess/hbase_fields_clean.tsv",
    )
    parser.add_argument(
        "--user_lastn_path",
        type=str,
        default="user_lastn_my.json",
        help="if exists, only keep doc_ids referenced by these users",
    )
    parser.add_argument(
        "--out_path", type=str, default="note_info_my.json"
    )
    args = parser.parse_args()

    keep = load_keep_ids(args.user_lastn_path)
    if keep is not None:
        print(f"[build_note_info] keep_set size from user_lastn: {len(keep)}")
    else:
        print("[build_note_info] no user_lastn filter, keeping all rows")

    csv.field_size_limit(sys.maxsize)

    note_infos = []
    n_in = 0
    n_skip_no_doc = 0
    n_skip_filter = 0
    n_skip_dup = 0
    seen = set()

    with open(args.clean_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_in += 1
            doc_id = (row.get("doc_id") or "").strip()
            if not doc_id:
                n_skip_no_doc += 1
                continue
            if keep is not None and doc_id not in keep:
                n_skip_filter += 1
                continue
            if doc_id in seen:
                n_skip_dup += 1
                continue
            seen.add(doc_id)

            note_infos.append({
                "note_id":  doc_id,
                "title":   (row.get("title")   or "").strip(),
                # the official template expects a `content` field; we feed the
                # AI summary into it (best free-text we have for these notes)
                "content": (row.get("summary") or "").strip(),
                "ocr":     (row.get("ocr")     or "").strip(),
                "fimg_url": "",
            })

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(note_infos, f, ensure_ascii=False)

    print(
        f"[build_note_info] in={n_in} kept={len(note_infos)} "
        f"skip_no_doc={n_skip_no_doc} skip_filter={n_skip_filter} skip_dup={n_skip_dup}"
    )
    print(f"[build_note_info] -> {args.out_path}")


if __name__ == "__main__":
    main()
