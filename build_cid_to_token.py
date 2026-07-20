"""Build a per-cid category-token lookup array for explicit multi-interest training.

Purpose
-------
We are moving from kmeans-based latent interest tokens
(cluster_based_matching in redrec.py) to an explicit category-to-token
assignment. At training time, given a pos item's cid, we need an O(1)
lookup that returns which of the 3 interest tokens the item belongs to.

CSV lookups at every batch are far too slow (3.08M rows), so we
preprocess the mapping ONCE into a numpy array whose row ordering
matches the memmap pool's cids.npy. At training time the dataloader
memmap-opens this file and does a single indexed read.

Output file: <memmap-dir>/cid_to_token.npy
    shape:  [num_items]                           (== len(cids.npy))
    dtype:  int8
    values: 0/1/2  -- token index (matches the token_1/2/3 lists in
                       config/category_to_token.json, 1-based -> 0-based)
            -1     -- item exists in memmap pool but has no valid first-
                       level category in the CSV. Per the "lazy filter"
                       plan (Option B), the dataloader will skip these
                       items when picking pos/seq, so no unlabeled row
                       ever reaches the model.

Row-order invariant
-------------------
The i-th entry of cid_to_token.npy corresponds to the i-th cid in
cids.npy. This mirrors how embeddings.bin already relates to cids.npy
(row i of embeddings.bin is the embedding for cids[i]). Any downstream
code that does `emb_row = embeddings[cid2idx[cid]]` can therefore do
`token = cid_to_token[cid2idx[cid]]` with the same index -- no separate
lookup structure needed.

Usage
-----
  python build_cid_to_token.py \
      --csv <path>/cids_union_subject.csv \
      --memmap-dir /group/40094/jingweidong/user_sequential_feature_recall/qbfeed_action_flow/preprocessed64d \
      --assignment config/category_to_token.json

  # optional flags:
  #   --output <path>       (default: <memmap-dir>/cid_to_token.npy)
  #   --dry-run             (compute + print stats but do not write)

At the end the script prints:
  * How many pool cids fall into each token (should match the JSON's
    _totals.distribution counts).
  * How many pool cids are unlabeled (-1) and why (empty CSV first
    field vs. cid not present in CSV at all).
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np


TOKEN_UNLABELED = -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True,
                   help='Path to cids_union_subject.csv (doc_id + '
                        'union_subject_result_first columns).')
    p.add_argument('--memmap-dir', required=True,
                   help='Path to the item-pool memmap dir (must contain '
                        'cids.npy). Output cid_to_token.npy lands here too.')
    p.add_argument('--assignment', required=True,
                   help='Path to category_to_token.json.')
    p.add_argument('--output', default=None,
                   help='Where to write cid_to_token.npy. Defaults to '
                        '<memmap-dir>/cid_to_token.npy.')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute stats and print report; do NOT write file.')
    return p.parse_args()


def load_assignment(path):
    """Return a dict: category_name -> 0-based token index.

    Reads token_N_categories fields; underscore-prefixed metadata keys
    (_comment, _totals, ...) are ignored. Detects duplicate assignments
    and aborts, because dedup at this stage prevents silent corruption
    downstream.
    """
    with open(path, 'r', encoding='utf-8') as f:
        assignment = json.load(f)

    cat_to_token = {}
    n_tokens = 0
    for key, val in assignment.items():
        if key.startswith('_'):
            continue
        if not (key.startswith('token_') and key.endswith('_categories')):
            continue
        try:
            token_1based = int(key.split('_')[1])
        except (ValueError, IndexError):
            continue
        n_tokens = max(n_tokens, token_1based)
        if not isinstance(val, list):
            print(f'[error] {key} is not a list in {path}', file=sys.stderr)
            sys.exit(2)
        for cat in val:
            if cat in cat_to_token:
                print(f'[error] category {cat!r} assigned to multiple tokens '
                      f'in {path}', file=sys.stderr)
                sys.exit(2)
            cat_to_token[cat] = token_1based - 1  # 1-based -> 0-based
    if not cat_to_token:
        print(f'[error] no token_N_categories fields found in {path}',
              file=sys.stderr)
        sys.exit(2)
    print(f'[info] assignment loaded: {len(cat_to_token)} categories -> '
          f'{n_tokens} tokens', file=sys.stderr)
    return cat_to_token, n_tokens


def load_pool_cids(memmap_dir):
    """Load cids.npy from the memmap dir. Returns int64 ndarray, sorted
    ascending (that is the memmap layout invariant)."""
    cids_path = os.path.join(memmap_dir, 'cids.npy')
    if not os.path.isfile(cids_path):
        print(f'[error] cids.npy not found at {cids_path}', file=sys.stderr)
        sys.exit(2)
    cids = np.load(cids_path)
    cids = np.ascontiguousarray(cids, dtype=np.int64)
    print(f'[info] memmap pool loaded: {len(cids):,} cids from {cids_path}',
          file=sys.stderr)
    return cids


def build_csv_category_map(csv_path):
    """Read the CSV once and produce dict: doc_id_str -> first_level_category.

    CSV rows are quoted (see the header comment in cids_union_subject.csv);
    we use csv.reader, not naive split, to handle that safely. Empty
    first-level categories are dropped at this point -- the caller
    doesn't need to distinguish empty-first from missing-row.
    """
    csv_cat = {}
    empty_first = 0
    duplicate_doc_ids = 0
    total_rows = 0
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_doc = header.index('doc_id')
            i_first = header.index('union_subject_result_first')
        except ValueError as e:
            print(f'[error] unexpected CSV header: {header}. {e}',
                  file=sys.stderr)
            sys.exit(2)

        for row in reader:
            total_rows += 1
            if len(row) <= max(i_doc, i_first):
                empty_first += 1
                continue
            doc_id = row[i_doc].strip()
            first = row[i_first].strip()
            if not first:
                empty_first += 1
                continue
            if doc_id in csv_cat:
                duplicate_doc_ids += 1
                # First one wins; skip further occurrences.
                continue
            csv_cat[doc_id] = first
    print(f'[info] CSV parsed: {total_rows:,} rows -> {len(csv_cat):,} '
          f'labeled doc_ids ({empty_first:,} empty-first, '
          f'{duplicate_doc_ids:,} duplicate)', file=sys.stderr)
    return csv_cat, {'total_rows': total_rows, 'empty_first': empty_first,
                     'duplicate_doc_ids': duplicate_doc_ids}


def build_cid_to_token(pool_cids, csv_cat, cat_to_token):
    """Produce the [num_items] int8 array. Never returns -- either
    writes the array or aborts on an unexpected category.

    Categories that appear in the CSV but not in cat_to_token are
    treated as an error: it means the assignment JSON is incomplete
    with respect to this data. This is stricter than dataloader's
    silent skip; here we're preprocessing once, so surfacing the gap
    is much better than silently masking items to unlabeled.
    """
    num_items = len(pool_cids)
    out = np.full(num_items, TOKEN_UNLABELED, dtype=np.int8)
    stats = Counter()
    unknown_categories = Counter()  # category name -> #items

    for i, cid_int in enumerate(pool_cids):
        cid_str = str(int(cid_int))
        cat = csv_cat.get(cid_str, None)
        if cat is None:
            stats['not_in_csv'] += 1
            continue
        token = cat_to_token.get(cat, None)
        if token is None:
            unknown_categories[cat] += 1
            stats['unknown_category'] += 1
            continue
        out[i] = token
        stats[f'token_{token}'] += 1

    if unknown_categories:
        print('[error] CSV contains categories not assigned in JSON:',
              file=sys.stderr)
        for cat, cnt in unknown_categories.most_common():
            print(f'         - {cat} ({cnt:,} items)', file=sys.stderr)
        print(f'[error] {sum(unknown_categories.values()):,} pool items would '
              f'have no token assignment. Fix config/category_to_token.json '
              f'and re-run.', file=sys.stderr)
        sys.exit(3)

    return out, stats


def main():
    args = parse_args()

    # -------- load inputs --------
    cat_to_token, n_tokens = load_assignment(args.assignment)
    pool_cids = load_pool_cids(args.memmap_dir)
    csv_cat, csv_stats = build_csv_category_map(args.csv)

    # -------- build lookup array --------
    cid_to_token, stats = build_cid_to_token(pool_cids, csv_cat, cat_to_token)

    # -------- report --------
    num_items = len(pool_cids)
    print()
    print('=' * 72)
    print('cid_to_token summary')
    print('=' * 72)
    print(f'  memmap pool size          : {num_items:,}')
    print(f'  csv labeled doc_ids       : {len(csv_cat):,}')
    print(f'  csv empty-first rows      : {csv_stats["empty_first"]:,}')
    print()
    for t in range(n_tokens):
        cnt = stats.get(f'token_{t}', 0)
        share = cnt / max(1, num_items)
        print(f'  token_{t + 1}                  : {cnt:>10,} ({share:>6.2%})')
    print(f'  unlabeled (in pool, empty first in CSV or missing from CSV):')
    print(f'    -> "not in csv"         : {stats.get("not_in_csv", 0):>10,}')
    total_unlabeled = int(np.sum(cid_to_token == TOKEN_UNLABELED))
    print(f'    -> total unlabeled      : {total_unlabeled:>10,} '
          f'({total_unlabeled / max(1, num_items):.2%})')
    print()

    # -------- write --------
    if args.dry_run:
        print(f'[info] --dry-run set; not writing output.')
        return

    out_path = args.output or os.path.join(args.memmap_dir, 'cid_to_token.npy')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Save as .npy so downstream memmap-load is one call. shape==[num_items],
    # dtype=int8 (values fit in [-1, 127], plenty of headroom for future
    # >3-token setups).
    np.save(out_path, cid_to_token)
    on_disk = os.path.getsize(out_path)
    print(f'[info] wrote {out_path} ({on_disk:,} bytes, dtype={cid_to_token.dtype}, '
          f'shape={cid_to_token.shape})')

    # Quick round-trip check.
    reload = np.load(out_path, mmap_mode='r')
    assert reload.shape == cid_to_token.shape and reload.dtype == cid_to_token.dtype
    assert np.array_equal(np.asarray(reload), cid_to_token)
    print(f'[info] round-trip verify OK')


if __name__ == '__main__':
    main()
