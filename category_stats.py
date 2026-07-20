"""Analyze the category distribution of the 64-d item pool.

Purpose
-------
We plan to move from kmeans-based latent interest tokens to explicit
category-to-token assignment (e.g. token_1 = {娱乐, 电影, ...},
token_2 = {生活, ...}, token_3 = {搞笑, ...}). Before choosing which
categories go to which token, we need to see the actual distribution:

  * How many items per first-level (一级) category?
  * How many first-level categories are there? (this is the pool of
    "atoms" we'll bucket into 3 token groups)
  * How many items are unlabeled (empty union_subject_result_first)?
    Per the plan, unlabeled items will be dropped.
  * How many items in the CSV cannot be matched to the 64-d memmap pool?
    (sanity check that doc_id == cid semantically)
  * Second-level distribution inside the top-N first-levels, so we can
    decide whether a large first-level (e.g. 娱乐) needs to be split
    across tokens.

Two modes
---------
1. Distribution mode (no --assignment): the default. Prints category
   counts, greedy 3-bucket suggestion, etc.
2. Verification mode (--assignment <json>): reads a proposed
   category_to_token.json (see config/category_to_token.json for the
   schema) and cross-checks:
     * every one of the 55 first-level categories is assigned to
       exactly one token (no missing, no duplicate);
     * declared per-token counts in `_totals.distribution` agree with
       the counts derived from the CSV;
     * assignment covers every category the CSV actually contains
       (surfaces any surprise category the CSV has that the JSON
       forgot to route somewhere).

Usage
-----
  # distribution mode
  python category_stats.py \\
      --csv <path>/cids_union_subject.csv \\
      --memmap-dir /group/40094/jingweidong/user_sequential_feature_recall/qbfeed_action_flow/preprocessed64d \\
      --top-secondary 5

  # verification mode
  python category_stats.py \\
      --csv <path>/cids_union_subject.csv \\
      --assignment config/category_to_token.json

The CSV format assumed:
    "doc_id","rowkey","union_subject_result_first","union_subject_result_second"
    "3665939421521259","00068313847924ah","网红达人","网红达人_原创短剧"
    ...
Empty categories are represented as empty strings ("").
Values are quoted so we use csv.reader (not naive split).
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True,
                   help='Path to cids_union_subject.csv')
    p.add_argument('--memmap-dir', default=None,
                   help='Optional. If set, cross-check CSV doc_ids against '
                        'the 64-d memmap pool (cids.npy) and report the '
                        'coverage rate.')
    p.add_argument('--top-secondary', type=int, default=5,
                   help='For each top-K first-level category, print the '
                        'top-N second-level subcategories underneath.')
    p.add_argument('--out-json', default=None,
                   help='Optional path to dump the full stats as JSON, '
                        'for downstream token-assignment scripts.')
    p.add_argument('--assignment', default=None,
                   help='Path to a category_to_token.json to verify against '
                        'the actual CSV distribution. Loads the token_N_categories '
                        'lists, cross-checks completeness and per-token counts. '
                        'When set, distribution-mode output is still printed.')
    return p.parse_args()


def load_memmap_cids(memmap_dir):
    """Load cids.npy from a memmap dir; return a Python set of str cids
    for fast membership testing against CSV doc_ids."""
    import numpy as np
    cids_path = os.path.join(memmap_dir, 'cids.npy')
    if not os.path.isfile(cids_path):
        print(f'[warn] cids.npy not found at {cids_path}, skipping memmap check',
              file=sys.stderr)
        return None
    cids = np.load(cids_path)
    # cids in memmap are typically int64; jsonl / CSV store them as str.
    # Convert to str set for uniform comparison.
    cids_str = {str(int(c)) if isinstance(c, (int, float)) or hasattr(c, 'item') else str(c)
                for c in cids}
    print(f'[info] memmap pool: {len(cids_str):,} cids loaded from {cids_path}',
          file=sys.stderr)
    return cids_str


def load_assignment(path):
    """Load a category_to_token.json and return the parsed structure PLUS
    a flat {category -> token_idx (1-based)} dict for downstream use.

    Ignores '_'-prefixed metadata fields. Detects any category being
    assigned to more than one token (returns them in `duplicates`).
    """
    with open(path, 'r', encoding='utf-8') as f:
        assignment = json.load(f)

    cat_to_token = {}
    duplicates = []  # (category, [token_ids])
    token_cats = {}  # 1-based idx -> [categories]
    for key, val in assignment.items():
        if key.startswith('_'):
            continue
        if not (key.startswith('token_') and key.endswith('_categories')):
            continue
        # 'token_1_categories' -> 1
        try:
            token_id = int(key.split('_')[1])
        except (ValueError, IndexError):
            continue
        if not isinstance(val, list):
            continue
        token_cats[token_id] = list(val)
        for cat in val:
            if cat in cat_to_token and cat_to_token[cat] != token_id:
                duplicates.append((cat, sorted({cat_to_token[cat], token_id})))
            cat_to_token[cat] = token_id

    return assignment, cat_to_token, token_cats, duplicates


def main():
    args = parse_args()

    memmap_cids = None
    if args.memmap_dir:
        memmap_cids = load_memmap_cids(args.memmap_dir)

    # -------- pass 1: read CSV, count everything --------
    first_counter = Counter()             # first-level -> count
    second_counter = Counter()            # second-level -> count
    first_to_secondary = defaultdict(Counter)  # first -> (second -> count)
    total_rows = 0
    empty_first = 0                       # rows with empty first-level
    empty_second_only = 0                 # rows with first but empty second
    not_in_memmap = 0                     # CSV doc_ids not in memmap pool
    in_memmap_and_labeled = 0             # good rows (kept for training)
    duplicate_doc_ids = 0
    seen_doc_ids = set()

    with open(args.csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_doc = header.index('doc_id')
            i_first = header.index('union_subject_result_first')
            i_second = header.index('union_subject_result_second')
        except ValueError as e:
            print(f'[error] unexpected CSV header: {header}. {e}', file=sys.stderr)
            sys.exit(1)

        for row in reader:
            total_rows += 1
            if len(row) <= max(i_doc, i_first, i_second):
                # malformed row, count as unlabeled
                empty_first += 1
                continue

            doc_id = row[i_doc].strip()
            first = row[i_first].strip()
            second = row[i_second].strip()

            # duplicate detection
            if doc_id in seen_doc_ids:
                duplicate_doc_ids += 1
            else:
                seen_doc_ids.add(doc_id)

            # memmap coverage
            in_memmap = True
            if memmap_cids is not None and doc_id not in memmap_cids:
                not_in_memmap += 1
                in_memmap = False

            # label buckets
            if not first:
                empty_first += 1
                continue
            if not second:
                empty_second_only += 1

            first_counter[first] += 1
            if second:
                second_counter[second] += 1
                first_to_secondary[first][second] += 1

            if in_memmap and first:
                in_memmap_and_labeled += 1

    # -------- report --------
    print('=' * 72)
    print('CSV summary')
    print('=' * 72)
    print(f'  total rows            : {total_rows:,}')
    print(f'  unique doc_ids        : {len(seen_doc_ids):,}')
    print(f'  duplicate doc_ids     : {duplicate_doc_ids:,}')
    print(f'  empty first-level     : {empty_first:,} ({empty_first / max(1, total_rows):.2%}) '
          f'-- these will be dropped from training')
    print(f'  first-level ok, empty second : {empty_second_only:,} '
          f'({empty_second_only / max(1, total_rows):.2%})')
    if memmap_cids is not None:
        print(f'  memmap pool size      : {len(memmap_cids):,}')
        print(f'  CSV rows not in memmap: {not_in_memmap:,} '
              f'({not_in_memmap / max(1, total_rows):.2%})')
        print(f'  usable for training   : {in_memmap_and_labeled:,} '
              f'(labeled AND in memmap)')

    # -------- first-level distribution --------
    print()
    print('=' * 72)
    print(f'First-level (union_subject_result_first) distribution')
    print(f'  {len(first_counter)} distinct first-level categories total')
    print('=' * 72)
    print(f'{"rank":>4} {"count":>10} {"pct":>7} {"cum_pct":>8}  category')
    print('-' * 72)
    labeled_total = sum(first_counter.values())
    cum = 0
    for rank, (name, cnt) in enumerate(first_counter.most_common(), 1):
        pct = cnt / labeled_total
        cum += pct
        print(f'{rank:>4} {cnt:>10,} {pct:>6.2%} {cum:>7.2%}   {name}')

    # -------- top-K first x top-N second --------
    K = min(15, len(first_counter))  # dive into the top-15 first-levels
    N = args.top_secondary
    print()
    print('=' * 72)
    print(f'Top-{N} second-level under the top-{K} first-level categories')
    print('=' * 72)
    for first_name, first_cnt in first_counter.most_common(K):
        print(f'\n  [{first_name}]  ({first_cnt:,} items, '
              f'{first_cnt / labeled_total:.2%})')
        sub = first_to_secondary[first_name]
        for sub_name, sub_cnt in sub.most_common(N):
            print(f'      {sub_cnt:>8,} ({sub_cnt / first_cnt:>6.2%})  {sub_name}')

    # -------- guidance for token assignment --------
    print()
    print('=' * 72)
    print('Token-assignment guidance (target: 3 balanced buckets)')
    print('=' * 72)
    # Balanced target: split labeled_total roughly into 3 equal parts.
    target_per_token = labeled_total / 3.0
    print(f'  labeled item count      : {labeled_total:,}')
    print(f'  target per token (equal): {int(target_per_token):,}')
    print()
    print(f'  Greedy assignment (bin-packing by descending count):')
    print(f'    Iterate first-levels from largest to smallest, drop each into')
    print(f'    the currently-smallest token bucket. Not optimal in general')
    print(f'    but produces a decent balance for skewed head distributions.')
    print()
    buckets = [[] for _ in range(3)]
    bucket_sums = [0, 0, 0]
    for name, cnt in first_counter.most_common():
        target = min(range(3), key=lambda i: bucket_sums[i])
        buckets[target].append((name, cnt))
        bucket_sums[target] += cnt

    for i in range(3):
        share = bucket_sums[i] / labeled_total
        member_preview = ', '.join(name for name, _ in buckets[i][:6])
        if len(buckets[i]) > 6:
            member_preview += f', ... (+{len(buckets[i]) - 6} more)'
        print(f'    token_{i + 1}: {bucket_sums[i]:>10,} items ({share:.2%})')
        print(f'             members ({len(buckets[i])} categories): {member_preview}')
    print()
    print('  NOTE: This is only a starting point. You may want to override')
    print('  based on category SEMANTICS (e.g. put "娱乐"+"电影"+"综艺"+"网红达人"')
    print('  in the same "entertainment" token even if that makes the token')
    print('  imbalanced). Post-run analysis of the current kmeans model may')
    print('  also inform which categories naturally co-cluster.')

    # -------- assignment verification (only if --assignment) --------
    verification_had_errors = False
    if args.assignment:
        print()
        print('=' * 72)
        print(f'Assignment verification against {args.assignment}')
        print('=' * 72)
        try:
            assignment, cat_to_token, token_cats, duplicates = load_assignment(
                args.assignment
            )
        except Exception as e:
            print(f'[error] failed to load assignment JSON: {e}')
            sys.exit(2)

        n_tokens = len(token_cats)
        print(f'  tokens declared         : {n_tokens} '
              f'({", ".join(f"token_{i}" for i in sorted(token_cats))})')
        print(f'  categories declared     : {len(cat_to_token)}')

        csv_cats = set(first_counter.keys())
        json_cats = set(cat_to_token.keys())

        # ---- structural checks ----
        # (a) duplicates within JSON
        if duplicates:
            verification_had_errors = True
            print()
            print(f'  [FAIL] {len(duplicates)} category assigned to multiple tokens:')
            for cat, tids in duplicates:
                print(f'         - {cat} -> tokens {tids}')

        # (b) missing from JSON but present in CSV
        missing = sorted(csv_cats - json_cats,
                         key=lambda c: -first_counter[c])
        if missing:
            verification_had_errors = True
            print()
            print(f'  [FAIL] {len(missing)} categories present in CSV but NOT '
                  f'assigned in JSON. These items would fall through the '
                  f'category_to_token lookup at training time:')
            for cat in missing:
                print(f'         - {cat} ({first_counter[cat]:,} items)')

        # (c) extra in JSON but not in CSV
        extra = sorted(json_cats - csv_cats)
        if extra:
            # Not fatal; the JSON may include categories that will appear
            # in future CSVs. Warn but do not fail.
            print()
            print(f'  [WARN] {len(extra)} categories declared in JSON but '
                  f'NOT observed in CSV (harmless if they show up later):')
            for cat in extra:
                print(f'         - {cat}')

        # ---- per-token count verification ----
        print()
        print(f'  Per-token counts (JSON declaration vs CSV-derived):')
        # Header
        print(f'    {"token":>7}  {"declared":>12}  {"actual":>12}  '
              f'{"delta":>10}  {"share_actual":>12}  n_cat')
        totals_block = assignment.get('_totals', {})
        distribution_block = totals_block.get('distribution', {})

        for token_id in sorted(token_cats):
            token_label = f'token_{token_id}'
            declared = distribution_block.get(token_label, {}).get('items', None)
            actual = sum(first_counter.get(cat, 0) for cat in token_cats[token_id])
            n_cat = len(token_cats[token_id])
            share_actual = actual / max(1, labeled_total)
            if declared is None:
                print(f'    {token_label:>7}  {"(none)":>12}  {actual:>12,}  '
                      f'{"n/a":>10}  {share_actual:>12.2%}  {n_cat}')
            else:
                delta = actual - declared
                print(f'    {token_label:>7}  {declared:>12,}  {actual:>12,}  '
                      f'{delta:>+10,}  {share_actual:>12.2%}  {n_cat}')
                if delta != 0:
                    verification_had_errors = True

        # ---- overall coverage check ----
        actual_covered = sum(first_counter.get(cat, 0)
                             for cat in json_cats & csv_cats)
        print()
        print(f'  Coverage: {actual_covered:,} / {labeled_total:,} = '
              f'{actual_covered / max(1, labeled_total):.4%} of labeled items '
              f'are routed to a token.')
        if actual_covered != labeled_total:
            verification_had_errors = True
            print(f'  [FAIL] {labeled_total - actual_covered:,} labeled items '
                  f'have no token assignment.')

        # ---- final verdict ----
        print()
        if verification_had_errors:
            print('  [FAIL] Assignment has issues (see above). Fix before use.')
        else:
            print('  [OK]   Assignment is complete, consistent, and covers '
                  '100% of labeled items.')

    # -------- optional JSON dump --------
    if args.out_json:
        payload = {
            'total_rows': total_rows,
            'unique_doc_ids': len(seen_doc_ids),
            'duplicate_doc_ids': duplicate_doc_ids,
            'empty_first': empty_first,
            'empty_second_only': empty_second_only,
            'not_in_memmap': not_in_memmap if memmap_cids is not None else None,
            'labeled_total': labeled_total,
            'first_level_counts': dict(first_counter),
            'second_level_counts_top1000': dict(second_counter.most_common(1000)),
            'greedy_balanced_buckets': [
                {'token': i + 1, 'items': bucket_sums[i], 'share': bucket_sums[i] / labeled_total,
                 'categories': [name for name, _ in buckets[i]]}
                for i in range(3)
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'\n[info] full stats dumped to {args.out_json}')

    # Non-zero exit code lets CI / scripts detect verification failure.
    if args.assignment and verification_had_errors:
        sys.exit(3)


if __name__ == '__main__':
    main()


def load_memmap_cids(memmap_dir):
    """Load cids.npy from a memmap dir; return a Python set of str cids
    for fast membership testing against CSV doc_ids."""
    import numpy as np
    cids_path = os.path.join(memmap_dir, 'cids.npy')
    if not os.path.isfile(cids_path):
        print(f'[warn] cids.npy not found at {cids_path}, skipping memmap check',
              file=sys.stderr)
        return None
    cids = np.load(cids_path)
    # cids in memmap are typically int64; jsonl / CSV store them as str.
    # Convert to str set for uniform comparison.
    cids_str = {str(int(c)) if isinstance(c, (int, float)) or hasattr(c, 'item') else str(c)
                for c in cids}
    print(f'[info] memmap pool: {len(cids_str):,} cids loaded from {cids_path}',
          file=sys.stderr)
    return cids_str


def main():
    args = parse_args()

    memmap_cids = None
    if args.memmap_dir:
        memmap_cids = load_memmap_cids(args.memmap_dir)

    # -------- pass 1: read CSV, count everything --------
    first_counter = Counter()             # first-level -> count
    second_counter = Counter()            # second-level -> count
    first_to_secondary = defaultdict(Counter)  # first -> (second -> count)
    total_rows = 0
    empty_first = 0                       # rows with empty first-level
    empty_second_only = 0                 # rows with first but empty second
    not_in_memmap = 0                     # CSV doc_ids not in memmap pool
    in_memmap_and_labeled = 0             # good rows (kept for training)
    duplicate_doc_ids = 0
    seen_doc_ids = set()

    with open(args.csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            i_doc = header.index('doc_id')
            i_first = header.index('union_subject_result_first')
            i_second = header.index('union_subject_result_second')
        except ValueError as e:
            print(f'[error] unexpected CSV header: {header}. {e}', file=sys.stderr)
            sys.exit(1)

        for row in reader:
            total_rows += 1
            if len(row) <= max(i_doc, i_first, i_second):
                # malformed row, count as unlabeled
                empty_first += 1
                continue

            doc_id = row[i_doc].strip()
            first = row[i_first].strip()
            second = row[i_second].strip()

            # duplicate detection
            if doc_id in seen_doc_ids:
                duplicate_doc_ids += 1
            else:
                seen_doc_ids.add(doc_id)

            # memmap coverage
            in_memmap = True
            if memmap_cids is not None and doc_id not in memmap_cids:
                not_in_memmap += 1
                in_memmap = False

            # label buckets
            if not first:
                empty_first += 1
                continue
            if not second:
                empty_second_only += 1

            first_counter[first] += 1
            if second:
                second_counter[second] += 1
                first_to_secondary[first][second] += 1

            if in_memmap and first:
                in_memmap_and_labeled += 1

    # -------- report --------
    print('=' * 72)
    print('CSV summary')
    print('=' * 72)
    print(f'  total rows            : {total_rows:,}')
    print(f'  unique doc_ids        : {len(seen_doc_ids):,}')
    print(f'  duplicate doc_ids     : {duplicate_doc_ids:,}')
    print(f'  empty first-level     : {empty_first:,} ({empty_first / max(1, total_rows):.2%}) '
          f'-- these will be dropped from training')
    print(f'  first-level ok, empty second : {empty_second_only:,} '
          f'({empty_second_only / max(1, total_rows):.2%})')
    if memmap_cids is not None:
        print(f'  memmap pool size      : {len(memmap_cids):,}')
        print(f'  CSV rows not in memmap: {not_in_memmap:,} '
              f'({not_in_memmap / max(1, total_rows):.2%})')
        print(f'  usable for training   : {in_memmap_and_labeled:,} '
              f'(labeled AND in memmap)')

    # -------- first-level distribution --------
    print()
    print('=' * 72)
    print(f'First-level (union_subject_result_first) distribution')
    print(f'  {len(first_counter)} distinct first-level categories total')
    print('=' * 72)
    print(f'{"rank":>4} {"count":>10} {"pct":>7} {"cum_pct":>8}  category')
    print('-' * 72)
    labeled_total = sum(first_counter.values())
    cum = 0
    for rank, (name, cnt) in enumerate(first_counter.most_common(), 1):
        pct = cnt / labeled_total
        cum += pct
        print(f'{rank:>4} {cnt:>10,} {pct:>6.2%} {cum:>7.2%}   {name}')

    # -------- top-K first x top-N second --------
    K = min(15, len(first_counter))  # dive into the top-15 first-levels
    N = args.top_secondary
    print()
    print('=' * 72)
    print(f'Top-{N} second-level under the top-{K} first-level categories')
    print('=' * 72)
    for first_name, first_cnt in first_counter.most_common(K):
        print(f'\n  [{first_name}]  ({first_cnt:,} items, '
              f'{first_cnt / labeled_total:.2%})')
        sub = first_to_secondary[first_name]
        for sub_name, sub_cnt in sub.most_common(N):
            print(f'      {sub_cnt:>8,} ({sub_cnt / first_cnt:>6.2%})  {sub_name}')

    # -------- guidance for token assignment --------
    print()
    print('=' * 72)
    print('Token-assignment guidance (target: 3 balanced buckets)')
    print('=' * 72)
    # Balanced target: split labeled_total roughly into 3 equal parts.
    target_per_token = labeled_total / 3.0
    print(f'  labeled item count      : {labeled_total:,}')
    print(f'  target per token (equal): {int(target_per_token):,}')
    print()
    print(f'  Greedy assignment (bin-packing by descending count):')
    print(f'    Iterate first-levels from largest to smallest, drop each into')
    print(f'    the currently-smallest token bucket. Not optimal in general')
    print(f'    but produces a decent balance for skewed head distributions.')
    print()
    buckets = [[] for _ in range(3)]
    bucket_sums = [0, 0, 0]
    for name, cnt in first_counter.most_common():
        target = min(range(3), key=lambda i: bucket_sums[i])
        buckets[target].append((name, cnt))
        bucket_sums[target] += cnt

    for i in range(3):
        share = bucket_sums[i] / labeled_total
        member_preview = ', '.join(name for name, _ in buckets[i][:6])
        if len(buckets[i]) > 6:
            member_preview += f', ... (+{len(buckets[i]) - 6} more)'
        print(f'    token_{i + 1}: {bucket_sums[i]:>10,} items ({share:.2%})')
        print(f'             members ({len(buckets[i])} categories): {member_preview}')
    print()
    print('  NOTE: This is only a starting point. You may want to override')
    print('  based on category SEMANTICS (e.g. put "娱乐"+"电影"+"综艺"+"网红达人"')
    print('  in the same "entertainment" token even if that makes the token')
    print('  imbalanced). Post-run analysis of the current kmeans model may')
    print('  also inform which categories naturally co-cluster.')

    # -------- optional JSON dump --------
    if args.out_json:
        payload = {
            'total_rows': total_rows,
            'unique_doc_ids': len(seen_doc_ids),
            'duplicate_doc_ids': duplicate_doc_ids,
            'empty_first': empty_first,
            'empty_second_only': empty_second_only,
            'not_in_memmap': not_in_memmap if memmap_cids is not None else None,
            'labeled_total': labeled_total,
            'first_level_counts': dict(first_counter),
            'second_level_counts_top1000': dict(second_counter.most_common(1000)),
            'greedy_balanced_buckets': [
                {'token': i + 1, 'items': bucket_sums[i], 'share': bucket_sums[i] / labeled_total,
                 'categories': [name for name, _ in buckets[i]]}
                for i in range(3)
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'\n[info] full stats dumped to {args.out_json}')


if __name__ == '__main__':
    main()
