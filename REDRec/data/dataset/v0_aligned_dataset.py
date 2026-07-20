"""V0-aligned dataset for the official REDRec framework.

This module re-implements the V0Simple data protocol on top of the REDRec
batch-dict format that ``model.forward_precomputed_embedding`` consumes.

Key alignment points (vs. V0Simple ``v0_simple/dataset.py``):

  * Reads the V0 user_last/*.jsonl layout, where ``history`` items carry one
    of three labels:
        - ``seq``     : input behavior sequence (kept in original order, NOT
                        sorted by ts -- many seq events have ts=0).
        - ``pos``     : ground-truth target candidates.
        - ``hardneg`` : per-user impression-no-click negatives. Currently
                        DISABLED to match V0's ablation (rely on in-batch +
                        global random negs only). Wired up so we can flip
                        it on later for a fair hardneg comparison.
  * Embedding pool uses the V0 memmap layout produced by
    ``RedSeqRecV0Simple/tools/preprocess_embedding.py`` (a directory with
    ``embeddings.bin`` / ``cids.npy`` / ``meta.json``). All DataLoader
    workers and DDP ranks share a single OS page cache; cid lookup is a
    sorted-int64 ``np.searchsorted`` (O(log N)). The legacy dict-in-npy
    layout is also accepted for back-compat.
  * Training target strategy is ``random`` (V0 default): pick exactly one
    pos as the target each step, the seq items remain the input. This gives
    epoch-wise per-user augmentation (~7.6 effective epochs per user-pass).
  * Online shuffle buffer (per worker, V0 reservoir-style swap-and-emit) is
    supported. The V0 train.shuffled.jsonl is already globally shuffled, so
    a small buffer (1024) is plenty.
  * Yields the *exact* batch-dict schema expected by
    ``REDRec.model.redrec.RedRec.forward_precomputed_embedding``::

        precomputed_input_embeds   [B, L, D]
        precomputed_attention_mask [B, L]
        precomputed_target_embeds  [B, 1, D]   # query_nums=1
        precomputed_target_mask    [B, 1]
        precomputed_neg_embeds     [NEG, D]    # per-rank random neg pool

    so we do NOT need to touch the model code.
"""

import json
import os
import random
from logging import getLogger

import numpy as np
import torch

# Schema-tolerant adapter shared with the train-side dataset, so train and
# online eval interpret heterogeneous (flat 512-d / array-form 64-d) jsonl
# records identically. See dataset.py top-of-module docstring for the
# canonical description of both schemas and the split rule.
from .dataset import normalize_history_record

# ---------------------------------------------------------------------------
# Embedding loader (memmap layout preferred; legacy dict-in-npy supported).
# Lifted from RedSeqRecV0Simple/v0_simple/dataset.py to keep this module
# self-contained -- no dependency on the V0 codebase.
# ---------------------------------------------------------------------------

class _SortedCidIndex:
    """Dict-like cid -> row-index lookup backed by a sorted int64 array."""

    __slots__ = ('cids_sorted',)

    def __init__(self, cids_sorted: np.ndarray):
        if cids_sorted.dtype != np.int64:
            cids_sorted = cids_sorted.astype(np.int64, copy=False)
        self.cids_sorted = cids_sorted

    def __len__(self):
        return int(self.cids_sorted.size)

    @staticmethod
    def _to_int(cid):
        if isinstance(cid, (int, np.integer)):
            return int(cid)
        return int(cid)

    def __contains__(self, cid):
        try:
            x = self._to_int(cid)
        except (ValueError, TypeError):
            return False
        idx = np.searchsorted(self.cids_sorted, x)
        return idx < self.cids_sorted.size and int(self.cids_sorted[idx]) == x

    def __getitem__(self, cid):
        x = self._to_int(cid)
        idx = np.searchsorted(self.cids_sorted, x)
        if idx >= self.cids_sorted.size or int(self.cids_sorted[idx]) != x:
            raise KeyError(cid)
        return int(idx)

    def get(self, cid, default=None):
        try:
            return self[cid]
        except (KeyError, ValueError, TypeError):
            return default


class _MemmapEmbeddings:
    """Read-only ndarray-like wrapper over a numpy memmap."""

    __slots__ = ('_mm', 'shape', 'dtype', '_path')

    def __init__(self, bin_path: str, shape, dtype):
        self._path = bin_path
        self._mm = np.memmap(bin_path, dtype=dtype, mode='r', shape=tuple(shape))
        self.shape = self._mm.shape
        self.dtype = self._mm.dtype

    @property
    def ndim(self):
        return self._mm.ndim

    def __len__(self):
        return self._mm.shape[0]

    def __getitem__(self, idx):
        out = self._mm[idx]
        if isinstance(idx, (int, np.integer)):
            return np.asarray(out)
        return out

    def __iter__(self):
        for i in range(self._mm.shape[0]):
            yield np.asarray(self._mm[i])

    def as_full_array(self) -> np.ndarray:
        return np.asarray(self._mm)

    def __getstate__(self):
        return {'_path': self._path, 'shape': self.shape, 'dtype': self.dtype}

    def __setstate__(self, state):
        self._path = state['_path']
        self.shape = state['shape']
        self.dtype = np.dtype(state['dtype'])
        self._mm = np.memmap(self._path, dtype=self.dtype, mode='r', shape=tuple(self.shape))


def _is_memmap_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return all(os.path.isfile(os.path.join(path, f))
               for f in ('embeddings.bin', 'cids.npy', 'meta.json'))


def load_v0_embeddings(path: str):
    """Load item embeddings, auto-detecting memmap-dir vs legacy npy.

    Returns:
        cids:        1-D int64 numpy array (sorted ascending in memmap layout).
        embeddings:  ndarray-like with ``.shape``, ``[i]``, iteration.
                     Memmap layout returns ``_MemmapEmbeddings``;
                     legacy layout returns plain ``np.ndarray``.
        cid_index:   dict-like with ``cid in obj`` and ``obj[cid]``.
    """
    logger = getLogger()

    # ---- (A) memmap layout ----
    if _is_memmap_dir(path):
        meta_path = os.path.join(path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        n = int(meta['num_items'])
        d = int(meta['embed_dim'])
        dtype = np.dtype(meta.get('dtype', 'float32'))
        bin_path = os.path.join(path, 'embeddings.bin')
        cids_path = os.path.join(path, 'cids.npy')

        logger.info(f'[v0_aligned] load memmap embeddings: dir={path} '
                    f'num={n} dim={d} dtype={dtype}')
        cids_sorted = np.load(cids_path, mmap_mode='r')
        cids_sorted = np.ascontiguousarray(cids_sorted, dtype=np.int64)
        embeddings = _MemmapEmbeddings(bin_path, shape=(n, d), dtype=dtype)
        cid_index = _SortedCidIndex(cids_sorted)
        return cids_sorted, embeddings, cid_index

    # ---- (B) legacy dict-in-npy layout ----
    logger.info(f'[v0_aligned] load legacy npy embeddings: {path}')
    logger.warning('[v0_aligned] each worker will hold a private copy of '
                   'the embedding matrix; for large datasets prefer the memmap '
                   'layout produced by RedSeqRecV0Simple/tools/preprocess_embedding.py.')
    data = np.load(path, allow_pickle=True).item()
    cids = np.asarray(data['cid']).astype(str)
    embeddings = np.asarray(data['embedding'], dtype=np.float32)
    cid2idx = {cid: idx for idx, cid in enumerate(cids)}
    logger.info(f'[v0_aligned] loaded num={len(cids)} dim={embeddings.shape[1]}')
    return cids, embeddings, cid2idx


# ---------------------------------------------------------------------------
# IterableDataset (training)
# ---------------------------------------------------------------------------

class REDRecV0AlignedDataset(torch.utils.data.IterableDataset):
    """Streaming train dataset aligned with V0Simple, yielding REDRec batch dicts.

    Per-user logic:
      input_cids  = items with label == seq_label, kept in history order,
                    last ``max_seq_len`` items if longer.
      target_cid  = one item with label == positive_label
                    (random pick when ``target_strategy == 'random'``;
                     last when ``'last'``).
      hard_negs   = items with label in ``hard_neg_labels`` (default empty;
                    flip on later for hardneg ablation).

    Sharding mirrors V0: ``line_no % world_size == rank`` then
    ``(line_no // world_size) % num_workers == worker_id``.

    Yields a batch dict (NOT per-sample dict). DataLoader is constructed with
    ``batch_size=None`` so the framework consumes whatever this iterable
    produces verbatim.
    """

    def __init__(self, global_rank, world_size, config):
        super().__init__()
        self.config = config
        self.global_rank = int(global_rank or 0)
        self.world_size = int(world_size or 1)
        self.logger = getLogger()

        d = config.data
        # ---- paths ----
        # Prefer the V0 memmap directory; fall back to legacy npy.
        emb_path = (d.get('v0_embedding_dir', None)
                    or d.get('embedding_dir', None)
                    or d.get('precomputed_embedding_npy', None))
        if not emb_path:
            raise ValueError(
                'config.data must set one of: v0_embedding_dir / embedding_dir / '
                'precomputed_embedding_npy')
        self.train_jsonl = (d.get('v0_train_jsonl', None)
                            or d.get('train_jsonl', None)
                            or d.get('precomputed_user_history_jsonl', None))
        if not self.train_jsonl:
            raise ValueError('config.data must set v0_train_jsonl (or train_jsonl).')

        # ---- sequence / target / hardneg config ----
        self.max_seq_len = int(d.get('max_seq_len', 200))
        self.seq_label = str(d.get('seq_label', 'seq'))
        self.positive_label = str(d.get('positive_label', 'pos'))
        # Hard-neg labels: default DISABLED (V0 ablation), pass a list/string to enable.
        self.hard_neg_labels = self._normalize_label_set(d.get('hard_neg_label', []))
        self.max_hard_neg = int(d.get('max_hard_neg', 0))
        self.min_history_len = int(d.get('min_history_len', 4))
        self.sample_lastn = bool(d.get('sample_lastn', True))
        ts_str = str(d.get('train_target_strategy', 'random')).lower()
        if ts_str not in ('random', 'last', 'multi'):
            raise ValueError(f"train_target_strategy must be 'random'|'last'|'multi', got {ts_str!r}")
        self.target_strategy = ts_str
        self.loop_forever = bool(d.get('loop_forever', True))
        self.shuffle_buffer_size = max(0, int(d.get('shuffle_buffer_size', 1024)))
        self.max_lines = d.get('max_lines', None)

        # ---- multi-interest mode (V0Simple 64-d experiment alignment) ----
        # When enabled, the dataset emits all `target_tail_len` positives per
        # user as a (B, T, D) target tensor instead of a single (B, 1, D)
        # target. The model side (forward_precomputed_embedding) already
        # supports N>=query_nums via cluster_based_matching: T=10 positives
        # are kmeans-clustered to query_nums=3 centers, then 1-to-1 matched
        # to the 3 user queries via Hungarian (linear_sum_assignment).
        mi_cfg = d.get('multi_interest', None) or {}
        self.multi_interest_enabled = bool(mi_cfg.get('enabled', False))
        # If train_target_strategy='multi' is set, force-enable multi-interest
        # for backward-compat with people who only flip the strategy flag.
        if self.target_strategy == 'multi':
            self.multi_interest_enabled = True
        self.target_tail_len = int(mi_cfg.get('target_tail_len', 10))
        if self.multi_interest_enabled and self.target_tail_len <= 0:
            raise ValueError('multi_interest.target_tail_len must be > 0 when enabled.')

        # ---- training batch / negatives ----
        self.train_batch_size = int(d.get('train_batch_size', 8))
        # ``neg_samples_per_gpu`` keeps the framework's per-rank random neg pool
        # (the model's forward_precomputed_embedding all-gathers them across
        # ranks, so effective negatives = neg_samples_per_gpu * world_size +
        # in-batch pos targets). Set to 0 to disable.
        self.neg_samples_per_gpu = int(d.get('neg_samples_per_gpu', 400))
        self.stats_interval = int(d.get('precomputed_stats_interval', 10000))

        # See dataset.py::normalize_history_record. Only applies when the
        # input jsonl uses the array-form (64-d datav2) schema; flat
        # (datav1) records ignore this value entirely.
        self.pos_n_for_64d = int(d.get('precomputed_64d_pos_n', 10))

        # ---- load embeddings ----
        cids_or_arr, embeddings, cid2idx = load_v0_embeddings(emb_path)
        self.cids = cids_or_arr
        self.embeddings = embeddings
        self.cid2idx = cid2idx
        self.embed_dim = int(embeddings.shape[1])
        self.num_items = int(embeddings.shape[0])

        # ---- explicit multi-interest: category-to-token lookup ----
        # If category_assignment_path is set (points at a cid_to_token.npy
        # produced by build_cid_to_token.py), we enable EXPLICIT assignment
        # instead of the model-side kmeans+Hungarian path. Effects:
        #   * _select_window drops pos/seq items whose token is -1
        #     (unlabeled items in the CSV); the model never sees them.
        #   * _build_batch emits an extra tensor `precomputed_target_token_ids`
        #     of shape [B, T] with values in {0, 1, ..., n_tokens-1}, one
        #     token idx per target pos. The model uses these to gather the
        #     matching interest-token embedding directly, skipping the
        #     kmeans+linear_sum_assignment CPU roundtrip entirely.
        #
        # The array is memmap-opened (mode='r'): ~3 MB fits happily in the
        # OS page cache and is shared across all DataLoader workers and DDP
        # ranks on the same host.
        cat_path = d.get('category_assignment_path', None)
        if cat_path:
            if not os.path.isfile(cat_path):
                raise FileNotFoundError(
                    f'category_assignment_path is set but the file does not '
                    f'exist: {cat_path}. Generate it with build_cid_to_token.py.'
                )
            self.cid_to_token = np.load(cat_path, mmap_mode='r')
            if self.cid_to_token.shape != (self.num_items,):
                raise ValueError(
                    f'cid_to_token shape {self.cid_to_token.shape} does not '
                    f'match embedding pool ({self.num_items},). The lookup '
                    f'was produced against a different memmap; regenerate.'
                )
            self.explicit_multi_interest = True
            n_labeled = int((np.asarray(self.cid_to_token) != -1).sum())
            self.logger.info(
                f'[v0_aligned] explicit multi-interest enabled; loaded '
                f'{cat_path} shape={self.cid_to_token.shape} '
                f'labeled={n_labeled:,}/{self.num_items:,} '
                f'({n_labeled / max(1, self.num_items):.2%})'
            )
        else:
            self.cid_to_token = None
            self.explicit_multi_interest = False

        self.logger.info(
            f'[v0_aligned rank={self.global_rank}/{self.world_size}] '
            f'jsonl={self.train_jsonl} max_seq_len={self.max_seq_len} '
            f'min_hist={self.min_history_len} target_strategy={self.target_strategy} '
            f'hard_neg_labels={sorted(self.hard_neg_labels)} max_hard_neg={self.max_hard_neg} '
            f'shuffle_buf={self.shuffle_buffer_size} bs={self.train_batch_size} '
            f'neg_per_gpu={self.neg_samples_per_gpu} '
            f'embed: num={self.num_items} dim={self.embed_dim} '
            f'explicit_mi={self.explicit_multi_interest}'
        )

    @staticmethod
    def _normalize_label_set(raw):
        if raw is None:
            return set()
        if isinstance(raw, (list, tuple, set)):
            return {str(x).strip() for x in raw if str(x).strip()}
        s = str(raw).strip()
        if not s:
            return set()
        return {tok.strip() for tok in s.split(',') if tok.strip()}

    def __len__(self):
        return 1_000_000_000

    # ---- per-user sample selection (mirrors V0._select_window) ----
    def _select_window(self, history):
        """Return (input_cids, target_cid_or_list, hard_neg_cids).

        In single-target mode (default): target_cid_or_list is a `str`.
        In multi-interest mode: target_cid_or_list is a `list[str]` of length
          `target_tail_len`, taken from the LAST T positives of the user.
        Returns (None, None, None) when the user fails the gating check.

        Filtering order (matters for explicit_multi_interest):
          1. cid_index membership filter  -- items whose cid is not in the
             memmap pool are dropped. This is symmetric between explicit
             and kmeans paths: neither path can use an unrecognizable cid.
          2. min_history_len gate         -- uses the POST-index-filter but
             PRE-explicit-filter counts, so users with tail unlabeled items
             (~0.33% of pool) don't get demoted below the gate just because
             they used the wrong content. This keeps the training user
             population identical between explicit and kmeans runs, so
             recall A/B tests measure the training method, not a
             population shift.
          3. explicit unlabeled filter    -- drops unlabeled items from
             seq_items / pos_items so downstream _build_batch never has
             to emit a target_token_id of -1.
          4. target_tail_len check        -- LABELED pos count must fill T
             slots (multi-interest mode only).
        """
        seq_items, pos_items, hard_neg_items = [], [], []
        for it in history:
            cid = it.get('cid')
            if cid is None:
                continue
            cid = str(cid)
            if cid not in self.cid2idx:
                continue
            label = it.get('label')
            if label == self.seq_label:
                seq_items.append(cid)
            elif label == self.positive_label:
                pos_items.append(cid)
            elif self.hard_neg_labels and label in self.hard_neg_labels:
                hard_neg_items.append(cid)

        if not pos_items:
            return None, None, None
        # V0Simple-aligned gate: total clicks (input + GT) >= min_history_len.
        # Applied BEFORE the explicit unlabeled filter so the gate operates
        # on the same click count the kmeans path sees. Users with 25 total
        # clicks (3 of which happen to be unlabeled) still pass the >=20
        # gate here; the 3 unlabeled items are dropped only in the next
        # block. Without this ordering, explicit runs would silently
        # exclude ~1% of borderline users the kmeans path keeps, biasing
        # any train/eval A/B comparison.
        if (len(seq_items) + len(pos_items)) < self.min_history_len:
            return None, None, None
        if not seq_items:
            # Need at least 1 input item; otherwise the user_llm sees an
            # all-padded sequence which is not a useful training example.
            return None, None, None

        # Explicit-mode lazy filter (after the gate).
        # cid_to_token is memmap-opened; one indexed read per remaining
        # history item, ~O(hist_len) with negligible overhead.
        if self.explicit_multi_interest:
            seq_items = [
                c for c in seq_items
                if int(self.cid_to_token[self.cid2idx[c]]) >= 0
            ]
            pos_items = [
                c for c in pos_items
                if int(self.cid_to_token[self.cid2idx[c]]) >= 0
            ]
            # Note: the filter can now produce ``seq_items == []`` even
            # after the gate passed (unlikely: would require a user whose
            # entire pool-matched history is unlabeled). Guard again.
            if not pos_items or not seq_items:
                return None, None, None

        L = self.max_seq_len
        if len(seq_items) > L:
            input_cids = seq_items[-L:] if self.sample_lastn else seq_items[:L]
        else:
            input_cids = list(seq_items)

        if self.multi_interest_enabled:
            T = self.target_tail_len
            if len(pos_items) < T:
                # Not enough LABELED positives to fill the (B, T, D)
                # target tensor. In explicit mode this can happen when a
                # user has T total pos items but some are unlabeled and
                # were filtered above; in kmeans mode it means the user
                # simply doesn't have T pos items at all.
                return None, None, None
            target = pos_items[-T:]
        else:
            if self.target_strategy == 'random':
                target = random.choice(pos_items)
            else:
                target = pos_items[-1]

        hard_neg_cids = []
        if hard_neg_items and self.max_hard_neg > 0:
            seen = set()
            uniq = []
            target_set = set(target) if isinstance(target, list) else {target}
            for c in hard_neg_items:
                if c in target_set or c in seen:
                    continue
                seen.add(c)
                uniq.append(c)
            if len(uniq) > self.max_hard_neg:
                uniq = random.sample(uniq, self.max_hard_neg)
            hard_neg_cids = uniq

        return input_cids, target, hard_neg_cids

    # ---- batch construction (REDRec dict schema) ----
    def _build_batch(self, samples):
        B = len(samples)
        L = self.max_seq_len
        D = self.embed_dim
        # T=1 in single-target mode, T=target_tail_len in multi-interest mode.
        T = self.target_tail_len if self.multi_interest_enabled else 1

        input_embeds = np.zeros((B, L, D), dtype=np.float32)
        attention_mask = np.zeros((B, L), dtype=np.int64)
        target_embeds = np.zeros((B, T, D), dtype=np.float32)
        target_mask = np.ones((B, T), dtype=np.int64)
        # target_token_ids[b, j] = 0..Q-1, one token per target pos.
        # Initialised to -1 as a sentinel; the value only becomes a valid
        # token index in the explicit-multi-interest branch below. If any
        # downstream consumer accidentally reads target_token_ids on a
        # non-explicit batch (or on a padded slot when we start allowing
        # pos < T in the future), the -1 will fail the model-side bounds
        # check instead of silently steering every padded pos to token_0.
        target_token_ids = np.full((B, T), -1, dtype=np.int64)
        user_ids, target_cids = [], []

        for row, sample in enumerate(samples):
            user_ids.append(sample['user_id'])
            seq_cids = sample['input_cids'][-L:]
            start = L - len(seq_cids)
            for offset, cid in enumerate(seq_cids):
                input_embeds[row, start + offset] = self.embeddings[self.cid2idx[cid]]
                attention_mask[row, start + offset] = 1
            tgt = sample['target_cid']
            if isinstance(tgt, list):
                # multi-interest: T positives.
                target_cids.append(list(tgt))
                for j, cid in enumerate(tgt[:T]):
                    idx = self.cid2idx[cid]
                    target_embeds[row, j] = self.embeddings[idx]
                    if self.explicit_multi_interest:
                        # cid was already required non-unlabeled by
                        # _select_window, so this is guaranteed >= 0.
                        target_token_ids[row, j] = int(self.cid_to_token[idx])
            else:
                target_cids.append([tgt])
                idx = self.cid2idx[tgt]
                target_embeds[row, 0] = self.embeddings[idx]
                if self.explicit_multi_interest:
                    target_token_ids[row, 0] = int(self.cid_to_token[idx])

        # Per-rank random neg pool (model.all_gather across ranks downstream).
        if self.neg_samples_per_gpu > 0:
            replace = self.num_items < self.neg_samples_per_gpu
            neg_idx = np.random.randint(0, self.num_items, size=self.neg_samples_per_gpu) \
                if not replace else np.random.choice(self.num_items, size=self.neg_samples_per_gpu, replace=True)
            neg_embeds = np.asarray(self.embeddings[neg_idx], dtype=np.float32)
        else:
            neg_embeds = np.zeros((0, D), dtype=np.float32)

        batch = {
            'precomputed_input_embeds': torch.from_numpy(input_embeds),
            'precomputed_attention_mask': torch.from_numpy(attention_mask),
            'precomputed_target_embeds': torch.from_numpy(target_embeds),
            'precomputed_target_mask': torch.from_numpy(target_mask),
            'precomputed_neg_embeds': torch.from_numpy(neg_embeds),
            'user_ids': user_ids,
            'target_cids': target_cids,
        }
        if self.explicit_multi_interest:
            batch['precomputed_target_token_ids'] = torch.from_numpy(target_token_ids)
        return batch

    # ---- iteration with V0-style reservoir shuffle buffer ----
    def _generate(self, worker_id, num_workers):
        emitted = 0
        skipped = 0
        rng = random.Random(0x9E3779B9 ^ (self.global_rank * 100003 + (worker_id or 0)))

        B = self.shuffle_buffer_size
        buf = []  # list of per-sample dicts {user_id, input_cids, target_cid}
        out_batch = []  # samples queued for the next emitted batch

        def _flush_batch_if_ready():
            nonlocal out_batch, emitted
            if len(out_batch) >= self.train_batch_size:
                emitted += len(out_batch)
                if self.stats_interval and emitted % self.stats_interval == 0:
                    self.logger.info(
                        f'[v0_aligned rank={self.global_rank} worker={worker_id}] '
                        f'emitted_samples={emitted} skipped={skipped} buf={len(buf)}'
                    )
                yield_batch = self._build_batch(out_batch[:self.train_batch_size])
                out_batch = out_batch[self.train_batch_size:]
                return yield_batch
            return None

        while True:
            with open(self.train_jsonl, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f):
                    if self.max_lines is not None and line_no >= int(self.max_lines):
                        break
                    if line_no % self.world_size != self.global_rank:
                        continue
                    if num_workers and num_workers > 0:
                        if (line_no // self.world_size) % num_workers != worker_id:
                            continue

                    try:
                        rec = json.loads(line)
                        history = normalize_history_record(rec, pos_n=self.pos_n_for_64d)
                        if history is None:
                            skipped += 1
                            continue
                        input_cids, target_cid, _hn = self._select_window(history)
                        if input_cids is None:
                            skipped += 1
                            continue
                        sample = {
                            'user_id': rec.get('qimei36', ''),
                            'input_cids': input_cids,
                            'target_cid': target_cid,
                        }
                    except Exception:
                        skipped += 1
                        continue

                    # Reservoir-style shuffle buffer (per worker).
                    if B <= 0:
                        out_batch.append(sample)
                    else:
                        if len(buf) < B:
                            buf.append(sample)
                        else:
                            j = rng.randrange(B)
                            out_batch.append(buf[j])
                            buf[j] = sample

                    batch = _flush_batch_if_ready()
                    if batch is not None:
                        yield batch

            if not self.loop_forever:
                # Drain remaining buffer in random order.
                if B > 0 and buf:
                    rng.shuffle(buf)
                    out_batch.extend(buf)
                    buf.clear()
                while len(out_batch) >= self.train_batch_size:
                    emitted += self.train_batch_size
                    yield self._build_batch(out_batch[:self.train_batch_size])
                    out_batch = out_batch[self.train_batch_size:]
                if out_batch:
                    emitted += len(out_batch)
                    yield self._build_batch(out_batch)
                    out_batch = []
                return
            # loop_forever: re-open the file; keep ``buf`` populated so samples
            # mix across epoch boundaries.

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self._generate(0, 1)
        return self._generate(info.id, info.num_workers)


# ---------------------------------------------------------------------------
# Hold-out eval set builder (mirrors V0Simple ``build_eval_set``).
# Used by trainer to compute top1/top10/top100 recall against the full
# item pool every ``eval_step`` steps.
# ---------------------------------------------------------------------------

def build_v0_eval_pack(config, embeddings, cid_index, logger=None):
    """Build a PER-USER eval pack against the full item pool.

    Layout (one row == one user, NOT one (user, pos) sample):
        seq_cid_idx   : (U, L) int64 cpu  -- row index of each input cid into
                                             ``embeddings_ref``, -1 for padding.
        mask          : (U, L) uint8 cpu  -- 1 valid / 0 pad (left-padded).
        hist_lens     : (U,)   int64 cpu  -- #valid input items per user.
        pos_idx_lists : list[list[int]] (length U) -- deduped ground-truth
                                                       item rows per user.
        embeddings_ref / embed_dim / num_items -- pool meta.

    Per-user metric definition (what evaluate_v0_recall consumes):
        recall@K_u   = |TopK_u ∩ G_u| / |G_u|
        hit_rate@K_u = 1{TopK_u ∩ G_u != empty}
        final value  = mean over valid users.

    Backward-compatibility note on ``eval_target_strategy``:
        The old per-sample pack expanded each user's pos list according to
        this flag ('all' / 'last' / 'first'). In the per-user formulation
        the ground truth is intrinsically a SET, so the flag no longer has
        a meaningful effect. We keep the yaml key so existing configs still
        load; values other than 'all' just print a deprecation note.
    """
    if logger is None:
        logger = getLogger()
    d = config.data

    eval_jsonl = (d.get('v0_eval_jsonl', None) or d.get('eval_jsonl', None))
    if not eval_jsonl:
        raise ValueError('config.data must set v0_eval_jsonl (or eval_jsonl).')

    max_seq_len = int(d.get('max_seq_len', 200))
    min_history_len = int(d.get('min_history_len', 4))
    seq_label = str(d.get('seq_label', 'seq'))
    positive_label = str(d.get('positive_label', 'pos'))
    # When multi_interest is enabled, eval ground truth must be the SAME
    # tail-T positives the trainer optimizes against. We therefore prefer
    # `multi_interest.target_tail_len` over the legacy `precomputed_64d_pos_n`
    # so train target == eval GT, mirroring V0Simple's 64-d experiment.
    mi_cfg = d.get('multi_interest', None) or {}
    mi_enabled = bool(mi_cfg.get('enabled', False))
    if mi_enabled:
        pos_n_for_64d = int(mi_cfg.get('target_tail_len',
                                       d.get('precomputed_64d_pos_n', 10)))
    else:
        pos_n_for_64d = int(d.get('precomputed_64d_pos_n', 10))
    eval_users = int(d.get('eval_users', 50000))
    strat = str(d.get('eval_target_strategy', 'all')).lower()
    if strat not in ('all', 'last', 'first'):
        raise ValueError(f"eval_target_strategy must be one of all/last/first, got {strat!r}")
    if strat != 'all':
        logger.warning(
            f"[v0_eval] eval_target_strategy={strat!r} is ignored under "
            f"the per-user evaluation protocol; the entire pos set is used "
            f"as ground truth for every user."
        )

    use_full_file = eval_users <= 0
    if not use_full_file:
        with open(eval_jsonl, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        target_n = eval_users
        if target_n >= total_lines:
            stride = 1
            target_n = total_lines
        else:
            stride = max(1, total_lines // target_n)
        logger.info(
            f'[v0_eval] stride-sample plan: total_lines={total_lines} '
            f'eval_users={target_n} stride={stride}'
        )
    else:
        stride = 1
        target_n = None

    # Memory-efficient layout (avoids the original ~234 GB / rank explosion):
    #   seq_cid_idx : (U, L) int64  -- row index of each input cid in `embeddings`,
    #                                  -1 for padding slots. Materializing the
    #                                  actual (L, D) fp32 sequence is deferred to
    #                                  the eval loop (per-batch GPU lookup).
    #   mask        : (U, L) uint8  -- 1 for real positions, 0 for padding.
    #   hist_lens   : (U,)   int64  -- #valid items in the (untruncated) history,
    #                                  used by evaluate_v0_recall to bucketize
    #                                  per-user metrics by sequence length.
    #   pos_idx_lists : list[list[int]] (length U) -- ground-truth item rows
    #                                                 (deduplicated). Variable
    #                                                 length, so kept as a
    #                                                 python-list-of-lists; the
    #                                                 per-rank shard is small
    #                                                 (~6e4 users * ~10 pos)
    #                                                 so this is fine.
    #
    # For 50000 users * 200 seq_len -> seq_cid_idx ~ 80 MB, mask ~ 10 MB.
    # item_pool is NOT materialized here; we pass `embeddings` through and let
    # evaluate_v0_recall L2-normalize it on the GPU once per eval.
    seq_cid_idxs, masks, hist_lens_list, pos_idx_lists = [], [], [], []
    L = max_seq_len
    skipped = 0
    n_users_read = 0
    n_users_kept = 0

    with open(eval_jsonl, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            if line_no % stride != 0:
                continue
            if (target_n is not None) and (n_users_read >= target_n):
                break
            n_users_read += 1
            try:
                rec = json.loads(line)
                history = normalize_history_record(rec, pos_n=pos_n_for_64d)
                if history is None:
                    skipped += 1
                    continue
                seq_cids, pos_cids = [], []
                for it in history:
                    cid = it.get('cid')
                    if cid is None:
                        continue
                    cid = str(cid)
                    if cid not in cid_index:
                        continue
                    label = it.get('label')
                    if label == seq_label:
                        seq_cids.append(cid)
                    elif label == positive_label:
                        pos_cids.append(cid)

                # V0Simple-aligned gating: require
                #   total clicks (input + GT) >= min_history_len, AND input >= 1.
                # Here ``seq_cids`` is the input part (history minus the tail-T
                # positives, since normalize_history_record already split them);
                # ``pos_cids`` is the GT part. Earlier we required
                # ``len(seq_cids) >= min_history_len`` which raised the bar to
                # "total clicks >= min_history_len + T", dropping ~11% of users
                # whose input length was 10..19. V0Simple's
                # ``build_eval_set_latest`` checks ``len(kept) >= 20`` BEFORE
                # carving out the tail-T, so a user with 25 total clicks (15
                # input + 10 GT) is kept by V0 but was dropped by us. Match V0.
                if not pos_cids or not seq_cids:
                    skipped += 1
                    continue
                if (len(seq_cids) + len(pos_cids)) < min_history_len:
                    skipped += 1
                    continue
                # Track raw (pre-truncation) history length for length-bucket stats;
                # the bucketization should reflect "user activity", not the model's
                # context window. Use total clicks (input + GT) so the bucket
                # definition matches V0's "kept" length convention.
                full_hist_len = len(seq_cids) + len(pos_cids)
                if len(seq_cids) > L:
                    seq_cids = seq_cids[-L:]

                seq_idx = np.full((L,), -1, dtype=np.int64)
                mask = np.zeros((L,), dtype=np.uint8)
                start = L - len(seq_cids)
                for i, c in enumerate(seq_cids):
                    seq_idx[start + i] = int(cid_index[c])
                    mask[start + i] = 1

                # Deduplicate positives to a SET (preserve first-seen order).
                pos_idx_set = []
                seen = set()
                for c in pos_cids:
                    idx = int(cid_index[c])
                    if idx not in seen:
                        seen.add(idx)
                        pos_idx_set.append(idx)
                if not pos_idx_set:
                    skipped += 1
                    continue

                seq_cid_idxs.append(seq_idx)
                masks.append(mask)
                hist_lens_list.append(int(full_hist_len))
                pos_idx_lists.append(pos_idx_set)
                n_users_kept += 1
            except Exception:
                skipped += 1
                continue

    avg_pos = (sum(len(p) for p in pos_idx_lists) / max(1, len(pos_idx_lists)))
    avg_hist = (sum(hist_lens_list) / max(1, len(hist_lens_list)))
    logger.info(
        f'[v0_eval] users_read={n_users_read} users_kept={n_users_kept} '
        f'skipped={skipped} avg_hist_len={avg_hist:.1f} '
        f'avg_pos_per_user={avg_pos:.2f} (max_seq_len={L})'
    )
    if not seq_cid_idxs:
        return None

    seq_cid_idx_t = torch.from_numpy(np.stack(seq_cid_idxs, axis=0))   # (U, L) int64
    mask_t = torch.from_numpy(np.stack(masks, axis=0))                  # (U, L) uint8
    hist_lens_t = torch.tensor(hist_lens_list, dtype=torch.int64)       # (U,)

    n_seq_bytes = seq_cid_idx_t.numel() * seq_cid_idx_t.element_size()
    n_mask_bytes = mask_t.numel() * mask_t.element_size()
    logger.info(
        f'[v0_eval] eval pack indices built: seq_cid_idx={tuple(seq_cid_idx_t.shape)} '
        f'(~{n_seq_bytes / 1024 / 1024 / 1024:.2f} GB) '
        f'mask={tuple(mask_t.shape)} (~{n_mask_bytes / 1024 / 1024:.0f} MB) '
        f'pos_idx_lists=list[len={len(pos_idx_lists)}]'
    )

    # Item pool: keep as a reference, NOT a materialized fp32 tensor.
    # Memmap-backed embeddings share OS page cache across all 8 ranks
    # (~6.7 GB total, not 53 GB), and L2-normalization is deferred to
    # evaluate_v0_recall where it runs once on the GPU.
    return {
        'seq_cid_idx': seq_cid_idx_t,
        'mask': mask_t,
        'hist_lens': hist_lens_t,
        'pos_idx_lists': pos_idx_lists,
        'embeddings_ref': embeddings,  # the same _MemmapEmbeddings used by training
        'embed_dim': int(embeddings.shape[1]),
        'num_items': int(embeddings.shape[0]),
    }
