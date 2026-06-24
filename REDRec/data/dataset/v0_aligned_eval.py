"""V0-aligned online eval helpers (per-user formulation).

For each user U_i we compute ONE user embedding (via any of the
strategies below), retrieve the top-K items from the FULL item pool,
then compare against the user's ground-truth pos SET G_i:

    recall@K_u   = |TopK_u  intersect G_u| / |G_u|
    hit_rate@K_u = 1{TopK_u intersect G_u != empty}

The reported metric for each strategy / k is the simple mean over all
valid users (and additionally per history-length bucket).

Strategies:
  * ``mean_pool``   : zero-parameter mean over valid input positions.
  * ``last_pool``   : zero-parameter last valid position.
  * ``last8_pool``  : mean over the last 8 valid positions.
  * ``last32_pool`` : mean over the last 32 valid positions.
  * ``redrec``      : the REDRec user tower (input_embedding_projector ->
                      user_llm -> note_embedding_head, with N learnable
                      queries appended). Same forward path that
                      ``model.forward_precomputed_embedding`` uses, sans
                      the NCE loss.

All five strategies score against the SAME L2-normalized item pool so
the recall / hit_rate numbers are directly comparable. mean / last /
last8 / last32 are zero-cost baselines that do not change over training
(we evaluate them once and cache).

Distributed strategy
--------------------
Eval users are sharded round-robin across ranks. Each rank computes
its local user embeddings + per-user recall / hit_rate sums, then a
single ``all_reduce(SUM)`` combines across ranks. Final reported value
= sum(per_user_metric) / sum(valid_users).
"""

from logging import getLogger
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


# History-length buckets for per-user metric breakdown. Each tuple is
# [lo, hi) where hi=None means open-ended on the right. Ranges chosen
# to match the v0 reference protocol (powers of two, 4 -> 8 -> ... -> 128+).
HIST_LEN_BUCKETS: Tuple[Tuple[int, Optional[int], str], ...] = (
    (4, 8, '4-7'),
    (8, 16, '8-15'),
    (16, 32, '16-31'),
    (32, 64, '32-63'),
    (64, 128, '64-127'),
    (128, None, '128+'),
)


def _bucketize_hist_lens(hist_lens: torch.Tensor) -> torch.Tensor:
    """Map (U,) int64 history lengths to (U,) int64 bucket id.

    bucket id == len(HIST_LEN_BUCKETS) means "out of range" (e.g. < 4); such
    users are excluded from per-bucket reporting (they shouldn't exist
    because build_v0_eval_pack already enforces min_history_len, but we
    handle it defensively).
    """
    out = torch.full_like(hist_lens, fill_value=len(HIST_LEN_BUCKETS))
    for bid, (lo, hi, _) in enumerate(HIST_LEN_BUCKETS):
        if hi is None:
            sel = hist_lens >= lo
        else:
            sel = (hist_lens >= lo) & (hist_lens < hi)
        out[sel] = bid
    return out


@torch.no_grad()
def _mean_pool(seq_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions. (B, L, D), (B, L) -> (B, D)."""
    m = mask.unsqueeze(-1).to(seq_emb.dtype)
    s = (seq_emb * m).sum(dim=1)
    n = m.sum(dim=1).clamp_min(1.0)
    return s / n


@torch.no_grad()
def _last_pool(seq_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Last valid position. (B, L, D), (B, L) -> (B, D).

    Mask is left-padded (V0/V0-aligned convention), so the last position is
    always L-1 when there is at least one valid item.
    """
    return seq_emb[:, -1, :]


def _last_n_pool(seq_emb: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
    """Mean over the LAST n valid positions. (B, L, D), (B, L) -> (B, D).

    Mirrors the v0 baseline ``last8_pool`` / ``last32_pool`` definition: take
    the last ``n`` mask-valid positions of each user, mean-pool them. We pick
    the suffix window ``[L-n : L]`` (left-padded convention, so this slice is
    the most recent n positions; padding inside it is suppressed by ``mask``).
    Falls back to ``_last_pool`` when n <= 1.
    """
    if n <= 1:
        return _last_pool(seq_emb, mask)
    L = seq_emb.size(1)
    n = min(n, L)
    seq_tail = seq_emb[:, -n:, :]                          # (B, n, D)
    mask_tail = mask[:, -n:].unsqueeze(-1).to(seq_emb.dtype)  # (B, n, 1)
    s = (seq_tail * mask_tail).sum(dim=1)                  # (B, D)
    cnt = mask_tail.sum(dim=1).clamp_min(1.0)              # (B, 1)
    return s / cnt


@torch.no_grad()
def _redrec_user_emb(model, seq_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run the REDRec user tower forward to produce the per-user embedding.

    Mirrors the user-side path of ``model.forward_precomputed_embedding``::

        user_feats = input_embedding_projector(input_embeds)            # 512 -> 1536
        user_feats = cat([user_feats, query_embedding], dim=1)          # +N queries
        attn_mask  = cat([attn_mask, ones[:, query_nums]], dim=1)
        user_hid   = user_llm(inputs_embeds=user_feats, attn_mask=...)
        user_emb   = note_embedding_head(user_hid)                      # 1536 -> 512
        user_emb   = user_emb[:, -query_nums:]                          # take query slot
        user_emb   = L2-normalize(user_emb, dim=-1)

    For query_nums == 1 the result is (B, 1, D) -> we squeeze to (B, D).
    For query_nums > 1 we take the mean over the query slots (matches the
    ``cluster_based_matching`` degenerate case when target_num=1).

    The base RedRec module is resolved via attribute access on whatever
    wrapper (Fabric / DeepSpeed) ``model`` currently is. We rely on Fabric's
    standard attribute-forwarding for ``input_embedding_projector`` etc.;
    fall back to a hand-rolled walk through common wrapper attrs only if
    direct attribute access fails.
    """
    def _resolve_base(m):
        # Try attribute-forwarded access first (Fabric does this automatically).
        if hasattr(m, 'input_embedding_projector') and hasattr(m, 'user_llm'):
            return m
        # Common Fabric / DDP / DeepSpeed wrapper paths.
        for attr in ('module', '_forward_module', '_original_module'):
            sub = getattr(m, attr, None)
            if sub is not None and hasattr(sub, 'input_embedding_projector'):
                return sub
        return m  # last resort -- will AttributeError below if truly missing

    base = _resolve_base(model)

    proj_dtype = base.input_embedding_projector[0].weight.dtype
    inp = seq_emb.to(dtype=proj_dtype)
    am = mask.long()

    user_feats = base.input_embedding_projector(inp)
    if base.query_nums > 0:
        bsz = user_feats.shape[0]
        q = base.query(torch.arange(base.query_nums, device=user_feats.device))
        user_feats = torch.cat(
            [user_feats, q.unsqueeze(0).repeat(bsz, 1, 1)], dim=1
        )
        am = torch.cat(
            [am, torch.ones((bsz, base.query_nums), dtype=am.dtype, device=am.device)],
            dim=1,
        )

    hid = base.user_llm(inputs_embeds=user_feats, attention_mask=am).hidden_states[-1]
    emb = base.note_embedding_head(hid)
    # Lift from user_output_dim (64) to user_final_dim (512) via output_mlp.
    if hasattr(base, 'output_mlp'):
        emb = base.output_mlp(emb)
    if base.query_nums > 0:
        emb = emb[:, -base.query_nums:, :]
    else:
        emb = emb[:, -1:, :]
    # query_nums >= 1: mean over query slots (degenerates to identity for q=1).
    emb = emb.mean(dim=1)
    return emb


@torch.no_grad()
def _topk_metrics_local(
    user_emb: torch.Tensor,
    pos_sets_padded: torch.Tensor,
    pos_sets_lens: torch.Tensor,
    item_pool_T: torch.Tensor,
    bucket_ids: torch.Tensor,
    n_buckets: int,
    ks: Sequence[int] = (1, 10, 50, 100, 500),
    score_chunk: int = 1024,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Per-user top-k recall + hit_rate against the full item pool.

    Args:
        user_emb        : (U, D)         user embeddings (unnormalized OK,
                                         normalized internally).
        pos_sets_padded : (U, P_max)     int64 ground-truth item rows,
                                         padded with -1 to a common width.
        pos_sets_lens   : (U,)           int64 number of real pos per user
                                         (>= 1 for valid users).
        item_pool_T     : (D, N)         L2-normalized item pool.
        bucket_ids      : (U,)           int64 history-length bucket id;
                                         n_buckets == "out of range".
        n_buckets       : int            number of in-range buckets.
        valid_mask      : (U,)           bool, False for padding rows
                                         appended to keep ranks lock-step.

    Returns dict with float64 sum-tensors on `user_emb.device`:
        recall_sum_overall   : (len(ks),)
        hitrate_sum_overall  : (len(ks),)
        valid_count_overall  : scalar (== valid users contributing).
        recall_sum_bucket    : (n_buckets, len(ks))
        hitrate_sum_bucket   : (n_buckets, len(ks))
        valid_count_bucket   : (n_buckets,)
    All sums are NOT divided by valid count; reduction (across ranks) +
    final division happens in evaluate_v0_recall.
    """
    device = user_emb.device
    nks = len(ks)
    out = {
        'recall_sum_overall':  torch.zeros(nks, dtype=torch.float64, device=device),
        'hitrate_sum_overall': torch.zeros(nks, dtype=torch.float64, device=device),
        'valid_count_overall': torch.zeros((), dtype=torch.float64, device=device),
        'recall_sum_bucket':   torch.zeros((n_buckets, nks), dtype=torch.float64, device=device),
        'hitrate_sum_bucket':  torch.zeros((n_buckets, nks), dtype=torch.float64, device=device),
        'valid_count_bucket':  torch.zeros(n_buckets, dtype=torch.float64, device=device),
    }
    if user_emb.numel() == 0:
        return out

    user_emb = F.normalize(user_emb.float(), dim=-1)
    max_k = max(ks)
    U = user_emb.size(0)

    for s in range(0, U, score_chunk):
        e = min(U, s + score_chunk)
        u = user_emb[s:e]                              # (b, D)
        scores = u @ item_pool_T                       # (b, N)
        topk = scores.topk(max_k, dim=1).indices       # (b, max_k) int64

        pos_b = pos_sets_padded[s:e]                   # (b, P_max) int64 (-1 padded)
        pos_len_b = pos_sets_lens[s:e].to(torch.float64)  # (b,)
        bid_b = bucket_ids[s:e]                        # (b,)
        if valid_mask is not None:
            vm_b = valid_mask[s:e]
        else:
            vm_b = torch.ones(e - s, dtype=torch.bool, device=device)
        # Force non-valid rows to NOT contribute regardless of length.
        vm_b = vm_b & (pos_len_b > 0)

        # match: (b, max_k, P_max) bool. Padded entries (-1) won't match any
        # topk index because topk indices are in [0, N).
        match = topk.unsqueeze(-1).eq(pos_b.unsqueeze(1))      # (b, max_k, P_max)

        for ki, k in enumerate(ks):
            top_k = match[:, :k, :]                            # (b, k, P_max)
            # per-user, per-pos: did pos j land in top-k?
            hit_per_pos = top_k.any(dim=1)                     # (b, P_max) bool
            # recall_u = (#matched pos) / |G_u|
            n_hit = hit_per_pos.to(torch.float64).sum(dim=1)   # (b,)
            recall_u = n_hit / pos_len_b.clamp_min(1.0)        # (b,)
            hit_u = (n_hit > 0).to(torch.float64)              # (b,)

            # mask out padding/invalid users.
            valid_f = vm_b.to(torch.float64)
            recall_u = recall_u * valid_f
            hit_u = hit_u * valid_f

            out['recall_sum_overall'][ki] += recall_u.sum()
            out['hitrate_sum_overall'][ki] += hit_u.sum()

            # bucketize: scatter-add into (n_buckets,)
            for bid in range(n_buckets):
                sel = (bid_b == bid) & vm_b
                if sel.any():
                    out['recall_sum_bucket'][bid, ki] += recall_u[sel].sum()
                    out['hitrate_sum_bucket'][bid, ki] += hit_u[sel].sum()

        # valid_count counted ONCE per chunk (not per k).
        out['valid_count_overall'] += vm_b.to(torch.float64).sum()
        for bid in range(n_buckets):
            sel = (bid_b == bid) & vm_b
            if sel.any():
                out['valid_count_bucket'][bid] += sel.to(torch.float64).sum()

    return out


@torch.no_grad()
def evaluate_v0_recall(
    model,
    eval_pack: dict,
    device: torch.device,
    rank: int,
    world_size: int,
    user_batch: int = 256,
    score_chunk: int = 1024,
    ks=(1, 10, 50, 100, 500),
    eval_baselines: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate per-user recall@K and hit_rate@K against the full item pool.

    For each strategy ('redrec', 'mean_pool', 'last_pool', 'last8_pool',
    'last32_pool') we encode each user ONCE, retrieve the top-max(ks) items,
    and report:
        recall@K_u   = |TopK_u  intersect G_u| / |G_u|
        hit_rate@K_u = 1{TopK_u intersect G_u != empty}
    averaged over valid users (and additionally per history-length bucket).

    ``eval_pack`` (built by ``build_v0_eval_pack`` on each rank):
        seq_cid_idx    : (U, L) int64 cpu  -- row index into embeddings_ref,
                                              -1 for padding slots.
        mask           : (U, L) uint8 cpu  -- 1 valid / 0 pad (left-padded).
        hist_lens      : (U,)   int64 cpu  -- raw (pre-truncation) history
                                              length per user, used for
                                              bucketed reporting.
        pos_idx_lists  : list[list[int]] (length U) -- ground-truth item rows.
        embeddings_ref : ndarray-like      -- shared memmap (NOT a copy).
        embed_dim      : int
        num_items      : int

    Each rank processes ``[rank::world_size]`` slice of users; per-user
    metric SUMS are all-reduced (SUM) across ranks. Final value =
    sum(per_user_metric) / sum(valid_users).

    Returns:
        {strategy_name: {
            'top{k}_recall'    : float,
            'top{k}_hit_rate'  : float,
            'top{k}_recall_bucket'   : { bucket_label: float, ... },
            'top{k}_hit_rate_bucket' : { bucket_label: float, ... },
            ...
            '_n_users_overall' : int,
            '_n_users_bucket'  : { bucket_label: int, ... },
        }}
    """
    logger = getLogger()
    seq_cid_idx_full = eval_pack['seq_cid_idx']        # (U, L) int64 cpu
    mask_full = eval_pack['mask']                      # (U, L) uint8 cpu
    hist_lens_full = eval_pack['hist_lens']            # (U,)   int64 cpu
    pos_idx_lists = eval_pack['pos_idx_lists']         # list[list[int]] len U
    embeddings = eval_pack['embeddings_ref']           # memmap or ndarray
    embed_dim = int(eval_pack['embed_dim'])
    num_items = int(eval_pack['num_items'])

    U_total = seq_cid_idx_full.size(0)
    L = int(seq_cid_idx_full.size(1))

    # ---- pad pos_idx_lists to a (U, P_max) int64 tensor ----
    # Padding sentinel is -1, which can never collide with a valid item row.
    # P_max here is the global max (across all users in the pack); for the
    # 64-d run with pos_n_for_64d=10 + dedup it's typically <= 10, well
    # within memory. If a future run blows this up (>1000) we'd want to
    # switch to per-batch padding instead.
    pos_lens_np = np.asarray([len(p) for p in pos_idx_lists], dtype=np.int64)
    P_max = int(pos_lens_np.max()) if len(pos_lens_np) > 0 else 0
    pos_padded_np = np.full((U_total, max(1, P_max)), fill_value=-1, dtype=np.int64)
    for i, lst in enumerate(pos_idx_lists):
        if lst:
            pos_padded_np[i, :len(lst)] = np.asarray(lst, dtype=np.int64)
    pos_padded_full = torch.from_numpy(pos_padded_np)  # (U, P_max) int64 cpu
    pos_lens_full = torch.from_numpy(pos_lens_np)      # (U,)       int64 cpu

    # ---- shard users round-robin across ranks, PADDED to equal length ----
    # Padding rationale: every rank must call _redrec_user_emb the same number
    # of times so DeepSpeed's collective forwards stay in lock-step. Padded
    # rows reuse user 0 as a harmless placeholder (their hits are masked off).
    if world_size > 1:
        per_rank = (U_total + world_size - 1) // world_size  # ceil
        idx_full = torch.arange(U_total, dtype=torch.long)
        idx_rank = idx_full[rank::world_size]
        valid_mask_local = torch.ones(idx_rank.numel(), dtype=torch.bool)
        if idx_rank.numel() < per_rank:
            pad_n = per_rank - idx_rank.numel()
            idx_rank = torch.cat([idx_rank, torch.zeros(pad_n, dtype=torch.long)])
            valid_mask_local = torch.cat(
                [valid_mask_local, torch.zeros(pad_n, dtype=torch.bool)]
            )
        idx_local = idx_rank
    else:
        idx_local = torch.arange(U_total, dtype=torch.long)
        valid_mask_local = torch.ones(idx_local.numel(), dtype=torch.bool)

    # ---- materialize item_pool on GPU and L2-normalize there ----
    if rank == 0:
        logger.info(
            f'[v0_eval] uploading item_pool ({num_items} x {embed_dim}) to GPU '
            f'and normalizing in chunks ...'
        )
    item_pool_d = torch.empty((num_items, embed_dim), dtype=torch.float32, device=device)
    chunk = 200_000
    if hasattr(embeddings, 'as_full_array'):
        emb_np = embeddings.as_full_array()
    else:
        emb_np = embeddings
    for s in range(0, num_items, chunk):
        e = min(num_items, s + chunk)
        block = np.ascontiguousarray(np.asarray(emb_np[s:e], dtype=np.float32))
        block_t = torch.from_numpy(block).to(device, non_blocking=True)
        item_pool_d[s:e] = F.normalize(block_t, dim=-1)
        del block, block_t
    item_pool_T = item_pool_d.t().contiguous()         # (D, N)

    # Per-batch-of-users scoped state on device.
    pos_padded_local = pos_padded_full[idx_local].to(device)   # (U_local, P_max)
    pos_lens_local = pos_lens_full[idx_local].to(device)       # (U_local,)
    hist_lens_local = hist_lens_full[idx_local].to(device)     # (U_local,)
    bucket_ids_local = _bucketize_hist_lens(hist_lens_local).to(device)
    valid_mask_local_d = valid_mask_local.to(device)
    n_buckets = len(HIST_LEN_BUCKETS)
    bucket_labels = [b[2] for b in HIST_LEN_BUCKETS]

    results: Dict[str, Dict[str, float]] = {}

    def _reduce_to_metrics(local_acc: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """SUM-reduce sums across ranks and divide by valid counts.

        Returns a flat dict with:
            top{k}_recall, top{k}_hit_rate            : overall floats
            top{k}_recall_bucket, top{k}_hit_rate_bucket : bucket -> float
            _n_users_overall                          : int
            _n_users_bucket                           : bucket -> int
        """
        # Pack everything into a single tensor for one all_reduce.
        nks = len(ks)
        if world_size > 1:
            buf = torch.cat([
                local_acc['recall_sum_overall'].reshape(-1),         # nks
                local_acc['hitrate_sum_overall'].reshape(-1),        # nks
                local_acc['valid_count_overall'].reshape(1),         # 1
                local_acc['recall_sum_bucket'].reshape(-1),          # n_buckets * nks
                local_acc['hitrate_sum_bucket'].reshape(-1),         # n_buckets * nks
                local_acc['valid_count_bucket'].reshape(-1),         # n_buckets
            ]).to(torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            offset = 0
            recall_overall = buf[offset:offset + nks].cpu().tolist(); offset += nks
            hitrate_overall = buf[offset:offset + nks].cpu().tolist(); offset += nks
            n_overall = float(buf[offset].item()); offset += 1
            recall_bucket = buf[offset:offset + n_buckets * nks].reshape(
                n_buckets, nks).cpu().tolist(); offset += n_buckets * nks
            hitrate_bucket = buf[offset:offset + n_buckets * nks].reshape(
                n_buckets, nks).cpu().tolist(); offset += n_buckets * nks
            n_bucket = buf[offset:offset + n_buckets].cpu().tolist()
        else:
            recall_overall = local_acc['recall_sum_overall'].cpu().tolist()
            hitrate_overall = local_acc['hitrate_sum_overall'].cpu().tolist()
            n_overall = float(local_acc['valid_count_overall'].item())
            recall_bucket = local_acc['recall_sum_bucket'].cpu().tolist()
            hitrate_bucket = local_acc['hitrate_sum_bucket'].cpu().tolist()
            n_bucket = local_acc['valid_count_bucket'].cpu().tolist()

        out: Dict[str, float] = {}
        denom_overall = max(1.0, n_overall)
        for ki, k in enumerate(ks):
            out[f'top{k}_recall'] = recall_overall[ki] / denom_overall
            out[f'top{k}_hit_rate'] = hitrate_overall[ki] / denom_overall
        for ki, k in enumerate(ks):
            r_b = {}
            h_b = {}
            for bid, lab in enumerate(bucket_labels):
                d = max(1.0, n_bucket[bid])
                r_b[lab] = recall_bucket[bid][ki] / d
                h_b[lab] = hitrate_bucket[bid][ki] / d
            out[f'top{k}_recall_bucket'] = r_b
            out[f'top{k}_hit_rate_bucket'] = h_b
        out['_n_users_overall'] = int(round(n_overall))
        out['_n_users_bucket'] = {
            lab: int(round(n_bucket[bid])) for bid, lab in enumerate(bucket_labels)
        }
        return out

    # ---- raw item_pool (un-normalized) for input-side lookups ----
    if rank == 0:
        logger.info(
            f'[v0_eval] uploading raw item_pool ({num_items} x {embed_dim}) '
            f'to GPU for input-side lookups ...'
        )
    item_pool_raw_d = torch.empty(
        (num_items, embed_dim), dtype=torch.float32, device=device
    )
    for s in range(0, num_items, chunk):
        e = min(num_items, s + chunk)
        block = np.ascontiguousarray(np.asarray(emb_np[s:e], dtype=np.float32))
        item_pool_raw_d[s:e].copy_(torch.from_numpy(block).to(device, non_blocking=True))
        del block

    def _gather_seq_batch(idx_batch: torch.Tensor):
        """Look up the input sequence embeddings for a batch of users.

        idx_batch: (b,) int64 row indices into seq_cid_idx_full.
        Returns:
            seq_b  : (b, L, D) fp32 on device
            mask_b : (b, L)    int64 on device
        """
        cid_idx_b = seq_cid_idx_full[idx_batch].to(device, non_blocking=True)  # (b, L)
        mask_b = mask_full[idx_batch].to(device, non_blocking=True).long()     # (b, L)
        # -1 is the padding sentinel; clamp to 0 so embedding_lookup is valid
        # (those positions are zero-masked downstream by mask_b).
        safe_idx = cid_idx_b.clamp_min(0)
        seq_b = item_pool_raw_d[safe_idx]              # (b, L, D)
        # zero-out padded positions to avoid leaking item 0's embedding
        seq_b = seq_b * mask_b.unsqueeze(-1).to(seq_b.dtype)
        return seq_b, mask_b

    def _empty_acc():
        nks = len(ks)
        return {
            'recall_sum_overall':  torch.zeros(nks, dtype=torch.float64, device=device),
            'hitrate_sum_overall': torch.zeros(nks, dtype=torch.float64, device=device),
            'valid_count_overall': torch.zeros((), dtype=torch.float64, device=device),
            'recall_sum_bucket':   torch.zeros((n_buckets, nks), dtype=torch.float64, device=device),
            'hitrate_sum_bucket':  torch.zeros((n_buckets, nks), dtype=torch.float64, device=device),
            'valid_count_bucket':  torch.zeros(n_buckets, dtype=torch.float64, device=device),
        }

    def _add_acc(dst, src):
        for k_ in dst:
            dst[k_] = dst[k_] + src[k_]

    # ---- redrec strategy ----
    model_was_training = False
    base = model
    for attr in ('module', '_forward_module', '_original_module'):
        sub = getattr(base, attr, None)
        if sub is not None and hasattr(sub, 'input_embedding_projector'):
            base = sub
            break
    if base.training:
        model_was_training = True
    model.eval()

    redrec_acc = _empty_acc()
    n_local = idx_local.numel()
    if n_local > 0:
        for s in range(0, n_local, user_batch):
            e = min(n_local, s + user_batch)
            sl = idx_local[s:e]
            seq_b, mask_b = _gather_seq_batch(sl)
            ue = _redrec_user_emb(model, seq_b, mask_b)
            pos_b = pos_padded_local[s:e]
            plen_b = pos_lens_local[s:e]
            bid_b = bucket_ids_local[s:e]
            vm_b = valid_mask_local_d[s:e]
            chunk_acc = _topk_metrics_local(
                ue, pos_b, plen_b, item_pool_T, bid_b, n_buckets,
                ks=ks, score_chunk=score_chunk, valid_mask=vm_b,
            )
            _add_acc(redrec_acc, chunk_acc)
    results['redrec'] = _reduce_to_metrics(redrec_acc)

    # ---- zero-param baselines ----
    # Aligned with the v0 reference run:
    #   mean_pool   : mean over ALL valid positions
    #   last_pool   : the last valid position
    #   last8_pool  : mean over the last 8 valid positions
    #   last32_pool : mean over the last 32 valid positions
    # All four are zero-parameter and target-time invariant, so we only need
    # to evaluate them once (cached in trainer).
    if eval_baselines:
        mean_acc = _empty_acc()
        last_acc = _empty_acc()
        last8_acc = _empty_acc()
        last32_acc = _empty_acc()
        if n_local > 0:
            for s in range(0, n_local, user_batch):
                e = min(n_local, s + user_batch)
                sl = idx_local[s:e]
                seq_b, mask_b = _gather_seq_batch(sl)
                pos_b = pos_padded_local[s:e]
                plen_b = pos_lens_local[s:e]
                bid_b = bucket_ids_local[s:e]
                vm_b = valid_mask_local_d[s:e]

                mp = _mean_pool(seq_b, mask_b)
                _add_acc(mean_acc, _topk_metrics_local(
                    mp, pos_b, plen_b, item_pool_T, bid_b, n_buckets,
                    ks=ks, score_chunk=score_chunk, valid_mask=vm_b,
                ))

                lp = _last_pool(seq_b, mask_b)
                _add_acc(last_acc, _topk_metrics_local(
                    lp, pos_b, plen_b, item_pool_T, bid_b, n_buckets,
                    ks=ks, score_chunk=score_chunk, valid_mask=vm_b,
                ))

                lp8 = _last_n_pool(seq_b, mask_b, n=8)
                _add_acc(last8_acc, _topk_metrics_local(
                    lp8, pos_b, plen_b, item_pool_T, bid_b, n_buckets,
                    ks=ks, score_chunk=score_chunk, valid_mask=vm_b,
                ))

                lp32 = _last_n_pool(seq_b, mask_b, n=32)
                _add_acc(last32_acc, _topk_metrics_local(
                    lp32, pos_b, plen_b, item_pool_T, bid_b, n_buckets,
                    ks=ks, score_chunk=score_chunk, valid_mask=vm_b,
                ))
        results['mean_pool'] = _reduce_to_metrics(mean_acc)
        results['last_pool'] = _reduce_to_metrics(last_acc)
        results['last8_pool'] = _reduce_to_metrics(last8_acc)
        results['last32_pool'] = _reduce_to_metrics(last32_acc)

    # Free GPU buffers (~13 GB on H20 -> we don't want them lingering across
    # 1000-step gaps; the next eval can rebuild from page-cached memmap fast).
    del item_pool_T, item_pool_d, item_pool_raw_d
    torch.cuda.empty_cache()

    if model_was_training:
        model.train()

    if rank == 0:
        ks_list = list(ks)
        for name, m in results.items():
            recall_str = ' '.join(f'r@{k}={m[f"top{k}_recall"]:.4f}' for k in ks_list)
            hit_str = ' '.join(f'h@{k}={m[f"top{k}_hit_rate"]:.4f}' for k in ks_list)
            n_users = m.get('_n_users_overall', 0)
            logger.info(f'[v0_eval] {name:<11} | n={n_users} | {recall_str} | {hit_str}')

    return results


def format_recall_table(results: Dict[str, Dict[str, float]],
                        ks=(1, 10, 50, 100, 500)) -> str:
    """Pretty-print the per-user comparison table (rank-0 only).

    Layout:
        1) overall recall@K table (one row per strategy)
        2) overall hit_rate@K table (one row per strategy)
        3) per-history-length-bucket recall@K table for the redrec strategy
           and each baseline (mean_pool / last_pool / last8_pool / last32_pool)
        4) lift line: redrec top-K recall vs the best pooling baseline
    """
    ks_list = list(ks)
    header_cols = [f'top{k}' for k in ks_list]

    def _table(metric_key_fn, title):
        lines = [f'== {title} ==']
        header = f"{'strategy':<12} | " + ' | '.join(f'{c:>10}' for c in header_cols)
        lines.append(header)
        lines.append('-' * len(header))
        for name, m in results.items():
            cells = ' | '.join(
                f'{m.get(metric_key_fn(k), 0.0):>10.4f}' for k in ks_list
            )
            lines.append(f"{name:<12} | {cells}")
        return lines

    lines: List[str] = []
    lines.extend(_table(lambda k: f'top{k}_recall', 'recall@K (per-user mean)'))
    lines.append('')
    lines.extend(_table(lambda k: f'top{k}_hit_rate', 'hit_rate@K (per-user mean)'))

    # Per-bucket recall breakdown (only if at least one strategy carries it).
    any_bucket_keys = False
    for m in results.values():
        if any(f'top{k}_recall_bucket' in m for k in ks_list):
            any_bucket_keys = True
            break
    if any_bucket_keys:
        # Pick the bucket label list from the first strategy that has them.
        bucket_labels: List[str] = []
        for m in results.values():
            for k in ks_list:
                rb = m.get(f'top{k}_recall_bucket')
                if isinstance(rb, dict) and rb:
                    bucket_labels = list(rb.keys())
                    break
            if bucket_labels:
                break
        for name, m in results.items():
            n_bucket_dict = m.get('_n_users_bucket', {})
            lines.append('')
            lines.append(f'== {name} | recall@K by hist_len bucket ==')
            header_b = f"{'bucket':<10} {'n_users':>9} | " + \
                ' | '.join(f'{c:>10}' for c in header_cols)
            lines.append(header_b)
            lines.append('-' * len(header_b))
            for lab in bucket_labels:
                n_u = int(n_bucket_dict.get(lab, 0))
                cells = []
                for k in ks_list:
                    rb = m.get(f'top{k}_recall_bucket', {})
                    cells.append(f'{rb.get(lab, 0.0):>10.4f}')
                lines.append(f"{lab:<10} {n_u:>9} | " + ' | '.join(cells))

    if 'redrec' in results:
        lift_k = 10 if 10 in ks_list else (min(ks_list) if ks_list else None)
        if lift_k is not None:
            lift_key = f'top{lift_k}_recall'
            baselines = []
            if 'mean_pool' in results and lift_key in results['mean_pool']:
                baselines.append(results['mean_pool'][lift_key])
            if 'last_pool' in results and lift_key in results['last_pool']:
                baselines.append(results['last_pool'][lift_key])
            if baselines:
                best = max(baselines)
                v = results['redrec'].get(lift_key, 0.0)
                if best > 0:
                    lines.append('')
                    lines.append(
                        f'>>> redrec top{lift_k} recall lift over best pooling baseline: '
                        f'{(v - best) / best * 100:+.2f}%'
                    )
    return '\n'.join(lines)
