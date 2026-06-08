"""V0-aligned online eval helpers.

Computes top-{1,10,50,100,500} recall over the full item pool for three
user-embedding strategies:

  * ``mean_pool`` : zero-parameter mean over valid input positions.
  * ``last_pool`` : zero-parameter last valid position.
  * ``redrec``    : the REDRec user tower (input_embedding_projector ->
                    user_llm -> note_embedding_head, with N learnable
                    queries appended). Runs the same forward path that
                    ``model.forward_precomputed_embedding`` uses, but
                    returns only the user embedding without the NCE loss.

All three strategies score against the SAME L2-normalized item pool, so
the recall numbers are directly comparable. The redrec strategy is used
to track training progress; mean_pool / last_pool are zero-cost baselines
that do not change over training (we evaluate them once and cache).

Distributed strategy
--------------------
Eval samples are sharded round-robin across ranks; each rank computes
its local user embeddings + topk hits, then a single ``all_reduce(SUM)``
combines counts. The item_pool is held on rank-0 only at build time,
then broadcast to all ranks (or alternatively each rank keeps a copy
loaded from the same memmap file -- both work; we go with the simpler
"each rank loads its own pool" path since the memmap is page-cached).
"""

from logging import getLogger
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


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
def _topk_hits_local(
    user_emb: torch.Tensor,
    target_idx: torch.Tensor,
    item_pool_T: torch.Tensor,
    ks=(1, 10, 50, 100, 500),
    score_chunk: int = 1024,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[int, int]:
    """For each user in this rank's shard, compute hit@k against the full item pool.

    Returns a dict ``{k: total_hits_int}`` (NOT divided by U).

    Args:
        user_emb:    (U, D)  user embeddings (cosine-normalize before topk).
        target_idx:  (U,)    int64 row index in the item pool.
        item_pool_T: (D, N)  L2-normalized item pool, transposed for matmul.
        valid_mask:  (U,)    bool, True for real users / False for padding
                             slots (padded so that DeepSpeed Stage3 sees
                             equal forward count across ranks). Padding hits
                             are excluded from the per-k count.
    """
    if user_emb.numel() == 0:
        return {k: 0 for k in ks}
    user_emb = F.normalize(user_emb.float(), dim=-1)
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    U = user_emb.size(0)
    for s in range(0, U, score_chunk):
        e = min(U, s + score_chunk)
        u = user_emb[s:e]
        t = target_idx[s:e]
        scores = u @ item_pool_T  # (b, N)
        topk = scores.topk(max_k, dim=1).indices  # (b, max_k)
        for k in ks:
            hit = (topk[:, :k] == t.unsqueeze(1)).any(dim=1)
            if valid_mask is not None:
                hit = hit & valid_mask[s:e]
            hits[k] += int(hit.sum().item())
    return hits


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
    """Evaluate redrec (and optionally mean_pool / last_pool) recall on a hold-out set.

    ``eval_pack`` (built by ``build_v0_eval_pack`` on each rank):
        seq_cid_idx    : (U, L) int64 cpu  -- row index into embeddings_ref,
                                              -1 for padding slots.
        mask           : (U, L) uint8 cpu  -- 1 valid / 0 pad.
        target_idx     : (U,)   int64 cpu  -- row index of the pos target.
        embeddings_ref : ndarray-like      -- shared memmap (NOT a copy);
                                              evaluate_v0_recall slices and
                                              uploads only the rows it needs
                                              for each user batch.
        embed_dim      : int
        num_items      : int

    Each rank processes ``[rank::world_size]`` slice of users; hit counts
    are all-reduced (SUM) across ranks. Final recall = total_hits / U.

    Returns:
        {strategy_name: {f'top{k}_recall': float, ...}, ...}
    """
    logger = getLogger()
    seq_cid_idx_full = eval_pack['seq_cid_idx']        # (U, L) int64 cpu
    mask_full = eval_pack['mask']                      # (U, L) uint8 cpu
    target_idx_full = eval_pack['target_idx']          # (U,)   int64 cpu
    embeddings = eval_pack['embeddings_ref']           # memmap or ndarray
    embed_dim = int(eval_pack['embed_dim'])
    num_items = int(eval_pack['num_items'])

    U_total = seq_cid_idx_full.size(0)
    L = int(seq_cid_idx_full.size(1))

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
    # This is the ONLY full (N, D) tensor we put on the GPU. For Qwen2.5-1.5B
    # + Stage1 we have ~50-60 GB free per H20, plenty for a 6.67 GB fp32
    # item pool. We chunk the upload so the host-side fp32 staging buffer
    # also stays bounded.
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

    target_idx_local = target_idx_full[idx_local].to(device)
    valid_mask_local_d = valid_mask_local.to(device)

    results: Dict[str, Dict[str, float]] = {}

    def _reduce_hits_to_recall(hits_local: Dict[int, int]) -> Dict[str, float]:
        if world_size > 1:
            buf = torch.tensor(
                [hits_local[k] for k in ks], dtype=torch.long, device=device
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            totals = buf.tolist()
        else:
            totals = [hits_local[k] for k in ks]
        return {f'top{k}_recall': totals[i] / max(1, U_total)
                for i, k in enumerate(ks)}

    # ---- helper: load a batch of (seq_emb, mask) onto the GPU ----
    # We use item_pool_d (already on GPU, but normalized) ONLY for normalized
    # negatives -- for the input sequence we need the *un*-normalized
    # embeddings, because the redrec input projector was trained on raw
    # embeddings. So we keep a parallel raw item_pool on GPU as well IF the
    # user ever asks for the redrec strategy. For mean/last pool, normalized
    # is fine because we re-normalize the user emb downstream anyway.
    # For simplicity and correctness we materialize a raw (un-normalized)
    # item_pool on GPU as well. Total GPU pool memory: 2 * 6.67 GB = 13.3 GB
    # (still well within budget). NOTE: we could index `embeddings` per batch
    # from CPU to save GPU memory, but uploading 768 (bs) * 200 (L) = 153K
    # rows per batch = 0.3 GB transfer per batch is comparable to a single
    # one-time upload, and the CPU memmap path is slower due to PCIe round-trips.
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

    redrec_hits = {k: 0 for k in ks}
    n_local = idx_local.numel()
    if n_local > 0:
        for s in range(0, n_local, user_batch):
            e = min(n_local, s + user_batch)
            sl = idx_local[s:e]
            seq_b, mask_b = _gather_seq_batch(sl)
            ue = _redrec_user_emb(model, seq_b, mask_b)
            t_b = target_idx_local[s:e]
            vm_b = valid_mask_local_d[s:e]
            h = _topk_hits_local(
                ue, t_b, item_pool_T, ks=ks,
                score_chunk=score_chunk, valid_mask=vm_b,
            )
            for k in ks:
                redrec_hits[k] += h[k]
    results['redrec'] = _reduce_hits_to_recall(redrec_hits)

    # ---- zero-param baselines ----
    # Aligned with the v0 reference run (see
    # /apdcephfs_gy4/share_303218624/jingweidong/output/logs/nohup_20260602_114853.log):
    #   mean_pool   : mean over ALL valid positions
    #   last_pool   : the last valid position
    #   last8_pool  : mean over the last 8 valid positions
    #   last32_pool : mean over the last 32 valid positions
    # All four are zero-parameter and target-time invariant, so we only need to
    # evaluate them once (cached in trainer).
    if eval_baselines:
        mean_hits = {k: 0 for k in ks}
        last_hits = {k: 0 for k in ks}
        last8_hits = {k: 0 for k in ks}
        last32_hits = {k: 0 for k in ks}
        if n_local > 0:
            for s in range(0, n_local, user_batch):
                e = min(n_local, s + user_batch)
                sl = idx_local[s:e]
                seq_b, mask_b = _gather_seq_batch(sl)
                t_b = target_idx_local[s:e]
                vm_b = valid_mask_local_d[s:e]

                mp = _mean_pool(seq_b, mask_b)
                h_m = _topk_hits_local(
                    mp, t_b, item_pool_T, ks=ks,
                    score_chunk=score_chunk, valid_mask=vm_b,
                )
                for k in ks:
                    mean_hits[k] += h_m[k]

                lp = _last_pool(seq_b, mask_b)
                h_l = _topk_hits_local(
                    lp, t_b, item_pool_T, ks=ks,
                    score_chunk=score_chunk, valid_mask=vm_b,
                )
                for k in ks:
                    last_hits[k] += h_l[k]

                lp8 = _last_n_pool(seq_b, mask_b, n=8)
                h_l8 = _topk_hits_local(
                    lp8, t_b, item_pool_T, ks=ks,
                    score_chunk=score_chunk, valid_mask=vm_b,
                )
                for k in ks:
                    last8_hits[k] += h_l8[k]

                lp32 = _last_n_pool(seq_b, mask_b, n=32)
                h_l32 = _topk_hits_local(
                    lp32, t_b, item_pool_T, ks=ks,
                    score_chunk=score_chunk, valid_mask=vm_b,
                )
                for k in ks:
                    last32_hits[k] += h_l32[k]
        results['mean_pool'] = _reduce_hits_to_recall(mean_hits)
        results['last_pool'] = _reduce_hits_to_recall(last_hits)
        results['last8_pool'] = _reduce_hits_to_recall(last8_hits)
        results['last32_pool'] = _reduce_hits_to_recall(last32_hits)

    # Free GPU buffers (~20 GB on H20 -> we don't want them lingering across
    # 1000-step gaps; the next eval can rebuild from page-cached memmap fast).
    del item_pool_T, item_pool_d, item_pool_raw_d
    torch.cuda.empty_cache()

    if model_was_training:
        model.train()

    if rank == 0:
        ks_list = list(ks)
        for name, m in results.items():
            metric_str = ' '.join(f'top{k}={m[f"top{k}_recall"]:.4f}' for k in ks_list)
            logger.info(f'[v0_eval] {name:<10} | {metric_str}')

    return results


def format_recall_table(results: Dict[str, Dict[str, float]],
                        ks=(1, 10, 50, 100, 500)) -> str:
    """Pretty-print the comparison table (rank-0 only)."""
    header_cols = [f'top{k}' for k in ks]
    header = f"{'strategy':<12} | " + ' | '.join(f'{c:>10}' for c in header_cols)
    lines = [header, '-' * len(header)]
    for name, m in results.items():
        cells = ' | '.join(f'{m[f"top{k}_recall"]:>10.4f}' for k in ks)
        lines.append(f"{name:<12} | {cells}")
    if 'redrec' in results:
        baselines = []
        if 'mean_pool' in results:
            baselines.append(results['mean_pool']['top10_recall'])
        if 'last_pool' in results:
            baselines.append(results['last_pool']['top10_recall'])
        if baselines:
            best = max(baselines)
            v = results['redrec']['top10_recall']
            if best > 0:
                lines.append('')
                lines.append(
                    f'>>> redrec top10 lift over best pooling baseline: '
                    f'{(v - best) / best * 100:+.2f}%'
                )
    return '\n'.join(lines)
