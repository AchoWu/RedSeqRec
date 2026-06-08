"""
recall_top10_compare.py
=======================

For each user in `emb/<tag>/user_embedding_pr.pkl`, perform Top-K retrieval
against TWO candidate pools using TWO user-embedding methods, and dump the
results to a single JSONL file for side-by-side human inspection.

Methods
-------
- model  : the 3-query user embedding produced by the trained model
           (loaded from emb/<tag>/user_embedding_pr.pkl).
- kmeans : KMeans(k=3) cluster centers over the user's real homefeed history,
           computed on the SAME pool that's currently being retrieved
           (so for each pool we re-fit kmeans in that pool's embedding space).

Pools
-----
- my_pool   : emb/<tag>/base_item_embedding_pr.pkl
              (your custom 84812 docs; we have title/content for these)
- demo_pool : emb/demo_multiscene/base_item_embedding_pr.pkl
              (the official ~778k-doc test set; only note_id is known)

Output
------
A JSONL file (default: emb/<tag>/recall_top10_compare.jsonl), one line per
user. Each line:

{
  "user_id": "...",
  "seq_len": 53,
  "history_note_ids": [...],         # real (non-pad) history doc ids
  "future_target_ids": [...],        # ground-truth future clicks (in my_pool)
  "recall": {
    "my_pool":   {
      "model":   [{"rank":1,"note_id":...,"score":0.71,"title":"...","content":"..."}, ...],
      "kmeans":  [...]
    },
    "demo_pool": {
      "model":   [{"rank":1,"note_id":...,"score":0.65,"url":"https://..."}, ...],
      "kmeans":  [...]
    }
  }
}

Usage
-----
    cd eval
    CUDA_VISIBLE_DEVICES=0 python recall_top10_compare.py \\
        --config_path ../config/my_data.yaml \\
        --tag_name my_data \\
        --topk 10 \\
        --max_users 200          # optional: only dump first N for a quick look
"""
import os
import json
import pickle
import argparse

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
from sklearn.cluster import KMeans


# ----------------------------- IO helpers --------------------------------- #

def _to_numpy(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", dtype=torch.float32).numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def normalize(x, eps=1e-8):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)


def load_user_embeds(path):
    with open(path, "rb") as f:
        rows = pickle.load(f)
    out = {}
    for r in rows:
        uid = str(r["id"])
        emb = _to_numpy(r["embed"]).astype(np.float32, copy=False)
        if emb.ndim == 1:
            assert emb.size % 64 == 0, f"user {uid}: bad embed dim {emb.size}"
            emb = emb.reshape(-1, 64)
        out[uid] = emb
    return out


def load_note_embeds(path):
    print(f"  loading note embeds from {path}")
    with open(path, "rb") as f:
        rows = pickle.load(f)
    ids, embs = [], []
    for r in rows:
        ids.append(str(r["id"]))
        embs.append(_to_numpy(r["embed"]).astype(np.float32, copy=False))
    embs = np.stack(embs, axis=0)
    embs = normalize(embs)
    print(f"  -> {embs.shape[0]} notes, dim={embs.shape[1]}")
    return embs, ids


def load_user_lastn(user_lastn_path):
    """uid -> {targets:[note_id], history:[note_id], seq_len:int}"""
    with open(user_lastn_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = {}
    for uid, v in obj.items():
        if v.get("eval_doc_ids"):
            tgt = [str(x) for x in v["eval_doc_ids"]]
        else:
            tgt = [str(n["note_id"]) for n in v.get("ads_click_noteid_target", [])]
        lastn = v.get("homefeed_noteid_lastn", []) or []
        hist = [str(it.get("note_id")) for it in lastn
                if str(it.get("note_id")) != "-1"]
        out[str(uid)] = {
            "targets":  tgt,
            "history":  hist,
            "seq_len":  len(hist),
        }
    return out


def load_note_text(note_info_path):
    """Returns dict: note_id (str) -> {'title':..., 'content':...}.
    The file is a JSON list of dicts with at least note_id/title/content."""
    if not note_info_path or not os.path.exists(note_info_path):
        return {}
    print(f"  loading note text from {note_info_path}")
    with open(note_info_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    m = {}
    for item in arr:
        nid = str(item.get("note_id"))
        if not nid:
            continue
        m[nid] = {
            "title":   item.get("title", ""),
            "content": item.get("content", ""),
        }
    print(f"  -> note text entries: {len(m)}")
    return m


# ----------------------------- KMeans builder ----------------------------- #

def build_kmeans_user_embed(history_idx, note_embs, k=3, seed=0):
    n = history_idx.size
    if n < k:
        return None
    H = note_embs[history_idx]
    if n == k:
        return normalize(H.astype(np.float32))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    km.fit(H)
    centers = km.cluster_centers_.astype(np.float32)
    labels = km.labels_
    rng = np.random.default_rng(seed)
    for c in range(k):
        if (labels == c).sum() == 0:
            centers[c] = H[rng.integers(0, n)]
    return normalize(centers)


# ----------------------------- retrieval ---------------------------------- #

def topk_retrieval_per_user(u_emb_np, note_t, note_ids, topk, device):
    """u_emb_np: [n_query, 64]. Returns (top_ids[list[str]], top_scores[list[float]])."""
    u_t = torch.from_numpy(u_emb_np).to(device).half()         # [Q, 64]
    sims = (u_t @ note_t.T).max(dim=0).values                  # [N]
    top = sims.topk(topk, dim=-1)
    idx = top.indices.cpu().numpy().tolist()
    sc  = top.values.float().cpu().numpy().tolist()
    return [note_ids[i] for i in idx], [float(s) for s in sc]


def retrieve_pool(
    pool_label,
    pool_path,
    user_embeds_model,
    user_history,
    target_users,
    note_text_map,
    topk,
    device,
    seed,
    add_url=False,
):
    """Returns dict: uid -> {'model': [...], 'kmeans': [...]}.

    For users where kmeans cannot be built in this pool (history too short
    after filtering by this pool's id space), the 'kmeans' field will be
    omitted (i.e. set to None).
    """
    note_embs, note_ids = load_note_embeds(pool_path)
    note_id2idx = {nid: i for i, nid in enumerate(note_ids)}
    note_t = torch.from_numpy(note_embs).to(device).half()

    out = {}
    n_skip_kmeans = 0
    desc = f"retrieve[{pool_label}]"
    for uid in tqdm(target_users, desc=desc):
        rec = {"model": None, "kmeans": None}

        # -------- model --------
        if uid in user_embeds_model:
            top_ids, top_sc = topk_retrieval_per_user(
                user_embeds_model[uid], note_t, note_ids, topk, device
            )
            rec["model"] = _format_topk(top_ids, top_sc, note_text_map, add_url)

        # -------- kmeans (per-pool) --------
        hist = user_history.get(uid, {}).get("history", [])
        idxs = [note_id2idx[h] for h in hist if h in note_id2idx]
        idxs = np.asarray(sorted(set(idxs)), dtype=np.int64)
        if idxs.size >= 3:
            ce = build_kmeans_user_embed(idxs, note_embs, k=3, seed=seed)
            if ce is not None:
                top_ids, top_sc = topk_retrieval_per_user(
                    ce, note_t, note_ids, topk, device
                )
                rec["kmeans"] = _format_topk(top_ids, top_sc, note_text_map, add_url)
            else:
                n_skip_kmeans += 1
        else:
            n_skip_kmeans += 1

        out[uid] = rec

    print(f"  [{pool_label}] users without valid kmeans (history<3 in this pool): "
          f"{n_skip_kmeans}/{len(target_users)}")

    # free GPU memory before loading next pool
    del note_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _format_topk(top_ids, top_sc, note_text_map, add_url):
    rows = []
    for r, (nid, sc) in enumerate(zip(top_ids, top_sc), 1):
        item = {"rank": r, "note_id": nid, "score": round(float(sc), 4)}
        info = note_text_map.get(nid)
        if info is not None:
            t = info.get("title", "") or ""
            c = info.get("content", "") or ""
            # truncate so the JSONL stays human-readable
            if len(t) > 200: t = t[:200] + "..."
            if len(c) > 600: c = c[:600] + "..."
            item["title"] = t
            item["content"] = c
        if add_url:
            item["url"] = f"https://www.xiaohongshu.com/explore/{nid}"
        rows.append(item)
    return rows


# ----------------------------- main --------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, default="../config/my_data.yaml")
    ap.add_argument("--tag_name", type=str, default="my_data")
    ap.add_argument(
        "--demo_pool_path", type=str,
        default="emb/demo_multiscene/base_item_embedding_pr.pkl",
        help="Official xhs evaluation pool (base_item_embedding_pr.pkl).",
    )
    ap.add_argument(
        "--note_info_path", type=str, default="note_info_my.json",
        help="JSON list with {note_id,title,content} for my_pool docs.",
    )
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20251231)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--max_users", type=int, default=0,
        help="If >0, only dump the first N users (sorted by user_id).",
    )
    ap.add_argument(
        "--out_path", type=str, default="",
        help="Where to write the JSONL output. "
             "Default: emb/<tag>/recall_top10_compare.jsonl",
    )
    args = ap.parse_args()

    with open(args.config_path, "r") as f:
        cfg = edict(yaml.safe_load(f))

    user_lastn_path = cfg.eval.user_eval.user_lastn_path
    my_pool_path    = f"emb/{args.tag_name}/base_item_embedding_pr.pkl"
    user_embed_path = f"emb/{args.tag_name}/user_embedding_pr.pkl"
    out_path        = (args.out_path
                       or f"emb/{args.tag_name}/recall_top10_compare.jsonl")

    if not os.path.exists(args.demo_pool_path):
        raise FileNotFoundError(
            f"demo_pool not found: {args.demo_pool_path}\n"
            f"Please pass --demo_pool_path."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[recall] loading user lastn ...")
    user_lastn = load_user_lastn(user_lastn_path)
    print(f"  -> {len(user_lastn)} users")

    print("[recall] loading model user embeds ...")
    user_embeds_model = load_user_embeds(user_embed_path)
    print(f"  -> {len(user_embeds_model)} model user embeds, "
          f"n_query={next(iter(user_embeds_model.values())).shape[0]}")

    print("[recall] loading note text (my_pool) ...")
    note_text_map = load_note_text(args.note_info_path)

    # decide which users to dump
    target_users = sorted(
        uid for uid in user_embeds_model.keys() if uid in user_lastn
    )
    if args.max_users and args.max_users > 0:
        target_users = target_users[: args.max_users]
    print(f"[recall] dumping {len(target_users)} users")

    # -------- retrieve in my_pool --------
    print("\n[recall] === my_pool ===")
    rec_my = retrieve_pool(
        "my_pool", my_pool_path,
        user_embeds_model, user_lastn, target_users,
        note_text_map=note_text_map,
        topk=args.topk, device=device, seed=args.seed,
        add_url=False,
    )

    # -------- retrieve in demo_pool --------
    print("\n[recall] === demo_pool (official xhs) ===")
    rec_demo = retrieve_pool(
        "demo_pool", args.demo_pool_path,
        user_embeds_model, user_lastn, target_users,
        note_text_map={},        # no text available for demo pool
        topk=args.topk, device=device, seed=args.seed,
        add_url=True,
    )

    # -------- write JSONL --------
    print(f"\n[recall] writing -> {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for uid in tqdm(target_users, desc="writing"):
            info = user_lastn[uid]
            line = {
                "user_id": uid,
                "seq_len": info["seq_len"],
                "history_note_ids":  info["history"][:50],
                "future_target_ids": info["targets"],
                "recall": {
                    "my_pool":   rec_my.get(uid,   {"model": None, "kmeans": None}),
                    "demo_pool": rec_demo.get(uid, {"model": None, "kmeans": None}),
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"[recall] done. {len(target_users)} users written to {out_path}")


if __name__ == "__main__":
    main()
