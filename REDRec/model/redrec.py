# model for multi-scene

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from transformers import AutoConfig, AutoModelForCausalLM
from scipy.optimize import linear_sum_assignment

from REDRec.utils.enum_type import InputType
from REDRec.model.basemodel import BaseModel, all_gather

# ----------- Utility Functions -----------
def batch_cosine_matching(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched cosine similarity and Hungarian assignment.
    Matches every row in A (N) to best rows in B (n) in a batch, possibly with fallback.
    """
    batch_size, N, dim = A.shape
    n = B.shape[1]
    device = A.device

    sim_matrix = torch.bmm(A, B.transpose(1, 2))  # (bs, N, n)
    matched_B = []

    for b in range(batch_size):
        cur_sim = sim_matrix[b]
        cost_matrix = -cur_sim.detach().float().cpu().numpy()
        cost_matrix = np.nan_to_num(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = torch.full((N,), -1, dtype=torch.long, device=device)
        matches[row_ind] = torch.tensor(col_ind, device=device)
        # Unmatched: greedy max
        unmatched = (matches == -1)
        if unmatched.any():
            _, remaining = torch.max(cur_sim[unmatched], dim=1)
            matches[unmatched] = remaining
        matched_B.append(B[b][matches])
    return torch.stack(matched_B, dim=0)

def cos_kmeans(X: torch.Tensor, n_clusters: int, num_iter: int = 10):
    """
    Batched cosine k-means clustering.
    Returns cluster centers and labels.
    """
    bs, N, dim = X.shape
    device = X.device
    indices = np.random.choice(N, size=n_clusters, replace=False)
    cluster_centers = X[:, indices].clone()
    for _ in range(num_iter):
        similarities = torch.bmm(X, cluster_centers.transpose(1, 2))
        labels = torch.argmax(similarities, dim=-1)
        new_centers = torch.zeros_like(cluster_centers)
        for k in range(n_clusters):
            mask = (labels == k).float().unsqueeze(-1)
            sum_vectors = (X * mask).sum(dim=1)
            norm = torch.norm(sum_vectors, p=2, dim=-1, keepdim=True)
            new_centers[:, k] = sum_vectors / (norm + 1e-8)
        cluster_centers = new_centers
    return cluster_centers, labels

def cluster_based_matching(A: torch.Tensor, B: torch.Tensor, num_iter: int = 5) -> torch.Tensor:
    """
    Matches each vector in A to B by first clustering A and then linear assignment.
    """
    batch_size, N, dim = A.shape
    n = B.shape[1]
    device = A.device
    cluster_centers, labels = cos_kmeans(A, n_clusters=n, num_iter=num_iter)
    matched_B = torch.zeros_like(A)
    for b in range(batch_size):
        sim_matrix = torch.mm(cluster_centers[b], B[b].T)
        cost_matrix = -sim_matrix.detach().float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cluster_to_B = torch.zeros(n, dtype=torch.long, device=device)
        cluster_to_B[row_ind] = torch.tensor(col_ind, device=device)
        B_indices = cluster_to_B[labels[b]]
        matched_B[b] = B[b][B_indices]
    return matched_B

# ----------- Model Heads & AE -----------
class ProjectionHead(nn.Module):
    """Nonlinear projection with residual to reduce embedding dimension."""
    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.projection(x)
        x = self.gelu(proj)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + proj
        x = self.layer_norm(x)
        return x

class ClassificationHead(nn.Module):
    """Simple feedforward head for classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class LatentProjEncoder(nn.Module):
    """Compression encoder."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 8), nn.GELU(),
            nn.Linear(latent_dim * 8, latent_dim * 4), nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(),
        )
    def forward(self, x): return self.encoder(x)

class LatentProjDecoder(nn.Module):
    """Compression decoder."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2), nn.GELU(),
            nn.Linear(latent_dim*2, latent_dim*4), nn.GELU(),
            nn.Linear(latent_dim*4, latent_dim*8), nn.GELU(),
            nn.Linear(latent_dim*8, input_dim)
        )
    def forward(self, x): return self.decoder(x)

class LlamaRMSNorm(nn.Module):
    """Root Mean Square Norm used in Llama."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# ----------- Losses -----------
def topk_multi_positive_accuracy(logits: torch.Tensor, pos_num: int, k_list=[1, 10, 100]) -> dict:
    """
    Computes multi-positive top-k accuracy. Assumes positives at front of logits.
    """
    probs = torch.softmax(logits, dim=-1)
    _, pred_indices = torch.topk(probs, k=max(k_list), dim=-1)
    pos_mask = torch.zeros_like(probs).bool().to(logits.device)
    pos_mask[:, :pos_num] = True
    topk_acc = {}
    for k in k_list:
        pred_k = pred_indices[:, :k]
        # Did we predict any positive in top-k?
        correct = torch.any(pos_mask.gather(1, pred_k), dim=1)
        topk_acc[f"top{k}_acc"] = correct.float().mean()
    return topk_acc

def multilabel_categorical_crossentropy(y_true, y_pred):
    """Multi-label cross-entropy for multi-hot targets."""
    y_pred = (1 - 2 * y_true) * y_pred
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred - y_true * 1e12, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred - (1 - y_true) * 1e12, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss

class REDRec(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.logger = getLogger('REDRec')
        self.config = config
        self._parse_config(config)
        self._build_model()
        # First-batch check flag for the explicit-token-id branch: we only
        # want to verify max(token_ids) < query_nums once per training run
        # (the check requires a GPU->CPU sync and is redundant after batch 1).
        self._explicit_token_id_bounds_checked = False

    def _parse_config(self, config):
        mcfg = config.model
        dcfg = config.data
        self.item_pretrain_dir = mcfg['item_pretrain_dir']
        self.user_pretrain_dir = mcfg['user_pretrain_dir']
        self.gradient_checkpointing = mcfg.get('gradient_checkpointing', False)
        self.use_ft_flash_attn = mcfg.get('use_ft_flash_attn', False)
        self.max_input_token_len = dcfg.get('max_input_token_len', 128)
        self.loss_type = config.get('loss', 'nce')
        self.nce_thres = config.get('nce_thres', 0.99)
        self.add_item_action_embed = mcfg.get('add_item_action_embed', False)
        self.add_hour_embed = mcfg.get('add_hour_embed', False)
        self.add_position_embed = mcfg.get('add_position_embed', False)
        self.predict_action = mcfg.get('predict_action', False)
        self.AE_compress_dim = mcfg.get('AE_compress_dim', -1)
        self.AE_compress_decay_rate = mcfg.get('AE_compress_decay_rate', 0.9995)
        self.learnable_interest_query = mcfg.get('learnable_interest_query', -1)
        self.query_nums = mcfg.get('query_nums', -1)
        self.window_size = mcfg.get('window_pos', -1)
        self.engage_action_n = mcfg.get('engage_action_n', 5)
        self.llm_init_item = mcfg.get('item_llm_init', True)
        self.llm_init_user = mcfg.get('user_llm_init', True)
        # Both 'precomputed_embedding' (official) and 'v0_aligned' (V0Simple-aligned)
        # datasets feed pre-encoded item embeddings into the user tower, so they share
        # the same model topology: item_llm is dropped, input_embedding_projector is
        # added to lift the precomputed dim to user_llm.hidden_size, and forward goes
        # through forward_precomputed_embedding.
        self.use_precomputed_embedding = dcfg.get('dataset_type', None) in (
            'precomputed_embedding', 'v0_aligned',
        )
        self.precomputed_input_dim = mcfg.get('precomputed_input_dim', 512)

    def _build_model(self):
        # ----- Item/User LLM -----
        if self.use_precomputed_embedding:
            self.item_llm = None
        else:
            self.item_llm = self._create_llm(self.item_pretrain_dir, self.llm_init_item, freeze=self.config.training.get('freeze_item', False))
        self.user_llm = self._create_llm(self.user_pretrain_dir, self.llm_init_user, freeze=False)
        self.projection_dim = self.user_llm.config.hidden_size if self.item_llm is None else self.item_llm.config.hidden_size
        
        # ----- Embedding heads -----
        # user_output_dim controls the intermediate user-tower output dim (default 64).
        # user_final_dim controls the final output dim after the output MLP (default 512),
        # which must match the precomputed item embedding dim for NCE and eval.
        self.user_output_dim = int(self.config.model.get('user_output_dim', 64))
        self.user_final_dim = int(self.config.model.get('user_final_dim', 512))
        self.note_embedding_head = ProjectionHead(self.projection_dim, output_dim=self.user_output_dim)
        # MLP to lift user_output_dim (64) to user_final_dim (512) for NCE loss & eval.
        if self.user_output_dim != self.user_final_dim:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.user_output_dim, self.user_final_dim * 2),
                nn.GELU(),
                nn.Linear(self.user_final_dim * 2, self.user_final_dim),
                nn.LayerNorm(self.user_final_dim),
            )
        else:
            self.output_mlp = nn.Identity()
        self.item_emb_token_n = 1
        self.item_emb_tokens = nn.Parameter(torch.zeros(1, self.item_emb_token_n, self.projection_dim))
        self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
        
        if self.add_item_action_embed:
            self.item_action_embedding = nn.Parameter(torch.zeros(self.engage_action_n, self.projection_dim))
            self.item_action_embedding.data.normal_(mean=0.0, std=0.02)

        if self.add_hour_embed:
            self.hour_embeddings = nn.Embedding(24, self.projection_dim)

        if self.learnable_interest_query and self.learnable_interest_query > 0:
            self.learnable_interest_query_embedding = nn.Parameter(
                torch.zeros(1, self.learnable_interest_query, self.user_llm.config.hidden_size)
            )
            self.learnable_interest_query_embedding.data.normal_(mean=0.0, std=0.02)

        if self.query_nums and self.query_nums > 0:
            self.query = nn.Embedding(self.query_nums, self.projection_dim)

        if self.AE_compress_dim > 0:
            self.latent_proj_encoder = LatentProjEncoder(self.projection_dim, self.AE_compress_dim)
            self.latent_proj_decoder = LatentProjDecoder(self.projection_dim, self.AE_compress_dim)
            self.ae_decay = 1.0
        else:
            self.ae_decay = -1.0

        if self.predict_action:
            self.action_predict_head = ClassificationHead(self.projection_dim, hidden_dim=512, num_classes=self.engage_action_n)

        if self.loss_type == 'nce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            raise NotImplementedError("Only NCE loss is supported.")

        if self.add_position_embed:
            self.position_embeddings = nn.Embedding(200, self.projection_dim)

        if self.use_precomputed_embedding:
            # 使用 mlp 升维对齐 qb embedding
            # 512 -> 1536（qb embedding dim）
            self.input_embedding_projector = nn.Sequential(
                nn.Linear(self.precomputed_input_dim, self.projection_dim),
                nn.GELU(),
                nn.LayerNorm(self.projection_dim),
            )

    def _create_llm(self, pretrain_dir, init=True, freeze=False):
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.output_hidden_states = True
        hf_config.return_dict = True
        hf_config.use_cache = False

        # NOTE: newer `transformers` (>=4.40-ish) forbids the bare
        # `AutoModelForCausalLM(config=...)` ctor and requires going through
        # one of the two factory methods. We use `from_config` for the
        # random-init branch (no pretrained weights, model lives on CPU
        # initially -- DeepSpeed will move/shard it during engine init,
        # which is also why we drop the explicit `.cuda()` here: under
        # ZeRO-3 forcing CUDA materialization here defeats stage-3 param
        # partitioning and risks OOM on rank 0).
        llm = (
            AutoModelForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            if init else
            AutoModelForCausalLM.from_config(hf_config)
        )

        if freeze:
            self.logger.info("Freezing parameters of LLM: {}".format(pretrain_dir))
            for param in llm.parameters(): param.requires_grad = False
        return llm

    def _get_positional_embeddings(self, seq_len, device):
        return self.position_embeddings(torch.arange(seq_len, dtype=torch.long, device=device))
    
    # ---------------------- Main Forward Logic ----------------------

    def forward_item_emb(self, input_ids, position_ids, attention_mask, emb_tokens, llm, max_batch_size=150):
        total_batch_size = input_ids.shape[0]
        all_embs = []
        for start_idx in range(0, total_batch_size, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_batch_size)
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_position_ids = position_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]
            inputs_embeds = llm.get_input_embeddings()(batch_input_ids)
            special_pos_mask = (batch_input_ids == 0) & batch_attention_mask
            inputs_embeds[special_pos_mask] = emb_tokens.squeeze(0).squeeze(0)
            model_out = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids
            )
            hidden_states = model_out.hidden_states[-1]
            batch_embedding = []
            
            for i in range(hidden_states.size(0)):
                special_positions = torch.where(special_pos_mask[i])[0]
                if len(special_positions) > 0:
                    special_pos = special_positions[-1].item()
                    emb = hidden_states[i, special_pos]
                else:
                    valid_positions = batch_attention_mask[i]
                    last_valid_pos = torch.where(valid_positions)[0][-1].item()
                    emb = hidden_states[i, last_valid_pos]
                batch_embedding.append(emb)
            
            batch_embedding = torch.stack(batch_embedding)
            all_embs.append(batch_embedding)
        return torch.cat(all_embs, dim=0)

    def apply_autoencoder(self, features, neg_feature):
        
        if self.AE_compress_dim > 0:
            reconstruct = self.latent_proj_decoder(self.latent_proj_encoder(features))
            loss_pos = F.mse_loss(reconstruct, features.detach())
            reconstruct_neg = self.latent_proj_decoder(self.latent_proj_encoder(neg_feature))
            loss_neg = F.mse_loss(reconstruct_neg, neg_feature.detach())
            loss = loss_pos + loss_neg
            out_features = self.ae_decay * features + (1 - self.ae_decay) * reconstruct
            self.ae_decay *= self.AE_compress_decay_rate
        else:
            loss = 0
            out_features = features
        
        return out_features, loss

    def get_user_inputs(self, interaction):
        features = []
        masks = []
        homefeed_len = self.config.data.get('lastn_max_click_note_num_homefeed', 64)
        ads_len = self.config.data.get('lastn_max_click_note_num_ads', 64)
        
        # homefeed
        if homefeed_len > 0:
            feat = self.forward_item_emb(
                interaction['pos_input_ids_homefeed'], interaction['pos_position_ids_homefeed'],
                interaction['pos_attention_mask_homefeed'], self.item_emb_tokens, self.item_llm
            )
            features.append(feat)
            masks.append(torch.ones(feat.size(0), dtype=torch.bool))
        
        # ads
        if ads_len > 0:
            feat = self.forward_item_emb(
                interaction['pos_input_ids_ads'], interaction['pos_position_ids_ads'],
                interaction['pos_attention_mask_ads'], self.item_emb_tokens, self.item_llm
            )
            features.append(feat)
            masks.append(torch.ones(feat.size(0), dtype=torch.bool))
        
        user_feats = torch.cat(features, dim=0).unsqueeze(0)
        user_mask = torch.cat(masks, dim=0).unsqueeze(0)

        # AE compress
        reconstruct_loss = 0
        
        if self.AE_compress_dim > 0:
            user_feats, reconstruct_loss = self.apply_autoencoder(user_feats, None)

        # Add item action/hour/position/query embedding if needed
        # (add similar logic here as needed, for brevity skip action logic; see previous code for action handling)
        if self.add_hour_embed:
            pos_hour_labels_homefeed = interaction.get('pos_hour_labels_homefeed')
            pos_hour_labels_ads = interaction.get('pos_hour_labels_ads')
            hour_feats = torch.cat(
                [self.hour_embeddings(pos_hour_labels_homefeed), self.hour_embeddings(pos_hour_labels_ads)], dim=1
            )
            user_feats = user_feats + hour_feats
        
        if self.add_position_embed:
            pos_homefeed = self._get_positional_embeddings(features[0].shape[1], self.user_llm.device)
            pos_ads = self._get_positional_embeddings(features[1].shape[1], self.user_llm.device)
            user_feats = user_feats + torch.cat([pos_homefeed.unsqueeze(0), pos_ads.unsqueeze(0)], dim=1)
        
        if self.query_nums > 0:
            N = user_feats.size(0)
            query_embedding = self.query(torch.arange(self.query_nums, device=user_feats.device))
            user_feats = torch.cat([user_feats[:, :-self.window_size], query_embedding.unsqueeze(0).repeat(N, 1, 1)], dim=1)
            user_mask = torch.cat([user_mask[:, :-self.window_size], torch.ones((N, self.query_nums), dtype=user_mask.dtype, device=user_mask.device)], dim=1)

        return user_feats, user_mask, reconstruct_loss

    def forward(self, interaction):
        if self.use_precomputed_embedding or 'precomputed_input_embeds' in interaction:
            return self.forward_precomputed_embedding(interaction)

        user_feats, mask, reconstruct_loss = self.get_user_inputs(interaction)
        user_hidden = self.user_llm(inputs_embeds=user_feats, attention_mask=mask).hidden_states[-1]
        user_emb64 = self.note_embedding_head(user_hidden)
        # Negatives
        neg_feat = self.forward_item_emb(
            interaction['neg_input_ids'], interaction['neg_position_ids'],
            interaction['neg_attention_mask'], self.item_emb_tokens, self.item_llm
        )
        if self.AE_compress_dim > 0:
            neg_inputs = self.ae_decay * neg_feat + (1 - self.ae_decay) * self.latent_proj_decoder(self.latent_proj_encoder(neg_feat))
            neg_emb64 = self.note_embedding_head(neg_inputs)
        else:
            neg_emb64 = self.note_embedding_head(neg_feat)
        
        # NCE window loss
        target_group_pos = user_emb64[:, -self.window_size:]
            
        with torch.no_grad(): 
            self.logit_scale.clamp_(0, np.log(100))
        
        logit_scale = self.logit_scale.exp()
        D = neg_emb64.size(-1)
        output_embs = user_emb64[:, -self.query_nums:] / user_emb64[:, -self.query_nums:].norm(dim=-1, keepdim=True)
        target_group_pos = target_group_pos / target_group_pos.norm(dim=-1, keepdim=True)
        neg_emb64 = neg_emb64 / neg_emb64.norm(dim=-1, keepdim=True)
        output_embs = cluster_based_matching(target_group_pos, output_embs)
        pos_logits = F.cosine_similarity(output_embs, target_group_pos, dim=-1).unsqueeze(-1)
        neg_embedding_all = all_gather(neg_emb64, sync_grads=True).reshape(-1, D).transpose(-1, -2)
        neg_logits = torch.matmul(output_embs, neg_embedding_all)
        fix_logits = torch.matmul(target_group_pos, neg_embedding_all)
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = torch.flatten(logits, 0, 1) * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)

        user_embed_loss = F.cross_entropy(logits, labels)
        model_out = {
            'reconstruct_loss': reconstruct_loss,
            'user_embed_loss': user_embed_loss,
            'ae_decay': self.ae_decay,
            'loss': reconstruct_loss + user_embed_loss,
            'nce_samples': (logits > torch.finfo(logits.dtype).min / 100).sum(dim=1).float().mean()
        }
        
        # Top-k accuracies. Extended from official [1,10,100] to also include
        # 50/500 to align with V0Simple baseline. Keep increasing for `break`.
        for k in [1, 10, 50, 100, 500]:
            if k > logits.size(1): break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        
        return model_out

    def forward_precomputed_embedding(self, interaction):
        input_embeds = interaction['precomputed_input_embeds'].to(dtype=self.input_embedding_projector[0].weight.dtype)
        attention_mask = interaction['precomputed_attention_mask'].long()
        target_embeds = interaction['precomputed_target_embeds'].to(dtype=self.input_embedding_projector[0].weight.dtype)
        neg_embeds = interaction['precomputed_neg_embeds'].to(dtype=self.input_embedding_projector[0].weight.dtype)

        if target_embeds.dim() == 2:
            target_embeds = target_embeds.unsqueeze(1)

        target_mask = interaction.get('precomputed_target_mask', None)
        if target_mask is not None:
            target_mask = target_mask.bool()
        else:
            target_mask = torch.ones(target_embeds.shape[:2], dtype=torch.bool, device=target_embeds.device)

        # User side: project 512->1536, pass through user_llm, then note_embedding_head to 64-d
        user_feats = self.input_embedding_projector(input_embeds)

        if self.query_nums > 0:
            # 拼接 self.query_nums 个 learnable query token 到末尾
            batch_size = user_feats.shape[0]
            query_embedding = self.query(torch.arange(self.query_nums, device=user_feats.device))
            user_feats = torch.cat([user_feats, query_embedding.unsqueeze(0).repeat(batch_size, 1, 1)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, self.query_nums), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=1)

        user_hidden = self.user_llm(inputs_embeds=user_feats, attention_mask=attention_mask).hidden_states[-1]
        user_emb64 = self.note_embedding_head(user_hidden)

        # Lift user embedding from user_output_dim (64) to user_final_dim (512)
        # via the output MLP so it aligns with the full-dim item embeddings.
        user_emb64 = self.output_mlp(user_emb64)

        # Item side: slice by user_final_dim (512) -- effectively a no-op when
        # precomputed_input_dim == user_final_dim.
        D_out = self.user_final_dim
        target_emb64 = target_embeds[:, :, :D_out]
        neg_emb64 = neg_embeds[:, :, :D_out] if neg_embeds.dim() == 3 else neg_embeds[:, :D_out]

        target_num = target_emb64.size(1)
        if self.query_nums > 0:
            output_embs = user_emb64[:, -self.query_nums:]
        else:
            output_embs = user_emb64[:, -target_num:]

        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))

        logit_scale = self.logit_scale.exp()
        output_embs = output_embs / output_embs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_emb64 = target_emb64 / target_emb64.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        neg_emb64 = neg_emb64 / neg_emb64.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.query_nums > 0:
            # Explicit multi-interest fast path (preferred when available):
            # dataloader ships precomputed_target_token_ids [B, T] telling us
            # which of the Q learnable interest tokens each target pos should
            # be matched against. Given output_embs [B, Q, D_out] and token
            # ids [B, T], gather at dim=1 to produce matched embeddings
            # [B, T, D_out] where matched[b, t] = output_embs[b, token_ids[b, t]].
            # This skips the kmeans + linear_sum_assignment CPU roundtrip in
            # cluster_based_matching entirely, and gives every pos a stable
            # (dataloader-declared) token assignment across steps -- token
            # semantics no longer depend on batch content.
            explicit_ids = interaction.get('precomputed_target_token_ids', None)
            if explicit_ids is not None:
                # target_token_ids: [B, T] int64 on cpu; move to device once.
                token_ids = explicit_ids.to(output_embs.device, dtype=torch.long)
                B_ = output_embs.size(0)
                T_ = token_ids.size(1)
                D_ = output_embs.size(-1)
                # One-time bounds check: yaml's model.query_nums and the
                # cid_to_token.npy referenced by data.category_assignment_path
                # are produced by two different processes; when they disagree
                # (e.g. yaml Q=2 but cid_to_token contains id=2 for the
                # 社会知识 group), gathering with an oversized id is either
                # (a) silently clamped down and semantics scramble, or (b)
                # crashes deep inside CUDA with no useful traceback.
                # We surface the mismatch with a clear Python-level error on
                # the first batch and skip the check on all subsequent
                # batches (it requires a GPU->CPU sync each call).
                if not self._explicit_token_id_bounds_checked:
                    max_id = int(token_ids.max().item())
                    min_id = int(token_ids.min().item())
                    Q = output_embs.size(1)
                    if not (0 <= min_id and max_id < Q):
                        raise ValueError(
                            f'precomputed_target_token_ids out of range for '
                            f'query_nums={Q}: observed min={min_id} max={max_id}. '
                            f'Check that yaml model.query_nums matches the '
                            f'number of token_N_categories entries in the '
                            f'JSON referenced by data.category_assignment_path.'
                        )
                    self._explicit_token_id_bounds_checked = True
                gather_idx = token_ids.unsqueeze(-1).expand(B_, T_, D_)
                matched_output_embs = output_embs.gather(1, gather_idx)
            elif output_embs.size(1) == target_emb64.size(1):
                # Legacy fast path: query_nums == target_num (typical case
                # query_nums=1, target_num=1); cluster_based_matching would
                # degenerate to identity on a 1x1 assignment, so skip the
                # per-sample CPU call to avoid GPU<->CPU sync overhead.
                matched_output_embs = output_embs
            else:
                # Legacy kmeans + Hungarian path (multi-interest without an
                # explicit category assignment). Kept for backward compat with
                # runs that don't set data.category_assignment_path.
                matched_output_embs = cluster_based_matching(target_emb64, output_embs)
        else:
            matched_output_embs = output_embs[:, -target_num:]

        pos_logits = F.cosine_similarity(matched_output_embs, target_emb64, dim=-1).unsqueeze(-1)

        D = neg_emb64.size(-1)
        neg_embedding_all = all_gather(neg_emb64, sync_grads=True).reshape(-1, D).transpose(-1, -2)
        neg_logits = torch.matmul(matched_output_embs, neg_embedding_all)
        fix_logits = torch.matmul(target_emb64, neg_embedding_all)
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[target_mask] * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        user_embed_loss = F.cross_entropy(logits, labels)

        model_out = {
            'reconstruct_loss': torch.zeros((), device=logits.device, dtype=user_embed_loss.dtype),
            'user_embed_loss': user_embed_loss,
            'ae_decay': self.ae_decay,
            'loss': user_embed_loss,
            'nce_samples': (logits > torch.finfo(logits.dtype).min / 100).sum(dim=1).float().mean()
        }
        # Top-k accuracies. Extended from official [1,10,100] to also include
        # 50/500 so we can compare apples-to-apples with the V0Simple baseline,
        # which logs top1/top50/top100/top500. Keep the list strictly increasing
        # so that the `break` skips only out-of-range k.
        for k in [1, 10, 50, 100, 500]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out
    
    # ----------------- Deployment/Embedding APIs -----------------

    @torch.no_grad()
    def compute_user_embedding_homefeed(self, batch, note_id2idx, raw_embeds, device):
        """Batch user embedding from sequence note_id, using raw_embeds."""
        raw_embeds = np.array(raw_embeds)
        input_dim = raw_embeds.shape[1]
        lastns_homefeed = np.array(batch["note_seqs_homefeed"])
        attention_mask_homefeed = torch.from_numpy(lastns_homefeed != '-1').int().to(device)
        batch_size, seq_len_homefeed = attention_mask_homefeed.shape
        features_homefeed = torch.zeros([batch_size, seq_len_homefeed, input_dim], dtype=torch.float32, device=device)
        
        for idx in range(batch_size):
            for note_idx, note_id in enumerate(lastns_homefeed[idx]):
                if note_id != '-1' and note_id in note_id2idx:
                    cur_note_embed = torch.from_numpy(raw_embeds[note_id2idx[note_id]]).float().to(device)
                    features_homefeed[idx, note_idx] = cur_note_embed
        
        features = features_homefeed
        if hasattr(self, 'input_embedding_projector'):
            features = self.input_embedding_projector(features.to(dtype=self.input_embedding_projector[0].weight.dtype))
        else:
            features = features.to(dtype=torch.bfloat16)
        mask = attention_mask_homefeed
        
        if self.query_nums > 0:
            N = features.shape[0]
            query_embedding = self.query(torch.arange(self.query_nums, device=device))
            features = torch.cat([features, query_embedding.unsqueeze(0).repeat(N, 1, 1)], dim=1)
            mask = torch.cat([mask, torch.ones((N, self.query_nums), dtype=mask.dtype, device=device)], dim=1)
        
        user_hidden = self.user_llm(inputs_embeds=features, attention_mask=mask).hidden_states[-1]
        user_emb64 = self.note_embedding_head(user_hidden)
        user_emb64 = self.output_mlp(user_emb64)
        user_emb_final = user_emb64[:, -self.query_nums:] if self.query_nums > 0 else user_emb64[:, -1:]
        normed_embs = user_emb_final / user_emb_final.norm(dim=-1, keepdim=True)
        normed_embs = normed_embs.flatten(1, 2)
        return {'user_embed_final_64d_norm': normed_embs}

    @torch.no_grad()
    def compute_user_embedding(self, batch, note_id2idx, raw_embeds, device):
        """
        batch: collated batch return {'user_ids', 'note_seqs_homefeed', ...}, batch list of list
        note_id2idx: dict, note_raw_id(str) -> token id(int)
        item_emb_tokens: embedding table (nn.Embedding/nn.Parameter/np.ndarray)
        item_llm: forward item embedding
        device: cuda/cpu
        """
        # import pdb;pdb.set_trace()
        raw_embeds = np.array(raw_embeds)
        input_dim = raw_embeds.shape[1]
        lastns_homefeed = np.array(batch["note_seqs_homefeed"])
        lastns_ads = np.array(batch["note_seqs_ads"])
        attention_mask_homefeed = (lastns_homefeed != '-1')
        attention_mask_ads = (lastns_ads != '-1')
        attention_mask_homefeed = torch.from_numpy(attention_mask_homefeed).int().to(device)
        attention_mask_ads = torch.from_numpy(attention_mask_ads).int().to(device)
        batch_size, seq_len_ads = attention_mask_ads.shape
        batch_size, seq_len_homefeed = attention_mask_homefeed.shape
        features_homefeed = torch.zeros([batch_size, seq_len_homefeed, input_dim], dtype=torch.float32, device=device)
        features_ads = torch.zeros([batch_size, seq_len_ads, input_dim], dtype=torch.float32, device=device)
        
        all_exist_flag = []
        for idx in range(batch_size):
            cur_user_lastn_homefeed = lastns_homefeed[idx]
            cur_user_lastn_ads = lastns_ads[idx]
            
            for note_idx, note_id in enumerate(cur_user_lastn_homefeed):
                if note_id == '-1':
                    continue
                if note_id in note_id2idx:
                    cur_note_embed = raw_embeds[note_id2idx[note_id]]
                    cur_note_embed = torch.from_numpy(cur_note_embed).float().to(device)
                    features_homefeed[idx, note_idx] = cur_note_embed

            for note_idx, note_id in enumerate(cur_user_lastn_ads):
                if note_id == '-1':
                    continue
                if note_id in note_id2idx:
                    cur_note_embed = raw_embeds[note_id2idx[note_id]]
                    cur_note_embed = torch.from_numpy(cur_note_embed).float().to(device)
                    features_ads[idx, note_idx] = cur_note_embed

        user_lastn_raw_features = torch.cat([features_homefeed, features_ads], dim=1).to(device)
        attention_mask = torch.cat([attention_mask_homefeed, attention_mask_ads], dim=1).to(device)

        if hasattr(self, 'input_embedding_projector'):
            user_lastn_raw_features = self.input_embedding_projector(
                user_lastn_raw_features.to(dtype=self.input_embedding_projector[0].weight.dtype)
            )
        else:
            user_lastn_raw_features = user_lastn_raw_features.to(dtype=torch.bfloat16)

        if self.add_item_action_embed:
            user_lastn_action_features = torch.zeros_like(user_lastn_raw_features)
            
            action_seqs_homefeed = batch.get('action_seqs_homefeed', [])
            action_seqs_ads = batch.get('action_seqs_ads', [])
            
            for idx in range(batch_size):
                if idx < len(action_seqs_homefeed):
                    for pos, actions in enumerate(action_seqs_homefeed[idx]):
                        if pos < seq_len_homefeed and actions:
                            for action_index, action in enumerate(actions):
                                if action and action_index < len(self.item_action_embedding):
                                    user_lastn_action_features[idx][pos] += self.item_action_embedding[action_index]
                
                if idx < len(action_seqs_ads):
                    for pos, actions in enumerate(action_seqs_ads[idx]):
                        if pos < seq_len_ads and actions:
                            for action_index, action in enumerate(actions):
                                if action and action_index < len(self.item_action_embedding):
                                    user_lastn_action_features[idx][seq_len_homefeed + pos] += self.item_action_embedding[action_index]
            
            final_user_inputs = user_lastn_raw_features + user_lastn_action_features
        else:
            final_user_inputs = user_lastn_raw_features

        if self.add_hour_embed:
            hour_labels_homefeed = batch.get('hour_labels_homefeed', [])
            hour_labels_ads = batch.get('hour_labels_ads', [])
            
            if hour_labels_homefeed and hour_labels_ads:
                hour_labels_homefeed = torch.tensor(hour_labels_homefeed, device=device)
                hour_labels_ads = torch.tensor(hour_labels_ads, device=device)
                
                hour_embeddings_homefeed = self.hour_embeddings(hour_labels_homefeed)
                hour_embeddings_ads = self.hour_embeddings(hour_labels_ads)
                hour_embeddings_combined = torch.cat([hour_embeddings_homefeed, hour_embeddings_ads], dim=1)
                
                final_user_inputs = final_user_inputs + hour_embeddings_combined.to(device)

        if self.add_position_embed:
            pos_embeddings_homefeed = self.get_positional_embeddings(seq_len_homefeed, device)
            pos_embeddings_ads = self.get_positional_embeddings(seq_len_ads, device)
            
            pos_embeddings_homefeed = pos_embeddings_homefeed.unsqueeze(0).expand(batch_size, -1, -1)
            pos_embeddings_ads = pos_embeddings_ads.unsqueeze(0).expand(batch_size, -1, -1)
            combined_pos = torch.cat([pos_embeddings_homefeed, pos_embeddings_ads], dim=1)
            
            final_user_inputs = final_user_inputs + combined_pos.to(device)

        features = final_user_inputs

        if self.query_nums > 0:
            N = features.shape[0]
            query_embedding = self.query(torch.arange(self.query_nums, device=device))
            features = torch.cat([features, query_embedding.unsqueeze(0).repeat(N, 1, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((N, self.query_nums), 
                                                                dtype=attention_mask.dtype, device=device)], dim=1)

        # User LLM encoding
        user_hidden_states = self.user_llm(inputs_embeds=features, attention_mask=attention_mask).hidden_states[-1]
        user_embed_64d = self.note_embedding_head(user_hidden_states)
        user_embed_64d = self.output_mlp(user_embed_64d)

        if self.query_nums > 0:
            user_embed_final_64d = user_embed_64d[:, -self.query_nums:]
        else:
            user_embed_final_64d = user_embed_64d[:, -1:]

        # Norm
        user_embed_final_64d_norm = user_embed_final_64d / user_embed_final_64d.norm(dim=-1, keepdim=True)
        user_embed_final_64d_norm = user_embed_final_64d_norm.flatten(1, 2)  # [bs, nq*64]
        return {'user_embed_final_64d_norm': user_embed_final_64d_norm}
    
    @torch.no_grad()
    def compute_user_embedding_deploy(self, lastns, actions, note_id2idx, raw_embeds, device):
        import json
        raw_embeds = np.array(raw_embeds)
        lastns = [json.loads(lastn["lastn"]) for lastn in lastns]
        
        def process_feed(feed_key, length):
            batched_feed = []
            for lastn in lastns:
                if feed_key not in lastn:
                    batched_feed.append(['-1'] * length)
                else:
                    feed = lastn[feed_key][:length][::-1]
                    note_ids = [t["note_id"] for t in feed] + ['-1'] * (length - len(feed))
                    batched_feed.append(note_ids)
            return np.array(batched_feed)
        
        lastns_homefeed = process_feed("homefeed", 96)
        lastns_ads = process_feed("ads", 32)
        attention_mask_homefeed = torch.from_numpy(lastns_homefeed != '-1').int()
        attention_mask_ads = torch.from_numpy(lastns_ads != '-1').int()
        batch_size, seq_len_homefeed = attention_mask_homefeed.shape
        batch_size, seq_len_ads = attention_mask_ads.shape
        features_homefeed = torch.zeros([batch_size, seq_len_homefeed, self.item_llm.config.hidden_size], dtype=torch.bfloat16)
        features_ads = torch.zeros([batch_size, seq_len_ads, self.item_llm.config.hidden_size], dtype=torch.bfloat16)
        
        for idx in range(batch_size):
            
            for note_idx, note_id in enumerate(lastns_homefeed[idx]):
                if note_id != '-1' and note_id in note_id2idx:
                    cur_note_embed = torch.from_numpy(raw_embeds[note_id2idx[note_id]]).bfloat16()
                    features_homefeed[idx, note_idx] = cur_note_embed
            
            for note_idx, note_id in enumerate(lastns_ads[idx]):
                if note_id != '-1' and note_id in note_id2idx:
                    cur_note_embed = torch.from_numpy(raw_embeds[note_id2idx[note_id]]).bfloat16()
                    features_ads[idx, note_idx] = cur_note_embed
        
        features_homefeed = features_homefeed.to(device)
        features_ads = features_ads.to(device)
        user_model_inputs = torch.concat([features_homefeed, features_ads], dim=1)
        attention_mask = torch.concat([attention_mask_homefeed, attention_mask_ads], dim=1)
        if self.query_nums > 0:
            N = user_model_inputs.shape[0]
            query_embedding = self.query(torch.arange(self.query_nums, device=user_model_inputs.device))
            user_model_inputs = torch.cat([user_model_inputs, query_embedding.unsqueeze(0).repeat(N, 1, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((N, self.query_nums), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        user_hidden = self.user_llm(inputs_embeds=user_model_inputs, attention_mask=attention_mask).hidden_states[-1]
        user_embed_64d = self.note_embedding_head(user_hidden)
        user_embed_64d = self.output_mlp(user_embed_64d)
        if self.predict_action:
            action_logits = self.action_predict_head(user_hidden)[:, -1]
        else:
            action_logits = None
        
        user_embed_final_64d = user_embed_64d[:, -self.query_nums:] if self.query_nums > 0 else user_embed_64d[:, -1]
        normed_embs = user_embed_final_64d / user_embed_final_64d.norm(dim=-1, keepdim=True)
        return normed_embs.tolist()

    @torch.no_grad()
    def compute_item(self, interaction):
        pos_input_ids, pos_position_ids, attention_mask = (
            interaction['pos_input_ids'], interaction['pos_position_ids'], interaction['pos_attention_mask']
        )
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, attention_mask, self.item_emb_tokens, self.item_llm, max_batch_size=512)
        embed = self.latent_proj_decoder(self.latent_proj_encoder(pos_embedding)) if self.AE_compress_dim > 0 else pos_embedding
        embed_64d = self.note_embedding_head(pos_embedding)
        return embed, embed_64d
