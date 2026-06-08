# RedSeqRec 用户塔训练文档

> **目标**：训练一个用户塔，把"用户最近的正向行为序列"映射成 64 维向量，
> 与线上 item 检索池**完全对齐**——线上池子中每个 item 也是把 512 维多模态向量
> 取前 64 维再做 L2 归一化得到的，因此训练完的用户塔可以直接和现有 item 池
> 做余弦相似度检索召回，**无需重刷海量 item embedding**。

---

## 1. 训练范式总览

| 维度 | 说明 |
|---|---|
| 任务类型 | 序列推荐 / 双塔召回 |
| 用户侧 | 1.5B Qwen2.5 + 序列输入投影层 + interest query，输出 64 维 |
| Item 侧 | **不可训练**：512 维 → 取前 64 维 → L2 归一化 |
| 监督信号 | InfoNCE（in-batch + 全局负采样池） |
| 训练精度 | bfloat16 mixed |
| 分布式 | DeepSpeed ZeRO-3，单机 8 卡 |
| 入口 | [run.py](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/run.py) |
| 配置 | [config/precomputed_embedding_train.yaml](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/config/precomputed_embedding_train.yaml) |

---

## 2. 输入数据格式

### 2.1 用户行为序列：JSONL

每行一个用户：

```json
{
  "qimei36": "<user_id>",
  "history": [
    {"cid": "<note_id>", "ts": 1716700000, "label": "pos"},
    {"cid": "<note_id>", "ts": 1716700123, "label": "pos"},
    ...
  ]
}
```

字段说明：

| 字段 | 类型 | 含义 |
|---|---|---|
| `qimei36` | str | 用户唯一 ID（仅日志和 debug 用） |
| `history` | list | 行为列表，**不要求预排序**，代码内部会按 `ts` 升序 |
| `history[*].cid` | str | 物料（笔记/视频）ID，必须能在 item embedding npy 中找到 |
| `history[*].ts` | int | UNIX 秒级时间戳 |
| `history[*].label` | str | 标签，目前只采纳 `"pos"` 作为正样本 |

> 后续扩展（见 §7）：将引入 `"hard_neg"` 作为困难负样本标签。

当前试跑数据：
- 路径：`eval/data_prep/train_users_500k.jsonl`
- 规模：50w 用户
- 用途：**冒烟跑 / 调参 / 验证 loss 走势**，跑通后会替换为百万–千万级正式数据。

### 2.2 物料向量库：NPY

```python
data = np.load(path, allow_pickle=True).item()
data["cid"]        # np.ndarray[str], shape=(N,)
data["embedding"]  # np.ndarray[float32], shape=(N, 512)
```

当前文件：`embedding/cc_ai_video_multimodal_embedding.npy`，包含 1,443,475 个物料、维度 512。

> ⚠️ **JSONL 中出现但不在 npy 中的 cid 会被静默丢弃**，请保证两份数据时间一致。

---

## 3. 正负样本构造

代码位置：[REDRecPrecomputedEmbeddingDataset](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/REDRec/data/dataset/dataset.py)

### 3.1 单用户样本生成（`_select_training_window`）

针对一个用户的 history：

1. **过滤**：仅保留 `label == "pos"` 且 cid 在 npy 中的记录；
2. **排序**：按 `ts` 升序；
3. **截断/采样**：
   - 若 `len > max_seq_len(=96)`：随机抽 96 条再排序（论文官方做法）；
   - 否则取最近 96 条；
4. **窗口拆分**：
   - 末尾 `window_pos=3` 条 → **正样本目标**（target）
   - 前面剩余的部分 → **输入上下文序列**（最长 93）
5. **过滤**：若上下文 < `min_history_len=4`，整条样本丢弃。

> ✦ 核心思想：用前面的 click 序列预测最后 3 个 click，监督信号天然对齐"召回下一个用户会点的 item"。

### 3.2 负样本（当前版本：随机负采样）

代码位置：`_sample_negative_embeddings`

- 每个 GPU 每步 **从全量物料池（1.44M）随机抽 `neg_samples_per_gpu = 400` 个**；
- 训练时通过 `all_gather` 拼接 8 卡，**每个 query 实际有 3200 负样本**；
- 用 `nce_thres = 0.98` 屏蔽假阴性：若某负样本与正样本余弦 > 0.98，将其 logit 置为最小，避免学反；
- 由于物料池 1.44M ≫ 400，撞采概率近似 0，等效"无放回"。

### 3.3 Batch 组装（`_build_batch`）

每个 batch 输出：

| key | shape | 含义 |
|---|---|---|
| `precomputed_input_embeds` | (B, 93, 512) | 上下文序列 item 向量，左侧补零 |
| `precomputed_attention_mask` | (B, 93) | 1=有效 item，0=padding |
| `precomputed_target_embeds` | (B, 3, 512) | 末尾 3 个正样本向量 |
| `precomputed_target_mask` | (B, 3) | 1=有效正样本 |
| `precomputed_neg_embeds` | (400, 512) | 当前 GPU 抽到的负池 |
| `user_ids` / `target_cids` | list | 仅 debug 用 |

`B = train_batch_size = 8`（单卡）。

---

## 4. 模型结构

代码位置：[REDRec](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/REDRec/model/redrec.py)

```
input: (B, 93, 512)  ← 用户历史 item 多模态向量
   │
   ├─ input_embedding_projector:  Linear(512→1536) → GELU → LayerNorm    [全新初始化]
   │
   ├─ + 3 个可学习 interest query (1, 3, 1536)                            [LLM ckpt 加载]
   │
   ├─ user_llm: Qwen2.5-1.5B（28 层 Transformer，hidden=1536）            [LLM ckpt 加载]
   │      └─ 启用 gradient_checkpointing，启用 FlashAttention
   │
   ├─ note_embedding_head:  ProjectionHead(1536→64, 含残差+LayerNorm)     [LLM ckpt 加载]
   │
   ├─ 取末尾 3 个位置 → (B, 3, 64) → L2 归一化  ← 用户向量
   │
   └─ 与 (B, 3, 64) 正样本 + (3200, 64) 负池 计算 InfoNCE
```

### 4.1 Item 侧

**完全没有可训练参数**。流程：

```
原始 512 维 → emb[:, :64]  →  L2 norm  →  64 维 item 向量
```

这一步与线上检索池构造方式完全一致，确保**无需重刷 item embedding**。

---

## 5. 损失函数

### 5.1 Cluster-based matching（query ↔ target 配对）

3 个 user query 与 3 个 target 之间用**匈牙利算法**做最优一对一配对（按 cosine），
避免硬性按位置对齐导致的不必要错配。

### 5.2 InfoNCE 主损失

对每对 `(matched_user_q, target)`：

```
pos_logit  = cos(q, target)                              # 标量
neg_logits = q · neg_pool^T                              # (3200,)
neg_logits[ cos(target, neg) > 0.98 ] = -inf             # 屏蔽假阴性
logits = [pos_logit, neg_logits] * exp(logit_scale)      # (3201,)
loss   = CrossEntropy(logits, label=0)                   # 正样本永远在第 0 位
```

`logit_scale` 是可学习温度（初始 `log(1/0.07) ≈ 2.66`），上限 clamp 到 `ln(100)`。

### 5.3 监控指标

每步打印 / 写入 TensorBoard：

| 指标 | 含义 |
|---|---|
| `loss` | InfoNCE 主损失 |
| `nce_samples` | 实际生效的负样本数（被 mask 掉的不算） |
| `nce_top1_acc` | 正样本排在第 1 的比例 |
| `nce_top10_acc` | 正样本排在前 10 的比例 |
| `nce_top100_acc` | 正样本排在前 100 的比例 |

---

## 6. 可训练参数 & 优化策略

### 6.1 哪些参数会更新

| 模块 | 初始化 | 是否更新 | 学习率 |
|---|---|---|---|
| `input_embedding_projector` | **从零（标准正态）** | ✅ | 1e-4（基础 × 10） |
| `query`（3 × 1536） | Red-Mmu-Rec ckpt | ✅ | 1e-5 |
| `user_llm`（~1.5B 参数） | Red-Mmu-Rec ckpt | ✅ | 1e-5 |
| `note_embedding_head` | Red-Mmu-Rec ckpt | ✅ | 1e-5 |
| `logit_scale`（标量） | log(1/0.07) | ✅ | 1e-5 |
| **item_llm** | — | ❌ **未构建** | — |

> ckpt 加载时会出现 339 个 `item_llm.*` 的 unexpected key，**这是预期行为**。
> 因为 item 侧使用预计算向量，无需 LLM。

### 6.2 优化器与调度

| 项 | 值 |
|---|---|
| 优化器 | AdamW |
| 基础学习率 | 1e-5 |
| 投影层学习率倍率 | 10×（仅 `input_embedding_projector` 享受） |
| 权重衰减 | 0.01 |
| Scheduler | cosine，warmup 400 步 |
| 梯度裁剪 | 1.0 |
| 累积步数 | 1（不开累积） |
| 总步数（试跑） | 20000 |

### 6.3 分布式与显存

| 项 | 值 |
|---|---|
| 框架 | Lightning Fabric + DeepSpeed ZeRO-3 |
| 精度 | bf16-mixed |
| Gradient Checkpointing | 开启 |
| 单卡 batch | 8 |
| 卡数 | 8 |
| 等效正样本对/步 | 8 × 8 × 3 = **192** |
| 等效负样本/查询 | 8 × 400 = **3200** |

---

## 7. 后续扩展：困难负样本

> 业务侧将提供两类困难负样本：
> 1. **曝光未点击**（impression but no click）
> 2. **点击但观看时长 < 5s**（low-dwell click）
>
> 两者都是"用户看到了却不喜欢"的强信号，比随机负采样更接近线上分布。

### 7.1 推荐的接入方式

**方案 A（推荐）：在 JSONL 中扩展 label，dataset 抽样时混入**

JSONL 增加 label：

```json
{
  "qimei36": "u123",
  "history": [
    {"cid": "n1", "ts": ..., "label": "pos"},
    {"cid": "n2", "ts": ..., "label": "hard_neg"},   // ← 新增
    ...
  ]
}
```

数据集类需要小幅改动（修改点已标注，便于后续接入）：

1. `_select_training_window`：保留 pos 时也单独收集 `hard_neg` 列表；
2. `_build_batch`：增加 `precomputed_hard_neg_embeds`，每个用户给定 K 个困难负；
3. `forward_precomputed_embedding`：将困难负 logits 与随机负 logits **拼接**后再做 CE。
   - 形如 `logits = [pos, hard_negs(K), random_negs(3200)]`
   - 同样要用 `nce_thres=0.98` 屏蔽对应的假阴性。

**方案 B（轻量过渡）：先在数据准备阶段把困难负 cid 混进随机负池**

让 `_sample_negative_embeddings` 以 `(p, 1-p)` 比例从困难负池和全量池采样，
不改 batch 结构。优点是改动小，缺点是无法做 user-specific 困难负。

**强烈建议走方案 A**：困难负样本必须是"该用户曾经曝光但负向反馈"的那个 item，做成 user-level 才有意义。

### 7.2 推荐参数

- 每用户困难负数量：`K = 4 ~ 8`
- 困难负 loss 权重：先与随机负**等权混合**（直接拼 logits），观察后再考虑加权（例如 hard_neg 权重 1.5）；
- 仍然保留 3200 全局随机负：避免模型只学"区分用户内部困难负"，而忘了"区分整池"。

---

## 8. 运行步骤

### 8.1 一次性准备：把 ZeRO 切片 ckpt 转单文件

加载 ZeRO 切片每次启动要 ~25 分钟，转成单文件后只需 ~13 分钟（torch.load）。

```bash
python utils/zero_to_fp32.py \
  /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/eval/pre_trained_ckpts/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b/checkpoint \
  /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/eval/pre_trained_ckpts/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b/pytorch_model.bin
```

`run.py` 会自动判断：传入路径如果是文件就走 `torch.load`，是目录就走 ZeRO 重组。

### 8.2 启动训练

```bash
cd /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec

torchrun --nproc_per_node=8 --master_port=29500 \
  run.py --config_path config/precomputed_embedding_train.yaml
```

输出位置：

| 内容 | 路径 |
|---|---|
| 文本日志 | `expr/precomputed_cc_ai_video_multimodal_user_tower/logger/<timestamp>.log` |
| TensorBoard | `expr/precomputed_cc_ai_video_multimodal_user_tower/tensorboard/<timestamp>/` |
| 模型 ckpt | `expr/precomputed_cc_ai_video_multimodal_user_tower/checkpoint-<step>/` |

### 8.3 实时查看 loss

```bash
LOG=$(ls -t expr/precomputed_cc_ai_video_multimodal_user_tower/logger/*.log | head -1)
grep "lr:" $LOG | awk -F'loss: ' '{print $2}' | awk -F',' '{print $1}' | tail -50
```

或开 TensorBoard：

```bash
tensorboard --logdir expr/precomputed_cc_ai_video_multimodal_user_tower/tensorboard --port 6006
```

---

## 9. 健康基线（试跑 50w 用户 / 20000 步）

| 阶段 | 步数 | loss 区间 | top1 acc | top10 acc | top100 acc |
|---|---|---|---|---|---|
| 起步 | 0–199 | 7.9 | < 0.05 | < 0.10 | < 0.30 |
| 5k 步 | ~5000 | 6.5 ± 0.2 | ~0.10 | ~0.20 | ~0.45 |
| 15k 步 | ~15000 | 6.0 ± 0.2 | **~0.21** | **~0.36** | **~0.61** |
| 20k 步 | ~20000 | 仍下降中 | 仍上涨 | 仍上涨 | 仍上涨 |

**判断训练是否健康的快速准则：**

- ✅ 200 步内 loss 应该从 7.9 降到 7.2 以下；
- ✅ 5000 步 top1_acc 应该 > 0.08；
- ❌ 出现 NaN / loss 突然抬升 5+ 量级 → 检查 grad clip、检查负池是否有重复正样本；
- ❌ top1_acc 长期为 0 → 检查 item 侧的 64 维归一化是否生效。

---

## 10. 配置文件关键字段速查

[config/precomputed_embedding_train.yaml](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/config/precomputed_embedding_train.yaml) 的关键字段：

```yaml
data:
  dataset_type: precomputed_embedding             # ← 必须是这个
  precomputed_user_history_jsonl: <jsonl 路径>     # ← 用户行为
  precomputed_embedding_npy:      <npy 路径>      # ← item 向量库
  precomputed_positive_label: pos                  # ← 哪个 label 当正样本
  precomputed_min_history_len: 4                   # ← 用户最少正样本数
  precomputed_sample_lastn: true                   # ← >96 时随机抽 96
  lastn_max_click_note_num_homefeed: 96            # ← 序列上限
  train_batch_size: 8                              # ← 单卡 batch
  neg_samples_per_gpu: 400                         # ← 单卡负池
  train_num_workers: 8

model:
  model_name: REDRec
  precomputed_input_dim: 512                       # ← item 向量维度
  user_pretrain_dir: <Qwen2.5-1.5B 目录>           # ← 必须本地存在
  query_nums: 3                                    # ← interest query 个数 = target 长度
  window_pos: 10                                   # ← 仅推理用，训练受 query_nums 控制
  use_ft_flash_attn: true
  gradient_checkpointing: true

training:
  total_step: 20000
  eval_step: 999999                                # ⚠️ 当前永不存盘，正式跑请改成 5000
  load_pretrained_model: <pytorch_model.bin 路径>
  optim_args:
    learning_rate: 1.0e-5
    lr_mult_prefix: ['input_embedding_projector']
    lr_mult_rate: 10.0
  scheduler_args:
    type: cosine
    warmup_steps: 400
  strategy: deepspeed
  stage: 3

precision: bf16-mixed
loss: nce
nce_thres: 0.98
```

---

## 11. 常见问题

**Q1：负池里如果碰巧抽到了正样本怎么办？**
答：被 `nce_thres=0.98` 自动屏蔽（cos 大于阈值 → logit 置 -inf），不会学反。

**Q2：为什么 `cluster_based_matching` 走 CPU + scipy？BFloat16 报错怎么办？**
答：scipy 的 `linear_sum_assignment` 不支持 BF16，需要先 `.float().cpu().numpy()`。
代码已经修过这个 bug。

**Q3：模型保存后怎么用？**
答：保存的是 ZeRO 切片，需要先用 `utils/zero_to_fp32.py` 转单文件，
然后修改推理脚本里的 `load_pretrained_model` 指向单文件即可。

**Q4：要不要冻结 LLM？**
答：当前不冻结。理由：item 多模态向量是新的 512 维空间，LLM 需要少量微调
才能匹配。试跑结果显示放开比冻结收敛更好。如显存吃紧可以试 LoRA（`use_lora: true`）。

**Q5：50w 用户跑 20000 步会过拟合吗？**
答：单卡 batch=8 × 8 卡 = 64 用户/步，20000 步 = 128w 用户次曝光，
50w 用户大概看了 2.56 遍。不算严重过拟合，但**正式跑请用百万–千万级数据**。

---

## 12. 数据流性能分析与优化路线

### 12.1 当前实现 & 单步耗时拆解（试跑配置：B=8, seq=93, neg=400/卡）

| 工作 | 量级 | 单步耗时（粗估） |
|---|---|---|
| 读 8 行 jsonl | 8 行 | < 1 ms |
| 排序 + 截窗 96 | 8 个 list | < 1 ms |
| 取 ~800 个 item embedding（dict 查 + 单行拷贝） | 8×(93+3)≈800 | 2–4 ms |
| 随机抽 400 个负样本（`np.random.choice` + 索引） | 400 | 1–2 ms |
| `torch.from_numpy` | 5 个张量 | < 1 ms |
| **数据侧合计** | | **~5–8 ms / step** |
| **GPU 1.5B fwd+bwd（bf16, B=8, seq=93）** | — | **~400–600 ms / step** |

**结论**：数据侧只占 GPU 计算的 1–2%，加上 `num_workers=8` 的预取，**当前不是瓶颈**。
此前 GPU 利用率 30% 是 batch 太小、计算密度低导致的，与数据流无关。

### 12.2 三个隐患（数据规模上去后会暴露）

#### 隐患 A：1.44M item embedding 在每个 worker 进程被复制一份

```python
self.embeddings = np.asarray(data['embedding'], dtype=np.float32)  # ≈ 2.95 GB
self.cid2idx = {cid: idx for idx, cid in enumerate(self.cids)}     # ≈ 200 MB
```

`DataLoader(num_workers=8)` × 8 卡 = **64 个 worker**，理论占用 200GB。
Linux COW 能让 numpy 数组共享物理页，**但 Python dict 一旦被读就触发 refcount 写，全部复制**。
**这是数据量翻倍后最先暴露的问题。**

#### 隐患 B：`_get_embedding` 是逐元素拷贝

每条样本约 99 次 `self.embeddings[self.cid2idx[cid]]`：

- 现在 batch=8 → 800 次/步，5–8ms 还撑得住；
- batch=32 + hard_neg(K=8) → 3500+ 次/步，逼近 30–50ms；
- 如果未来上 LoRA 让 GPU 提速，数据侧就开始拖后腿。

#### 隐患 C：每个 worker 把整个 jsonl 从头读到尾，64 个 worker = 64× 磁盘 IO

```python
for line_no, line in enumerate(f):
    if line_no % self.world_size != self.global_rank: continue
    if (line_no // self.world_size) % nw != worker_id: continue
```

50w 行（< 1GB）无所谓；几千万行（几十 GB）+ 分布式存储 → IO 重复 64×。

### 12.3 优化路线（按优先级）

#### 🔴 P0：item embedding 改 mmap / 共享内存（数据规模一加就要做）

```python
# REDRecPrecomputedEmbeddingDataset.__init__ 里
# 旧：self.embeddings = np.asarray(data['embedding'], dtype=np.float32)
# 新：分两个文件预存盘
#   embedding/cc_ai_video_multimodal.cids.npy      （N,）
#   embedding/cc_ai_video_multimodal.embs.npy      （N, 512）float32
self.embeddings = np.load(embs_path, mmap_mode='r')  # 不进 RAM，多 worker 共享 page cache
self.cids = np.load(cids_path, allow_pickle=True)
```

并把 `cid2idx` 改成 **fork 后只读**的容器（例如 `numpy` 字符串数组 + `np.searchsorted`），
避免 Python dict 触发 COW。

预期收益：64 worker 总内存从 ~200GB 降到 ~3GB（page cache 共享）。

#### 🟡 P1：`_build_batch` 改成一次性 fancy index（hard_neg 接入或 batch 翻倍后做）

```python
# 旧：双重 for 循环，800 次单行拷贝
# 新：
flat_cids = []        # 长度 = B * (context_max_len + target_window_size)
positions = []        # 记录每个 cid 在 (row, col, kind) 的位置
for row, s in enumerate(samples):
    for col, c in enumerate(s['input_cids']):  ...
    for col, c in enumerate(s['target_cids']): ...
flat_idxs = np.fromiter((cid2idx[c] for c in flat_cids), dtype=np.int64)
flat_embs = self.embeddings[flat_idxs]   # 一次性拷贝
# 再 reshape 回 (B, T, 512)
```

预期收益：数据侧耗时降 5–10×。

#### 🟢 P2：jsonl 预切片（数据涨到千万行后做）

数据准备阶段把 jsonl 按 `rank × worker` 切成 64 个小文件：

```bash
python eval/data_prep/split_jsonl_by_shard.py \
  --input train_users.jsonl \
  --num_shards 64 \
  --output_dir shards/
```

dataset 里读 `shards/rank{R}_worker{W}.jsonl`，磁盘 IO 直接降到 1×。

#### 🟢 P2：离线一次性做 ts 排序 + 缺失 cid 过滤

在 data_prep 阶段就把 history 按 ts 排好序、过滤掉 npy 里没有的 cid，
`_select_training_window` 里就只剩 `random.sample` + 切片，再降一半数据侧耗时。

### 12.4 验证是否真为瓶颈的探针

在 [trainer.py](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/REDRec/trainer.py) 训练循环里插桩：

```python
import time
data_t = time.time()
for step, data in enumerate(train_dl):
    data_wait_ms = (time.time() - data_t) * 1000
    t = time.time()
    losses = self.model(data); ...
    gpu_ms = (time.time() - t) * 1000
    if step % 100 == 0:
        self.logger.info(f"data_wait={data_wait_ms:.1f}ms gpu={gpu_ms:.1f}ms ratio={data_wait_ms/gpu_ms:.2%}")
    data_t = time.time()
```

判定：
- `ratio < 5%`  → 完全不是瓶颈，无需优化；
- `ratio 5%–20%` → 可做 P1，但不紧急；
- `ratio > 20%` → 必须做 P0 + P1。

---

## 13. 关于"正样本数量太少"与"变长正样本"

> 你的观察很准：**当前每个用户只取末 3 个 pos 当 target，相比 3200 个负样本，
> 正负比 ≈ 1:1067**。这其实是 InfoNCE 的常态（负样本本来就要远多于正样本，
> 才能学到判别性），但**正样本太少会让"用户实际有几十上百个 pos"的信号没被充分利用**。

### 13.1 增加正样本数量的两种思路

#### 思路 A：固定增大 `query_nums` / `window_pos`

把 `query_nums: 3` 改成 `query_nums: 8` 或 `16`：

| 改动点 | 影响 |
|---|---|
| `model.query_nums` | 可学习 query 数量增加，参数轻微增加（K × 1536 ≈ 25k 参数） |
| `model.window_pos` | 取末尾 K 个作 target |
| 数据侧 `target_window_size` | 跟随 `window_pos`，自动调整 |
| context_max_len | `= max_seq_len - window_pos`，会变短（96-16=80） |
| 计算量 | InfoNCE 是 `K × 3201` logit，K=16 时增长 5× CE 时间，**实际占比 < 5% 总耗时** |
| 收敛 | 正样本对数从 3 → 16，**梯度信噪比显著提升**，loss 应该降得更快 |

**强烈建议正式跑就把 `query_nums` 提到 8 或 16**，几乎零成本但收益明显。

⚠️ 唯一注意：`query_nums` 越大，`min_history_len + target_window_size` 越大，
会过滤掉更多短序列用户。可适当下调 `precomputed_min_history_len`（4 → 2）补偿。

#### 思路 B：变长正样本（每用户用不同 K_i）

这是你问的"如果每个用户用不同的正样本数好不好实现"。

**短答：实现不算难，但有个根本约束——所有用户的 K_i 必须 ≤ 模型固定的 query_nums。**

原因：`query` 是模型里**预先定义**的可学习参数 `(1, K, 1536)`，K 是结构参数；
batch 内每条样本经过 LLM 后会输出 K 个 user embedding。**K 不能 per-sample 不同**。

#### 折中方案（推荐）：固定 K_max + 变长有效目标 + mask

实现思路：

1. **模型侧**：把 `query_nums` 固定成一个上限，比如 `K_max = 16`；
2. **数据侧**：
   - 每个用户根据自己 history 长度决定 `K_i = min(L_i // 4, K_max)`
     （比如序列长 12 取 3 个，长 60 取 15 个，长 200 取 16 个）；
   - `target_embeds` 仍 padding 到 `(B, K_max, 512)`；
   - `target_mask[b, k] = 1 if k < K_i else 0`；
3. **loss 侧**：cluster_based_matching 只在 `target_mask=1` 的位置上做匹配，
   `target_mask=0` 的位置不参与 InfoNCE loss。

这样实现的好处：

| 优点 | 说明 |
|---|---|
| **充分利用长序列用户** | 长 history 用户能贡献 16 个监督信号，短的也不浪费 |
| **GPU 计算图固定** | tensor shape 不变，无 dynamic shape 问题，DeepSpeed/编译器友好 |
| **平均 loss 计算** | 改成按 mask 求平均，避免长用户主导梯度 |

### 13.2 具体代码改动点（变长方案）

需要改 3 处，**改动量很小**：

#### 1️⃣ [dataset.py](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/REDRec/data/dataset/dataset.py) 的 `_select_training_window`

```python
# 旧：固定取末尾 self.target_window_size 个
target_cids = pos_cids[-self.target_window_size:]
input_cids  = pos_cids[:-self.target_window_size]

# 新：根据序列长度动态决定 K_i
seq_len = len(pos_cids)
K_max = self.target_window_size                # 模型上限
K_min = self.config.data.get('precomputed_target_min', 1)
target_ratio = self.config.data.get('precomputed_target_ratio', 0.25)
K_i = max(K_min, min(K_max, int(seq_len * target_ratio)))
target_cids = pos_cids[-K_i:]
input_cids  = pos_cids[:-K_i]
```

`_build_batch` 不用改太多，因为现在的代码已经在用 `target_mask`，
只需要把 `for offset, cid in enumerate(cur_target_cids)` 的循环里
`offset` 从 **0 开始填到 K_i-1**，剩下 `[K_i, K_max)` 自动是 0（因为 `np.zeros` 初始化）。

#### 2️⃣ [redrec.py](/group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/REDRec/model/redrec.py) 的 `forward_precomputed_embedding`

```python
# 现在的 cluster_based_matching 已经是处理 (B, K, D) 的，
# 只需要在配对前把 target_mask=0 的位置剔除：
valid_mask = precomputed_target_mask.bool()  # (B, K)
# 对每个样本独立做匈牙利匹配（已有逻辑），但只在 valid 的位置上配对
# loss 求和后再除以 valid_mask.sum() 而不是 B*K
```

#### 3️⃣ config 增加两个超参

```yaml
data:
  precomputed_target_min: 1
  precomputed_target_ratio: 0.25     # 取末 25% 当 target，但上限 query_nums
```

### 13.3 推荐配置（综合上面两节）

> 给正式跑（百万–千万用户）的推荐配置：

```yaml
model:
  query_nums: 16                     # ← 从 3 提到 16，K_max
  window_pos: 16
data:
  precomputed_min_history_len: 2     # ← 短序列也保留
  precomputed_target_min: 1          # ← 至少给 1 个 target
  precomputed_target_ratio: 0.25     # ← 长序列用户给到 16 个 target
  train_batch_size: 16               # ← 配合 GPU 利用率
  neg_samples_per_gpu: 600           # ← 适当扩大负池
```

**预期收益**（相对当前 K=3, B=8）：
- 等效正样本对/步：3 × 64 = 192 → **16 × 128 = 2048**（约 10×）
- loss 收敛速度：**预计 5000 步达到当前 15000 步的 top10/top100 水平**

### 13.4 关于"正负比 1:1067"的本质

补充澄清：**InfoNCE 的"正负比"和分类任务里说的正负比不是一回事**。
InfoNCE 的目标是让 query 在 N+1 个候选中找到**那一个**正样本，
所以负样本越多越能拉开判别性。1:1000 ~ 1:10000 是 SimCLR/CLIP 等成熟方法的标配，
**正负不平衡不是这里的问题，问题是"每个用户只用 3 个监督信号太浪费"**。

把 K_max 从 3 提到 16 之后，监督信号密度提升，单 user 多视角对比学习，
对长序列用户的特征学习帮助巨大。这才是核心改动。

