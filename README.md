> **本次改动（64d → 64d 用户塔训练）说明**
>
> 本次改造保留了 RedSeqRec 原始训练框架（Qwen2.5-1.5B 用户塔 + 3 query 多兴趣 + V0 风格的 per-user recall@K 在线评估），仅做了**适配自有数据**和**输入/输出对齐到 64 维**所必需的最小修改，目的是利用预训练好的 1.5B 用户塔权重，单独 fine-tune **用户塔**（item 侧使用预先算好的 64 维 embedding，不再做 item 塔联合训练）。
>
> ## 一、关键改动点
>
> 1. **数据加载（自有数据接入）**
>    - 数据协议对齐 `RedSeqRecV0Simple`：每行一个用户，`history` 字段中以 `label ∈ {seq, pos}` 区分输入序列与目标尾段；也兼容 datav2（数组形式 `content_ids`）。
>    - 通过 `dataset_type: v0_aligned` 走 [v0_aligned_dataset.py](REDRec/data/dataset/v0_aligned_dataset.py)，从 `v0_train_jsonl` 流式读取训练样本，从 `v0_eval_jsonl` 抽取 `eval_users=50000` 个 hold-out 用户作为评估集。
>    - Item embedding 池来自 `v0_embedding_dir`（`cids.npy + embeddings.bin + meta.json` 的 64d memmap），训练 / 评估共享同一份 OS page cache，零拷贝。
>    - `min_history_len=20`、`target_tail_len=10`：每个用户最后 10 个 pos 作为预测目标，剩余作为输入序列（与 V0Simple 完全一致，便于和小模型方案对齐对比）。
>
> 2. **输入 / 输出对齐到 64 维**
>    - 输入 adapter：`Linear(64 → 1536) + GELU + LayerNorm(1536)`，把 64d item embedding 升维进入 Qwen2.5-1.5B。
>    - 输出 adapter：`ProjectionHead(1536 → 64, residual + LN)`，最终 `output_mlp = Identity`，`user_output_dim = user_final_dim = 64`，与 item 池的 64d 对齐做点积召回。
>    - 配置开关：`model.precomputed_input_dim: 64` / `user_output_dim: 64` / `user_final_dim: 64`。
>    - **不做 item 塔联合训练**：`item_llm_init: false`，item 侧只用预计算 emb；`user_llm_init: true` 加载预训练用户塔权重。
>
> 3. **评测方式（per-user recall@K + hit_rate@K）**
>    - 训练循环里挂了 V0 风格的在线评估（`Trainer._run_v0_eval`）：每 `training.eval_interval` 步对 50000 个 hold-out 用户做一次召回，对 **完整 item 池** 计算 `top1 / top50 / top100 / top500` 的 per-user recall 和 hit_rate。
>    - 同时输出 4 个零参数 baseline（`mean_pool / last_pool / last8_pool / last32_pool`）便于横向对比；它们与训练步数无关，只在第一次 eval 时计算并缓存。
>    - 训练正式开始前会先跑一轮 step-0 baseline eval（随机初始化 redrec + 4 个 baseline），方便观察 lift。
>
> 4. **TensorBoard**
>    - 训练 / 评测的所有标量都写入 TensorBoard：`loss`、`lr`、`nce_top{1,10,50,100,500}_acc`、`eval_recall/{strategy}_top{k}`、`eval_hit_rate/{strategy}_top{k}`。
>    - 当 `saver.v0_style: true` 时，TB 目录为 `<log_dir>/<saved_model_name>_<RUN_TS>/tb/`（与 V0Simple 一致，便于多次 run 对照）。
>
> 5. **Checkpoint Top-K 保留**
>    - `training.ckpt_keep_top_k: 5`：按 `redrec.top500_recall` 自动保留 Top-5 ckpt，其余被裁剪；最后一步的 ckpt 永远不裁。
>
> ## 二、目录与产物路径
>
> ```text
> 训练输入数据：
>   /group/40094/jingweidong/user_sequential_feature_recall/qbfeed_action_flow/
>     ├── user_latest/train.jsonl        # 训练用户序列（v0_train_jsonl）
>     ├── user_latest/test.jsonl         # 评估用户序列（v0_eval_jsonl）
>     └── preprocessed64d/               # 64d item embedding memmap（cids.npy/embeddings.bin/meta.json）
>
> 预训练用户塔权重：
>   /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/eval/pre_trained_ckpts/Qwen2.5-1.5B
>
> 训练产物输出（saver.output_dir）：
>   /apdcephfs_gy4/share_303218624/jingweidong/output_xhs/
>     ├── redrec_64d_3query_stage1_<RUN_TS>/    # stage1 输出根目录（v0_style）
>     │     ├── tb/                              # TensorBoard 事件文件
>     │     ├── checkpoint-<step>/               # DeepSpeed ZeRO 分片 ckpt（按 top500_recall Top-5 保留）
>     │     └── ...
>     └── redrec_64d_3query_stage2_<RUN_TS>/    # stage2 输出根目录（同上结构）
> ```
>
> ## 三、如何执行训练
>
> 环境：`conda activate redseqrec`（已验证可跑通训练 / 测评的环境）。所有命令都在仓库根目录执行。
>
> ### 3.1 Stage1：冻结 user_llm，仅训 adapter / query / logit_scale
>
> - 配置文件：[config/train_64d_stage1.yaml](config/train_64d_stage1.yaml)
> - 启动脚本：[start_train_64d_stage1.sh](start_train_64d_stage1.sh)
> - 关键设置：ZeRO-2、`lr=1e-3`（cosine + 2k warmup）、`total_step=100000`、`eval_interval=save_step=1000`、每 rank micro batch=8。
>
> ```bash
> # 8 卡完整训练
> bash start_train_64d_stage1.sh
>
> # 单卡冒烟（200 步）
> DEBUG=1 SANITY_STEPS=200 bash start_train_64d_stage1.sh
>
> # 8 卡跑 500 步快速验证流程
> SANITY_STEPS=500 bash start_train_64d_stage1.sh
> ```
>
> ### 3.2 Stage2：解冻 user_llm，与 adapter 联合 fine-tune
>
> - 配置文件：[config/train_64d_stage2.yaml](config/train_64d_stage2.yaml)
> - 启动脚本：[start_train_64d_stage2.sh](start_train_64d_stage2.sh)
> - 关键设置：ZeRO-3 + grad ckpt、`lr=2e-5`（cosine + 2k warmup）、`total_step=100000`、每 rank micro batch=4 / accum=8（有效 bs=32/rank）。
> - **必填**：通过 `STAGE1_CKPT` 环境变量指定 stage1 输出的最佳 ckpt 目录（或预转的 fp32 `pytorch_model.bin`）。
>
> ```bash
> # 8 卡完整训练（必须传 STAGE1_CKPT）
> export STAGE1_CKPT=/apdcephfs_gy4/share_303218624/jingweidong/output_xhs/redrec_64d_3query_stage1_<RUN_TS>/checkpoint-<step>
> bash start_train_64d_stage2.sh
>
> # 单卡冒烟
> DEBUG=1 SANITY_STEPS=200 STAGE1_CKPT=$STAGE1_CKPT bash start_train_64d_stage2.sh
> ```
>
> ### 3.3 查看训练过程
>
> ```bash
> tensorboard --logdir /apdcephfs_gy4/share_303218624/jingweidong/output_xhs/redrec_64d_3query_stage1_<RUN_TS>/tb
> ```
>
> 关注：
> - `loss`、`lr`、`nce_top{1,10,50,100,500}_acc`：训练拟合情况；
> - `eval_recall/redrec_top{1,50,100,500}`、`eval_hit_rate/redrec_top{1,50,100,500}`：用户塔在 hold-out 集上的召回能力，对比同图的 `mean_pool / last_pool / last8_pool / last32_pool` 4 条 baseline。
>
> ---
>
> 以下是原项目（RedSeqRec）的官方 README，未做修改。
>
> ---
>
## Cross-Scenario Unified Modeling of User Interests at Billion Scale  

![intro](intro.png)

https://arxiv.org/abs/2510.14788

This repository contains the Qwen-adapted implementation, tested on the Qwen 2.5 series. It achieves stable convergence and demonstrates superior performance compared to the Llama version under identical training steps.

## Model and Data
### **Model:** 
Modelscope: [Red-MMU-Rec-Multiscene-Qwen2.5-1.5b](https://modelscope.cn/models/xumanjie/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b)   
Huggingface: [Red-MMU-Rec-Multiscene-Qwen2.5-1.5b](https://huggingface.co/RedMMURec/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b)  
(Apache License Version 2.0)

### **Data:** 
Modelscope: [Red-MMU-Data](https://modelscope.cn/datasets/xumanjie/Red-MMU-Data)  
Huggingface: [Red-MMU-Data](https://huggingface.co/datasets/RedMMURec/Red-MMU-Data)
- Training Data: 
Contains large-scale user–item interaction histories collected from 1.08 million users, including `note_embeddings` and `multiscene_lastn` parquet files for pretraining and fine-tuning multimodal recommendation models. (CC BY-NC-ND 4.0)

- Test Data: Includes user_embedding, item_embedding, and user_lastn.json for evaluation. (CC BY-NC-ND 4.0)
Includes `user_embedding`, `item_embedding`, and `user_lastn.json` (CC BY-NC-ND 4.0).

### Training Data: 

1. note_embeddings

| Column              | Type          | Description                                                                                               |
| ------------------- | ------------- | --------------------------------------------------------------------------------------------------------- |
| `encrypted_note_id` | `string`      | Hashed unique identifier of a note.                                                                       |
| `note_idx`          | `int64`       | Numerical index used for cross-referencing.                                                               |
| `note_feature`      | `list<float>` | Embedding vector (64D) representing the note’s latent semantics or visual-textual features. |

2. multiscene_lastn

| Column              | Type           | Description                                               |
| ------------------- | -------------- | --------------------------------------------------------- |
| `encrypted_user_id` | `string`       | Hashed user identifier.                                   |
| `user_idx`          | `int64`        | Integer index of the user.                                |
| `homefeed_list`     | `list<struct>` | Sequence of interactions in homefeed context. |
| `ads_list`          | `list<struct>` | Sequence of interactions with ads.            |

note_idx serves as the join key between note_embeddings and both lists (homefeed_list / ads_list). 

#### Test Data:

Includes user_embedding, item_embedding, and user_lastn.json for evaluation. (CC BY-NC-ND 4.0)
Includes `user_embedding`, `item_embedding`, and `user_lastn.json`.

Due to company policy, we can only open-source a small portion of the notes from Xiaohongshu; specifically, all notes in the test set (~0.8 million) are publicly available, and you can view each note at https://www.xiaohongshu.com/explore/{note_id}
, while the remaining notes in the training data are released as their semantic embeddings only.


## Replicating the Results

1. Download the pretrained checkpoints (`pre_trained_ckpts`) and data files as listed above.
2. Place all the files under the `eval` folder. The folder structure should be like: 

    ```text
    eval/
    ├── emb/
    │   └── demo_multiscene/
    │       ├── base_item_embedding_pr.pkl
    │       ├── lastn_item_embedding_pr.pkl
    │       └── user_embedding_pr.pkl
    ├── pre_trained_ckpts/
    │   ├── Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b
    │   └── Qwen2.5-1.5B
    └── user_lastn.json
    ```
3. Run the evaluation script:

    ```bash
    python eval.py --tag_name demo_multiscene
    ```

The results from this repository may be slightly higher than those reported in the original paper, as outdated records have been removed from the recall pool.

## Further Instructions
### 0: Environment Setup
1. docker：cuda12.4-ofed5.8-nccl2.20.5-torch2.3-2.5
2. `pip install -r requirements.txt`

### 1: Training
```shell
bash start_train.sh [args]
```
### 2: Debugging
```shell
DEBUG=1 bash start_train.sh [args]
```

### 3: Testing
Before testing, update the eval section in the config file.

- model_path: Path to model you’re using
- user_eval.user_lastn_feature_root: Path to lastn features

#### Step 1: Extract lastn note features
```shell
cd eval
python generate_lastn_item_embedding.py --config_path xxx --world_size 4 --global_shift 0 --global_rank 0
```

#### step-2: Extract base pool note embeddings
```shell
cd eval
python generate_base_item_embedding.py  --config_path xxx --world_size 4 --global_shift 0 --global_rank 0 
```

#### step-3: Extract user embeddings
(Download the lastn features locally first)

```shell
cd eval
python generate_user_embedding.py  --config_path xxx --gpu_id 0
```

##### Evaluation script
```shell
cd eval
bash eval.sh
```

You can replicate our results in the paper through our pretrained models and embeddings. Download them from []() and run 'bash eval.sh', you should see results like:

```
Processed 7811 user embeddings
Processed 778111 note embeddings
Note embedding shape: torch.Size([778111, 64])
Valid users for evaluation: 7811
Batch processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [00:02<00:00, 11.82it/s]
------- Compatible Optimized Results -------
Total valid users: 7811
NDCG@10: 0.0180
NDCG@100: 0.0386
NDCG@1000: 0.0653
HR@10: 0.0699
HR@100: 0.2310
HR@1000: 0.5033
MRR: 0.0304
```

## Reference
https://github.com/bytedance/HLLM  
https://github.com/meta-recsys/generative-recommenders  
https://github.com/QwenLM/Qwen  

## Cite Us
```text
@article{xu2025cross,
  title={Cross-Scenario Unified Modeling of User Interests at Billion Scale},
  author={Xu, Manjie and Chen, Cheng and Jia, Xin and Zhou, Jingyi and Wu, Yongji and Wang, Zejian and Zhang, Chi and Zuo, Kai and Chen, Yibo and Tang, Xu and Hu, Yao and Zhu, Yixin},
  journal={arXiv preprint arXiv:2510.14788},
  year={2025}
}
```
