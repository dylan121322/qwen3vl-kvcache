# Qwen3VL 图片 KV 缓存 — 使用与维护文档

> 2026-08-11（v2 重写）| 供其他 agent 使用
> 涉及文件：`<ComfyUI根目录>\ComfyUI\comfy\text_encoders\{qwen3vl.py, llama.py}`
> 场景：MiniMax H3 视频生成的 conditioning CLIP（Qwen3-VL-32B，nvfp4 量化，CPU 推理）

## 1. 功能与实测收益

为 H3 的 Qwen3-VL-32B conditioning 编码器实现**图片区间 KV 缓存**：同一组参考图重复编码时，跳过 LLM 层图片 token 的全部计算（q/k/v/o 投影、MLP、attention），只计算文本行与最后一层。

| 场景 | 耗时 | 收益 |
|---|---|---|
| 3 ref 图（4 图整体区间，旧语义）| MISS 332.6s → HIT 201.3s | 省 39% |
| **段2+ 真实场景**（3 ref 固定 + first_frame 每段变）| MISS 316.2s → HIT 224.5s | 省 92s（29%），**每段持续命中** |

输出一致性：**位级一致 0.000e+00**（flash 分段路径 vs 正常路径）。

## 2. 启用方式（默认关闭）

```bash
set QWEN3VL_KV_CACHE=1   # Windows，启动 ComfyUI / h3_runner 前设置
```

- 默认关闭：未设置时行为与修改前完全一致（零侵入）。
- 修改 `qwen3vl.py`/`llama.py` 后需**重启进程**生效。
- 缓存目录：`<缓存目录>（默认 ComfyUI根目录\h3_kv_cache，env QWEN3VL_KV_CACHE_DIR 可覆盖）`（~4.3GB/图组，手动清理，无自动淘汰）。

## 3. 工作原理

### 3.1 布局与因果前提（为什么能缓存）

H3 的 tokenize 布局（`minimax.py`）：

```
[ "<Picture 1>: " + 图1块 + "<Picture 2>: " + 图2块 + ... ] + [ "<Picture k>: " + first_frame块 ] + [ prompt ]
         └────────────── 固定段（ref 图，跨段不变）──────────────┘   └─ 可变段（每段新）─┘
```

- 编码走 `Llama2_.forward`，**无条件因果 mask**（`triu(1)`）；
- 图片 token 在 prompt 之前，**只 attend 前面的固定内容**（标签文本 + 前面图块）；
- mrope position_ids 的图区间只由 (start, grid) 决定，与后续文本无关。

→ **固定段每层的 KV 与输出 hidden 完全独立于 first_frame 与 prompt**（合成验证：跨文本 diff = 0）。这是缓存的理论基础。

### 3.2 为什么"每图独立缓存"不可行（重要设计约束）

图 N 的 layer 0 输入 = 自己的 embeds（可独立），但 **layer 1+ 的输入 = 前一层输出，其中图 N token 的 attention 查询了前面所有图的 KV**（因果）→ **图 N 的每层 KV/hidden 都依赖图 1..N-1 的内容**。

推论：
- 图 N 的缓存 key 必须包含前缀指纹 → 等于把前缀纳入 → **等价于整段一个区间**；
- 任何一张 ref 图变化 → 其后的所有缓存**必然失效**（架构决定，非实现限制）；
- 因此"固定段 + 可变段"的二分是**理论最大可缓存范围**，不是简化妥协。

### 3.3 缓存区间语义（2026-08-11 定版）

**缓存区间 = 首图 start → 倒数第二张图 end（排除末图）**，含图间文本（"<Picture 2>: " 等）。

- **末图 = first_frame**（H3 约定：`also_ref_first_frame` 把首帧追加为最后一个 Picture）→ 每段变化，重算；
- 前 N-1 张固定 ref 图独立缓存 → **first_frame 变化不破坏命中**；
- 单图/无图 → 不缓存（无可缓存的固定部分）；
- key = `md5(区间 embeds 字节) + start + 固定 ref 图 grid 摘要`；
- 段 1（纯 3 ref 无 first_frame）也排除第 3 张（无法区分"末图是否 first_frame"）→ 少缓存 1/3，~30-40s，每部电影一次，可接受。

> **first_frame 约定**：该区间语义基于 H3 布局（first_frame 总是序列最后一张图）。若使用场景布局不同（末图非关键帧、多关键帧、关键帧置首等），可修改 `_image_kv_cache_key` 中的 `end = images[-1]["index"]` 调整"哪些图进入缓存区间"——区间内内容不变即命中，与区间后内容无关。

### 3.4 缓存内容

每个 key 一个文件（torch.save）：`{"kv": [...], "h": [...]}`

| 字段 | 内容 | 3 图组大小 |
|---|---|---|
| `kv` | 50 层 × (k, v) 图区间张量 (1, 8, L, 128)（rope 后）| ~1.2GB |
| `h` | 50 层 × 图区间输出 hidden (1, L, 5120)（deepstack 注入后）| ~3.1GB |

L = 区间 token 数（3 ref 为 6037）。**缓存 h 是必需的**：conditioning 需要全序列输出，图区间 hidden 链必须完整——h 让复用路径跳过图区间的层间传播。

### 3.5 复用路径（HIT）

- 前 49 层：图区间输入 = 缓存 `h[i-1]`；q/k/v 只对文本行（前缀+后缀）投影；图区间 KV 注入（rope 后直接拼）；attention 用优化内核（flash，text 行 q × 全 KV）；图区间输出 = 缓存 `h[i]`；**跳过 deepstack 注入**（缓存 h 已含 deepstack 特征）；
- **最后一层不跳过**（全量计算）：H3 conditioning 需要图区间 hidden 真值输出给 DiT；其 KV 重算，与缓存位级一致。

### 3.6 保存路径（MISS）

首次编码时自建 past_key_values（encode 路径默认不传，否则 Attention 不回传 KV），从每层 Attention 返回的 KV slice 出区间，连同每层输出 hidden 落盘。

## 4. 代码结构

### `comfy/text_encoders/qwen3vl.py`

| 符号 | 说明 |
|---|---|
| `_QWEN3VL_KV_CACHE_ENABLED` | 开关：`QWEN3VL_KV_CACHE=1` |
| `_image_kv_cache_dir` | `<缓存目录>（默认 ComfyUI根目录\h3_kv_cache，env QWEN3VL_KV_CACHE_DIR 可覆盖）` |
| `_image_kv_cache_key(embeds, embeds_info)` | 返回 (key, start, end)；区间 = 首图 start → **倒数第二图 end**（排除末图=first_frame）；key = md5(区间 embeds) + start + ref grid；单图/无图 → None |
| `_image_kv_cache_load / _image_kv_cache_save` | 磁盘读写 {"kv", "h"} |
| `Qwen3VL.forward` | prefill 自动管理：HIT → `kv_restore=(start,end,kvs,hs)`；MISS → `save_image_kv`，返回后落盘；save 模式自建 past_key_values；带 KV 时 3 元组还原 2 元组（encode 路径契约：SDClipModel 把 outputs[2] 当 pooled） |
| `Qwen3VLClipModel.generate` | 生成路径同样集成（`BaseGenerate.generate` 透传） |

### `comfy/text_encoders/llama.py`

| 符号 | 说明 |
|---|---|
| `Attention.forward(kv_restore)` | 非 None → `_forward_segmented` |
| `Attention._forward_segmented(hidden_states, attention_mask, freqs_cis, kv_restore, optimized_attention)` | 文本行 q/k/v + KV 注入 + flash（优先）/原生 matmul（fallback）双路径；mask 兼容 4 维/2 维 |
| `Llama2_.forward(kv_restore, save_image_kv)` | 分段循环：i>0 图输入=h[i-1]；非最后层传 kv_restore + 输出覆盖 h[i]；最后一层全量；segmented 时跳过 deepstack 注入；保存 `self._saved_image_kv=(key, kv_layers, h_layers)` |
| `BaseGenerate.generate(image_kv_restore, image_kv_save)` | step 0 透传给 model.forward |

## 5. 验证记录

| 层 | 方法 | 结果 |
|---|---|---|
| 合成小模型 | 4 层小 Llama2_（GQA/mrope/q/k_norm 结构保留）| KV 跨文本位级不变（diff=0）；分段 vs 正常输出 4e-8（fp32）|
| 真实 32B 单图 | 随机图 256 token | z1 vs z3 = 0.000e+00 |
| 真实 32B H3 三图 | 3 张参考图 + 中文 prompt | MISS 332.6s → HIT 201.3s；z1 vs z3 = 0.000e+00 |
| 合成多图分离 | 3 图（2 ref 固定 + 末图变）| HIT 输出位级一致；ref 变 → key 变（MISS）；末图变 → key 不变（HIT）|
| 真实段2 场景 | 3 ref 固定 + first_frame 每段变 | MISS 316.2s → HIT 224.5s；z2 vs z3 = 0.000e+00 |

## 6. 限制与注意事项

1. **命中条件**：缓存区间（前 N-1 图）embeds 位级一致（同图 + 同前缀布局 + 同 start）。ref 图/标签/顺序变化 → 新 key（MISS 重存）。末图（first_frame）变化不影响命中。
2. **单图不缓存**；段1 只缓存前 2 张 ref（见 3.3）。
3. **旧缓存作废**：2026-08-11 区间语义修改前的缓存 key 不匹配，清理 `<缓存目录>（默认 ComfyUI根目录\h3_kv_cache，env QWEN3VL_KV_CACHE_DIR 可覆盖）` 旧文件。
4. **CPU 瓶颈未触及**：nvfp4 权重每次 forward 反量化 ~25s 固定成本（纯文本 20 token 也 25.4s）。KV 缓存只省图 token 计算。进一步提速：int8 权重（与 KV 缓存正交）。
5. **内存**：加载缓存 ~8.5GB；与模型（15.7GB 量化 + 反量化临时）共存注意峰值（48GB 机器实测可行）。
6. **数值**：flash 分段路径与正常路径位级一致；若用原生 matmul fallback，差异 ~8e-5（bf16 噪声，可接受）。
7. **开关关闭时**：全部走原逻辑，零行为变化。

## 7. 调试

- 日志：`[KV-CACHE] HIT/MISS/saved`、`[VITCACHE]`。
- 快速验证（脚本已清理，可重建）：
  - 合成：`Llama2_` 小 config 跑 `kv_restore` vs 正常路径输出对比（<1e-4）；
  - 真实：**必须** `minimax.te(**llama_detect(sd))` 加载（量化 metadata 路径）——直接 `MiniMaxH3ClipModel()` 会因 nvfp4 打包形状不匹配报错（nvfp4 每 2 值打包 1 字节，shape 列减半）；
  - 测试进程须在 import 前设 `os.environ["QWEN3VL_KV_CACHE"]="1"`。
- 恢复原始文件：`*.bak_kvcache`（2026-08-11 20:25 备份）。
