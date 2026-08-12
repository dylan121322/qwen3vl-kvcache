# Qwen3VL Image KV Cache

> 为 ComfyUI 内置 Qwen3-VL 编码器（MiniMax H3 conditioning，nvfp4 量化 CPU 推理）实现的**图片区间 KV 缓存**：
> 固定参考图重复编码时，跳过 LLM 层图片 token 的全部计算，只计算文本行与最后一层。

## 实测收益

| 状态 | 耗时（3 ref + first_frame 每段变，CPU）|
|---|---|
| 无缓存（MISS）| 325.8s |
| 命中（HIT）| **223.8s（省 102s，31%）** |
| 输出一致性 | **位级一致 0.000e+00** |

## 工作原理（30 秒版）

```
H3 布局: [固定前缀 + 图1..图N-1] [first_frame] [prompt]
                ↑ 缓存区间              ↑ 每段重算

因果注意力 + 图在前 + 前缀固定
  → 图区间的每层 KV 与输出 hidden 独立于 first_frame/prompt
  → 缓存 50 层 (KV + hidden)，拆层存储（每层一个文件 ~70MB）
  → HIT 时逐层惰性加载，只算文本行；最后一层不跳过（conditioning 需要真值）
```

三个不可简化的设计约束：

1. **不能逐图独立缓存**——图 N 的 layer 1+ 依赖前面所有图的 KV（因果链），key 含前缀指纹即等价整段；
2. **缓存必须含每层输出 hidden**——图区间 hidden 链必须完整，否则最终 conditioning 输出无真值；
3. **必须拆层存储**——整包 8.5GB 一次加载会挤爆 49GB 内存触发 swap（实测峰值页面文件 72.5GB），CPU 拉不满、收益被换页吃掉（v2 修复）。

安全设计（2026-08-12 审查修复）：

- **key 含前缀 + end + 配置指纹 + schema 版本**：`v2_<config>_<md5(embeds[0:end])>_<start>_<end>_<grid>`——前缀同长改写、末图尺寸变化、模型配置/代码版本变化均生成新 key，杜绝静默陈旧命中；
- **deepstack 注入修复**：HIT 时仅跳过缓存区间内的注入（h 已含），区间外（first_frame）照常注入——MISS 与 HIT 在 ff 行输出位级一致；
- **损坏缓存回退**：逐层加载失败抛 `CacheCorruptError` → 删除坏缓存目录 → 本次走普通路径，不崩溃；
- **原子写盘**：每层文件 tmp + rename，防半写文件。

## 关于 first_frame（重要约定）

- `first_frame` 是 H3 多段视频生成的**首帧关键帧**：段 2+ 把上一段的末帧作为本段起始帧传入 CLIP（FL2VA 关键帧模式），保证段间画面衔接。
- **它每段都不同**（每段是上一段的末帧），所以无法缓存、每次重算。
- **缓存区间 = 前 N-1 张图（排除末图）**——该设计基于 H3 的布局约定：**first_frame 总是序列中最后一张图**（`also_ref_first_frame` 把首帧追加为最后一个 Picture）。
- 参考图（前 N-1 张）跨段不变 → 独立缓存 → 每段命中。

**如果你的使用场景布局不同**（例如：末图不是 first_frame、多个 first_frame、first_frame 置首/居中、或根本没有关键帧概念），可自行调整 `qwen3vl.py` 中 `_image_kv_cache_key` 的区间选择：

```python
# 当前：排除最后一张图（H3 first_frame 约定）
end = images[-1]["index"]

# 例：全部图都固定（无 first_frame）→ 缓存全部图区间
end = max(e["index"] + e["size"] for e in images)

# 例：前 2 张固定、其余可变 → end = images[1]["index"] + images[1]["size"]
```

> 注意：区间后的一切（未缓存图 + 文本）每次重算且依赖缓存区间的 KV——只要"区间内内容不变"就能命中，与区间后内容无关。

## 安装

### 前置条件

| 项 | 要求 |
|---|---|
| ComfyUI | Windows portable 版（2026-08-03 之后，含 MiniMax H3 节点支持）|
| 环境路径 | `<ComfyUI根目录>\`（本补丁按此路径开发验证）|
| CLIP 权重 | `<H3权重路径>\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（nvfp4 量化）|
| Python | 使用 ComfyUI 自带的 `python_embeded\python.exe`，无需额外依赖 |

### 安装步骤

**1. 备份原文件**（重要——回滚靠它）：

```bat
cd /d <ComfyUI根目录>\ComfyUI\comfy\text_encoders
copy /y qwen3vl.py qwen3vl.py.bak_kvcache
copy /y llama.py llama.py.bak_kvcache
copy /y qwen35.py qwen35.py.bak_kvcache
```

**2. 复制补丁文件**（来自本仓库 `patch/`）：

```bat
copy /y patch\qwen3vl.py <ComfyUI根目录>\ComfyUI\comfy\text_encoders\qwen3vl.py
copy /y patch\llama.py   <ComfyUI根目录>\ComfyUI\comfy\text_encoders\llama.py
```

> `qwen35.py` 未改动，无需复制（仓库内保留仅为完整性）。

**3. 校验语法**（可选但推荐）：

```bat
<ComfyUI根目录>\python_embeded\python.exe -m py_compile <ComfyUI根目录>\ComfyUI\comfy\text_encoders\qwen3vl.py <ComfyUI根目录>\ComfyUI\comfy\text_encoders\llama.py
```

**4. 重启 ComfyUI / h3_runner**（必须——Python 模块已加载进内存，不重启不生效）。

## 使用

### 开启（默认关闭，三选一）

**方式 A —— 系统环境变量（永久）**：
```bat
setx QWEN3VL_KV_CACHE 1
```
（`setx` 后需重开终端/重启 ComfyUI；改回关闭用 `setx QWEN3VL_KV_CACHE 0`）

**方式 B —— 启动脚本内设置（推荐，仅对本次启动生效）**：
```bat
:: start_comfyui_persist.bat 开头加一行
set QWEN3VL_KV_CACHE=1
```

**方式 C —— 命令行临时**：
```bat
set QWEN3VL_KV_CACHE=1 && <ComfyUI根目录>\python_embeded\python.exe <h3_runner路径>\h3_runner.py ...
```

### 运行预期（看日志）

**第一段（MISS，自动保存缓存）**：
```
[KV-CACHE] MISS <key> start=7 end=6044 -> will save
[KV-CACHE] saved <key> (50 layers, per-layer files)
```
编码耗时 ~325s（全量）。

**段 2+（HIT，first_frame 变化不影响）**：
```
[KV-CACHE] HIT <key> start=7 end=6044
```
编码耗时 ~224s（省 102s）。输出与无缓存**位级一致**（0.000e+00）。

### 缓存维护

- 位置：`<缓存目录，默认 ComfyUI根目录\h3_kv_cache，可 set QWEN3VL_KV_CACHE_DIR 覆盖>\<key>\00.pt … 49.pt`（每个 key 一个目录，每层一个文件 ~70MB）
- 首次 MISS 后生成；同一组参考图跨段复用
- **换参考图 → 新 key**（MISS 重建，旧 key 成死数据）——手动删除旧目录释放磁盘：
  ```bat
  rmdir /s /q <缓存目录，默认 ComfyUI根目录\h3_kv_cache，可 set QWEN3VL_KV_CACHE_DIR 覆盖>\<旧key目录>
  ```
- 旧格式单文件缓存（`<key>.pt`，v1 时代遗留）已作废，可整体删除：`del <缓存目录，默认 ComfyUI根目录\h3_kv_cache，可 set QWEN3VL_KV_CACHE_DIR 覆盖>\*.pt`

### 关闭

取消环境变量后重启即可：`setx QWEN3VL_KV_CACHE 0`（或删除启动脚本里的 `set` 行）。未设置时代码行为与官方完全一致（零侵入）。

### 回滚

```bat
cd /d <ComfyUI根目录>\ComfyUI\comfy\text_encoders
copy /y qwen3vl.py.bak_kvcache qwen3vl.py
copy /y llama.py.bak_kvcache llama.py
rmdir /s /q <缓存目录，默认 ComfyUI根目录\h3_kv_cache，可 set QWEN3VL_KV_CACHE_DIR 覆盖>   :: 可选，删缓存
:: 重启 ComfyUI
```

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志无 `[KV-CACHE]` | 环境变量未设置 / 未重启 | 确认 `QWEN3VL_KV_CACHE=1` 且重启 |
| 一直 MISS | 参考图换了 / 前缀布局变了 | 正常（新 key）；确认旧缓存已删 |
| HIT 但耗时不降 | 内存不足触发 swap（可用 <10GB 时整包加载曾导致）| v2 已修复（逐层加载 +70MB）；仍异常则检查是否有其他进程占内存 |
| `size mismatch` 加载报错 | 用了非量化加载路径（测试脚本直接构造 ClipModel）| 必须经 `llama_detect` + `te()` 量化加载路径（见 docs）|

## 目录结构

```
qwen3vl-kvcache/
├── README.md                      本文件
├── docs/
│   ├── qwen3vl_kv_cache.md        详细原理 / 代码结构 / 验证记录 / 调试
│   └── Qwen3VL_KV_Cache_实现原理详解.html   图文图解（v2）
└── patch/
    ├── qwen3vl.py                 补丁后（缓存管理：开关 / key / 拆层存储 / 自动 HIT-MISS）
    ├── llama.py                   补丁后（分段注意力 / 逐层加载 / 保存）
    ├── qwen35.py                  未改动（同批备份）
    └── original/                  补丁前原始文件（回滚用）
```

## 版本历史

| 版本 | 时间 | 变化 |
|---|---|---|
| v1 | 2026-08-11 | 初版：整包缓存、图整体区间；合成 + 真实 32B 验证位级一致 |
| v1.1 | 2026-08-11 | first_frame 分离：区间排除末图，ff 每段变不破坏命中 |
| v2 | 2026-08-12 | 拆层存储 + 逐层惰性加载：HIT 峰值内存 +8.5GB → +70MB，修复实装 swap |

## 许可证

本仓库是 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（GPL-3.0）`comfy/text_encoders/{qwen3vl.py, llama.py}` 的派生修改，遵循 **GPL-3.0**。
