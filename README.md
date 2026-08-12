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

## 使用

```bash
set QWEN3VL_KV_CACHE=1    # 默认关闭；重启 ComfyUI / h3_runner 生效
```

- 第一段 MISS（自动保存缓存），段 2+ 命中；first_frame 每段变化不影响命中
- 缓存目录：`E:\ai\h3_kv_cache\<key>\00.pt … 49.pt`
- 命中条件：前 N-1 张参考图 + 前缀布局与缓存时位级一致

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
