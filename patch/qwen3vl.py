import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Tokenizer

from comfy import sd1_clip
import comfy.text_encoders.qwen_vl
from .qwen35 import Qwen35VisionModel
from .llama import BaseLlama, BaseQwen3, BaseGenerate, Llama2_, Qwen3VL_4BConfig, Qwen3VL_8BConfig, Qwen3VL_32BConfig


QWEN3VL_VISION = {
    "qwen3vl_4b": dict(hidden_size=1024, intermediate_size=4096, depth=24, deepstack_visual_indexes=[5, 11, 17]),
    "qwen3vl_8b": dict(hidden_size=1152, intermediate_size=4304, depth=27, deepstack_visual_indexes=[8, 16, 24]),
    "qwen3vl_32b": dict(hidden_size=1152, intermediate_size=4304, depth=27, deepstack_visual_indexes=[8, 16, 24]),
}
QWEN3VL_VISION_COMMON = dict(num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3,
                             spatial_merge_size=2, num_position_embeddings=2304)

QWEN3VL_CONFIGS = {"qwen3vl_4b": Qwen3VL_4BConfig, "qwen3vl_8b": Qwen3VL_8BConfig, "qwen3vl_32b": Qwen3VL_32BConfig}




# 【2026-08-11】ViT 视觉编码缓存（磁盘持久化）：同图同尺寸只跑一次视觉编码器
import os as _vos, hashlib as _vhash
_vit_cache_dir = r"E:\ai\h3_vit_cache"
try:
    _vos.makedirs(_vit_cache_dir, exist_ok=True)
except Exception:
    pass

def _cached_visual(self, image, grid):
    """缓存 self.visual(image, grid) 的输出（merged, deepstack）"""
    try:
        h = _vhash.md5(image.cpu().numpy().tobytes()).hexdigest()[:16]
        grid_str = "_".join(str(g) for g in grid) if grid is not None else "nogrid"
        dims = "_".join(str(s) for s in image.shape)
        key = f"{dims}_{h}_{grid_str}"
        path = _vos.path.join(_vit_cache_dir, key + ".pt")
        print(f"[VITCACHE] key={key[:60]} exists={_vos.path.exists(path)}")
        if _vos.path.exists(path):
            import torch as _vt
            data = _vt.load(path, map_location="cpu", weights_only=True)
            # 搬回原设备
            dev = image.device
            merged = data["merged"].to(dev)
            deepstack = [d.to(dev) for d in data["deepstack"]]
            return merged, deepstack
        merged, deepstack = self.visual(image.to(image.device, dtype=torch.float32), grid)
        import torch as _vt2
        _vt2.save({"merged": merged.cpu(), "deepstack": [d.cpu() for d in deepstack]}, path)
        return merged, deepstack
    except Exception as e:
        print(f"[VITCACHE] ERROR: {e}")
        import traceback; traceback.print_exc()
        # 失败回退原始调用
        return self.visual(image.to(image.device, dtype=torch.float32), grid)
# ================= 图片 KV 缓存（LLM 层，2026-08-11）=================
# 缓存图片区间 [start, end) 在每一层的 K/V（rope 后），命中时跳过图片 token 的
# 全部 LLM 计算（q/k/v 投影 + attention 查询）。复用条件：同图 + 同 start + 同 grid
# （mrope 图片区间位置只由这两者决定，与 text 内容无关 → 同图换提问可命中）。
import hashlib as _khash

# 【开关】默认关闭：QWEN3VL_KV_CACHE=1 启用图片 KV 缓存（CPU 场景收益有限，默认不改变行为）
_QWEN3VL_KV_CACHE_ENABLED = os.environ.get("QWEN3VL_KV_CACHE", "0") == "1"

_image_kv_cache_dir = r"E:\ai\h3_kv_cache"
try:
    os.makedirs(_image_kv_cache_dir, exist_ok=True)
except Exception:
    pass


def _image_kv_cache_key(embeds, embeds_info):
    """返回 (key, start, end)：缓存区间 = 首图 start 到【倒数第二张图】end（含图间文本）。
    排除最后一张图（H3 的 first_frame 关键帧总是最后且每段变化）：其每次重算，
    前 N-1 张固定 ref 图独立缓存 → first_frame 变化不破坏命中。
    单图/无图 → None（无可缓存的固定部分）。"""
    images = sorted([e for e in embeds_info if e.get("type") == "image"], key=lambda e: e["index"])
    if len(images) < 2:
        return None
    start = min(e["index"] for e in images)
    end = images[-1]["index"]  # 末图（first_frame）在缓存区间外
    if end <= start:
        return None
    seg = embeds[0, start:end].detach().cpu().numpy().tobytes()
    h = _khash.md5(seg).hexdigest()[:16]
    fixed = images[:-1]  # key 只含固定 ref 图
    extras = [e.get("extra") for e in fixed]
    if all(x is None for x in extras):
        grid_str = "nogrid"
    else:
        grid_str = "_".join(
            str(int(v))
            for e in fixed
            for g in ((e["extra"]["grid"] if isinstance(e["extra"], dict) else e["extra"]) if e.get("extra") is not None else [])
            for v in g
        )
    return f"{h}_{start}_{grid_str}", start, end


def _image_kv_cache_load(key):
    """返回每层缓存文件路径列表（惰性：不加载，逐层按需读，避免 8.5GB 峰值内存 → swap）。"""
    layer_dir = os.path.join(_image_kv_cache_dir, key)
    if not os.path.isdir(layer_dir):
        return None
    try:
        files = sorted(os.listdir(layer_dir))
        return [os.path.join(layer_dir, f) for f in files]
    except Exception as e:
        print(f"[KV-CACHE] load error: {e}")
        return None


def _image_kv_cache_save(key, kv_layers, h_layers):
    """拆层存储：每层一个 .pt（k/v/h 各 ~70MB）→ HIT 时逐层加载，峰值内存仅单层。"""
    layer_dir = os.path.join(_image_kv_cache_dir, key)
    try:
        os.makedirs(layer_dir, exist_ok=True)
        for i, ((k, v), h) in enumerate(zip(kv_layers, h_layers)):
            torch.save({"k": k, "v": v, "h": h}, os.path.join(layer_dir, f"{i:02d}.pt"))
        print(f"[KV-CACHE] saved {key} ({len(kv_layers)} layers, per-layer files)")
    except Exception as e:
        print(f"[KV-CACHE] save error: {e}")


class Qwen3VLDeepstackMerger(nn.Module):
    # DeepStack merger: postshuffle LayerNorm (applied after spatial merge), unlike the main merger.
    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size, device=None, dtype=None, ops=None):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = ops.LayerNorm(self.merge_dim, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(self.merge_dim, self.merge_dim, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(self.merge_dim, out_hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(Qwen35VisionModel):
    # Qwen3.5 vision + DeepStack
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.deepstack_visual_indexes = config["deepstack_visual_indexes"]
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLDeepstackMerger(self.hidden_size, self.spatial_merge_size, config["out_hidden_size"], device=device, dtype=dtype, ops=ops)
            for _ in self.deepstack_visual_indexes
        ])


class Qwen3VL(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    model_type = "qwen3vl_8b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = QWEN3VL_CONFIGS[self.model_type](**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        vision_config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION[self.model_type], "out_hidden_size": config.hidden_size}
        self.visual = Qwen3VLVisionModel(vision_config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            # Qwen3-VL normalizes to [-1, 1] (mean/std 0.5), unlike Qwen2.5-VL's CLIP normalization.
            image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(embed["data"], patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            merged, deepstack = _cached_visual(self, image.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return None, None

    def build_image_inputs(self, embeds, embeds_info):
        # Returns (position_ids, visual_pos_masks, deepstack) for the prompt
        images = sorted([e for e in embeds_info if e.get("type") == "image"], key=lambda e: e["index"])
        if len(images) == 0:
            return None, None, None

        device = embeds.device
        seq = embeds.shape[1]
        position_ids = comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids(embeds_info, seq, device)

        # DeepStack: mask of image positions + per-vision-layer features to inject there.
        visual_pos_masks = torch.zeros((1, seq), dtype=torch.bool, device=device)
        deepstack = None
        for e in images:
            start = e["index"]
            end = e["size"] + start
            visual_pos_masks[0, start:end] = True
            ds = e["extra"]["deepstack"]
            if deepstack is None:
                deepstack = [d for d in ds]
            else:
                deepstack = [torch.cat([deepstack[i], ds[i]], dim=0) for i in range(len(ds))]
        return position_ids, visual_pos_masks, deepstack

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[], **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_embeds = kwargs.pop("deepstack_embeds", None)
        kv_restore = kwargs.pop("kv_restore", None)
        save_image_kv = kwargs.pop("save_image_kv", None)
        if embeds is not None and position_ids is None:
            position_ids, visual_pos_masks, deepstack_embeds = self.build_image_inputs(embeds, embeds_info)
        # 【图片 KV 缓存】prefill（embeds 全量且含图）自动管理：命中 → 分段复用；未命中 → 保存
        if _QWEN3VL_KV_CACHE_ENABLED and embeds is not None and len(embeds_info) > 0 and kv_restore is None and save_image_kv is None:
            kv_info = _image_kv_cache_key(embeds, embeds_info)
            if kv_info is not None:
                key, start, end = kv_info
                files = _image_kv_cache_load(key)
                if files is not None:
                    if len(files) == self.num_layers:
                        kv_restore = (start, end, files)  # 逐层文件路径，Llama2_.forward 按层惰性加载
                        print(f"[KV-CACHE] HIT {key} start={start} end={end}")
                    else:
                        print(f"[KV-CACHE] layer mismatch {len(files)} vs {self.num_layers}, miss")
                        save_image_kv = (start, end, key)
                else:
                    save_image_kv = (start, end, key)
                    print(f"[KV-CACHE] MISS {key} start={start} end={end} -> will save")
        # save 模式需要 past_key_values 让 Attention 回传 KV（encode 路径默认不传）
        if save_image_kv is not None:
            past_key_values = kwargs.pop("past_key_values", None)
            if past_key_values is None:
                config = self.model.config
                max_len = embeds.shape[1]
                past_key_values = [
                    (torch.empty(1, config.num_key_value_heads, max_len, config.head_dim, device=embeds.device, dtype=embeds.dtype),
                     torch.empty(1, config.num_key_value_heads, max_len, config.head_dim, device=embeds.device, dtype=embeds.dtype), 0)
                    for _ in range(self.num_layers)
                ]
            kwargs["past_key_values"] = past_key_values
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype,
            position_ids=position_ids,
            embeds_info=embeds_info,
            visual_pos_masks=visual_pos_masks,
            deepstack_embeds=deepstack_embeds,
            kv_restore=kv_restore,
            save_image_kv=save_image_kv,
            **kwargs,
        )
        # encode 路径（SDClipModel）期待 (z, intermediate) 二元组；带 KV 时为 3 元组（KV 已存 _saved_image_kv）
        if (save_image_kv is not None or kv_restore is not None) and isinstance(out, tuple) and len(out) == 3:
            out = out[:2]
        if save_image_kv is not None:
            saved = getattr(self.model, "_saved_image_kv", None)
            if saved is not None and saved[0] == save_image_kv[2]:
                _image_kv_cache_save(saved[0], saved[1], saved[2])
            self.model._saved_image_kv = None
        return out


def _make_qwen3vl_model(model_type):
    class Qwen3VL_(Qwen3VL):
        pass
    Qwen3VL_.model_type = model_type
    return Qwen3VL_


class Qwen3VLClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-1, dtype=None, attention_mask=True, model_options={}, model_type="qwen3vl_8b"):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=_make_qwen3vl_model(model_type), enable_attention_masks=attention_mask,
                         return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = self.process_tokens(tokens_only, self.execution_device)
        position_ids, visual_pos_masks, deepstack = self.transformer.build_image_inputs(embeds, embeds_info)

        image_kv_restore = None
        image_kv_save = None
        kv_info = _image_kv_cache_key(embeds, embeds_info) if _QWEN3VL_KV_CACHE_ENABLED else None
        if kv_info is not None:
            key, start, end = kv_info
            files = _image_kv_cache_load(key)
            if files is not None:
                if len(files) == self.transformer.num_layers:
                    image_kv_restore = (start, end, files)
                    print(f"[KV-CACHE] HIT {key} start={start} end={end}")
                else:
                    image_kv_save = (start, end, key)
            else:
                image_kv_save = (start, end, key)

        out = self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                                         presence_penalty=presence_penalty, position_ids=position_ids,
                                         visual_pos_masks=visual_pos_masks, deepstack_embeds=deepstack,
                                         image_kv_restore=image_kv_restore, image_kv_save=image_kv_save)
        if image_kv_save is not None:
            saved = getattr(self.transformer.model, "_saved_image_kv", None)
            if saved is not None and saved[0] == image_kv_save[2]:
                _image_kv_cache_save(saved[0], saved[1], saved[2])
            self.transformer.model._saved_image_kv = None
        return out


class Qwen3VLTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, model_type="qwen3vl_8b"):
        clip_model = lambda **kw: Qwen3VLClipModel(**kw, model_type=model_type)
        super().__init__(device=device, dtype=dtype, name=model_type, clip_model=clip_model, model_options=model_options)


class Qwen3VLSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, embedding_size=4096, embedding_key="qwen3vl_8b"):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Qwen3VLTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, model_type="qwen3vl_8b"):
        embedding_size = 2560 if model_type == "qwen3vl_4b" else 4096
        tokenizer = lambda *a, **kw: Qwen3VLSDTokenizer(*a, **kw, embedding_size=embedding_size, embedding_key=model_type)
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=model_type, tokenizer=tokenizer)
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, skip_template=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]

        skip_template = skip_template or text.startswith('<|im_start|>')
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                template = llama_template
            elif len(images) == 0:
                template = self.llama_template
            else:
                template = self.llama_template_images
                if len(images) > 1:
                    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                    template = template.replace(vision_block, vision_block * len(images), 1)
            llama_text = template.format(text)
            if not thinking:  # Qwen3 convention: empty think block suppresses reasoning
                llama_text += "<think>\n\n</think>\n\n"

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        for r in tokens[key_name]:
            for i in range(len(r)):
                if isinstance(r[i][0], (int, float)) and r[i][0] == 151655:  # <|image_pad|>
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


def tokenizer(model_type="qwen3vl_8b"):
    class Qwen3VLTokenizer_(Qwen3VLTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=model_type)
    return Qwen3VLTokenizer_


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen3vl_8b"):
    class Qwen3VLTEModel_(Qwen3VLTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, model_type=model_type)
    return Qwen3VLTEModel_
