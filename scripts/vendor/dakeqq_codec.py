"""Codec-only helpers derived from DakeQQ/Text-to-Speech-TTS-ONNX.
Source revision ae69a6c1cc9e43c855d7d906c80770581f49ee3c; Apache-2.0.
Changes: codec-only extraction; standard-op RMS normalization instead of custom symbolic op.
See docs/licenses/DakeQQ-NOTICE.txt. This module never loads AR weights.
"""
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from modeling_arktts_codec import ArkttsCausalConv1d,ArkttsCausalConvTranspose1d,ArkttsCodec,ArkttsConvNeXtBlock,ArkttsSnake1d
OUT_AUDIO_DTYPE="F32"
def simplified_layer_norm(x,scale,epsilon,axis=-1):
    return (x.float()*torch.rsqrt(x.float().square().mean(dim=axis,keepdim=True)+epsilon)*scale).to(scale.dtype)
def load_audio8_codec(model_path, config):
    """Mirror ArkttsModel.load_codec without its package-relative import."""
    codec = ArkttsCodec(config)
    state = torch.load(Path(model_path) / config.codec_filename, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("generator.") for key in state):
        state = {key[len("generator."):]: value for key, value in state.items() if key.startswith("generator.")}
    state = {key: value for key, value in state.items() if not key.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    codec = codec.float().eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False
    return codec

def _static_causal_conv_forward(self, x):
    """Causal Conv1d with a static zero prefix that exports as Concat.

    The original right `_extra_padding` is 0 whenever the input length is a multiple of the stride,
    which holds for every layer once the waveform is padded to a multiple of the codec frame size.
    """
    if self.padding > 0:
        x = torch.cat((self.left_padding, x), dim=-1)
    return self.conv(x)

def _static_causal_transpose_forward(self, x):
    """Causal ConvTranspose1d with a constant right-trim Slice."""
    x = self.conv(x)
    return x[..., :-self.right_trim] if self.right_trim > 0 else x

def _precomputed_snake_forward(self, x):
    return x + self.inv_alpha * torch.sin(self.alpha * x).pow(2)

def _fused_convnext_forward(self, x):
    residual = x
    x = self.dwconv(x).permute(0, 2, 1)
    x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
    return residual + x.permute(0, 2, 1)

def _remove_weight_norms(codec):
    parametrized = legacy = 0
    for module in list(codec.modules()):
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            parametrized += 1
        elif hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            torch.nn.utils.remove_weight_norm(module)
            legacy += 1
    return parametrized, legacy

def _fold_convnext(block):
    """Fold the LayerNorm affine into pwconv1 and the layer scale `gamma` into pwconv2."""
    dim = block.norm.normalized_shape[0]
    with torch.no_grad():
        block.pwconv1.bias.add_(block.pwconv1.weight @ block.norm.bias)
        block.pwconv1.weight.mul_(block.norm.weight.unsqueeze(0))
        block.pwconv2.weight.mul_(block.gamma.unsqueeze(1))
        block.pwconv2.bias.mul_(block.gamma)
    block.norm = torch.nn.LayerNorm(dim, eps=block.norm.eps, elementwise_affine=False)
    del block.gamma
    block.forward = types.MethodType(_fused_convnext_forward, block)

def prepare_codec_for_export(codec):
    """Remove weight norm and swap dynamic-shape codec layers for export-friendly exact equivalents."""
    parametrized, legacy = _remove_weight_norms(codec)
    stats = {
        "weight_norm_parametrized": parametrized,
        "weight_norm_legacy": legacy,
        "static_padding_convs": 0,
        "transpose_convs": 0,
        "snakes": 0,
        "fused_convnext": 0,
    }
    for module in list(codec.modules()):
        if isinstance(module, ArkttsCausalConv1d):
            padding = torch.zeros(1, module.conv.in_channels, module.padding)
            module.register_buffer("left_padding", padding, persistent=False)
            module.forward = types.MethodType(_static_causal_conv_forward, module)
            stats["static_padding_convs"] += int(module.padding > 0)
        elif isinstance(module, ArkttsCausalConvTranspose1d):
            module.right_trim = int(module.kernel_size - module.stride)
            module.forward = types.MethodType(_static_causal_transpose_forward, module)
            stats["transpose_convs"] += 1
        elif isinstance(module, ArkttsSnake1d):
            inv_alpha = (module.alpha.detach() + 1e-9).reciprocal()
            module.register_buffer("inv_alpha", inv_alpha, persistent=False)
            module.forward = types.MethodType(_precomputed_snake_forward, module)
            stats["snakes"] += 1
        elif isinstance(module, ArkttsConvNeXtBlock):
            _fold_convnext(module)
            stats["fused_convnext"] += 1
    return stats

def shape_dim_as_tensor(tensor, dim):
    """Return one dynamic dimension as a rank-1 int64 tensor."""
    return torch._shape_as_tensor(tensor)[dim].unsqueeze(0)

def half_split_permutation(num_heads, head_dim):
    """Row order that turns interleaved (x0, x1) RoPE pairs into the half-split layout, per head."""
    per_head = torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))
    return torch.cat([per_head + head * head_dim for head in range(num_heads)])

def half_split_rope_tables(max_len, head_dim, base, round_to_bf16):
    """cos/sin tables for the flip-based rotate_half: cos=cat(c, c), sin=cat(-s, s)."""
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    phases = torch.outer(torch.arange(max_len).float(), frequencies[: head_dim // 2])
    cos, sin = phases.cos(), phases.sin()
    if round_to_bf16:
        # Audio8 bakes its interleaved tables in bfloat16; match that rounding exactly.
        cos, sin = cos.bfloat16().float(), sin.bfloat16().float()
    return torch.cat((cos, cos), dim=-1), torch.cat((-sin, sin), dim=-1)

def flip_rotate_half(x, head_dim):
    """cat(x2, x1) via flip on the half axis; more efficient than split()+cat() in ONNX Runtime."""
    shape = x.shape
    return x.reshape(*shape[:-1], 2, head_dim // 2).flip(-2).reshape(shape)

class A8_CODEC_TRANSFORMER(torch.nn.Module):
    """Windowed causal codec transformer with norms, layer scales and the attention scale folded into weights.

    Mirrors ArkttsCodecWindowTransformer exactly: interleaved RoPE becomes half-split through a per-head
    q/k row permutation (q·k is invariant), causal/window mask is precomputed as an additive int8 buffer,
    and QK RoPE is pre-unsqueezed in half precision and applied jointly before splitting.
    """

    def __init__(self, window_module, max_len, channels_first_input=True, channels_first_output=None):
        super().__init__()
        for name in ("input_proj", "output_proj", "look_ahead_conv"):
            if not isinstance(getattr(window_module, name), torch.nn.Identity):
                raise ValueError(f"Codec transformer {name} is not Identity; folding is not implemented for it.")
        first = window_module.layers[0]
        attention = first.attention
        self.num_heads = attention.n_head
        self.num_kv_heads = attention.n_local_heads
        self.head_dim = attention.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qk_heads = self.num_heads + self.num_kv_heads
        self.window_size = window_module.window_size
        self.channels_first_input = channels_first_input
        self.channels_first_output = (
            window_module.channels_first if channels_first_output is None else channels_first_output
        )
        self.norm_eps = first.attention_norm.eps
        self.final_norm_eps = window_module.norm.eps
        dim = first.attention_norm.weight.shape[0]
        self.register_buffer("unit_scale", torch.ones(dim), persistent=False)
        self.register_buffer("final_scale", window_module.norm.weight.detach().clone(), persistent=False)
        cos, sin = half_split_rope_tables(max_len, self.head_dim, window_module.rope_base, round_to_bf16=True)
        self.register_buffer("rope_cos", cos.unsqueeze(1).half(), persistent=False)
        self.register_buffer("rope_sin", sin.unsqueeze(1).half(), persistent=False)
        # Precompute additive causal + sliding-window int8 mask: eliminates per-call dynamic mask logic
        positions = torch.arange(max_len)
        row, column = positions.unsqueeze(1), positions.unsqueeze(0)
        allowed = column <= row
        if self.window_size is not None:
            allowed = allowed & (column > row - self.window_size)
        mask = ((~allowed).to(torch.int8) * -128).reshape(1, 1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.qkv = torch.nn.ModuleList()
        self.o_proj = torch.nn.ModuleList()
        self.gate_up = torch.nn.ModuleList()
        self.down = torch.nn.ModuleList()
        q_permutation = half_split_permutation(self.num_heads, self.head_dim)
        k_permutation = half_split_permutation(self.num_kv_heads, self.head_dim)
        with torch.no_grad():
            for block in window_module.layers:
                self._fold_block(block, q_permutation, k_permutation)

    def _fold_block(self, block, q_permutation, k_permutation):
        attention, feed_forward = block.attention, block.feed_forward
        dim = attention.wqkv.in_features
        scale = self.head_dim ** -0.25
        weight = attention.wqkv.weight * block.attention_norm.weight.unsqueeze(0)
        q, k, v = weight.split((self.q_size, self.kv_size, self.kv_size), dim=0)
        qkv = torch.nn.Linear(dim, self.q_size + 2 * self.kv_size, bias=False)
        qkv.weight.copy_(torch.cat((q[q_permutation] * scale, k[k_permutation] * scale, v), dim=0))
        o_proj = torch.nn.Linear(self.q_size, dim, bias=False)
        o_proj.weight.copy_(attention.wo.weight * block.attention_layer_scale.gamma.unsqueeze(1))
        inter = feed_forward.w1.out_features
        gate_up = torch.nn.Linear(dim, 2 * inter, bias=False)
        gate_up_weight = torch.cat((feed_forward.w1.weight, feed_forward.w3.weight), dim=0)
        gate_up.weight.copy_(gate_up_weight * block.ffn_norm.weight.unsqueeze(0))
        down = torch.nn.Linear(inter, dim, bias=False)
        down.weight.copy_(feed_forward.w2.weight * block.ffn_layer_scale.gamma.unsqueeze(1))
        self.qkv.append(qkv)
        self.o_proj.append(o_proj)
        self.gate_up.append(gate_up)
        self.down.append(down)

    def forward(self, x):
        if self.channels_first_input:
            x = x.transpose(1, 2)
        length = x.shape[1]
        mask = self.causal_mask[..., :length, :length].to(x.dtype)
        cos = self.rope_cos[:length].to(x.dtype)
        sin = self.rope_sin[:length].to(x.dtype)
        for qkv, o_proj, gate_up, down in zip(self.qkv, self.o_proj, self.gate_up, self.down):
            hidden = simplified_layer_norm(x, self.unit_scale, self.norm_eps)
            qk, v = qkv(hidden).split((self.q_size + self.kv_size, self.kv_size), dim=-1)
            qk = qk.reshape(1, -1, self.qk_heads, self.head_dim)
            qk = qk * cos + flip_rotate_half(qk, self.head_dim) * sin
            q, k = qk.split((self.num_heads, self.num_kv_heads), dim=-2)
            q = q.reshape(1, -1, self.num_kv_heads, self.groups, self.head_dim).permute(0, 2, 3, 1, 4)
            k = k.permute(0, 2, 3, 1).unsqueeze(2)
            v = v.reshape(1, -1, self.num_kv_heads, self.head_dim).transpose(1, 2).unsqueeze(2)
            scores = torch.matmul(q, k) + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), v)
            attn = attn.permute(0, 3, 1, 2, 4).reshape(1, -1, self.q_size)
            x = x + o_proj(attn)
            hidden = simplified_layer_norm(x, self.unit_scale, self.norm_eps)
            gate, up = gate_up(hidden).chunk(2, dim=-1)
            x = x + down(F.silu(gate) * up)
        x = simplified_layer_norm(x, self.final_scale, self.final_norm_eps)
        return x.transpose(1, 2) if self.channels_first_output else x

def fold_codebook_tables(quantizer):
    """Fold each VQ out_proj into its codebook.

    Returns one stacked [sum(sizes), dim] table, the per-codebook row offsets and clamp limits,
    and the summed out_proj bias (semantic codebook first, then the residual codebooks).
    """
    vector_quantizers = list(quantizer.semantic_quantizer.quantizers) + list(quantizer.quantizer.quantizers)
    tables, offsets, limits = [], [], []
    bias = 0.0
    offset = 0
    with torch.no_grad():
        for vq in vector_quantizers:
            projection = vq.out_proj.weight[:, :, 0]
            tables.append(vq.codebook.weight @ projection.t())
            bias = bias + vq.out_proj.bias
            offsets.append(offset)
            limits.append(vq.codebook_size - 1)
            offset += vq.codebook_size
    return torch.cat(tables, dim=0), bias, offsets, limits

class A8_DECODER(torch.nn.Module):
    """Decode flat frame-major codec ids [1, num_codebooks * N] back to a raw audio waveform."""

    def __init__(self, codec, output_sample_rate, model_output_sample_rate, max_frames):
        super().__init__()
        quantizer = codec.quantizer
        table, bias, offsets, limits = fold_codebook_tables(quantizer)
        self.num_codebooks = len(offsets)
        self.codebook_table = torch.nn.Embedding(table.shape[0], table.shape[1])
        with torch.no_grad():
            self.codebook_table.weight.copy_(table)
        self.register_buffer("codebook_bias", bias.detach().reshape(1, 1, -1).clone(), persistent=False)
        self.register_buffer(
            "code_offsets", torch.tensor(offsets, dtype=torch.int32).reshape(1, 1, -1), persistent=False
        )
        self.register_buffer("code_max", torch.tensor(limits, dtype=torch.int32).reshape(1, 1, -1), persistent=False)
        self.register_buffer("code_min", torch.zeros(1, 1, len(limits), dtype=torch.int32), persistent=False)
        self.post_module = A8_CODEC_TRANSFORMER(quantizer.post_module, max_frames, channels_first_input=False)
        self.upsample = quantizer.upsample
        self.decoder = codec.decoder
        self.scale = output_sample_rate / model_output_sample_rate
        self.output_dtype = OUT_AUDIO_DTYPE.upper()
        self.static_conv_fusion_count = sum(
            int(isinstance(module, ArkttsCausalConv1d) and module.padding > 0)
            for module in list(self.upsample.modules()) + list(self.decoder.modules())
        )

    def forward(self, generated_codec):
        codes = generated_codec.reshape(1, -1, self.num_codebooks).int()
        codes = torch.minimum(torch.maximum(codes, self.code_min), self.code_max) + self.code_offsets
        latent = self.codebook_table(codes).sum(dim=2) + self.codebook_bias
        hidden = self.post_module(latent)
        generated_wav = self.decoder(self.upsample(hidden))
        if self.scale != 1.0:
            generated_wav = F.interpolate(
                generated_wav,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.output_dtype == "INT16":
            # The codec ends in Tanh, so the PCM scale cannot be folded into the last conv as in Qwen.
            generated_wav = (generated_wav * 32767.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)
        elif self.output_dtype == "F32":
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).float()
        else:
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).half()
        generated_len = shape_dim_as_tensor(generated_wav, -1)
        return generated_wav, generated_len
