# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging

from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def _load_lm_state_dict(
    filename: str,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    if filename.endswith(".safetensors"):
        dev = torch.device(device) if isinstance(device, str) else device
        # safetensors does not support mps directly
        load_device = "cpu" if dev.type == "mps" else dev.type
        return load_file(filename, device=load_device)
    with open(filename, "rb") as f:
        return torch.load(f, map_location=device)


def _get_expanded_source_name(name: str) -> str | None:
    to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
    for old_idx, new_idx in zip(range(8), range(8, 16)):
        for rep in to_replace:
            needle = f"{rep}.{new_idx}."
            if needle in name:
                return name.replace(needle, f"{rep}.{old_idx}.")
    return None


def _repeat_first_dim_to_shape(
    source: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor | None:
    if tuple(source.shape) == tuple(target_shape):
        return source.clone()
    if source.dim() != len(target_shape):
        return None
    if source.dim() > 1 and tuple(source.shape[1:]) != tuple(target_shape[1:]):
        return None
    if source.shape[0] <= 0:
        return None
    repeats = (target_shape[0] + source.shape[0] - 1) // source.shape[0]
    if repeats <= 0:
        return None
    rep_dims = (repeats,) + (1,) * (source.dim() - 1)
    expanded = source.repeat(rep_dims)
    return expanded[: target_shape[0]].clone()


def _patch_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_sd: dict[str, torch.Tensor],
    copy_missing_weights: bool,
) -> dict[str, torch.Tensor]:
    # Patch 1: expand depformer self_attn tensors when checkpoint was saved with fewer depformer steps.
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            target_shape = model_sd[name].shape
            if tensor.shape != target_shape:
                expanded = _repeat_first_dim_to_shape(tensor, target_shape)
                if expanded is not None:
                    logger.info(
                        "Expanding %s from %s to %s",
                        name,
                        tuple(tensor.shape),
                        tuple(target_shape),
                    )
                    state_dict[name] = expanded
                else:
                    logger.warning(
                        "Could not expand %s from %s to %s",
                        name,
                        tuple(tensor.shape),
                        tuple(target_shape),
                    )

    # Patch 2: fill missing module-list entries by copying 0..7 -> 8..15.
    if copy_missing_weights:
        for name, target_tensor in model_sd.items():
            if name in state_dict:
                continue
            source_name = _get_expanded_source_name(name)
            if source_name is None or source_name not in state_dict:
                continue
            source_tensor = state_dict[source_name]
            patched = _repeat_first_dim_to_shape(source_tensor, target_tensor.shape)
            if patched is None:
                logger.warning(
                    "Could not copy missing key %s from %s due to shape mismatch %s vs %s",
                    name,
                    source_name,
                    tuple(source_tensor.shape),
                    tuple(target_tensor.shape),
                )
                continue
            state_dict[name] = patched
            logger.info("Replacing %s <- %s", name, source_name)

    return state_dict


class MultiGPULMModel(transformer.StreamingContainer):
    """Wrapper for LMModel that shards transformer layers across multiple GPUs."""
    def __init__(self, model: LMModel, device_map: dict[str, int]):
        super().__init__()
        self._model = model
        self._device_map = device_map
        self._is_multi_gpu = True
        self._depformer_device = torch.device(f"cuda:{max(device_map.values())}")
        self._primary_device = torch.device("cuda:0")
        
        # Setup streams and events for sync
        self._gpu_streams = {str(i): torch.cuda.Stream(device=i) for i in set(device_map.values())}
        self._boundary_events = {str(i): torch.cuda.Event(enable_timing=False, interprocess=False) for i in set(device_map.values())}

    @property
    def device(self):
        return self._primary_device

    def __getattr__(self, name):
        if name in ["_model", "_device_map", "_is_multi_gpu", "_depformer_device", "_primary_device", "_gpu_streams", "_boundary_events"]:
            return super().__getattr__(name)
        return getattr(self._model, name)

    def parameters(self, recurse=True):
        # We ensure embeddings (on cuda:0) are returned first
        return self._model.parameters(recurse=recurse)

    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        return self._model.embed_codes(sequence.to(self._primary_device))

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        current_device = x.device
        
        # We need to manually iterate over transformer layers to handle transitions
        x = self._model.transformer.input_proj(x) if self._model.transformer.input_proj else x
        
        for i, layer in enumerate(self._model.transformer.transformer.layers):
            target_device_idx = self._device_map[f"layers.{i}"]
            target_device = torch.device(f"cuda:{target_device_idx}")
            
            if target_device != current_device:
                # Synchronization boundary
                prev_dev_str = str(current_device.index)
                target_dev_str = str(target_device_idx)
                
                with torch.cuda.stream(self._gpu_streams[prev_dev_str]):
                    self._boundary_events[prev_dev_str].record()
                
                with torch.cuda.stream(self._gpu_streams[target_dev_str]):
                    self._gpu_streams[target_dev_str].wait_event(self._boundary_events[prev_dev_str])
                    x = x.to(target_device, non_blocking=True)
                
                current_device = target_device
                
            with torch.cuda.stream(self._gpu_streams[str(current_device.index)]):
                x = layer(x)
        
        # Wait for last GPU to finish transformer before out_norm/linear
        torch.cuda.current_stream().wait_stream(self._gpu_streams[str(current_device.index)])
        
        if self._model.out_norm:
            x = self._model.out_norm(x)
        text_logits = self._model.text_linear(x)
        text_logits = text_logits[:, None]
        return x, text_logits

    def forward_depformer(self, cb_index, sequence, transformer_out, skip_transfer=False):
        if not skip_transfer:
            sequence = sequence.to(self._depformer_device, non_blocking=True)
            transformer_out = transformer_out.to(self._depformer_device, non_blocking=True)
        return self._model.forward_depformer(cb_index, sequence, transformer_out)


def _find_parent_real_device(model: torch.nn.Module, parent_name: str) -> torch.device | None:
    parent = model if parent_name == "" else model.get_submodule(parent_name)
    for p in parent.parameters(recurse=False):
        if p.device.type != "meta":
            return p.device
    for b in parent.buffers(recurse=False):
        if b.device.type != "meta":
            return b.device
    return None


def _set_tensor_on_module(
    model: torch.nn.Module,
    full_name: str,
    value: torch.Tensor,
    is_parameter: bool,
) -> None:
    if "." in full_name:
        parent_name, tensor_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent = model
        tensor_name = full_name

    if is_parameter:
        old_param = parent._parameters[tensor_name]
        parent._parameters[tensor_name] = torch.nn.Parameter(
            value, requires_grad=old_param.requires_grad
        )
    else:
        parent._buffers[tensor_name] = value


def _materialize_meta_tensors(
    model: torch.nn.Module,
    target_device: torch.device,
    copy_missing_weights: bool,
) -> None:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def _materialize(name: str, tensor: torch.Tensor, is_parameter: bool) -> None:
        if tensor.device.type != "meta":
            return

        source_name = _get_expanded_source_name(name) if copy_missing_weights else None
        source_tensor = None
        if source_name is not None:
            source_tensor = params.get(source_name)
            if source_tensor is None:
                source_tensor = buffers.get(source_name)

        value = None
        device = None
        if source_tensor is not None and source_tensor.device.type != "meta":
            device = source_tensor.device
            source_cast = source_tensor.to(device=device, dtype=tensor.dtype)
            value = _repeat_first_dim_to_shape(source_cast, tensor.shape)

        if value is None:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            device = _find_parent_real_device(model, parent_name) or target_device
            value = torch.zeros(tensor.shape, device=device, dtype=tensor.dtype)
            logger.warning("Materializing missing tensor %s with zeros on %s", name, device)

        _set_tensor_on_module(model, name, value, is_parameter=is_parameter)
        if is_parameter:
            params[name] = model.get_parameter(name)
        else:
            buffers[name] = model.get_buffer(name)

    for name, param in list(model.named_parameters()):
        _materialize(name, param, is_parameter=True)
    for name, buf in list(model.named_buffers()):
        _materialize(name, buf, is_parameter=False)

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_buffers:
        preview = (meta_params + meta_buffers)[:8]
        raise RuntimeError(
            f"Found unresolved meta tensors after materialization: {preview}"
        )


def _count_named_tensor_devices(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    meta_names: list[str] = []
    for name, tensor in named_tensors:
        dev_type = tensor.device.type
        counts[dev_type] = counts.get(dev_type, 0) + 1
        if dev_type == "meta":
            meta_names.append(name)
    return counts, meta_names


def get_model_device_summary(
    model: torch.nn.Module,
) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    """Return parameter and buffer device counts plus unresolved meta tensor names."""
    param_counts, meta_params = _count_named_tensor_devices(list(model.named_parameters()))
    buffer_counts, meta_buffers = _count_named_tensor_devices(list(model.named_buffers()))
    return param_counts, buffer_counts, meta_params, meta_buffers


def validate_no_meta_tensors(
    model: torch.nn.Module,
    *,
    log_prefix: str = "[PARAM_CHECK]",
    log_missing_examples: int = 8,
) -> bool:
    """Log device placement summary and return False if unresolved meta tensors remain."""
    param_counts, buffer_counts, meta_params, meta_buffers = get_model_device_summary(model)
    logger.info(
        "%s params=%s buffers=%s",
        log_prefix,
        param_counts,
        buffer_counts,
    )
    if not meta_params and not meta_buffers:
        return True

    logger.error(
        "[ERROR] Meta tensors detected after model load (%d params, %d buffers).",
        len(meta_params),
        len(meta_buffers),
    )
    preview = (meta_params + meta_buffers)[:log_missing_examples]
    logger.error("[ERROR] Examples: %s", ", ".join(preview))
    return False


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model
def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
    lowvram: bool = False,
    multi_gpu: bool = False,
    gpus: int | None = None,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
        lowvram: If True, use 4-bit quantization via bitsandbytes to reduce VRAM.
    """
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays

    if lowvram and filename is not None:
        return _get_moshi_lm_lowvram(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    if cpu_offload and filename is not None and not multi_gpu:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    if multi_gpu and filename is not None:
        return _get_moshi_lm_multi_gpu(
            filename, copy_missing_weights, device, dtype, lm_kwargs, gpus or torch.cuda.device_count()
        )

    if filename is not None and str(filename).endswith(".onnx"):
        return _get_moshi_lm_onnx(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    logger.info("[MODEL_LOAD] moshi initialized")
    logger.info(f"[MODEL_LOAD] target_device={device}")
    logger.info(f"[MODEL_LOAD] dtype={dtype}")

    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    if filename is None:
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    filename = str(filename)

    # Load and patch state_dict on CPU before moving to the target device.
    state_dict = _load_lm_state_dict(filename, device="cpu")
    model_sd = model.state_dict()
    state_dict = _patch_state_dict(
        state_dict=state_dict,
        model_sd=model_sd,
        copy_missing_weights=copy_missing_weights,
    )

    # Assign weights to target device
    dev = torch.device(device) if isinstance(device, str) else device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys:
        logger.warning(
            "Missing %d LM keys while loading %s (first 8: %s)",
            len(incompatible.missing_keys),
            filename,
            incompatible.missing_keys[:8],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected %d LM keys while loading %s (first 8: %s)",
            len(incompatible.unexpected_keys),
            filename,
            incompatible.unexpected_keys[:8],
        )

    _materialize_meta_tensors(
        model=model,
        target_device=dev,
        copy_missing_weights=copy_missing_weights,
    )
    model.eval()
    model._is_offloaded = False
    return model.to(device=device, dtype=dtype)


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
        import accelerate
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install <accelerate>"
        )

    filename = str(filename)
    logger.info("[MODEL_LOAD] Loading model with CPU offloading enabled (Accelerate)")
    logger.info(f"[MODEL_LOAD] filename={filename}")
    logger.info(f"[MODEL_LOAD] target_device={device}")
    logger.info(f"[MODEL_LOAD] Torch version: {torch.__version__}")
    try:
        logger.info(f"[MODEL_LOAD] Accelerate version: {accelerate.__version__}")
    except Exception:
        pass

    # 1. Create model on meta device to save RAM
    with init_empty_weights():
        model = LMModel(device="meta", dtype=dtype, **lm_kwargs)

    # 2. Determine target device
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model = LMModel(device=dev, dtype=dtype, **lm_kwargs)
        model_sd = model.state_dict()
        state_dict = _load_lm_state_dict(filename, device="cpu")
        state_dict = _patch_state_dict(
            state_dict=state_dict,
            model_sd=model_sd,
            copy_missing_weights=copy_missing_weights,
        )
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        if incompatible.missing_keys:
            logger.warning(
                "Missing %d LM keys while loading %s (first 8: %s)",
                len(incompatible.missing_keys),
                filename,
                incompatible.missing_keys[:8],
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "Unexpected %d LM keys while loading %s (first 8: %s)",
                len(incompatible.unexpected_keys),
                filename,
                incompatible.unexpected_keys[:8],
            )
        _materialize_meta_tensors(
            model=model,
            target_device=dev,
            copy_missing_weights=copy_missing_weights,
        )
        model.eval()
        return model

    # 3. Infer device map
    # We need to specify no_split_module_classes to prevent breaking layers across devices
    device_map = infer_auto_device_map(
        model,
        max_memory=None,
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # 4. Load on CPU, patch shape/key mismatches, and then dispatch per the inferred device map.
    logger.info(f"Dispatching model with device_map: {device_map}")

    model_sd = model.state_dict()
    state_dict = _load_lm_state_dict(filename, device="cpu")
    state_dict = _patch_state_dict(
        state_dict=state_dict,
        model_sd=model_sd,
        copy_missing_weights=copy_missing_weights,
    )
    for key in state_dict:
        state_dict[key] = state_dict[key].to(dtype=dtype)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys:
        logger.warning(
            "Missing %d LM keys while loading %s (first 8: %s)",
            len(incompatible.missing_keys),
            filename,
            incompatible.missing_keys[:8],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected %d LM keys while loading %s (first 8: %s)",
            len(incompatible.unexpected_keys),
            filename,
            incompatible.unexpected_keys[:8],
        )

    _materialize_meta_tensors(
        model=model,
        target_device=torch.device("cpu"),
        copy_missing_weights=copy_missing_weights,
    )

    model = dispatch_model(model, device_map=device_map)
    validate_no_meta_tensors(model, log_prefix="[POST_DISPATCH]")

    model.eval()
    model._is_offloaded = True
    return model


def _get_moshi_lm_lowvram(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with 4-bit quantization using bitsandbytes."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "lowvram mode requires the 'bitsandbytes' package. "
            "Install it with: pip install bitsandbytes"
        )

    filename = str(filename)
    logger.info("[MODEL_LOAD] Loading model with 4-bit quantization (bitsandbytes)")
    logger.info(f"[MODEL_LOAD] filename={filename}")

    # 1. Initialize model on CPU
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # 2. Identify and replace Linear layers with Linear4bit
    def replace_linear(m):
        for name, child in m.named_children():
            if isinstance(child, torch.nn.Linear):
                new_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=dtype,
                    quant_type="nf4",
                )
                setattr(m, name, new_layer)
            else:
                replace_linear(child)

    replace_linear(model)

    # 3. Load state dict on CPU
    state_dict = _load_lm_state_dict(filename, device="cpu")
    model_sd = model.state_dict()
    state_dict = _patch_state_dict(
        state_dict=state_dict,
        model_sd=model_sd,
        copy_missing_weights=copy_missing_weights,
    )

    # 4. Load weights. For bnb.nn.Linear4bit, we must NOT use assign=True 
    # to allow the layer to handle quantization during loading.
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning(
            "Missing %d LM keys while loading %s (first 8: %s)",
            len(incompatible.missing_keys),
            filename,
            incompatible.missing_keys[:8],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected %d LM keys while loading %s (first 8: %s)",
            len(incompatible.unexpected_keys),
            filename,
            incompatible.unexpected_keys[:8],
        )

    _materialize_meta_tensors(
        model=model,
        target_device=torch.device("cpu"),
        copy_missing_weights=copy_missing_weights,
    )

    # 5. Move to target device
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(device=dev)

    model.eval()
    model._is_offloaded = False
    return model
def _get_moshi_lm_multi_gpu(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
    num_gpus: int,
) -> MultiGPULMModel:
    """Load Moshi LM distributed across multiple GPUs."""
    filename = str(filename)
    logger.info(f"[MODEL_LOAD] Loading model sharded across {num_gpus} GPUs")

    # 1. Create model on CPU first
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
    
    # 2. Load and patch state dict on CPU
    state_dict = _load_lm_state_dict(filename, device="cpu")
    model_sd = model.state_dict()
    state_dict = _patch_state_dict(
        state_dict=state_dict,
        model_sd=model_sd,
        copy_missing_weights=copy_missing_weights,
    )

    # 3. Calculate layer sharding
    num_layers = lm_kwargs["num_layers"]
    # Simple even split for now, can be tuned with depformer_cost_ratio later
    layers_per_gpu = (num_layers + num_gpus - 1) // num_gpus
    
    device_map = {}
    for i in range(num_layers):
        gpu_idx = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"layers.{i}"] = gpu_idx
        
    logger.info(f"Layer sharding: {device_map}")

    # 4. Move layers to GPUs
    for i in range(num_layers):
        gpu_idx = device_map[f"layers.{i}"]
        model.transformer.transformer.layers[i].to(device=f"cuda:{gpu_idx}", dtype=dtype)
    
    # Text/Audio embeddings on cuda:0
    model.text_emb.to(device="cuda:0", dtype=dtype)
    for emb in model.emb:
        emb.to(device="cuda:0", dtype=dtype)
    if model.transformer.input_proj:
        model.transformer.input_proj.to(device="cuda:0", dtype=dtype)
        
    # Depformer and output heads on the last GPU
    last_gpu = f"cuda:{num_gpus - 1}"
    model.depformer.to(device=last_gpu, dtype=dtype)
    model.out_norm.to(device=last_gpu, dtype=dtype)
    model.text_linear.to(device=last_gpu, dtype=dtype)
    for head in model.linears:
        head.to(device=last_gpu, dtype=dtype)
    for emb in model.depformer_emb:
        emb.to(device=last_gpu, dtype=dtype)
    model.depformer_text_emb.to(device=last_gpu, dtype=dtype)
    for d_in in model.depformer_in:
        d_in.to(device=last_gpu, dtype=dtype)

    # 5. Load patched state_dict (strict=False because we sharded manually)
    model.load_state_dict(state_dict, strict=False, assign=True)
    
    _materialize_meta_tensors(
        model=model,
        target_device=torch.device("cpu"),
        copy_missing_weights=copy_missing_weights,
    )

    model.eval()
    return MultiGPULMModel(model, device_map)


def _get_moshi_lm_onnx(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM optimized with ONNX Runtime and OpenVINO Heterogeneous Execution."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX execution requires the 'onnxruntime' package. "
            "Install it with: pip install onnxruntime-openvino"
        )

    filename = str(filename)
    logger.info(f"[MODEL_LOAD] Loading ONNX model from {filename} with OpenVINO Heterogeneous Execution")
    
    # Generate PyTorch stub model
    model = LMModel(device="meta", dtype=dtype, **lm_kwargs)
    model.eval()
    
    # Setup OpenVINO Heterogeneous properties
    options = {"device_type": "HETERO:GPU,CPU", "precision": "FP16"}
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        ("OpenVINOExecutionProvider", options),
        "CPUExecutionProvider"
    ]
    
    logger.info(f"[MODEL_LOAD] Initializing ONNX InferenceSession with providers: {providers}")
    session = ort.InferenceSession(filename, sess_options=session_options, providers=providers)
    
    # Monkeypatching the critical context wrapper
    model._ort_session = session
    model._is_offloaded = False
    
    def onnx_forward_codes(sequence: torch.Tensor):
        logger.warning("ONNX _ort_session was invoked but full ONNX forward pass logic needs matching I/O definitions.")
        raise NotImplementedError("ONNX forward pass signature must be matched to exported inputs.")
        
    model.forward_codes = onnx_forward_codes
    return model
