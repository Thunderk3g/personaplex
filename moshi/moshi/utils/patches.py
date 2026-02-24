# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import logging
from dataclasses import fields, replace, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)

def move_to_safe_device(obj: Any, target_device: torch.device) -> Any:
    """Recursively move tensors from meta device to a real device."""
    if isinstance(obj, torch.Tensor):
        if obj.device.type == 'meta':
            return torch.zeros(obj.shape, device=target_device, dtype=obj.dtype)
        return obj.to(target_device)
    elif isinstance(obj, dict):
        return {k: move_to_safe_device(v, target_device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_safe_device(v, target_device) for v in obj]
    elif is_dataclass(obj) and not isinstance(obj, type):
        new_values = {}
        for field in fields(obj):
            val = getattr(obj, field.name)
            new_values[field.name] = move_to_safe_device(val, target_device)
        return replace(obj, **new_values)
    elif hasattr(obj, "asdict"):
        # Handler for types like RingKVCache that have asdict
        d = obj.asdict()
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                new_v = v
                if v.device.type == 'meta':
                    new_v = torch.zeros(v.shape, device=target_device, dtype=v.dtype)
                else:
                    new_v = v.to(target_device)
                
                # Try to set it back if it's an attribute
                if hasattr(obj, k):
                    setattr(obj, k, new_v)
        return obj
    return obj

def apply_meta_tensor_patch():
    """Apply monkey patches to fix meta tensor issues during inference warmup."""
    from ..modules.streaming import StreamingModule
    from ..modules.transformer import StreamingTransformer
    from ..modules.rope import RotaryEmbedding

    logger.info("Applying Meta Tensor Patch to StreamingModule and StreamingTransformer")

    # Patch 1: Global StreamingModule initialization
    original_start_streaming = StreamingModule._start_streaming
    
    def patched_start_streaming(self, batch_size: int):
        # Call original initialization which might create meta tensors
        original_start_streaming(self, batch_size)
        
        def _fix_state(name: str, module: StreamingModule):
            if module._streaming_state is not None:
                # Determine a safe "real" device for this module
                safe_device = torch.device("cpu")
                # Try to find a real device from parameters
                for p in module.parameters():
                    if p.device.type != 'meta':
                        safe_device = p.device
                        break
                
                # Recursively fix meta tensors in the newly created state
                module._streaming_state = move_to_safe_device(module._streaming_state, safe_device)
        
        self._apply_named_streaming(_fix_state)

    StreamingModule._start_streaming = patched_start_streaming

    # Patch 2: Break RotaryEmbedding sharing in StreamingTransformer
    # Accelerate offloading creates issues when RoPE is shared across layers on different devices.
    original_transformer_init = StreamingTransformer.__init__
    
    def patched_transformer_init(self, *args, **kwargs):
        pos_emb_type = kwargs.get('positional_embedding', 'sin')
        max_period = kwargs.get('max_period', 10_000)
        
        # Call original init
        original_transformer_init(self, *args, **kwargs)
        
        # If RoPE is used, ensure each layer has its own instance to avoid device affinity issues
        if pos_emb_type in {"rope", "sin_rope"}:
            for layer in self.layers:
                # Get the device of the layer's parameters
                try:
                    p = next(layer.parameters())
                    dev, dtype = p.device, p.dtype
                except StopIteration:
                    dev, dtype = torch.device("cpu"), torch.float32
                
                # Assign a fresh RoPE instance to each layer
                layer.self_attn.rope = RotaryEmbedding(max_period=max_period).to(
                    device=dev, dtype=dtype
                )

    StreamingTransformer.__init__ = patched_transformer_init
    logger.info("Meta Tensor Patch applied successfully")
