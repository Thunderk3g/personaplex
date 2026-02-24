import torch
from moshi.models.lm import LMModel
import logging

# Set up dummy config for LMModel
lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
}

try:
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)
    
    print('Model created on meta.')
    params = list(model.named_parameters())
    print('Total parameters:', len(params))
    # Print some layer 19 params
    l19_params = [n for n, p in params if 'transformer.layers.19' in n]
    print('Layer 19 params:', l19_params[:5])
    
    # Check for depformer params
    depf_params = [n for n, p in params if 'depformer' in n]
    print('Depformer params example:', depf_params[:2])
    
except Exception as e:
    print(f'Error: {e}')
