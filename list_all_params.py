import torch
from moshi.models.lm import LMModel
import logging
from accelerate import init_empty_weights

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
    with init_empty_weights():
        model = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)
    
    print('All parameter names (subset):')
    names = [n for n, p in model.named_parameters()]
    print(names[:20])
    
    # Specifically look at depformer
    depf_names = [n for n in names if 'depformer' in n]
    print('Depformer parameters:', depf_names[:10])
    
except Exception as e:
    print(f'Error: {e}')
