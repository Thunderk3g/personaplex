from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    keys = load_file(path, device='cpu').keys()
    for k in ['linears.0.weight', 'depformer_in.0.weight', 'depformer_emb.0.weight', 'depformer_text_emb.weight']:
        print(f'Key {k} exists: {k in keys}')
        
    # Check with transformer prefix if needed
    for k in ['transformer.linears.0.weight', 'lm.linears.0.weight']:
        print(f'Key {k} exists: {k in keys}')
        
except Exception as e:
    print(f'Error: {e}')
