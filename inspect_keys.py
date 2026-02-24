from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    keys = load_file(path, device='cpu').keys()
    test_key1 = 'transformer.layers.0.self_attn.in_proj_weight'
    test_key2 = 'transformer.layers.19.self_attn.in_proj_weight'
    print(f'Layer 0 MHA Key: {test_key1 in keys}')
    print(f'Layer 19 MHA Key: {test_key2 in keys}')
    
    # Check for any depformer keys
    depf_keys = [k for k in keys if 'depformer' in k]
    print(f'Depformer keys count: {len(depf_keys)}')
    if depf_keys:
        print(f'First depf key: {depf_keys[0]}')
        
except Exception as e:
    print(f'Error: {e}')
