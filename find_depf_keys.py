from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    keys = load_file(path, device='cpu').keys()
    depf_keys = [k for k in keys if 'depformer' in k]
    print(f'Total keys with "depformer": {len(depf_keys)}')
    if depf_keys:
        print('Sample keys:', depf_keys[:20])
except Exception as e:
    print(f'Error: {e}')
