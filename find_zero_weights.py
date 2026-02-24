from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    keys = load_file(path, device='cpu').keys()
    zero_weight_keys = [k for k in keys if '.0.weight' in k]
    print(f'Total keys with .0.weight: {len(zero_weight_keys)}')
    for k in sorted(zero_weight_keys):
        print(k)
except Exception as e:
    print(f'Error: {e}')
