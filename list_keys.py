from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    keys = sorted(load_file(path, device='cpu').keys())
    with open('/app/all_keys.txt', 'w') as f:
        for k in keys:
            f.write(k + '\n')
    print(f'Wrote {len(keys)} keys to /app/all_keys.txt')
except Exception as e:
    print(f'Error: {e}')
