from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    state_dict = load_file(path, device='cpu')
    linear_keys = sorted([k for k in state_dict.keys() if k.startswith('linears.')])
    print(f'Total linears keys: {len(linear_keys)}')
    for k in linear_keys:
        print(f'{k}: {state_dict[k].shape}')
except Exception as e:
    print(f'Error: {e}')
