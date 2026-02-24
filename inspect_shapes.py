from safetensors.torch import load_file
import torch

path = '/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/31c4ae40e3710304737e6f7b70df23bd06917e5c/model.safetensors'
try:
    state_dict = load_file(path, device='cpu')
    key = 'transformer.layers.19.self_attn.in_proj_weight'
    if key in state_dict:
        print(f'Key {key} shape: {state_dict[key].shape}')
    else:
        print(f'Key {key} not found.')
        
    # Also check a depformer key
    depf_key = 'depformer.layers.0.linear_in.weight'
    if depf_key in state_dict:
         print(f'Key {depf_key} shape: {state_dict[depf_key].shape}')
        
except Exception as e:
    print(f'Error: {e}')
