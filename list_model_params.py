import torch
from moshi.models.lm import LMModel
from accelerate import init_empty_weights

lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 16,
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
    "depformer_multi_linear": True,
    "delays": [0] * 17,
}

with init_empty_weights():
    model = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)

names = [n for n, _ in model.named_parameters()]
print("Total parameters:", len(names))

# Print top level names
print("\nTop level parameters (non-layer):")
for n in names:
    if 'layers.' not in n:
        print(n)

# Print a few samples of layers
print("\nSample transformer.layers.0:")
for n in names:
    if n.startswith('transformer.layers.0.'):
        print(n)

print("\nSample depformer.layers.0:")
for n in names:
    if n.startswith('depformer.layers.0.'):
        print(n)
