"""
Robust Model Loading Script with Unsloth and Hugging Face Fallback
-----------------------------------------------------------------
This script is designed to load large language models (7B+) efficiently using Unsloth.
It includes 4-bit quantization to prevent OOM errors and a fallback mechanism 
for models with architectures not yet supported by Unsloth (e.g., Moshi/Helium).

Prerequisites:
# pip install unsloth
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
"""

import torch
import os
import psutil
import signal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Technical Safeguards for RTX 3050 (4GB) ---
# 1. Global Inductor Suppression
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# 2. Prevent TorchDynamo from attempting compilation
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not found. Falling back to standard Transformers.")

# --- Configuration & Bug Mitigation ---
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Example model, replace with yours
MAX_SEQ_LENGTH = 2048  # Supports auto-extension, but setting it saves VRAM
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Explicitly use 4-bit quantization to prevent OOM

# Handle GPU capability for bfloat16 (Optimization)
if torch.cuda.is_available():
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        # Ampere GPUs (RTX 30xx, A100+) support bfloat16 for better precision/speed
        print("GPU supports bfloat16 - using it for maximum performance.")
        DTYPE = torch.bfloat16
    else:
        # Older GPUs keep float16
        print("GPU does not support bfloat16 - using float16.")
        DTYPE = torch.float16

# --- Memory & Timeout Safeguards ---
def check_memory_safeguard(threshold_percent=80):
    """Proactive OOM monitoring logic."""
    mem = psutil.virtual_memory()
    if mem.percent > threshold_percent:
        print(f"CRITICAL: System RAM usage at {mem.percent}%. Triggering emergency stop to prevent SIGSEGV.")
        return False
    return True

def timeout_handler(signum, frame):
    raise TimeoutError("Model initialization/Step 0 stalled (likely Inductor hang).")

def apply_unsloth_fixes(model):
    """Ensure torch.compile is never called on the model object."""
    if hasattr(model, "compile"):
        print("Safeguard: Bypassing torch.compile for stability.")
        model.compile = lambda *args, **kwargs: model
    return model

def apply_meta_tensor_patch():
    """Apply stability patches to handle meta tensors correctly."""
    try:
        from moshi.utils.patches import apply_meta_tensor_patch as patch
        patch()
    except ImportError:
        print("Warning: Could not apply stability patches (moshi.utils.patches not found).")

def load_model():
    """
    Attempts to load the model using Unsloth for maximum speed/efficiency.
    If Unsloth rejects the architecture, it safely falls back to standard Transformers.
    """
    
    # 1. Try Unsloth (Maximum efficiency for supported architectures)
    if UNSLOTH_AVAILABLE:
        try:
            print(f"Attempting to load {MODEL_NAME} with Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = MODEL_NAME,
                max_seq_length = MAX_SEQ_LENGTH,
                dtype = DTYPE,
                load_in_4bit = LOAD_IN_4BIT,
                # token = os.getenv("HF_TOKEN") # Uncomment if using gated models
            )
            
            # Enable 2x faster inference (Ensure compile is bypassed)
            model = apply_unsloth_fixes(model)
            FastLanguageModel.for_inference(model)
            print("Successfully loaded model using Unsloth!")
            return model, tokenizer
            
        except Exception as e:
            print(f"Unsloth could not load this model: {e}")
            print("Falling back to standard Hugging Face AutoModel...")

    # 2. Fallback: Standard Hugging Face with 4-bit quantization (bitsandbytes)
    # This acts as a safety net for custom architectures like Moshi/Helium.
    
    # Configure 4-bit quantization for the fallback
    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4", # Highly recommended for weight distribution
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {MODEL_NAME} via standard Transformers with quantization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with device_map="auto" to handle multi-GPU or CPU offloading automatically
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=DTYPE,
    )
    
    print("Successfully loaded model using standard fallback!")
    return model, tokenizer

def run_dummy_inference(model, tokenizer):
    """
    Performs a quick test to ensure the model is responsive after loading.
    """
    print("\nRunning dummy inference test...")
    prompt = "The future of AI is"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate tokens
    outputs = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Inference Result: {result[0]}")
    print("Dummy inference completed successfully!")

if __name__ == "__main__":
    # Set a 5-minute timeout for initialization
    signal.signal(signal.SIGALRM, timeout_handler)
    if os.name != 'nt': # signal.alarm is not available on Windows, but this is for Linux container
        signal.alarm(300) 

    try:
        # Check RAM before loading
        if not check_memory_safeguard():
            exit(1)

        apply_meta_tensor_patch()
        model, tokenizer = load_model()
        
        # Disable alarm on success
        if os.name != 'nt':
            signal.alarm(0)

        run_dummy_inference(model, tokenizer)
        print("\nAll systems go. Your model is ready for inference or fine-tuning.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to start model service. {e}")
        # Hint for OOM errors
        if "out of memory" in str(e).lower():
            print("HINT: Try reducing MAX_SEQ_LENGTH or ensure no other processes are using VRAM.")
