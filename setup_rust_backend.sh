#!/bin/bash
set -e

# ==============================================================================
# setup_rust_backend.sh
# 
# Installs and runs the official Kyutai Rust `moshi-backend` (Moshika/Candle)
# for ultra low-latency audio streaming with 8-bit quantized weights.
# This prevents Python/PyTorch overhead and utilizes SIMD acceleration.
# ==============================================================================

echo "=> Cloning Kyutai Moshi Repository..."
if [ ! -d "moshi_upstream" ]; then
    git clone https://github.com/kyutai-labs/moshi.git moshi_upstream
fi

cd moshi_upstream/rust/moshi-backend

echo "=> Installing Rust (if missing)..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "=> Building the optimized Rust Moshi Backend..."
# Build in release mode to enable maximum optimizations for Candle
cargo build --release

echo "=> Rust Backend Built!"
echo ""
echo "=> To run the server with 8-bit quantized weights, use the following command:"
echo ""
echo '  cargo run --release --bin moshi-backend -- \
    --hf-repo "kyutai/moshika-candle-q8" \
    --host "0.0.0.0" \
    --port 8998'
echo ""
echo "=> Note: Ensure you have sufficient permissions or sudo if using port 80/443."
