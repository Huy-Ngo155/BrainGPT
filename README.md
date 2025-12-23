BrainGPT: High-Performance Transformer with Rotary Embeddings and Flash-Attention
BrainGPT is a specialized Large Language Model (LLM) architecture designed for high-throughput training and extended context handling. This project implements modern Transformer optimizations used in state-of-the-art models to achieve superior memory efficiency and scaling capabilities.

Core Technical Features
Rotary Positional Embeddings (RoPE): Implements a frequency-based rotation mechanism to capture long-range token dependencies effectively, supporting context lengths up to 32,768 tokens.

Flash-Attention Integration: Supports the flash_attn_qkvpacked_func kernel to reduce attention complexity from O(N 
2
 ) to O(N), significantly optimizing VRAM usage on compatible GPUs.

Distributed Data Parallel (DDP): Fully compatible with torch.distributed for multi-GPU training environments, ensuring seamless model scaling.

Precision and Scaling: Utilizes GradScaler for mixed-precision training and includes a robust BrainTrainer class with gradient clipping and AdamW optimization.

Memory Profiling: Includes a built-in MemoryProfiler to track peak VRAM allocation and system resources during execution.

Model Specifications
Maximum Sequence Length: 2,048 (default).

Vocabulary Size: 50,000.

Embedding Dimension: 768.

Transformer Layers: 12.

Attention Heads: 12.

Installation
Bash
pip install torch torchvision torchaudio
# Optional: Install Flash Attention for hardware acceleration
pip install flash-attn --no-build-isolation
Usage and Testing
Training with DDP

To launch distributed training on multiple GPUs:

Bash
torchrun --nproc_per_node=[NUM_GPUS] ul.py
Validation and Export

Memory Benchmarking: The system automatically measures memory usage upon initialization to ensure VRAM efficiency.

ONNX Export: Use the export_model function to generate an optimized brain_model.onnx file with dynamic axes for production deployment.
