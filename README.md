# BrainGPT: High-Performance Transformer with Rotary Embeddings and Flash-Attention

BrainGPT is a specialized Large Language Model (LLM) architecture designed for high-throughput training and extended context handling. This project implements modern Transformer optimizations to achieve superior memory efficiency and scaling capabilities.

## Core Technical Features

* **Rotary Positional Embeddings (RoPE):** Implements a frequency-based rotation mechanism to capture long-range token dependencies effectively, supporting context lengths up to 32,768 tokens.
* **Flash-Attention Integration:** Supports the `flash_attn_qkvpacked_func` kernel to reduce attention complexity from $O(N^2)$ to $O(N)$, significantly optimizing VRAM usage on compatible GPUs.
* **Distributed Data Parallel (DDP):** Fully compatible with `torch.distributed` for multi-GPU training environments, ensuring seamless model scaling.
* **Memory Profiling:** Includes a built-in `MemoryProfiler` class to monitor peak VRAM allocation and system resources during execution.

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Maximum Sequence Length** | 2,048 (Default) |
| **Vocabulary Size** | 50,000 |
| **Embedding Dimension** | 768 |
| **Transformer Layers** | 12 |
| **Attention Heads** | 12 |

## Installation

Ensure you have a CUDA-capable environment and the following dependencies installed:

```bash
pip install torch torchvision torchaudio
# Optional: Install Flash Attention for hardware acceleration
pip install flash-attn --no-build-isolation
