# BrainGPT: High-Performance Transformer with Rotary Embeddings

> **Abstract:** We present BrainGPT, a scalable causal Transformer architecture designed with an emphasis on efficiency, long-context generalization, and deployment readiness[cite: 1]. [cite_start]The model integrates Rotary Positional Embeddings (RoPE), explicit key-value (KV) caching for fast autoregressive inference, and optional FlashAttention kernels[cite: 1].

## 📄 Technical Paper
For a detailed architectural analysis and system-level design principles, please refer to the official report:
* [**Download BrainGPT Technical Paper (PDF)**](BrainGPT_Technical_Paper.pdf)

## Core Technical Features
* [cite_start]**Rotary Positional Embeddings (RoPE):** Encodes positional information via rotations in query and key subspaces, enabling context extrapolation[cite: 1].
* [cite_start]**Key-Value (KV) Caching:** Reduces inference complexity from quadratic to linear time by maintaining per-layer caches during decoding[cite: 1].
* [cite_start]**FlashAttention Integration:** Leverages optimized kernels to reduce memory consumption and improve throughput when available[cite: 1].
* [cite_start]**Deployment Readiness:** Prioritizes modularity and transparency, making it suitable for both research and practical deployment[cite: 1].

## Architectural Comparison
| Component | GPT-2 | LLaMA | **BrainGPT** |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Absolute | RoPE | **RoPE (dynamic)** |
| **KV Cache** | No | Yes | **Yes** |
| **FlashAttention** | No | Yes | **Optional** |
| **Long Context** | Limited | Strong | **Strong** |
| **Deployment Export** | No | No | **ONNX-ready** |
[cite_start]*(Ref: BrainGPT Technical Paper, Table 1)* [cite: 1]

## Installation & Usage
```bash
# Install core dependencies
pip install torch torchvision torchaudio

# Run model
python braingpt.py
