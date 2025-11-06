# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Host Information

**Host:** Nvidia DGX Spark
**Hostname:** spark.thefootonline.local

## Hardware Specifications

### NVIDIA DGX Spark - GB10 Grace Blackwell Superchip

**Processor:**
- **CPU:** 20-core Arm (10 Cortex-X925 + 10 Cortex-A725)
- **GPU:** NVIDIA Blackwell Architecture
  - 6,144 CUDA cores
  - 5th Gen Tensor Cores
  - 4th Gen RT Cores
  - 2 Copy Engines (simultaneous data transfers)

**Memory Architecture:**
- **Total Memory:** 128 GB LPDDR5x unified system memory
- **Memory Bandwidth:** 273 GB/s (shared between CPU and GPU)
- **Interface:** 256-bit, 4266 MHz
- **Channels:** 16 channels LPDDR5X 8533
- **Key Feature:** Coherent unified memory - no separate VRAM, entire 128GB pool available to GPU without system-to-VRAM transfer overhead

**AI Performance:**
- Up to 1 PFLOP sparse FP4 tensor performance
- Up to 1,000 TOPS (trillion operations per second) inference
- Supports AI models up to 200 billion parameters (single device)
- Can cluster 2 units for up to 405 billion parameter models

**Storage:**
- 1 TB or 4 TB NVMe M.2 with self-encryption

**Networking:**
- 1x RJ-45 (10 GbE)
- ConnectX-7 Smart NIC
- 2x QSFP Network connectors (ConnectX-7) - 200 Gbps aggregate
- Wi-Fi 7, Bluetooth 5.4

**Power:**
- TDP: 140W (GB10 SOC)
- Total Power: 240W external PSU (required - do not substitute)
- Operating temp: 0°C to 35°C (32°F to 95°F)

**Performance Characteristics:**
- **Bottleneck:** Memory bandwidth (273 GB/s) is the primary limiting factor for inference
- **Thermal:** Excellent sustained performance without throttling due to external PSU design
- **Best Use Case:** Prototyping, experimentation, smaller model serving with batching

## Overview

This repository contains Docker Compose configurations for running GPU-accelerated inference services. Two services are available:

1. **ollama** - General-purpose LLM inference server
2. **vllm-qwen3-32b** - vLLM server running the Qwen3-32B model

Both services are configured to use all available NVIDIA GPU resources on the unified memory architecture.

## Common Commands

### Start specific service
```bash
docker compose up -d vllm-qwen3-32b
```

### Start all services
```bash
docker compose up -d
```

### Stop services
```bash
docker compose down
```

### View logs
```bash
docker compose logs -f [service-name]
```

### Restart a service
```bash
docker compose restart [service-name]
```

## Model Specifications

### Qwen3-32B

**Architecture:**
- **Parameters:** 32.8 billion parameters
- **Layers:** 64 transformer layers
- **Attention:** Grouped Query Attention (GQA)
  - 64 query heads
  - 8 key-value heads
- **Positional Encoding:** Rotary Positional Embeddings (RoPE)
- **Activation Function:** SwiGLU
- **Normalization:** RMSNorm with pre-normalization

**Context Length:**
- Native: 32,768 tokens
- Extended (with YaRN): 131,072 tokens
- Recommended production: 32K tokens for optimal performance

**Memory Requirements by Precision:**
- **FP16 (16-bit):** ~80 GB - NOT FEASIBLE on DGX Spark (128GB total shared memory)
- **FP8 (8-bit):** ~40 GB - Recommended for DGX Spark
- **INT4 (4-bit):** ~20 GB - Best for DGX Spark, leaves memory for KV cache and batching

**Recommended Quantization for DGX Spark:**
Use FP8 or INT4 quantization to maximize available memory for KV cache and concurrent request handling.

## Service Details

### vllm-qwen3-32b
- **Port:** 8000
- **Model:** Qwen/Qwen3-32B
- **Current Config:**
  - Max context: 16384 tokens
  - GPU memory utilization: 55% (~70.4 GB of 128 GB)
  - Cache location: /opt/hf (mounted to container)
- **Environment:** Requires HF_TOKEN in .env file

### ollama
- **Port:** 11434
- **Keep alive:** 30 minutes after last use
- **Model storage:** /opt/ollama (mounted to container)

## vLLM Optimization for DGX Spark + Qwen3-32B

### Critical Optimization Parameters

**Memory Management:**
```bash
--gpu-memory-utilization 0.85-0.90
```
- **Current:** 0.55 (too conservative)
- **Recommended:** 0.85-0.90 for DGX Spark's unified memory
- DGX Spark has no separate VRAM, so higher utilization is safe
- Leaves ~13-19 GB for OS and overhead

**Context Length:**
```bash
--max-model-len 32000
```
- **Current:** 16384 tokens
- **Recommended:** 32000 tokens (Qwen3-32B native support)
- Matches model's native context window
- Can reduce if more concurrent requests needed

**Batch Processing:**
```bash
--max-num-batched-tokens 8192
```
- **Recommended:** 8192-16384 for throughput optimization
- Higher values improve throughput on memory-bandwidth-limited systems
- Critical for DGX Spark where bandwidth (273 GB/s) is the bottleneck

**Concurrent Requests:**
```bash
--max-num-seqs 32
```
- Allows up to 32 concurrent sequences
- Balance between throughput and latency
- Adjust based on typical request patterns

**Advanced Features:**
```bash
--enable-chunked-prefill    # Improved batching for long prompts
--enable-prefix-caching     # Cache common prompt prefixes (big win for similar requests)
--dtype bfloat16           # Or use float16
```

**Quantization (Recommended for DGX Spark):**
```bash
--quantization fp8          # Reduces memory to ~40GB
# OR
--quantization awq          # 4-bit quantization to ~20GB
```

### Benchmark-Informed Configuration

Based on community benchmarks, optimal vLLM configuration for Qwen3-32B:

```bash
vllm serve Qwen/Qwen3-32B \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --dtype bfloat16 \
  --trust-remote-code
```

**For FP8 Quantization (Recommended):**
```bash
vllm serve Qwen/Qwen3-32B \
  --quantization fp8 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code
```

### Performance Expectations

**DGX Spark Constraints:**
- Memory bandwidth (273 GB/s) is primary bottleneck, not compute
- Focus on maximizing batch sizes and caching
- Expect ~2-20 tokens/sec per request depending on batch size
- Throughput scales with batching: 1 request = ~20 tps, 32 requests = ~300+ tps aggregate

**Optimization Strategy:**
1. Use FP8 or INT4 quantization to free memory
2. Enable prefix caching for repeated prompt patterns
3. Maximize batch size with higher memory utilization
4. Use chunked prefill for long input sequences
5. Monitor memory usage and adjust max-model-len if needed

### Key vLLM Parameters Reference

| Parameter | Purpose | Default | Recommended |
|-----------|---------|---------|-------------|
| `--gpu-memory-utilization` | Fraction of GPU memory to use | 0.90 | 0.85-0.90 |
| `--max-model-len` | Maximum sequence length | Model max | 32000 |
| `--max-num-batched-tokens` | Max tokens per batch iteration | 2048 | 8192-16384 |
| `--max-num-seqs` | Max concurrent sequences | 256 | 32-64 |
| `--block-size` | KV cache block size | 16 | 16 (8,16,32 valid) |
| `--swap-space` | CPU swap space per GPU (GiB) | 4 | 0 (unified mem) |
| `--enable-chunked-prefill` | Batch prefill with decode | false | true |
| `--enable-prefix-caching` | Cache prompt prefixes | false | true |
| `--quantization` | Quantization method | none | fp8 or awq |

## Configuration

Environment variables are stored in `.env` file:
- `HF_TOKEN` - Hugging Face API token for model downloads

## Model-Specific Documentation

Detailed configuration guides for deployed models:
- **Qwen3-32B:** See `docs/models/qwen3-32b.md` for complete deployment guide, troubleshooting, and performance tuning

## Monitoring and Troubleshooting

**Check memory usage:**
```bash
docker exec vllm-qwen3-32b nvidia-smi
```

**View detailed vLLM metrics:**
```bash
curl http://localhost:8000/metrics
```

**Common Issues:**
- **OOM errors:** Reduce `--max-model-len` or `--gpu-memory-utilization`
- **Low throughput:** Increase `--max-num-batched-tokens` and enable caching
- **High latency:** Reduce `--max-num-seqs` or batch size
