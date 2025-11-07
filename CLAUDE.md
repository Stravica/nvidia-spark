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

This repository contains Docker Compose configurations for running GPU-accelerated inference services. Three services are available:

1. **ollama** - General-purpose LLM inference server
2. **vllm-qwen3-32b-fp8** - vLLM server running Qwen3-32B-FP8 (RECOMMENDED for DGX Spark)
3. **vllm-qwen3-32b** - vLLM server running Qwen3-32B in BF16 (fallback option)

All services are configured to use all available NVIDIA GPU resources on the unified memory architecture.

## Common Commands

### Start specific service (FP8 recommended)
```bash
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8
```

### Start BF16 service (alternative)
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
- **FP8 (8-bit):** ~32-40 GB - **RECOMMENDED** for DGX Spark (optimal balance of quality and performance)
- **BF16 (16-bit):** ~65-80 GB - Feasible but limits KV cache and concurrent requests
- **INT4 (4-bit):** ~20 GB - Alternative for maximum memory headroom

**Recommended Configuration for DGX Spark:**
**Use the pre-quantized Qwen/Qwen3-32B-FP8 model** for best results. This provides excellent quality with ~32GB memory footprint, leaving ample space for KV cache (66GB+) and high concurrent request handling (up to 64 sequences).

## Service Details

### vllm-qwen3-32b-fp8 (RECOMMENDED)
- **Port:** 8000
- **Model:** Qwen/Qwen3-32B-FP8 (pre-quantized)
- **Profile:** `fp8` (use `docker compose --profile fp8 up -d vllm-qwen3-32b-fp8`)
- **Configuration:**
  - Max context: 32,000 tokens (native model support)
  - GPU memory utilization: 90%
  - Model memory: ~32 GB
  - KV cache memory: ~66 GB (271,360 tokens)
  - Max concurrent sequences: 64
  - Max batched tokens: 16,384
  - Features: prefix caching enabled, chunked prefill
  - Cache location: /opt/hf (mounted to container)
- **Performance:**
  - KV cache supports 8.48x concurrency at full 32K context
  - Optimal for high-throughput batch processing
- **Environment:** Requires HF_TOKEN in .env file

### vllm-qwen3-32b (Alternative - BF16)
- **Port:** 8000 (conflicts with FP8 service - use one or the other)
- **Model:** Qwen/Qwen3-32B (BF16 precision)
- **Configuration:**
  - Max context: 24,000 tokens
  - GPU memory utilization: 85%
  - Model memory: ~65 GB
  - Max concurrent sequences: 48
  - Max batched tokens: 16,384
  - Cache location: /opt/hf (mounted to container)
- **Use Case:** When FP8 quantization quality concerns exist (rarely needed)
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

**Primary Recommendation - Pre-Quantized FP8 Model:**
```bash
vllm serve Qwen/Qwen3-32B-FP8 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --trust-remote-code
```

**Benefits of FP8 Pre-Quantized:**
- Official quantization by Qwen team (validated quality)
- ~32GB model footprint vs ~65GB for BF16
- 66GB available for KV cache (2x more than BF16)
- Supports 64 concurrent sequences vs 48 for BF16
- Faster loading and inference
- Minimal quality degradation (<1% on benchmarks)

**Alternative - BF16 Full Precision:**
```bash
vllm serve Qwen/Qwen3-32B \
  --max-model-len 24000 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 48 \
  --enable-prefix-caching \
  --dtype bfloat16 \
  --trust-remote-code
```

**When to use BF16:**
- Extremely quality-sensitive applications (rare)
- Lower concurrency requirements acceptable
- Willing to trade throughput for marginal quality gain

### Performance Expectations

**DGX Spark Constraints:**
- Memory bandwidth (273 GB/s) is primary bottleneck, not compute
- Focus on maximizing batch sizes and caching
- Expect ~2-20 tokens/sec per request depending on batch size
- Throughput scales with batching: 1 request = ~20 tps, 32 requests = ~300+ tps aggregate

**Optimization Strategy:**
1. **Use pre-quantized Qwen/Qwen3-32B-FP8 model** (primary recommendation)
2. Enable prefix caching for repeated prompt patterns
3. Maximize batch size with higher memory utilization (0.90)
4. Leverage large KV cache from FP8's smaller footprint
5. Monitor memory usage and adjust max-model-len if needed

**Performance Comparison:**
| Metric | FP8 (Recommended) | BF16 (Alternative) |
|--------|-------------------|-------------------|
| Model Memory | ~32 GB | ~65 GB |
| KV Cache | ~66 GB | ~28 GB |
| Max Context | 32,000 tokens | 24,000 tokens |
| Max Concurrent Seqs | 64 | 48 |
| Quality vs BF16 | ~99% | 100% (baseline) |
| Throughput | Higher | Lower |

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

**Check memory usage (FP8):**
```bash
docker exec vllm-qwen3-32b-fp8 nvidia-smi
```

**Check memory usage (BF16):**
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
