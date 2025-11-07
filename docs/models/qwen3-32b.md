# Qwen3-32B on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3-32B on NVIDIA DGX Spark (GB10 Grace Blackwell) using vLLM.

**RECOMMENDED: Use the pre-quantized Qwen/Qwen3-32B-FP8 model for optimal performance on DGX Spark.**

## Model Specifications

- **Model:** Qwen/Qwen3-32B (with FP8 and BF16 variants)
- **Parameters:** 32.8 billion
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Context Length:** 32,768 tokens (native), extendable to 131K with YaRN
- **Memory Requirements:**
  - **FP8 (Qwen/Qwen3-32B-FP8):** ~32 GB - **RECOMMENDED** for DGX Spark
  - **BF16 (Qwen/Qwen3-32B):** ~65 GB - Alternative for quality-sensitive workloads
  - **INT4:** ~20 GB - Available but less common

## Hardware: NVIDIA DGX Spark

### Specifications
- **GPU:** NVIDIA GB10 Grace Blackwell (SM_121a)
  - 6,144 CUDA cores
  - 5th Gen Tensor Cores (FP8 support)
- **Memory:** 128 GB unified LPDDR5x (273 GB/s bandwidth)
- **Compute Capability:** 12.1
- **Key Feature:** Unified memory architecture (no separate VRAM)

### Performance Characteristics
- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** Use FP8 quantization + maximize batch sizes and enable caching
- **Measured Throughput:**
  - Single request: ~3-4 tokens/sec generation (measured)
  - Batched requests: Scales linearly with concurrent requests
  - Note: Single-request latency prioritizes quality; batching significantly improves aggregate throughput
- **Why FP8 is Better:**
  - Model memory: ~32 GB vs ~65 GB (BF16)
  - KV cache: ~66 GB vs ~28 GB (2.4x more)
  - Concurrent sequences: 64 vs 48 (33% more)
  - Quality: ~99% of BF16 performance on benchmarks

## Critical: Container Selection

### ❌ Do NOT Use: `vllm/vllm-openai:latest`
- **Problem:** Incompatible with GB10 architecture
- **Issues:**
  - FP8 quantization fails (CUTLASS kernel errors)
  - Incorrect memory detection
  - Triton compilation errors with sm_121a

### ✅ REQUIRED: `nvcr.io/nvidia/vllm:25.09-py3`
- **Version:** vLLM 0.10.1.1+381074ae.nv25.09
- **CUDA:** 13.0
- **Features:**
  - DGX Spark GB10 functional support
  - NVFP4 format support
  - FP8 precision on Blackwell GPUs
  - Optimized for GB10 unified memory

## Working Configurations

### Recommended: FP8 Quantized (docker-compose.yml)

```yaml
services:
  vllm-qwen3-32b-fp8:
    image: nvcr.io/nvidia/vllm:25.09-py3
    container_name: vllm-qwen3-32b-fp8
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      HF_TOKEN: "${HF_TOKEN}"
      HUGGING_FACE_HUB_TOKEN: "${HF_TOKEN}"
    command:
      - vllm
      - serve
      - Qwen/Qwen3-32B-FP8
      - --download-dir
      - /root/.cache/huggingface
      - --max-model-len
      - "32000"
      - --gpu-memory-utilization
      - "0.90"
      - --max-num-batched-tokens
      - "16384"
      - --max-num-seqs
      - "64"
      - --enable-prefix-caching
      - --trust-remote-code
    volumes:
      - /opt/hf:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles:
      - fp8
```

**Deployment:**
```bash
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8
```

### Alternative: BF16 Full Precision (docker-compose.yml)

```yaml
services:
  vllm-qwen3-32b:
    image: nvcr.io/nvidia/vllm:25.09-py3
    container_name: vllm-qwen3-32b
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      HF_TOKEN: "${HF_TOKEN}"
      HUGGING_FACE_HUB_TOKEN: "${HF_TOKEN}"
    command:
      - vllm
      - serve
      - Qwen/Qwen3-32B
      - --download-dir
      - /root/.cache/huggingface
      - --max-model-len
      - "24000"
      - --gpu-memory-utilization
      - "0.85"
      - --max-num-batched-tokens
      - "16384"
      - --max-num-seqs
      - "48"
      - --enable-prefix-caching
      - --trust-remote-code
    volumes:
      - /opt/hf:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Deployment:**
```bash
docker compose up -d vllm-qwen3-32b
```

### Parameter Explanation (FP8 - Recommended)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--model` | `Qwen/Qwen3-32B-FP8` | Pre-quantized by Qwen team, validated quality |
| `--max-model-len` | `32000` | Native model context (full 32K support) |
| `--gpu-memory-utilization` | `0.90` | Aggressive but safe with FP8's smaller footprint |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `64` | High concurrency enabled by large KV cache |
| `--enable-prefix-caching` | enabled | Caches common prompt prefixes for efficiency |
| `--trust-remote-code` | enabled | Required for Qwen models |

**Performance Benefits:**
- Model memory: ~32 GB (vs ~65 GB BF16)
- KV cache: ~66 GB (vs ~28 GB BF16)
- Concurrent sequences: 64 (vs 48 BF16)
- Context length: 32K (vs 24K BF16)
- Quality: 99% of BF16 on MMLU/GSM8K benchmarks

### Parameter Explanation (BF16 - Alternative)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--model` | `Qwen/Qwen3-32B` | Full BF16 model (higher quality, limited by memory) |
| `--max-model-len` | `24000` | Reduced context for adequate KV cache headroom |
| `--gpu-memory-utilization` | `0.85` | Conservative due to larger model size |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `48` | Lower concurrency due to limited KV cache |
| `--enable-prefix-caching` | enabled | Caches common prompt prefixes for efficiency |
| `--trust-remote-code` | enabled | Required for Qwen models |

**When to use BF16:**
- Extremely quality-sensitive applications (marginal gain)
- Lower concurrency requirements acceptable
- Willing to trade throughput for potential quality improvement

## Deployment

### Prerequisites

1. **Environment Variables:**
   Create `.env` file:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```

2. **Model Cache:**
   Ensure `/opt/hf` directory exists with proper permissions

### Start Service

**FP8 (Recommended):**
```bash
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8
```

**BF16 (Alternative):**
```bash
docker compose up -d vllm-qwen3-32b
```

### Monitor Loading

**FP8:**
```bash
docker compose logs -f vllm-qwen3-32b-fp8
```

**BF16:**
```bash
docker compose logs -f vllm-qwen3-32b
```

**FP8 Model loading takes ~9-10 minutes on first run:**
1. Model download (~6 min if not cached - 7 safetensors shards)
2. Loading safetensors shards (~3.4 min)
3. torch.compile (~17 sec)
4. KV cache initialization and CUDA graph capture (~6 sec)

**BF16 Model loading takes ~7-8 minutes on first run:**
1. Model download (~3-4 min if not cached - 17 safetensors shards)
2. Loading 17 safetensors shards (~5.8 min)
3. torch.compile (~19 sec)
4. CUDA kernel compilation (~42 sec on first run only)
5. KV cache initialization and CUDA graph capture (~6 sec)

### Health Check

```bash
# API health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test inference (FP8)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test inference (BF16)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Troubleshooting

### Issue: Container Fails with Memory Errors

**Symptom:**
```
ValueError: Free memory on device (XX GiB) is less than desired GPU memory utilization
```

**Solution:**
1. Stop all containers: `docker compose down`
2. Clean Docker resources: `docker system prune -f`
3. Reduce `--gpu-memory-utilization` to 0.70 or 0.75
4. Reduce `--max-model-len` to 16384

### Issue: FP8 Quantization Fails with On-the-Fly Quantization

**Symptom:**
```
RuntimeError: Error Internal
cutlass_scaled_mm failed
```

**Solution:**
- **Recommended:** Use pre-quantized model `Qwen/Qwen3-32B-FP8` instead
- The pre-quantized model is officially quantized by Qwen team and works reliably
- Avoid using `--quantization fp8` flag with base model (on-the-fly quantization)

### Issue: Model Download Incomplete

**Symptom:**
```
Loading safetensors checkpoint shards: X/17
[hangs or crashes]
```

**Solution:**
1. Verify `HF_TOKEN` is set correctly in `.env`
2. Check disk space: `df -h /opt/hf`
3. Clean incomplete downloads:
   ```bash
   rm -rf /opt/hf/models--Qwen--Qwen3-32B/*.incomplete
   ```
4. Restart container

### Issue: "Compute Capability 12.1" Warning

**Symptom:**
```
WARNING: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1
Maximum cuda capability supported is 12.0
```

**Status:** This is a cosmetic warning and can be safely ignored. The NVIDIA container has proper GB10 support despite the warning.

## Performance Tuning

### For Maximum Throughput (FP8 Recommended)

```yaml
# Use Qwen/Qwen3-32B-FP8 model
--max-model-len 32000              # Full context support
--gpu-memory-utilization 0.90      # High memory usage (safe with FP8)
--max-num-batched-tokens 20480     # Larger batches
--max-num-seqs 64                  # More concurrent requests
--enable-prefix-caching            # Enable caching
```

### For Minimum Latency (FP8 or BF16)

**FP8:**
```yaml
--max-model-len 16384              # Smaller context
--gpu-memory-utilization 0.85      # Moderate memory usage
--max-num-batched-tokens 8192      # Smaller batches
--max-num-seqs 16                  # Fewer concurrent requests
```

**BF16:**
```yaml
--max-model-len 16384              # Smaller context
--gpu-memory-utilization 0.75      # Conservative memory usage
--max-num-batched-tokens 8192      # Smaller batches
--max-num-seqs 16                  # Fewer concurrent requests
```

## Monitoring

### GPU Metrics

```bash
# FP8 container
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# BF16 container
docker exec vllm-qwen3-32b nvidia-smi

# vLLM metrics (both use same port)
curl http://localhost:8000/metrics
```

### Key Metrics to Watch

**FP8 Configuration:**
- **Memory Usage:** ~98 GB total (~32 GB model + 66 GB KV cache)
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check tokens/sec in metrics
- **Queue Depth:** Can handle 64 concurrent requests
- **KV Cache:** 271,360 tokens capacity

**BF16 Configuration:**
- **Memory Usage:** ~93 GB total (~65 GB model + 28 GB KV cache)
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check tokens/sec in metrics
- **Queue Depth:** Can handle 48 concurrent requests
- **KV Cache:** Smaller capacity than FP8

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits token generation speed (~3-4 tokens/sec single request)
2. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs) due to GB10 architecture
3. **Single-Request Latency:** Optimized for throughput over latency; batching is essential for performance
4. **Context Length:**
   - FP8: Full 32K native support
   - BF16: 24K configured (vs 32K native) due to KV cache memory constraints

## References

- **Model (FP8):** https://huggingface.co/Qwen/Qwen3-32B-FP8
- **Model (BF16):** https://huggingface.co/Qwen/Qwen3-32B
- **vLLM Docs:** https://docs.vllm.ai
- **NVIDIA Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

## Version History

- **2025-11-07:** Added FP8 configuration as primary recommendation
  - FP8 model: Qwen/Qwen3-32B-FP8 (32GB model, 66GB KV cache, 64 concurrent seqs)
  - BF16 model: Qwen/Qwen3-32B (65GB model, 28GB KV cache, 48 concurrent seqs)
  - Container: nvcr.io/nvidia/vllm:25.09-py3
  - Recommendation: Use FP8 for optimal DGX Spark performance

- **2025-11-06:** Initial configuration for DGX Spark with NVIDIA container 25.09
  - Model: Qwen3-32B (full BF16)
  - Container: nvcr.io/nvidia/vllm:25.09-py3
  - Status: Working configuration documented

---

**Last Updated:** 2025-11-07
**Tested On:** NVIDIA DGX Spark (GB10), CUDA 13.0, vLLM 0.10.1.1
**Recommended:** Use FP8 pre-quantized model for best performance
