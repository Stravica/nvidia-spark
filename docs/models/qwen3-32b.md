# Qwen3-32B on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3-32B on NVIDIA DGX Spark (GB10 Grace Blackwell) using vLLM.

## Model Specifications

- **Model:** Qwen/Qwen3-32B
- **Parameters:** 32.8 billion
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Context Length:** 32,768 tokens (native), extendable to 131K with YaRN
- **Memory Requirements:**
  - BF16: ~61 GB actual (works on DGX Spark with current config)
  - FP8: ~40 GB (alternative for more headroom)
  - INT4: ~20 GB (alternative for maximum concurrency)

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
- **Optimization Strategy:** Maximize batch sizes and enable caching
- **Measured Throughput:**
  - Single request: ~3-4 tokens/sec generation (measured)
  - Batched requests: Scales linearly with concurrent requests
  - Note: Single-request latency prioritizes quality; batching significantly improves aggregate throughput

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

## Working Configuration

### docker-compose.yml

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
    command: >
      --model Qwen/Qwen3-32B
      --download-dir /root/.cache/huggingface
      --max-model-len 24000
      --gpu-memory-utilization 0.85
      --max-num-batched-tokens 16384
      --max-num-seqs 48
      --enable-prefix-caching
      --trust-remote-code
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

### Parameter Explanation

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--model` | `Qwen/Qwen3-32B` | Full BF16 model (no quantization needed with NVIDIA container) |
| `--max-model-len` | `24000` | Conservative context for memory headroom |
| `--gpu-memory-utilization` | `0.85` | Safe for unified memory architecture |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `48` | Balance between throughput and latency |
| `--enable-prefix-caching` | enabled | Caches common prompt prefixes for efficiency |
| `--trust-remote-code` | enabled | Required for Qwen models |

### Alternative: FP8 Quantized Configuration

For even better performance, use pre-quantized model:

```yaml
command: >
  --model Qwen/Qwen3-32B-FP8
  --download-dir /root/.cache/huggingface
  --max-model-len 32000
  --gpu-memory-utilization 0.90
  --max-num-batched-tokens 16384
  --max-num-seqs 64
  --enable-prefix-caching
  --trust-remote-code
```

Benefits:
- ~32GB model size (vs ~63GB)
- More memory for KV cache
- Higher concurrent request handling
- Similar accuracy to BF16

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

```bash
docker compose up -d vllm-qwen3-32b
```

### Monitor Loading

```bash
docker compose logs -f vllm-qwen3-32b
```

Model loading takes ~7-8 minutes on first run:
1. Model download/verification (~2-3 min if cached, skip if already downloaded)
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

# Test inference
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

### Issue: FP8 Quantization Fails

**Symptom:**
```
RuntimeError: Error Internal
cutlass_scaled_mm failed
```

**Solution:**
- Remove `--quantization fp8` flag (NVIDIA container handles this automatically)
- Or use pre-quantized model: `Qwen/Qwen3-32B-FP8`

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

### For Maximum Throughput

```yaml
--max-model-len 20000              # Reduce context if not needed
--gpu-memory-utilization 0.90      # Increase memory usage
--max-num-batched-tokens 20480     # Larger batches
--max-num-seqs 64                  # More concurrent requests
--enable-prefix-caching            # Enable caching
```

### For Minimum Latency

```yaml
--max-model-len 16384              # Smaller context
--gpu-memory-utilization 0.75      # Leave headroom
--max-num-batched-tokens 8192      # Smaller batches
--max-num-seqs 16                  # Fewer concurrent requests
```

## Monitoring

### GPU Metrics

```bash
# Inside container
docker exec vllm-qwen3-32b nvidia-smi

# vLLM metrics
curl http://localhost:8000/metrics
```

### Key Metrics to Watch
- **Memory Usage:** Should be ~80-95% of allocated
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check tokens/sec in metrics
- **Queue Depth:** Monitor pending requests

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits token generation speed (~3-4 tokens/sec single request)
2. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs) due to GB10 architecture
3. **Single-Request Latency:** Optimized for throughput over latency; batching is essential for performance
4. **Context Length:** 24K configured (vs 32K native) to ensure adequate KV cache for concurrent requests

## References

- **Model:** https://huggingface.co/Qwen/Qwen3-32B
- **vLLM Docs:** https://docs.vllm.ai
- **NVIDIA Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

## Version History

- **2025-11-06:** Initial configuration for DGX Spark with NVIDIA container 25.09
- Model: Qwen3-32B (full BF16)
- Container: nvcr.io/nvidia/vllm:25.09-py3
- Status: Working configuration documented

---

**Last Updated:** 2025-11-06
**Tested On:** NVIDIA DGX Spark (GB10), CUDA 13.0, vLLM 0.10.1.1
