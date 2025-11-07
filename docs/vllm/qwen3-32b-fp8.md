# vLLM: Qwen3-32B-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3-32B-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** Pre-quantized `Qwen/Qwen3-32B-FP8` for optimal performance on DGX Spark unified memory architecture.

## Model Specifications

- **Model:** Qwen/Qwen3-32B-FP8 (officially pre-quantized by Qwen team)
- **Parameters:** 32.8 billion
- **Quantization:** FP8 (8-bit floating point)
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Context Length:** 32,768 tokens (native), extendable to 131K with YaRN
- **Memory Requirements:**
  - Model: ~32 GB
  - KV Cache: ~66 GB (with 32K context, 64 concurrent sequences)
  - Total: ~98 GB peak GPU memory usage

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics
- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + aggressive batching + prefix caching
- **Measured Performance:**
  - Single request: ~6-7 tokens/sec generation
  - Batched (64 concurrent): ~300-400 tokens/sec aggregate (estimated)
  - Time to first token: 200-500ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~32 GB (50% reduction vs BF16)
  - KV cache: ~66 GB (2.4x more than BF16 configuration)
  - Concurrent sequences: 64 (33% more than BF16)
  - Quality: ~99% of full precision on benchmarks

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

## Service Configuration

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
```

**Deployment:**
```bash
docker compose up -d vllm-qwen3-32b-fp8
```

### Parameter Explanation

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
- Model memory: ~32 GB (50% less than BF16)
- KV cache: ~66 GB (2.4x more than typical BF16 configurations)
- Concurrent sequences: 64 (optimal for DGX Spark)
- Context length: Full 32K native support
- Quality: 99% of full precision on MMLU/GSM8K benchmarks

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
# Start vLLM Qwen3-32B-FP8 service
docker compose up -d vllm-qwen3-32b-fp8
```

### Monitor Loading

```bash
# View logs
docker compose logs -f vllm-qwen3-32b-fp8

# Filter for key loading events
docker compose logs -f vllm-qwen3-32b-fp8 | grep -E "(Loading|INFO|Ready|Serving)"
```

**Model loading takes ~9-10 minutes on first run:**
1. Model download (~6 min if not cached - 7 safetensors shards, ~32GB)
2. Loading safetensors shards (~3.4 min)
3. torch.compile (~17 sec)
4. KV cache initialization and CUDA graph capture (~6 sec)

**Subsequent startups:** ~3-4 minutes (model already cached)

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
    "model": "Qwen/Qwen3-32B-FP8",
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

### For Maximum Throughput

```yaml
--max-model-len 32000              # Full context support
--gpu-memory-utilization 0.90      # High memory usage (safe with FP8)
--max-num-batched-tokens 20480     # Larger batches
--max-num-seqs 64                  # More concurrent requests
--enable-prefix-caching            # Enable caching
```

### For Minimum Latency

```yaml
--max-model-len 16384              # Smaller context
--gpu-memory-utilization 0.85      # Moderate memory usage
--max-num-batched-tokens 8192      # Smaller batches
--max-num-seqs 16                  # Fewer concurrent requests
```

### For Maximum Context (Lower Concurrency)

```yaml
--max-model-len 32000              # Full 32K context
--gpu-memory-utilization 0.85      # Moderate memory usage
--max-num-batched-tokens 16384     # Standard batches
--max-num-seqs 32                  # Moderate concurrency
```

## Monitoring

### GPU Metrics

```bash
# Check GPU utilization
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# Watch GPU status (updates every 2 seconds)
watch -n 2 docker exec vllm-qwen3-32b-fp8 nvidia-smi

# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# Container stats
docker stats vllm-qwen3-32b-fp8
```

### Key Metrics to Watch

- **Memory Usage:** ~98 GB total (~32 GB model + 66 GB KV cache)
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check `vllm:num_generation_tokens_total` in metrics
- **Queue Depth:** Monitor running vs waiting requests (max 64 concurrent)
- **KV Cache Usage:** 271,360 tokens capacity (check `vllm:gpu_cache_usage_perc`)
- **Prefix Cache Hit Rate:** Higher = better efficiency for repeated prompts

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits single-request token generation speed (~6-7 tokens/sec)
2. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs) due to GB10 architecture
3. **Throughput vs Latency Trade-off:** Optimized for aggregate throughput; batching is essential for maximum performance
4. **Context Length:** Full 32K native support, but reduces max concurrent sequences if all use full context

## API Reference

### OpenAI-Compatible Endpoints

- **Chat Completions:** `POST http://localhost:8000/v1/chat/completions`
- **Completions:** `POST http://localhost:8000/v1/completions`
- **Models:** `GET http://localhost:8000/v1/models`
- **Health:** `GET http://localhost:8000/health`
- **Metrics:** `GET http://localhost:8000/metrics` (Prometheus format)
- **OpenAPI Docs:** `GET http://localhost:8000/docs`

Full API documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

## Comparison with Ollama

For comparison with Ollama serving the same model, see [docs/ollama/qwen3-32b-fp8.md](../ollama/qwen3-32b-fp8.md).

**Quick Summary:**
- vLLM: Higher throughput (5-10x), more complex setup, OpenAI API
- Ollama: Simpler setup/API, lower concurrency, easier model management

## References

- **Model:** https://huggingface.co/Qwen/Qwen3-32B-FP8
- **vLLM Documentation:** https://docs.vllm.ai
- **NVIDIA vLLM Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Hardware:** [docs/nvidia-spark.md](../nvidia-spark.md)
- **Ollama Alternative:** [docs/ollama/qwen3-32b-fp8.md](../ollama/qwen3-32b-fp8.md)

---

**Last Updated:** 2025-11-07
**Tested On:** NVIDIA DGX Spark (GB10), CUDA 13.0, vLLM 0.10.1.1
**Model:** Qwen/Qwen3-32B-FP8 (officially pre-quantized)
