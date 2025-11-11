# vLLM: Qwen3-8B-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3-8B-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** Pre-quantized `Qwen/Qwen3-8B-FP8` for optimal performance on DGX Spark unified memory architecture.

## What is Qwen3-8B?

Qwen3-8B is a compact 8B parameter language model from Alibaba Cloud's Qwen team. It's part of the Qwen3 family, offering excellent quality-to-size ratio for resource-efficient deployments.

**Key Features:**
- **Efficient Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **High Quality:** Competitive performance with larger models on many benchmarks
- **Fast Inference:** Smaller size enables higher throughput on bandwidth-limited hardware
- **Multilingual:** Strong performance across English, Chinese, and other languages
- **Long Context:** Native 32K context window support

**Advantages on DGX Spark:**
- **Small Model Footprint:** Only ~8GB for model weights leaves more memory for KV cache
- **High Concurrency:** Can handle 64+ concurrent requests with 32K context
- **Fast Generation:** ~9-10 tokens/sec single-request (fastest in the platform)
- **Maximum Throughput:** ~450-500 tokens/sec aggregate with batching
- **Memory Efficient:** Enables larger batch sizes and more aggressive caching

## Good Use Cases

**Qwen3-8B-FP8 excels at:**

1. **High-Throughput Applications**
   - API services with many concurrent users
   - Batch processing of large document sets
   - Real-time chat applications with multiple simultaneous conversations

2. **Resource-Constrained Deployments**
   - Prototyping and development on limited hardware
   - Cost-sensitive production workloads
   - Multi-tenant environments requiring efficiency

3. **General-Purpose Tasks**
   - Question answering and information retrieval
   - Text summarization and classification
   - Code generation and debugging assistance
   - Content creation and editing

4. **Fast Iteration Requirements**
   - Development and testing workflows
   - A/B testing with rapid model swaps
   - Experimentation with prompts and parameters

## When NOT to Use This Model

**Consider alternatives when you need:**

- **Maximum Quality:** Use Llama-3.3-70B-FP8 for highest reasoning capability
- **Extreme Long Context:** Use Mistral-NeMo-12B-FP8 (65K-128K context support)
- **Domain Expertise:** Specialized models may outperform general-purpose models
- **Multilingual Excellence:** Qwen3-32B offers better multilingual capabilities

**Trade-offs:**
- Smaller model = lower quality ceiling vs 32B/70B models
- Best for speed/efficiency over maximum reasoning capability
- May struggle with very complex multi-step reasoning tasks

## Model Specifications

- **Model:** Qwen/Qwen3-8B-FP8 (officially pre-quantized by Qwen team)
- **Parameters:** 8.0 billion
- **Quantization:** FP8 (8-bit floating point)
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Context Length:** 32,768 tokens (native), extendable to 131K with YaRN
- **Memory Requirements:**
  - Model: ~8 GB
  - KV Cache: ~80-85 GB (with 32K context, 64 concurrent sequences)
  - Total: ~88-93 GB peak GPU memory usage

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics
- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + aggressive batching + prefix caching
- **Measured Performance:**
  - Single request: ~9-10 tokens/sec generation (fastest in platform)
  - Batched (64 concurrent): ~450-500 tokens/sec aggregate (estimated)
  - Time to first token: 100-300ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~8 GB (50% reduction vs BF16)
  - KV cache: ~80-85 GB (maximum cache allocation possible)
  - Concurrent sequences: 64+ (highest concurrency in platform)
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

## Docker Compose Service Configuration

```yaml
services:
  vllm-qwen3-8b-fp8:
    image: nvcr.io/nvidia/vllm:25.09-py3
    container_name: vllm-qwen3-8b-fp8
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      HF_TOKEN: "${HF_TOKEN}"
      HUGGING_FACE_HUB_TOKEN: "${HF_TOKEN}"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    command:
      - vllm
      - serve
      - Qwen/Qwen3-8B-FP8
      - --download-dir
      - /root/.cache/huggingface
      - --max-model-len
      - "32768"
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

## Configuration Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--model` | `Qwen/Qwen3-8B-FP8` | Pre-quantized by Qwen team, validated quality |
| `--max-model-len` | `32768` | Native model context (full 32K support) |
| `--gpu-memory-utilization` | `0.90` | Aggressive utilization enabled by small model size |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `64` | Maximum concurrency enabled by large KV cache |
| `--enable-prefix-caching` | enabled | Caches common prompt prefixes for efficiency |
| `--trust-remote-code` | enabled | Required for Qwen models |

**Performance Benefits:**
- Model memory: ~8 GB (75% less than 32B model)
- KV cache: ~80-85 GB (largest cache in platform)
- Concurrent sequences: 64+ (optimal for DGX Spark)
- Context length: Full 32K native support
- Single-request TPS: ~9-10 (fastest in platform)
- Batched TPS: ~450-500 (highest in platform)

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
# Stop any running vLLM services (all use port 8000)
docker compose stop vllm-qwen3-32b-fp8 vllm-qwen3-30b-a3b-fp8 vllm-llama33-70b-fp8

# Start vLLM Qwen3-8B-FP8 service
docker compose up -d vllm-qwen3-8b-fp8
```

### Monitor Loading

```bash
# View logs
docker compose logs -f vllm-qwen3-8b-fp8

# Filter for key loading events
docker compose logs -f vllm-qwen3-8b-fp8 | grep -E "(Loading|INFO|Ready|Serving)"
```

**Model loading takes ~6-8 minutes on first run:**
1. Model download (~4 min if not cached - smaller than 32B model, ~8GB)
2. Loading safetensors shards (~2 min)
3. torch.compile (~15 sec)
4. KV cache initialization and CUDA graph capture (~5 sec)

**Subsequent startups:** ~2-3 minutes (model already cached)

### Health Check

```bash
# API health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## API Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client pointing to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require API key
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### cURL (Chat Completions)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "How do I implement a binary search in Python?"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 500,
    "stream": true
  }'
```

## Performance Optimization

### For Maximum Throughput (Recommended)

```yaml
--max-model-len 32768              # Full context support
--gpu-memory-utilization 0.90      # Aggressive memory usage (safe with 8B model)
--max-num-batched-tokens 16384     # Large batches
--max-num-seqs 64                  # Maximum concurrent requests
--enable-prefix-caching            # Enable caching
```

**Expected Performance:**
- Batched throughput: ~450-500 tokens/sec aggregate
- Concurrent requests: 64
- KV cache: ~80-85 GB

### For Minimum Latency

```yaml
--max-model-len 16384              # Smaller context
--gpu-memory-utilization 0.85      # Moderate memory usage
--max-num-batched-tokens 8192      # Smaller batches
--max-num-seqs 16                  # Fewer concurrent requests
```

**Expected Performance:**
- Single-request latency: Lower TTFT
- Throughput: ~200-300 tokens/sec aggregate
- Trade-off: Lower total throughput for better individual request latency

### For Maximum Concurrency

```yaml
--max-model-len 32768              # Full 32K context
--gpu-memory-utilization 0.90      # Maximum memory usage
--max-num-batched-tokens 20480     # Larger batches
--max-num-seqs 96                  # Very high concurrency
```

**Expected Performance:**
- Concurrent requests: 96+ (highest possible)
- Best for: Many small requests
- KV cache: ~85 GB

## Monitoring

### GPU Metrics

```bash
# Check GPU utilization
docker exec vllm-qwen3-8b-fp8 nvidia-smi

# Watch GPU status (updates every 2 seconds)
watch -n 2 docker exec vllm-qwen3-8b-fp8 nvidia-smi

# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# Container stats
docker stats vllm-qwen3-8b-fp8
```

### Key Metrics to Watch

- **Memory Usage:** ~88-93 GB total (~8 GB model + 80-85 GB KV cache)
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check `vllm:num_generation_tokens_total` in metrics
- **Queue Depth:** Monitor running vs waiting requests (max 64+ concurrent)
- **KV Cache Usage:** ~350,000+ tokens capacity (check `vllm:gpu_cache_usage_perc`)
- **Prefix Cache Hit Rate:** Higher = better efficiency for repeated prompts

## Comparison with Other Models

| Model | Parameters | Model Memory | KV Cache | Single TPS | Batched TPS | Concurrency |
|-------|-----------|--------------|----------|------------|-------------|-------------|
| **Qwen3-8B-FP8** | 8B | ~8 GB | ~80-85 GB | ~9-10 | ~450-500 | 64+ |
| Qwen3-32B-FP8 | 32B | ~32 GB | ~66 GB | ~6-7 | ~300-400 | 64 |
| Qwen3-30B-A3B-FP8 | 30B (3B active) | ~30 GB | ~55-70 GB | ~7-9 | ~200-350 | 64 |
| Mistral-NeMo-12B-FP8 | 12B | ~12 GB | ~75-80 GB | ~8-9 | ~400-450 | 64 |
| Llama-3.3-70B-FP8 | 70B | ~35 GB | ~40-60 GB | ~5-7 | ~80-150 | 32 |

**Qwen3-8B-FP8 Advantages:**
- **Fastest single-request performance** in the platform
- **Highest batched throughput** (~450-500 tok/s)
- **Maximum concurrency** (64+ concurrent requests)
- **Largest KV cache** enables most aggressive batching
- **Fastest loading times** (smallest model)

**Trade-offs:**
- Lower quality ceiling vs 32B/70B models
- Not ideal for complex reasoning tasks
- Better for speed than maximum quality

## Troubleshooting

### Issue: Container Fails with Memory Errors

**Symptom:**
```
ValueError: Free memory on device (XX GiB) is less than desired GPU memory utilization
```

**Solution:**
1. Stop all containers: `docker compose down`
2. Clean Docker resources: `docker system prune -f`
3. Reduce `--gpu-memory-utilization` to 0.80 or 0.85
4. Reduce `--max-model-len` to 16384

### Issue: FP8 Quantization Fails

**Symptom:**
```
RuntimeError: Error Internal
cutlass_scaled_mm failed
```

**Solution:**
- **Recommended:** Use pre-quantized model `Qwen/Qwen3-8B-FP8` (default)
- The pre-quantized model is officially quantized by Qwen team and works reliably
- Avoid using `--quantization fp8` flag with base model

### Issue: Model Download Incomplete

**Symptom:**
```
Loading safetensors checkpoint shards: X/Y
[hangs or crashes]
```

**Solution:**
1. Verify `HF_TOKEN` is set correctly in `.env`
2. Check disk space: `df -h /opt/hf` (need ~10GB free)
3. Clean incomplete downloads:
   ```bash
   rm -rf /opt/hf/models--Qwen--Qwen3-8B-FP8/*.incomplete
   ```
4. Restart container

### Issue: "Compute Capability 12.1" Warning

**Symptom:**
```
WARNING: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1
Maximum cuda capability supported is 12.0
```

**Status:** This is a cosmetic warning and can be safely ignored. The NVIDIA container has proper GB10 support despite the warning.

### Issue: Lower Than Expected Throughput

**Solution:**
1. Verify batching is occurring: Check `vllm:num_running_requests` metric
2. Send concurrent requests to enable batching (single requests = lower throughput)
3. Increase `--max-num-batched-tokens` to 20480 or 32768
4. Verify prefix caching is enabled and working (check cache hit rate)
5. Monitor KV cache usage - should be 80-95% utilized

## Important Notes

1. **All vLLM services use port 8000** - Only one vLLM model can run at a time
2. **Batching is essential** - Single requests won't achieve maximum throughput
3. **First load is slow** - Expect 6-8 minutes for initial model download and compilation
4. **Subsequent starts faster** - Model cached at `/opt/hf` after first load
5. **Memory bandwidth bottleneck** - 273 GB/s limits single-request speed
6. **Smallest model = fastest** - 8B size enables highest throughput in platform
7. **Quality trade-off** - Optimized for speed/efficiency over maximum reasoning capability

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits single-request token generation speed
2. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs)
3. **Throughput vs Latency:** Optimized for aggregate throughput with batching
4. **Quality Ceiling:** 8B model has lower maximum quality vs 32B/70B models
5. **Context Length:** Full 32K support, but all concurrent sequences share KV cache

## API Reference

### OpenAI-Compatible Endpoints

- **Chat Completions:** `POST http://localhost:8000/v1/chat/completions`
- **Completions:** `POST http://localhost:8000/v1/completions`
- **Models:** `GET http://localhost:8000/v1/models`
- **Health:** `GET http://localhost:8000/health`
- **Metrics:** `GET http://localhost:8000/metrics` (Prometheus format)
- **OpenAPI Docs:** `GET http://localhost:8000/docs`

Full API documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

## References

- **Model:** https://huggingface.co/Qwen/Qwen3-8B-FP8
- **Model Card:** https://huggingface.co/Qwen/Qwen3-8B
- **vLLM Documentation:** https://docs.vllm.ai
- **NVIDIA vLLM Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Hardware:** [docs/nvidia-spark.md](../nvidia-spark.md)
- **Qwen3 Blog:** https://qwenlm.github.io/blog/qwen3/
- **Related Models:**
  - [Qwen3-32B-FP8](qwen3-32b-fp8.md) - Larger Qwen model
  - [Llama-3.1-8B-FP8](llama31-8b-fp8.md) - Alternative 8B model
  - [Mistral-NeMo-12B-FP8](mistral-nemo-12b-fp8.md) - 12B with long context

---

**Last Updated:** 2025-11-10
**Model:** Qwen/Qwen3-8B-FP8
**vLLM Version:** 0.10.1.1+381074ae.nv25.09
**Container:** nvcr.io/nvidia/vllm:25.09-py3
