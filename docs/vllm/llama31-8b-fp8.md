# vLLM: Llama-3.1-8B-Instruct-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running nvidia/Llama-3.1-8B-Instruct-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** NVIDIA's pre-quantized `nvidia/Llama-3.1-8B-Instruct-FP8` optimized for high-performance inference on NVIDIA hardware.

## What is Llama-3.1-8B?

Llama-3.1-8B-Instruct is Meta's compact instruction-tuned language model, part of the Llama 3.1 family. NVIDIA's FP8 version is specifically optimized for NVIDIA GPUs with Tensor Core acceleration.

**Key Features:**
- **Meta Architecture:** Dense Transformer with advanced attention mechanisms
- **NVIDIA Optimization:** Pre-quantized by NVIDIA for maximum performance
- **Instruction Following:** Fine-tuned for chat, instruction-following, and tool use
- **128K Context:** Native support for very long context windows
- **Safety:** Built-in safety features and responsible AI guidelines
- **Open License:** Llama 3.1 Community License allows commercial use

**Advantages on DGX Spark:**
- **NVIDIA Optimized:** Pre-quantized by NVIDIA specifically for their hardware
- **Small Footprint:** ~8GB model leaves maximum memory for KV cache
- **High Performance:** Fast single-request and batched throughput
- **Long Context:** Native 128K support (configurable to 32K-65K for DGX Spark)
- **Proven Quality:** Widely adopted and benchmarked across industry

## Good Use Cases

**Llama-3.1-8B-Instruct-FP8 excels at:**

1. **Conversational AI**
   - Chatbots and virtual assistants
   - Multi-turn dialogue systems
   - Customer support automation

2. **Instruction Following**
   - Task completion from natural language instructions
   - API and tool use scenarios
   - Structured output generation

3. **Code Generation**
   - Programming assistance and code completion
   - Debugging help and code explanation
   - Documentation generation

4. **Content Processing**
   - Document summarization
   - Text classification and analysis
   - Information extraction

5. **Long-Context Tasks** (with appropriate configuration)
   - Document analysis (up to 32K-65K tokens on DGX Spark)
   - Multi-document question answering
   - Extended conversation history

## When NOT to Use This Model

**Consider alternatives when you need:**

- **Maximum Quality:** Use Llama-3.3-70B-FP8 for highest reasoning capability
- **Extreme Long Context:** Use Mistral-NeMo-12B-FP8 (better optimized for 65K+ context)
- **Multilingual:** Use Qwen3-8B-FP8 or Qwen3-32B-FP8 for non-English languages
- **Specialized Domains:** Domain-specific models may outperform general models

**Trade-offs:**
- 8B size means lower quality ceiling vs 70B models
- Long context (128K native) limited to 32K-65K on DGX Spark memory constraints
- Best for speed/efficiency over maximum reasoning capability

## Model Specifications

- **Model:** nvidia/Llama-3.1-8B-Instruct-FP8 (NVIDIA pre-quantized)
- **Base Model:** meta-llama/Llama-3.1-8B-Instruct
- **Parameters:** 8.0 billion
- **Quantization:** FP8 (8-bit floating point, NVIDIA optimized)
- **Architecture:** Dense Transformer with GQA (Grouped Query Attention)
- **Context Length:** 128,000 tokens (native), configured to 32,768 for DGX Spark
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
  - Single request: ~9-10 tokens/sec generation
  - Batched (64 concurrent): ~450-500 tokens/sec aggregate (estimated)
  - Time to first token: 100-300ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~8 GB (50% reduction vs BF16)
  - KV cache: ~80-85 GB (maximum cache allocation)
  - Concurrent sequences: 64+ (highest concurrency)
  - Quality: ~99% of full precision on benchmarks
  - NVIDIA optimization: Kernels tuned for Blackwell architecture

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
  - NVIDIA model optimization support

## Docker Compose Service Configuration

```yaml
services:
  vllm-llama31-8b-fp8:
    image: nvcr.io/nvidia/vllm:25.09-py3
    container_name: vllm-llama31-8b-fp8
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
      - nvidia/Llama-3.1-8B-Instruct-FP8
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
| `--model` | `nvidia/Llama-3.1-8B-Instruct-FP8` | NVIDIA pre-quantized, optimized for NVIDIA hardware |
| `--max-model-len` | `32768` | Balanced context length (native 128K limited by DGX Spark memory) |
| `--gpu-memory-utilization` | `0.90` | Aggressive utilization enabled by small model size |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `64` | Maximum concurrency enabled by large KV cache |
| `--enable-prefix-caching` | enabled | Caches common prompt prefixes for efficiency |
| `--trust-remote-code` | enabled | May be required for some model features |

**Performance Benefits:**
- Model memory: ~8 GB (optimized by NVIDIA)
- KV cache: ~80-85 GB (largest cache in platform)
- Concurrent sequences: 64+ (optimal for DGX Spark)
- Context length: 32K configured (128K native max)
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

3. **Model Access:**
   Accept Llama 3.1 license on HuggingFace if required for NVIDIA version

### Start Service

```bash
# Stop any running vLLM services (all use port 8000)
docker compose stop vllm-qwen3-32b-fp8 vllm-qwen3-30b-a3b-fp8 vllm-llama33-70b-fp8 vllm-qwen3-8b-fp8

# Start vLLM Llama-3.1-8B-Instruct-FP8 service
docker compose up -d vllm-llama31-8b-fp8
```

### Monitor Loading

```bash
# View logs
docker compose logs -f vllm-llama31-8b-fp8

# Filter for key loading events
docker compose logs -f vllm-llama31-8b-fp8 | grep -E "(Loading|INFO|Ready|Serving)"
```

**Model loading takes ~6-8 minutes on first run:**
1. Model download (~4 min if not cached - ~8GB)
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
    "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
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
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a Python function to implement quicksort."}
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
    "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "How do I implement a linked list in Python?"}
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
    "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Explain machine learning"}],
    "max_tokens": 500,
    "stream": true
  }'
```

## Performance Optimization

### For Maximum Throughput (Recommended)

```yaml
--max-model-len 32768              # Full context support for DGX Spark
--gpu-memory-utilization 0.90      # Aggressive memory usage (safe with 8B model)
--max-num-batched-tokens 16384     # Large batches
--max-num-seqs 64                  # Maximum concurrent requests
--enable-prefix-caching            # Enable caching
```

**Expected Performance:**
- Batched throughput: ~450-500 tokens/sec aggregate
- Concurrent requests: 64
- KV cache: ~80-85 GB

### For Longer Context (Lower Concurrency)

```yaml
--max-model-len 65536              # Extended context (half of native 128K)
--gpu-memory-utilization 0.90      # Maximum memory usage
--max-num-batched-tokens 16384     # Standard batches
--max-num-seqs 32                  # Reduced concurrency for longer context
```

**Expected Performance:**
- Context: 65K tokens (vs 32K standard)
- Concurrent requests: 32 (vs 64 standard)
- Best for: Long document analysis

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
- Trade-off: Lower total throughput for better individual latency

## Monitoring

### GPU Metrics

```bash
# Check GPU utilization
docker exec vllm-llama31-8b-fp8 nvidia-smi

# Watch GPU status (updates every 2 seconds)
watch -n 2 docker exec vllm-llama31-8b-fp8 nvidia-smi

# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# Container stats
docker stats vllm-llama31-8b-fp8
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
| Qwen3-8B-FP8 | 8B | ~8 GB | ~80-85 GB | ~9-10 | ~450-500 | 64+ |
| **Llama-3.1-8B-FP8** | 8B | ~8 GB | ~80-85 GB | ~9-10 | ~450-500 | 64+ |
| Mistral-NeMo-12B-FP8 | 12B | ~12 GB | ~75-80 GB | ~8-9 | ~400-450 | 64 |
| Qwen3-32B-FP8 | 32B | ~32 GB | ~66 GB | ~6-7 | ~300-400 | 64 |
| Llama-3.3-70B-FP8 | 70B | ~35 GB | ~40-60 GB | ~5-7 | ~80-150 | 32 |

**Llama-3.1-8B-FP8 Advantages:**
- **NVIDIA-optimized** pre-quantized model
- **Comparable performance** to Qwen3-8B-FP8
- **Meta's proven architecture** widely adopted in industry
- **Strong instruction following** and tool use capabilities
- **Native 128K context** (configurable for hardware)
- **Commercial license** (Llama 3.1 Community License)

**vs Qwen3-8B-FP8:**
- Similar performance characteristics
- Llama: Better English, Meta architecture, NVIDIA-optimized
- Qwen: Better multilingual, newer architecture
- Choose based on use case and preference

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
- **Recommended:** Use NVIDIA pre-quantized model `nvidia/Llama-3.1-8B-Instruct-FP8` (default)
- Model is optimized by NVIDIA specifically for their hardware
- Avoid using `--quantization fp8` flag with base model

### Issue: Model Download Requires License Acceptance

**Symptom:**
```
Cannot access gated repo for url https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8
```

**Solution:**
1. Go to https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8
2. Accept the license agreement (may need to accept Meta's Llama 3.1 license first)
3. Ensure HF_TOKEN has read access to the model
4. Restart container

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
   rm -rf /opt/hf/models--nvidia--Llama-3.1-8B-Instruct-FP8/*.incomplete
   ```
4. Restart container

### Issue: "Compute Capability 12.1" Warning

**Symptom:**
```
WARNING: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1
Maximum cuda capability supported is 12.0
```

**Status:** This is a cosmetic warning and can be safely ignored. The NVIDIA container has proper GB10 support.

### Issue: Lower Than Expected Throughput

**Solution:**
1. Verify batching is occurring: Check `vllm:num_running_requests` metric
2. Send concurrent requests to enable batching
3. Increase `--max-num-batched-tokens` to 20480 or 32768
4. Verify prefix caching is enabled (check cache hit rate)
5. Monitor KV cache usage - should be 80-95% utilized

## Important Notes

1. **All vLLM services use port 8000** - Only one vLLM model can run at a time
2. **NVIDIA-optimized model** - Pre-quantized by NVIDIA for their hardware
3. **Long context available** - Native 128K, configured to 32K-65K for DGX Spark
4. **License required** - May need to accept Llama 3.1 Community License on HuggingFace
5. **Batching essential** - Single requests won't achieve maximum throughput
6. **First load slow** - Expect 6-8 minutes for initial download and compilation
7. **Subsequent starts faster** - Model cached at `/opt/hf` after first load
8. **Memory bandwidth bottleneck** - 273 GB/s limits single-request speed

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits single-request token generation speed
2. **Context Length on DGX Spark:** Native 128K limited to 32K-65K by memory constraints
3. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs)
4. **Throughput vs Latency:** Optimized for aggregate throughput with batching
5. **Quality Ceiling:** 8B model has lower maximum quality vs 70B models

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

- **Model:** https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8
- **Base Model:** https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **Llama 3.1 Announcement:** https://ai.meta.com/blog/meta-llama-3-1/
- **vLLM Documentation:** https://docs.vllm.ai
- **NVIDIA vLLM Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Hardware:** [docs/nvidia-spark.md](../nvidia-spark.md)
- **Related Models:**
  - [Qwen3-8B-FP8](qwen3-8b-fp8.md) - Alternative 8B model
  - [Mistral-NeMo-12B-FP8](mistral-nemo-12b-fp8.md) - 12B with long context
  - [Llama-3.3-70B-FP8](llama33-70b-fp8.md) - Larger Llama model

---

**Last Updated:** 2025-11-10
**Model:** nvidia/Llama-3.1-8B-Instruct-FP8
**vLLM Version:** 0.10.1.1+381074ae.nv25.09
**Container:** nvcr.io/nvidia/vllm:25.09-py3
