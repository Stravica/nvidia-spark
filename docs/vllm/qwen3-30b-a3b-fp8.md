# vLLM: Qwen3-30B-A3B-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** Pre-quantized `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` for optimal performance on DGX Spark unified memory architecture.

---

## What is Qwen3-30B-A3B?

**Qwen3-30B-A3B** is a Mixture-of-Experts (MoE) model from the Qwen3 family, designed for efficient inference with minimal quality trade-offs.

### Model Architecture

- **Total Parameters:** 30 billion
- **Active Parameters:** 3 billion per token (10% activation)
- **Architecture:** Sparse MoE with expert routing
- **Quantization:** FP8 (8-bit floating point)

### How MoE Works

Unlike dense models that activate all parameters for every token, MoE models use a gating mechanism to route each token to a subset of specialized "expert" networks. For Qwen3-30B-A3B:
- The model contains **30B total parameters** organized into expert networks
- For each token, only **3B parameters are activated** (the "A3B" designation)
- This provides **10x compute efficiency** compared to activating all 30B parameters

### Key Advantages

1. **Computational Efficiency:** Only 3B parameters active per token = faster inference than full 30B model
2. **Memory Efficiency:** FP8 quantization reduces model size to ~30GB (vs ~60GB for BF16)
3. **Quality:** Achieves ~95-98% of full dense 30B model performance
4. **Specialization:** Experts can specialize in different domains (coding, reasoning, creative writing)

### Good Use Cases

This model excels at:

- **General Instruction Following:** Strong performance across diverse tasks
- **Coding and Technical Writing:** Expert specialization for technical domains
- **Logical Reasoning:** Multi-step problem solving and analytical tasks
- **Mixed Workloads:** Efficient handling of varied request types
- **Moderate Context Tasks:** Up to 32K tokens with good performance
- **Batch Processing:** MoE efficiency enables better throughput with batching

### When NOT to Use This Model

Consider alternatives if you need:
- **Maximum long-context performance** → Use Llama 3.3 70B (128K context)
- **Highest absolute quality** → Use larger dense models
- **Minimal memory footprint** → Use smaller models (12B or less)
- **Multimodal capabilities** → Use Qwen3-VL series instead

---

## Model Specifications

- **Model:** Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (officially pre-quantized by Qwen team)
- **Total Parameters:** 30 billion
- **Active Parameters:** 3 billion per token
- **Quantization:** FP8 (8-bit floating point)
- **Architecture:** Sparse MoE with Grouped Query Attention (GQA)
- **Context Length:** 32,768 tokens (native), extendable to 40K
- **Memory Requirements:**
  - Model: ~30 GB
  - KV Cache: ~55-70 GB (with 32K context, 64 concurrent sequences)
  - Total: ~85-100 GB peak GPU memory usage

---

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics

- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + aggressive batching + prefix caching + MoE efficiency
- **Expected Performance:**
  - Single request: ~7-9 tokens/sec generation (better than dense 32B due to MoE)
  - Batched (32-64 concurrent): ~200-350 tokens/sec aggregate
  - Time to first token: 150-400ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~30 GB (50% reduction vs BF16)
  - KV cache: ~55-70 GB (more headroom than Qwen3-32B-FP8)
  - Concurrent sequences: 64 (excellent batching capacity)
  - Quality: ~99% of full FP8 precision on benchmarks

### MoE-Specific Benefits on DGX Spark

- **Reduced Memory Bandwidth Pressure:** Only 3B active params = 10x less data movement per token
- **Better Single-Request Latency:** Faster than dense 30B models despite similar memory footprint
- **Improved Batch Efficiency:** Less compute per token allows more concurrent requests

---

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
  - Native GB10 support
  - Optimized FP8 kernels for Blackwell
  - MoE-aware batching and scheduling
  - Correct unified memory handling

---

## Docker Compose Service

### Service Configuration

```yaml
vllm-qwen3-30b-a3b-fp8:
  image: nvcr.io/nvidia/vllm:25.09-py3
  container_name: vllm-qwen3-30b-a3b-fp8
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
    - Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
    - --download-dir
    - /root/.cache/huggingface
    - --max-model-len
    - "32768"
    - --gpu-memory-utilization
    - "0.85"
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

### Service Management

```bash
# Start service
docker compose up -d vllm-qwen3-30b-a3b-fp8

# View logs
docker compose logs -f vllm-qwen3-30b-a3b-fp8

# Stop service
docker compose stop vllm-qwen3-30b-a3b-fp8

# Restart service
docker compose restart vllm-qwen3-30b-a3b-fp8

# Check status
docker compose ps vllm-qwen3-30b-a3b-fp8
```

### Initial Startup

**First-time model download:** ~8-10 minutes
- Downloads ~30 GB from HuggingFace
- Caches to `/opt/hf`
- Loads model into GPU memory
- Compiles MoE kernels

**Subsequent startups:** ~30-60 seconds
- Loads from local cache
- Uses pre-compiled kernels
- Much faster after first run

---

## Configuration Parameters

### GPU Memory Utilization

```bash
--gpu-memory-utilization 0.85
```

**Why 85%?**
- Model: ~30 GB
- KV Cache: ~55 GB (at 85% utilization)
- Headroom: ~15 GB for system overhead
- **Higher than 32B model** due to MoE efficiency (less memory per active param)

**Tuning Guidelines:**
- **0.80:** More conservative, ~48 GB KV cache
- **0.85:** Recommended balance (default)
- **0.90:** Maximum throughput, ~66 GB KV cache (may cause OOM under heavy load)

### Context Length

```bash
--max-model-len 32768
```

**Native support:** 32,768 tokens
**Extendable:** Up to 40,000 tokens with performance degradation

**Per-Request Context:**
```python
# Client-side control
{
  "max_tokens": 2048,  # Generated tokens
  # Input prompt can be up to (32768 - max_tokens) tokens
}
```

### Batching Configuration

```bash
--max-num-seqs 64              # Concurrent requests
--max-num-batched-tokens 16384 # Tokens processed per batch
```

**64 concurrent sequences:**
- MoE efficiency allows more concurrency than dense models
- Each request can have different context lengths
- Automatic batching by vLLM scheduler

**16,384 batched tokens:**
- Processes up to 16K tokens in a single forward pass
- Balances latency vs throughput
- Optimal for DGX Spark memory bandwidth

### Prefix Caching

```bash
--enable-prefix-caching
```

**Critical for performance:**
- Caches common prompt prefixes (system prompts, few-shot examples)
- Reduces redundant computation by ~50-80% for repeated prefixes
- Especially valuable on bandwidth-limited DGX Spark
- **Essential for MoE models:** Avoids re-routing tokens through experts

---

## API Usage

### OpenAI-Compatible Endpoint

**Base URL:** `http://localhost:8000/v1`

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain how mixture-of-experts models work."}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": true
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a binary search algorithm in Python."}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming with Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=512,
    temperature=0.7,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

**Key Metrics:**
- `vllm:num_requests_running` - Active requests
- `vllm:num_requests_waiting` - Queued requests
- `vllm:gpu_cache_usage_perc` - KV cache utilization
- `vllm:time_to_first_token_seconds` - TTFT latency
- `vllm:time_per_output_token_seconds` - Generation speed

### GPU Monitoring

```bash
# Host GPU stats
nvidia-smi

# Container GPU stats
docker exec vllm-qwen3-30b-a3b-fp8 nvidia-smi

# Watch GPU usage
watch -n 1 'docker exec vllm-qwen3-30b-a3b-fp8 nvidia-smi'
```

### Container Logs

```bash
# Follow logs
docker compose logs -f vllm-qwen3-30b-a3b-fp8

# Last 100 lines
docker compose logs --tail=100 vllm-qwen3-30b-a3b-fp8

# Search logs for errors
docker compose logs vllm-qwen3-30b-a3b-fp8 | grep -i error
```

---

## Performance Optimization

### 1. Maximize Batching

**Problem:** Single requests underutilize GPU
**Solution:** Send concurrent requests

```bash
# Sequential requests (slow)
for i in {1..10}; do
  curl -X POST http://localhost:8000/v1/chat/completions -d '...' &
done

# Concurrent requests (5-10x faster aggregate throughput)
# vLLM automatically batches these
```

**Expected Results:**
- Single request: ~7-9 tok/s
- 16 concurrent: ~100-150 tok/s aggregate
- 64 concurrent: ~200-350 tok/s aggregate

### 2. Enable Prefix Caching (Already Enabled)

**Benefit:** 50-80% speedup for repeated prefixes

**Good candidates:**
- System prompts used across requests
- Few-shot examples
- Common instruction templates

**Example:**
```python
# This system prompt will be cached
messages = [
    {"role": "system", "content": "You are a helpful coding assistant with expertise in Python, JavaScript, and systems programming."},  # Cached
    {"role": "user", "content": "Write a function to..."}  # Different each time
]
```

### 3. Tune Context Length

**Problem:** Reserving full 32K context wastes KV cache memory
**Solution:** Use `--max-model-len` to match your workload

```bash
# If you only need 16K context:
--max-model-len 16384  # Doubles KV cache capacity for concurrent requests

# If you need full context:
--max-model-len 32768  # Default
```

### 4. Adjust GPU Memory Utilization

**Problem:** Running out of KV cache under high concurrency
**Solution:** Increase `--gpu-memory-utilization`

```bash
# Current: 85% (~55 GB KV cache)
--gpu-memory-utilization 0.85

# High concurrency: 90% (~66 GB KV cache)
--gpu-memory-utilization 0.90
```

**Warning:** Setting too high (>0.90) may cause OOM crashes.

### 5. Monitor KV Cache Usage

```bash
# Check cache utilization
curl http://localhost:8000/metrics | grep gpu_cache_usage_perc
```

**Interpretation:**
- **<80%:** Underutilized, can increase concurrency
- **80-95%:** Optimal utilization
- **>95%:** May cause queuing or OOM, reduce concurrency

---

## Comparison: Qwen3-30B-A3B vs Qwen3-32B

| Feature | Qwen3-30B-A3B-FP8 | Qwen3-32B-FP8 |
|---------|-------------------|----------------|
| **Architecture** | Sparse MoE | Dense Transformer |
| **Total Parameters** | 30B | 32.8B |
| **Active Parameters** | 3B per token | 32.8B per token |
| **Model Memory** | ~30 GB | ~32 GB |
| **KV Cache (85% util)** | ~55 GB | ~66 GB |
| **Total Memory** | ~85 GB | ~98 GB |
| **Context Length** | 32,768 | 32,768 |
| **Single Request TPS** | ~7-9 tok/s | ~6-7 tok/s |
| **Batched TPS (64 concurrent)** | ~200-350 tok/s | ~300-400 tok/s |
| **Quality** | 95-98% of dense 30B | Baseline |
| **Best For** | Mixed workloads, efficiency | Maximum throughput |

**Recommendations:**
- **Use Qwen3-30B-A3B-FP8** for: Better single-request latency, more memory headroom, diverse task types
- **Use Qwen3-32B-FP8** for: Maximum batched throughput, consistent high-concurrency workloads

---

## Troubleshooting

### Model Fails to Load

**Symptom:** Container exits with OOM or CUDA errors

**Solutions:**
1. Reduce GPU memory utilization:
   ```bash
   --gpu-memory-utilization 0.80
   ```

2. Reduce context length:
   ```bash
   --max-model-len 24576
   ```

3. Check GPU availability:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

### Slow Performance

**Symptom:** <5 tokens/sec generation

**Solutions:**
1. **Send concurrent requests** - MoE models benefit greatly from batching
2. **Check GPU utilization:**
   ```bash
   docker exec vllm-qwen3-30b-a3b-fp8 nvidia-smi
   # GPU util should be >80% during inference
   ```

3. **Enable prefix caching** (already enabled in config)
4. **Verify FP8 quantization is active:**
   ```bash
   docker compose logs vllm-qwen3-30b-a3b-fp8 | grep -i "quantization"
   # Should show: quantization=fp8
   ```

### High Latency / Queuing

**Symptom:** Long time to first token (>2 seconds)

**Solutions:**
1. **Check KV cache usage:**
   ```bash
   curl http://localhost:8000/metrics | grep cache_usage
   ```

2. **Increase GPU memory utilization:**
   ```bash
   --gpu-memory-utilization 0.90
   ```

3. **Reduce max sequences:**
   ```bash
   --max-num-seqs 32  # From 64
   ```

### Container Won't Start

**Symptom:** Service fails to start or immediately exits

**Solutions:**
1. **Check HF token:**
   ```bash
   echo $HF_TOKEN
   # Should show your token
   ```

2. **Verify disk space:**
   ```bash
   df -h /opt/hf
   # Need ~35 GB free for model
   ```

3. **Check logs:**
   ```bash
   docker compose logs vllm-qwen3-30b-a3b-fp8
   ```

4. **Ensure no other service is using port 8000:**
   ```bash
   docker compose stop vllm-qwen3-32b-fp8
   docker compose stop vllm-llama33-70b-fp8
   ```

---

## Important Notes

- **Never run multiple vLLM services simultaneously** - All use port 8000 and need significant GPU memory
- **First-time loading is slow** - Model download takes ~8-10 minutes
- **Subsequent starts are faster** - Models cached at `/opt/hf`
- **Memory bandwidth is the bottleneck** - Not compute capacity
- **Batching is essential for MoE** - Single-request performance doesn't fully leverage MoE efficiency
- **Prefix caching is critical** - Enables 50-80% speedup for repeated prompts
- **MoE provides better latency** - Faster than dense 30B models due to sparse activation

---

## References

- **Repository Root:** `/opt/inference/README.md`
- **Hardware Guide:** `/opt/inference/docs/nvidia-spark.md`
- **CLAUDE.md:** `/opt/inference/CLAUDE.md` (AI assistant guide)
- **Docker Compose:** `/opt/inference/docker-compose.yml`
- **vLLM Documentation:** https://docs.vllm.ai
- **Qwen3 Model Card:** https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

---

**Last Updated:** 2025-11-08
**Model:** Qwen3-30B-A3B-Instruct-2507-FP8
**vLLM Version:** 0.10.1.1+381074ae.nv25.09
**Container:** nvcr.io/nvidia/vllm:25.09-py3
