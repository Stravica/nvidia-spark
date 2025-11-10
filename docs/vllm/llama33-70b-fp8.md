# vLLM: Llama 3.3 70B Instruct FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running nvidia/Llama-3.3-70B-Instruct-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** Pre-quantized `nvidia/Llama-3.3-70B-Instruct-FP8` optimized for NVIDIA GPUs with native FP8 support.

---

## What is Llama 3.3 70B?

**Llama 3.3 70B** is Meta's latest and most capable open-weight language model in the Llama 3 series, released in December 2024.

### Model Architecture

- **Total Parameters:** 71 billion
- **Active Parameters:** 71 billion per token (dense model, no MoE)
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Quantization:** FP8 (8-bit floating point, quantized by NVIDIA)
- **Context Length:** 128,000 tokens (4x longer than most models)

### How Dense Models Work

Unlike sparse MoE models that activate only a subset of parameters, Llama 3.3 70B is a **dense model** where all 71B parameters are activated for every token:
- **Compute:** All layers process every token
- **Memory Bandwidth:** Higher data movement per token vs MoE
- **Quality:** No expert routing = more consistent quality across domains
- **Simplicity:** No gating mechanism, straightforward architecture

### Key Advantages

1. **Exceptional Quality:** State-of-the-art performance across benchmarks
2. **Massive Context:** 128K tokens = ~96,000 words = entire novels/codebases
3. **Consistent Performance:** No domain specialization gaps (vs MoE)
4. **Production-Ready:** Extensively tested by Meta and community
5. **Long-Context Reasoning:** Maintains coherence across very long conversations
6. **Instruction Following:** Highly tuned for following complex instructions

### Good Use Cases

This model excels at:

- **Long-Context Analysis:** Summarizing documents, legal contracts, research papers (up to 128K tokens)
- **Complex Reasoning:** Multi-step problem solving, mathematical proofs, logical deduction
- **Production Workloads:** Mission-critical applications requiring highest quality
- **Code Understanding:** Analyzing entire codebases, large files, architectural documentation
- **Extended Conversations:** Multi-turn dialogues with long history retention
- **Research and Technical Writing:** Detailed explanations, academic content
- **Multi-Document Tasks:** Comparing, synthesizing, cross-referencing multiple sources

### When NOT to Use This Model

Consider alternatives if you need:
- **Maximum throughput** → Use MoE models like Qwen3-30B-A3B (10x less compute per token)
- **Minimal latency** → Use smaller models (12B-32B range)
- **Very high concurrency** → Dense 70B uses more memory, limits batch size
- **Simple tasks** → Overkill for basic classification, short responses
- **Minimal memory footprint** → 70B requires ~35GB just for model weights

---

## Model Specifications

- **Model:** nvidia/Llama-3.3-70B-Instruct-FP8 (officially pre-quantized by NVIDIA)
- **Parameters:** 71 billion (all active, dense model)
- **Quantization:** FP8 (8-bit floating point)
- **Architecture:** Dense Transformer with Grouped Query Attention (GQA)
- **Context Length:** 128,000 tokens (native support, fully trained)
- **Memory Requirements:**
  - Model: ~35 GB
  - KV Cache: ~40-60 GB (depends on context length and concurrency)
  - Total: ~75-95 GB peak GPU memory usage

---

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics

- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + moderate batching + prefix caching + long-context support
- **Expected Performance:**
  - Single request (short context): ~5-7 tokens/sec generation
  - Single request (64K context): ~3-5 tokens/sec generation
  - Batched (16-32 concurrent): ~80-150 tokens/sec aggregate
  - Time to first token: 300-800ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~35 GB (50% reduction vs BF16)
  - Memory bandwidth: 2x better than BF16
  - Quality: ~99.5% of full BF16 precision on benchmarks
  - Essential for 128K context support on DGX Spark

### Dense Model Characteristics on DGX Spark

- **Higher Memory Bandwidth Usage:** All 71B params active = more data movement per token
- **Lower Batch Capacity:** Larger model = fewer concurrent requests vs 30B models
- **Excellent Single-Request Quality:** No expert routing = consistent performance
- **Long-Context Capable:** 128K context feasible with careful memory management

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
  - Long-context optimization for 128K tokens
  - Correct unified memory handling

---

## Docker Compose Service

### Service Configuration

```yaml
vllm-llama33-70b-fp8:
  image: nvcr.io/nvidia/vllm:25.09-py3
  container_name: vllm-llama33-70b-fp8
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
    - nvidia/Llama-3.3-70B-Instruct-FP8
    - --download-dir
    - /root/.cache/huggingface
    - --max-model-len
    - "65536"
    - --gpu-memory-utilization
    - "0.80"
    - --max-num-batched-tokens
    - "32768"
    - --max-num-seqs
    - "32"
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
docker compose up -d vllm-llama33-70b-fp8

# View logs
docker compose logs -f vllm-llama33-70b-fp8

# Stop service
docker compose stop vllm-llama33-70b-fp8

# Restart service
docker compose restart vllm-llama33-70b-fp8

# Check status
docker compose ps vllm-llama33-70b-fp8
```

### Initial Startup

**First-time model download:** ~10-12 minutes
- Downloads ~35 GB from HuggingFace
- Caches to `/opt/hf`
- Loads model into GPU memory
- Compiles optimized FP8 kernels

**Subsequent startups:** ~60-90 seconds
- Loads from local cache
- Uses pre-compiled kernels
- Much faster after first run

---

## Configuration Parameters

### GPU Memory Utilization

```bash
--gpu-memory-utilization 0.80
```

**Why 80%?**
- Model: ~35 GB
- KV Cache: ~40 GB (at 80% utilization)
- Headroom: ~20 GB for system overhead and long-context handling
- **Lower than smaller models** due to larger model size (71B vs 30B)

**Tuning Guidelines:**
- **0.75:** More conservative, ~35 GB KV cache, better stability
- **0.80:** Recommended balance (default)
- **0.85:** Higher throughput, ~48 GB KV cache (may cause OOM with very long contexts)

**Warning:** Don't exceed 0.85 with this model - risk of OOM on long contexts.

### Context Length

```bash
--max-model-len 65536
```

**Native support:** 128,000 tokens
**Configured:** 65,536 tokens (half of maximum)

**Why 65K instead of 128K?**
- **Memory Constraints:** 128K context requires ~80-100 GB KV cache (exceeds DGX Spark capacity)
- **Practical Balance:** 65K tokens = ~48,000 words = most use cases
- **Performance:** Lower context = better throughput and stability
- **Scaling:** Can increase to 96K if needed (reduce concurrency)

**Context Scaling Options:**
```bash
# Conservative (32K): More concurrent requests, faster
--max-model-len 32768

# Balanced (65K): Recommended default
--max-model-len 65536

# Extended (96K): Maximum for DGX Spark, reduce concurrency
--max-model-len 98304
--max-num-seqs 16  # Reduce to 16 concurrent
```

### Batching Configuration

```bash
--max-num-seqs 32              # Concurrent requests
--max-num-batched-tokens 32768 # Tokens processed per batch
```

**32 concurrent sequences:**
- Lower than Qwen models due to larger model size
- Each request can have different context lengths
- Automatic batching by vLLM scheduler

**32,768 batched tokens:**
- Balances latency vs throughput
- Optimal for DGX Spark memory bandwidth with 70B model
- Allows multiple long-context requests in batch

### Prefix Caching

```bash
--enable-prefix-caching
```

**Critical for long-context performance:**
- Caches common prompt prefixes (system prompts, document preambles)
- Reduces redundant computation by ~50-80% for repeated prefixes
- **Essential for 128K context model:** Avoids reprocessing long shared contexts
- Especially valuable on bandwidth-limited DGX Spark

**Example Use Case:**
```python
# Document analysis with cached document
messages = [
    {"role": "system", "content": "You are a legal document analyzer."},
    {"role": "user", "content": f"<document>{long_contract}</document>\n\nQuestion: {question}"}
    # The document part will be cached across multiple questions
]
```

---

## API Usage

### OpenAI-Compatible Endpoint

**Base URL:** `http://localhost:8000/v1`

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain the key differences between dense and sparse (MoE) language models."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": false
  }'
```

### Long-Context Analysis

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are a document analysis expert."},
      {"role": "user", "content": "Here is a research paper:\n\n[... 50K tokens of paper text ...]\n\nProvide a comprehensive summary highlighting key contributions, methodology, and conclusions."}
    ],
    "max_tokens": 2048,
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
    model="nvidia/Llama-3.3-70B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are an expert software architect."},
        {"role": "user", "content": "Design a scalable microservices architecture for a real-time analytics platform."}
    ],
    max_tokens=2048,
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
    model="nvidia/Llama-3.3-70B-Instruct-FP8",
    messages=[
        {"role": "user", "content": "Write a detailed technical blog post about designing distributed systems for high availability."}
    ],
    max_tokens=2048,
    temperature=0.7,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Long-Context Document Analysis

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Example: Analyze a large codebase or document
with open("large_document.txt", "r") as f:
    document = f.read()  # Up to ~50K tokens

response = client.chat.completions.create(
    model="nvidia/Llama-3.3-70B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a code review expert."},
        {"role": "user", "content": f"Review this codebase:\n\n{document}\n\nProvide a comprehensive analysis of architecture, potential issues, and improvement suggestions."}
    ],
    max_tokens=3072,
    temperature=0.7
)

print(response.choices[0].message.content)
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
- `vllm:prompt_tokens_total` - Total input tokens processed
- `vllm:generation_tokens_total` - Total output tokens generated

### GPU Monitoring

```bash
# Host GPU stats
nvidia-smi

# Container GPU stats
docker exec vllm-llama33-70b-fp8 nvidia-smi

# Watch GPU usage
watch -n 1 'docker exec vllm-llama33-70b-fp8 nvidia-smi'
```

### Container Logs

```bash
# Follow logs
docker compose logs -f vllm-llama33-70b-fp8

# Last 100 lines
docker compose logs --tail=100 vllm-llama33-70b-fp8

# Search logs for errors
docker compose logs vllm-llama33-70b-fp8 | grep -i error
```

---

## Performance Optimization

### 1. Leverage Prefix Caching for Long Documents

**Problem:** Re-processing long documents is expensive
**Solution:** Structure prompts to maximize prefix reuse

```python
# ✅ Good: Document in consistent position (cached)
messages = [
    {"role": "system", "content": "You are a legal analyst."},
    {"role": "user", "content": f"{long_document}\n\nQuestion: {question}"}
]

# ❌ Bad: Document position varies (no caching)
messages = [
    {"role": "user", "content": f"Question: {question}\n\nDocument: {long_document}"}
]
```

**Impact:** 50-80% reduction in processing time for repeated document queries.

### 2. Tune Context Length for Workload

**Problem:** Reserving 65K context may be excessive for short tasks
**Solution:** Adjust `--max-model-len` based on actual needs

```bash
# Short tasks (<10K context):
--max-model-len 16384  # Increases concurrency, faster throughput

# Medium tasks (10-30K context):
--max-model-len 32768  # Balanced

# Long tasks (30-65K context):
--max-model-len 65536  # Default

# Very long tasks (65-96K context):
--max-model-len 98304  # Reduce --max-num-seqs to 16
```

### 3. Batch Long-Context Requests

**Problem:** Single long-context request underutilizes GPU
**Solution:** Send multiple concurrent long-context requests

```bash
# Send 4-8 concurrent long-context requests
# vLLM will batch them efficiently
```

**Expected Results:**
- Single 32K context request: ~5-7 tok/s
- 8 concurrent 32K context requests: ~30-50 tok/s aggregate

### 4. Monitor KV Cache Usage

**Problem:** Running out of KV cache with long contexts
**Solution:** Monitor and adjust

```bash
# Check cache utilization
curl http://localhost:8000/metrics | grep gpu_cache_usage_perc
```

**Interpretation:**
- **<70%:** Underutilized, can increase concurrency or context length
- **70-90%:** Optimal utilization
- **>90%:** May cause queuing or OOM, reduce concurrency or context length

### 5. Optimize for Your Use Case

**High Quality, Low Throughput:**
```bash
--max-model-len 65536
--max-num-seqs 16
--gpu-memory-utilization 0.80
```

**Balanced:**
```bash
--max-model-len 32768
--max-num-seqs 32
--gpu-memory-utilization 0.80
```

**High Throughput, Shorter Context:**
```bash
--max-model-len 16384
--max-num-seqs 48
--gpu-memory-utilization 0.85
```

---

## Comparison: Llama 3.3 70B vs Qwen3-30B-A3B

| Feature | Llama 3.3 70B-FP8 | Qwen3-30B-A3B-FP8 |
|---------|-------------------|-------------------|
| **Architecture** | Dense Transformer | Sparse MoE |
| **Total Parameters** | 71B | 30B |
| **Active Parameters** | 71B per token | 3B per token |
| **Model Memory** | ~35 GB | ~30 GB |
| **KV Cache (80% util)** | ~40 GB | ~55 GB |
| **Total Memory** | ~75 GB | ~85 GB |
| **Context Length** | 65,536 (128K max) | 32,768 |
| **Single Request TPS** | ~5-7 tok/s | ~7-9 tok/s |
| **Batched TPS (32 concurrent)** | ~80-150 tok/s | ~150-250 tok/s |
| **Quality** | State-of-the-art | 95-98% of dense 30B |
| **Long-Context** | Excellent (128K native) | Good (32K native) |
| **Best For** | Highest quality, long-context | Efficiency, mixed workloads |

**Recommendations:**
- **Use Llama 3.3 70B-FP8** for: Maximum quality, long-context analysis, complex reasoning, production workloads
- **Use Qwen3-30B-A3B-FP8** for: Better throughput, more concurrency, shorter contexts, cost efficiency

---

## Comparison: Llama 3.3 70B vs Qwen3-32B

| Feature | Llama 3.3 70B-FP8 | Qwen3-32B-FP8 |
|---------|-------------------|----------------|
| **Architecture** | Dense Transformer | Dense Transformer |
| **Parameters** | 71B | 32.8B |
| **Model Memory** | ~35 GB | ~32 GB |
| **Context Length** | 65,536 (128K max) | 32,768 |
| **Single Request TPS** | ~5-7 tok/s | ~6-7 tok/s |
| **Batched TPS** | ~80-150 tok/s | ~300-400 tok/s |
| **Quality** | Higher (71B params) | High (32B params) |
| **Best For** | Quality + long-context | Maximum throughput |

**Recommendations:**
- **Use Llama 3.3 70B-FP8** for: Better quality, 2x longer context
- **Use Qwen3-32B-FP8** for: 2-3x better throughput with shorter contexts

---

## Troubleshooting

### Model Fails to Load

**Symptom:** Container exits with OOM or CUDA errors

**Solutions:**
1. Reduce GPU memory utilization:
   ```bash
   --gpu-memory-utilization 0.75
   ```

2. Reduce context length:
   ```bash
   --max-model-len 32768
   ```

3. Reduce max sequences:
   ```bash
   --max-num-seqs 16
   ```

4. Check GPU availability:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

### Slow Performance

**Symptom:** <4 tokens/sec generation

**Solutions:**
1. **Check GPU utilization:**
   ```bash
   docker exec vllm-llama33-70b-fp8 nvidia-smi
   # GPU util should be >70% during inference
   ```

2. **Enable prefix caching** (already enabled in config)

3. **Verify FP8 quantization is active:**
   ```bash
   docker compose logs vllm-llama33-70b-fp8 | grep -i "quantization"
   # Should show: quantization=fp8
   ```

4. **For long contexts:** Use prefix caching with repeated document prefixes

### High Latency / Queuing

**Symptom:** Long time to first token (>2 seconds) or requests queuing

**Solutions:**
1. **Check KV cache usage:**
   ```bash
   curl http://localhost:8000/metrics | grep cache_usage
   ```

2. **If cache >90%, reduce context length or concurrency:**
   ```bash
   --max-model-len 49152  # From 65536
   --max-num-seqs 24      # From 32
   ```

3. **Increase GPU memory utilization (careful):**
   ```bash
   --gpu-memory-utilization 0.85  # From 0.80
   ```

### Out of Memory with Long Contexts

**Symptom:** OOM errors when processing very long prompts

**Solutions:**
1. **Reduce max context length:**
   ```bash
   --max-model-len 49152  # From 65536
   ```

2. **Reduce concurrency:**
   ```bash
   --max-num-seqs 16  # From 32
   ```

3. **Check if multiple long requests are batched:**
   ```bash
   curl http://localhost:8000/metrics | grep num_requests_running
   # If >4 with long contexts, reduce concurrency
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
   # Need ~40 GB free for model
   ```

3. **Check logs:**
   ```bash
   docker compose logs vllm-llama33-70b-fp8
   ```

4. **Ensure no other service is using port 8000:**
   ```bash
   docker compose stop vllm-qwen3-32b-fp8
   docker compose stop vllm-qwen3-30b-a3b-fp8
   ```

---

## Important Notes

- **Never run multiple vLLM services simultaneously** - All use port 8000 and need significant GPU memory
- **First-time loading is slow** - Model download takes ~10-12 minutes
- **Subsequent starts are faster** - Models cached at `/opt/hf`
- **Memory bandwidth is the bottleneck** - 71B dense model = high bandwidth usage
- **Long-context requires careful tuning** - 128K context not feasible on DGX Spark, use 65K or less
- **Prefix caching is critical** - Essential for long-context performance
- **Lower throughput than smaller models** - Trade-off for higher quality and longer context
- **65K context = ~48,000 words** - Sufficient for most real-world use cases

---

## References

- **Repository Root:** `/opt/inference/README.md`
- **Hardware Guide:** `/opt/inference/docs/nvidia-spark.md`
- **CLAUDE.md:** `/opt/inference/CLAUDE.md` (AI assistant guide)
- **Docker Compose:** `/opt/inference/docker-compose.yml`
- **vLLM Documentation:** https://docs.vllm.ai
- **Llama 3.3 Model Card:** https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- **NVIDIA FP8 Model:** https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

---

**Last Updated:** 2025-11-08
**Model:** Llama-3.3-70B-Instruct-FP8
**vLLM Version:** 0.10.1.1+381074ae.nv25.09
**Container:** nvcr.io/nvidia/vllm:25.09-py3
