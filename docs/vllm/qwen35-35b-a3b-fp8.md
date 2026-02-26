# vLLM: Qwen3.5-35B-A3B-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen/Qwen3.5-35B-A3B-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** Pre-quantized `Qwen/Qwen3.5-35B-A3B-FP8` for optimal performance on DGX Spark unified memory architecture.

**Successor to:** Qwen3-30B-A3B-FP8 (same MoE family, new architecture)

---

## What is Qwen3.5-35B-A3B?

**Qwen3.5-35B-A3B** is a Mixture-of-Experts (MoE) model from the Qwen3.5 family, featuring a novel hybrid Gated DeltaNet architecture designed for efficient long-context inference.

### Model Architecture

- **Total Parameters:** 35 billion
- **Active Parameters:** 3 billion per token (~9% activation)
- **Architecture:** Sparse MoE with hybrid Gated DeltaNet + softmax attention
- **Quantization:** FP8 (fine-grained, block size 128)

### How Gated DeltaNet Works

Unlike the standard transformer architecture used in Qwen3-30B-A3B, Qwen3.5 uses a **hybrid attention** design with two types of layers arranged in a repeating 4-layer cycle:

- **3x Gated DeltaNet layers** (linear attention): O(n) scaling with sequence length, using recurrent hidden state instead of KV cache
- **1x Full softmax attention layer**: Quadratic attention providing global context and reasoning fidelity

This 3:1 hybrid ratio means:
- **30 of 40 layers** use linear attention (no KV cache needed, fixed-size state per sequence)
- **10 of 40 layers** use standard GQA attention (KV cache, 16 Q heads / 2 KV heads)

The result is dramatically reduced memory usage for long contexts compared to pure transformers.

### Mixture of Experts

- **Total Experts:** 256
- **Routed Experts per Token:** 8
- **Shared Experts:** 1 (always active)
- **Active Experts per Token:** 9 (8 routed + 1 shared)
- **Expert Intermediate Dimension:** 512

### Key Advantages Over Qwen3-30B-A3B

1. **Hybrid Attention:** DeltaNet linear layers scale O(n) instead of O(n^2) with context length
2. **Dramatically Reduced KV Cache:** Only 10/40 layers need KV cache (vs all layers in Qwen3)
3. **More Experts:** 256 experts (vs fewer in Qwen3) with finer-grained specialization
4. **Longer Native Context:** 262K tokens native (vs 32K in Qwen3-30B-A3B)
5. **Multi-Token Prediction:** Native MTP support for faster inference
6. **Thinking/Reasoning Mode:** Built-in chain-of-thought reasoning support

### Good Use Cases

This model excels at:

- **Long-Context Tasks:** Document analysis, code review, RAG with up to 65K+ context on Spark
- **Complex Reasoning:** Built-in thinking mode for multi-step problem solving
- **Coding:** Strong coding performance with tool-use support
- **General Instruction Following:** Improved quality over Qwen3-30B-A3B
- **Agentic Workflows:** Native tool calling with `qwen3_coder` parser
- **Batch Processing:** MoE efficiency enables high-throughput batching

### When NOT to Use This Model

Consider alternatives if you need:
- **Absolute maximum quality** -> Use Llama 3.3 70B (dense 70B)
- **Smallest memory footprint** -> Use Qwen3-8B or Llama-3.1-8B (8B models)
- **Proven NVIDIA container support** -> Use Qwen3-30B-A3B-FP8 (older, fully tested container)
- **Vision/multimodal on Spark** -> Vision encoder adds memory overhead; use `--language-model-only` or a dedicated VL model

---

## Model Specifications

- **Model:** Qwen/Qwen3.5-35B-A3B-FP8 (officially pre-quantized by Qwen team)
- **Total Parameters:** 35 billion
- **Active Parameters:** 3 billion per token
- **Quantization:** FP8 (fine-grained, block size 128)
- **Architecture:** Hybrid Gated DeltaNet + Softmax Attention, Sparse MoE with GQA
- **Hidden Dimension:** 2,048
- **Number of Layers:** 40 (30 DeltaNet + 10 full attention)
- **Vocabulary Size:** 248,320
- **Context Length:** 262,144 tokens (native), extensible to 1,010,000
- **Memory Requirements:**
  - Model weights: ~37.5 GB (FP8)
  - KV Cache + DeltaNet state: ~55-70 GB (with 32K context, 64 concurrent sequences)
  - Total: ~90-108 GB peak GPU memory usage

---

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics

- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + DeltaNet efficiency + batching + prefix caching + MoE
- **Expected Performance:**
  - Single request: ~8-12 tokens/sec generation (DeltaNet layers are faster than standard attention)
  - Batched (32-64 concurrent): ~200-400 tokens/sec aggregate
  - Time to first token: 100-300ms (DeltaNet O(n) prefill is faster than O(n^2) attention)
- **FP8 Advantages:**
  - Model memory: ~37.5 GB (50% reduction vs BF16)
  - Quality: Nearly identical to full precision (fine-grained FP8 with block size 128)

### DeltaNet-Specific Benefits on DGX Spark

- **Reduced KV Cache Memory:** Only 10/40 layers need KV cache, freeing memory for longer contexts
- **Fixed-Size DeltaNet State:** ~31 MB per sequence (fixed regardless of context length)
- **Faster Prefill:** DeltaNet layers have O(n) prefill vs O(n^2) for standard attention
- **Better Long-Context Scaling:** Memory grows linearly, not quadratically, for 75% of layers

---

## Critical: Container Selection

### Container Compatibility Note

Qwen3.5 was released on February 24, 2026 and uses a novel Gated DeltaNet architecture. Container support is evolving:

- **`nvcr.io/nvidia/vllm:25.09-py3`** - Does NOT support Qwen3.5 (too old)
- **`nvcr.io/nvidia/vllm:26.01-py3`** - May support Qwen3.5 (has DeltaNet support via Qwen3-Next, vLLM 0.11.1). Try this first.
- **`vllm/vllm-openai:cu130-nightly`** - Confirmed Qwen3.5 support for Blackwell GPUs (recommended fallback)

### Recommended: `nvcr.io/nvidia/vllm:26.01-py3`

- **Version:** vLLM 0.11.1
- **CUDA:** 13.1.1
- **Features:**
  - Native GB10 support
  - DeltaNet architecture support (added via Qwen3-Next in vLLM 0.10.2+)
  - MoE-aware batching and scheduling
  - Correct unified memory handling

### Fallback: `vllm/vllm-openai:cu130-nightly`

If the 26.01 container does not support Qwen3.5, use the official vLLM nightly build for Blackwell:

```bash
# In docker-compose.yml, change image to:
image: vllm/vllm-openai:cu130-nightly
```

**Note:** Nightly builds may have stability issues. Check the [vLLM GitHub](https://github.com/vllm-project/vllm) for known issues.

---

## Docker Compose Service

### Service Configuration

```yaml
vllm-qwen35-35b-a3b-fp8:
  image: nvcr.io/nvidia/vllm:26.01-py3
  container_name: vllm-qwen35-35b-a3b-fp8
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
    - Qwen/Qwen3.5-35B-A3B-FP8
    - --served-model-name
    - qwen35-35b-a3b-fp8
    - --download-dir
    - /root/.cache/huggingface
    - --max-model-len
    - "32768"
    - --gpu-memory-utilization
    - "0.80"
    - --max-num-batched-tokens
    - "16384"
    - --max-num-seqs
    - "64"
    - --enable-prefix-caching
    - --trust-remote-code
    - --enable-auto-tool-choice
    - --tool-call-parser
    - qwen3_coder
    - --reasoning-parser
    - qwen3
    - --language-model-only
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

### Key Configuration Differences from Qwen3-30B-A3B

| Parameter | Qwen3-30B-A3B | Qwen3.5-35B-A3B | Reason |
|-----------|---------------|------------------|--------|
| Container | `25.09-py3` | `26.01-py3` | DeltaNet architecture support |
| Model | `Qwen3-30B-A3B-Instruct-2507-FP8` | `Qwen3.5-35B-A3B-FP8` | New model |
| Tool parser | `hermes` | `qwen3_coder` | Qwen3.5 native tool calling |
| `--reasoning-parser` | (none) | `qwen3` | Thinking/reasoning mode support |
| `--language-model-only` | (none) | enabled | Skips vision encoder, saves ~2-3 GB |
| GPU memory util | `0.85` | `0.80` | Conservative for larger model (37.5 vs 30 GB) |

### Service Management

```bash
# Start service
docker compose up -d vllm-qwen35-35b-a3b-fp8

# View logs
docker compose logs -f vllm-qwen35-35b-a3b-fp8

# Stop service
docker compose stop vllm-qwen35-35b-a3b-fp8

# Restart service
docker compose restart vllm-qwen35-35b-a3b-fp8

# Check status
docker compose ps vllm-qwen35-35b-a3b-fp8
```

### Initial Startup

**First-time model download:** ~10-15 minutes
- Downloads ~37.5 GB from HuggingFace
- Caches to `/opt/hf`
- Loads model into GPU memory
- Compiles DeltaNet + MoE kernels (may take longer on first run)

**Subsequent startups:** ~30-90 seconds
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
- Model: ~37.5 GB (larger than Qwen3-30B-A3B's ~30 GB)
- KV Cache + DeltaNet state: ~65 GB (at 80% utilization)
- Headroom: ~25 GB for system overhead

**Tuning Guidelines:**
- **0.75:** Most conservative, ~58 GB for KV cache
- **0.80:** Recommended balance (default)
- **0.85:** Higher throughput, ~71 GB for KV cache
- **0.90:** Maximum throughput (may cause OOM under heavy load)

### Context Length

```bash
--max-model-len 32768
```

**Native support:** 262,144 tokens
**Configured:** 32,768 tokens (conservative for DGX Spark memory)

The model's native 262K context is designed for multi-GPU deployments. On DGX Spark with 128GB unified memory, a conservative 32K context maximizes concurrent request capacity. You can increase this:

```bash
# If you need longer context:
--max-model-len 65536   # 65K - good balance for document analysis

# If memory allows (reduce concurrency):
--max-model-len 131072  # 128K - for deep analysis tasks
```

**DeltaNet advantage:** Because 30/40 layers use fixed-size state instead of KV cache, increasing context length costs significantly less memory than with standard transformers.

### Batching Configuration

```bash
--max-num-seqs 64              # Concurrent requests
--max-num-batched-tokens 16384 # Tokens processed per batch
```

**64 concurrent sequences:**
- MoE efficiency allows high concurrency
- DeltaNet state is fixed-size per sequence (~31 MB each)
- Each request can have different context lengths

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
- **Note:** Prefix caching for DeltaNet/Mamba-style layers is experimental in current vLLM

### Language Model Only

```bash
--language-model-only
```

Skips loading the vision encoder, freeing ~2-3 GB of GPU memory for KV cache. Use this unless you need image/video processing.

### Reasoning Parser

```bash
--reasoning-parser qwen3
```

Enables the model's built-in thinking/reasoning mode. The model can produce chain-of-thought reasoning before its final answer, improving quality on complex tasks.

### Tool Call Parser

```bash
--tool-call-parser qwen3_coder
```

Enables native Qwen3.5 tool calling format, which is more capable than the `hermes` parser used with Qwen3 models.

---

## API Usage

### OpenAI-Compatible Endpoint

**Base URL:** `http://localhost:8000/v1`

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-35b-a3b-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain how the Gated DeltaNet architecture differs from standard transformers."}
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
    "model": "qwen35-35b-a3b-fp8",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": true
  }'
```

### Thinking/Reasoning Mode

The model supports a thinking mode where it reasons through problems before answering:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-35b-a3b-fp8",
    "messages": [
      {"role": "user", "content": "What is 23 * 47 + 156 / 12?"}
    ],
    "max_tokens": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20
  }'
```

**Recommended sampling for thinking mode:**
- Temperature: 1.0
- Top-P: 0.95
- Top-K: 20
- Presence penalty: 1.5

### Tool Calling

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-35b-a3b-fp8",
    "messages": [
      {"role": "user", "content": "What is the weather in London?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "max_tokens": 512
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
    model="qwen35-35b-a3b-fp8",
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
    model="qwen35-35b-a3b-fp8",
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
docker exec vllm-qwen35-35b-a3b-fp8 nvidia-smi

# Watch GPU usage
watch -n 1 'docker exec vllm-qwen35-35b-a3b-fp8 nvidia-smi'
```

### Container Logs

```bash
# Follow logs
docker compose logs -f vllm-qwen35-35b-a3b-fp8

# Last 100 lines
docker compose logs --tail=100 vllm-qwen35-35b-a3b-fp8

# Search logs for errors
docker compose logs vllm-qwen35-35b-a3b-fp8 | grep -i error
```

---

## Performance Optimization

### 1. Maximize Batching

**Problem:** Single requests underutilize GPU
**Solution:** Send concurrent requests

```bash
# Concurrent requests (5-10x faster aggregate throughput)
# vLLM automatically batches these
for i in {1..10}; do
  curl -X POST http://localhost:8000/v1/chat/completions -d '...' &
done
```

**Expected Results:**
- Single request: ~8-12 tok/s
- 16 concurrent: ~100-200 tok/s aggregate
- 64 concurrent: ~200-400 tok/s aggregate

### 2. Enable Prefix Caching (Already Enabled)

**Benefit:** 50-80% speedup for repeated prefixes

```python
# This system prompt will be cached
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},  # Cached
    {"role": "user", "content": "Write a function to..."}  # Different each time
]
```

### 3. Increase Context Length (If Needed)

The DeltaNet architecture makes longer contexts much cheaper than standard transformers:

```bash
# Conservative (default)
--max-model-len 32768

# Moderate - good for document analysis
--max-model-len 65536

# Extended - for deep analysis (reduce concurrency)
--max-model-len 131072
```

### 4. Enable Multi-Token Prediction

For faster single-request inference (at the cost of reduced batch throughput):

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

**Note:** MTP consumes additional KV cache memory. Reduce `--max-num-seqs` if using MTP.

### 5. Adjust GPU Memory Utilization

```bash
# Current: 80% (~65 GB KV cache)
--gpu-memory-utilization 0.80

# High concurrency: 85% (~71 GB KV cache)
--gpu-memory-utilization 0.85
```

**Warning:** Setting too high (>0.90) may cause OOM crashes on DGX Spark's unified memory.

### 6. Monitor KV Cache Usage

```bash
curl http://localhost:8000/metrics | grep gpu_cache_usage_perc
```

**Interpretation:**
- **<80%:** Underutilized, can increase concurrency or context length
- **80-95%:** Optimal utilization
- **>95%:** May cause queuing or OOM, reduce concurrency

---

## Comparison: Qwen3.5-35B-A3B vs Qwen3-30B-A3B

| Feature | Qwen3.5-35B-A3B-FP8 | Qwen3-30B-A3B-FP8 |
|---------|---------------------|-------------------|
| **Architecture** | Hybrid DeltaNet + Attention MoE | Standard Transformer MoE |
| **Total Parameters** | 35B | 30B |
| **Active Parameters** | 3B per token | 3B per token |
| **Experts** | 256 (8+1 active) | Fewer, larger experts |
| **Model Memory** | ~37.5 GB | ~30 GB |
| **KV Cache Layers** | 10/40 (DeltaNet reduces need) | All layers |
| **KV Cache Memory** | ~55-70 GB | ~55-70 GB |
| **Total Memory** | ~90-108 GB | ~85-100 GB |
| **Native Context** | 262,144 tokens | 32,768 tokens |
| **Configured Context** | 32,768 (expandable) | 32,768 |
| **Single Request TPS** | ~8-12 tok/s | ~7-9 tok/s |
| **Batched TPS (64)** | ~200-400 tok/s | ~200-350 tok/s |
| **Thinking Mode** | Native support | Not available |
| **Tool Calling** | `qwen3_coder` (native) | `hermes` (generic) |
| **Container** | `26.01-py3` | `25.09-py3` |
| **Quality** | Improved (outperforms Qwen3-235B) | Baseline |

**Recommendations:**
- **Use Qwen3.5-35B-A3B-FP8** for: Better quality, thinking/reasoning, native tool calling, longer context potential
- **Use Qwen3-30B-A3B-FP8** for: Proven stability, smaller model footprint, if container compatibility is a concern

---

## Troubleshooting

### Container Does Not Support Model

**Symptom:** Error about unknown model architecture or DeltaNet

**Solutions:**
1. Switch to the vLLM nightly container:
   ```yaml
   # In docker-compose.yml, change:
   image: vllm/vllm-openai:cu130-nightly
   ```

2. Or wait for the next NVIDIA container release (`26.02-py3` or later)

3. Check vLLM GitHub for DGX Spark compatibility:
   ```
   https://github.com/vllm-project/vllm/issues
   ```

### Model Fails to Load

**Symptom:** Container exits with OOM or CUDA errors

**Solutions:**
1. Reduce GPU memory utilization:
   ```bash
   --gpu-memory-utilization 0.75
   ```

2. Reduce context length:
   ```bash
   --max-model-len 16384
   ```

3. Ensure `--language-model-only` is set (skips vision encoder)

4. Check GPU availability:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

### FP8 Stream Capture Errors

**Symptom:** CUDA stream capture errors with FP8 models (known issue in 26.01)

**Solutions:**
1. Add `--enforce-eager` flag (disables CUDA graph, slower but stable):
   ```bash
   --enforce-eager
   ```

2. Or switch to nightly container which may have the fix

### Slow Performance

**Symptom:** <5 tokens/sec generation

**Solutions:**
1. **Send concurrent requests** - MoE models benefit greatly from batching
2. **Check GPU utilization:**
   ```bash
   docker exec vllm-qwen35-35b-a3b-fp8 nvidia-smi
   # GPU util should be >80% during inference
   ```
3. **Enable prefix caching** (already enabled in config)
4. **Verify FP8 quantization is active:**
   ```bash
   docker compose logs vllm-qwen35-35b-a3b-fp8 | grep -i "quantization"
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
   --gpu-memory-utilization 0.85
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
   ```

2. **Verify disk space:**
   ```bash
   df -h /opt/hf
   # Need ~40 GB free for model
   ```

3. **Check logs:**
   ```bash
   docker compose logs vllm-qwen35-35b-a3b-fp8
   ```

4. **Ensure no other service is using port 8000:**
   ```bash
   docker compose stop vllm-qwen3-30b-a3b-fp8
   docker compose stop vllm-qwen3-32b-fp8
   docker compose stop vllm-llama33-70b-fp8
   ```

---

## Important Notes

- **Never run multiple vLLM services simultaneously** - All use port 8000 and need significant GPU memory
- **Container compatibility is evolving** - Qwen3.5 was released Feb 24, 2026; container support may need nightly builds
- **First-time loading is slow** - Model download takes ~10-15 minutes (~37.5 GB)
- **Subsequent starts are faster** - Models cached at `/opt/hf`
- **Memory bandwidth is the bottleneck** - Not compute capacity
- **Batching is essential for MoE** - Single-request performance doesn't fully leverage MoE efficiency
- **Prefix caching is critical** - Enables 50-80% speedup for repeated prompts
- **DeltaNet enables longer context** - Can increase `--max-model-len` with lower memory cost than standard transformers
- **Use `--language-model-only`** - Saves memory by skipping the vision encoder

---

## References

- **Repository Root:** `/opt/inference/README.md`
- **Hardware Guide:** `/opt/inference/docs/nvidia-spark.md`
- **CLAUDE.md:** `/opt/inference/CLAUDE.md` (AI assistant guide)
- **Docker Compose:** `/opt/inference/docker-compose.yml`
- **vLLM Documentation:** https://docs.vllm.ai
- **vLLM Qwen3.5 Recipes:** https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- **Qwen3.5 Model Card:** https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8
- **Qwen3.5 GitHub:** https://github.com/QwenLM/Qwen3.5
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/
- **NVIDIA vLLM Container:** https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

---

**Last Updated:** 2026-02-26
**Model:** Qwen3.5-35B-A3B-FP8
**vLLM Version:** 0.11.1 (26.01-py3 container)
**Container:** nvcr.io/nvidia/vllm:26.01-py3
