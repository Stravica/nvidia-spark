# vLLM: Mistral-NeMo-12B-Instruct-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running neuralmagic/Mistral-Nemo-Instruct-2407-FP8 using vLLM on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** vLLM-optimized `neuralmagic/Mistral-Nemo-Instruct-2407-FP8` for long-context inference on DGX Spark.

## What is Mistral-NeMo-12B?

Mistral-NeMo-12B-Instruct is a state-of-the-art 12B parameter language model developed in collaboration between Mistral AI and NVIDIA. It's specifically designed for long-context understanding and instruction following.

**Key Features:**
- **Long Context Native:** 128K token context window (state-of-the-art for 12B models)
- **NVIDIA Collaboration:** Co-developed with NVIDIA for optimal performance
- **Tekken Tokenizer:** Efficient tokenizer with better multilingual support
- **Dense Architecture:** 12B parameters for strong reasoning capabilities
- **Commercial License:** Apache 2.0 open license
- **Quantization Aware:** Trained with quantization in mind for FP8 performance

**Advantages on DGX Spark:**
- **Longest Context:** Native 128K support (configured to 65K for DGX Spark stability)
- **Balanced Size:** 12B parameters balance quality and performance
- **High Efficiency:** Better tok/s than larger models while maintaining quality
- **Long Document Analysis:** Best in platform for extended context tasks
- **Competitive Performance:** State-of-the-art quality for 12B size class

## Good Use Cases

**Mistral-NeMo-12B-FP8 excels at:**

1. **Long-Context Analysis**
   - Multi-document question answering
   - Long-form document summarization (up to 65K tokens on DGX Spark)
   - Extended conversation history and context retention
   - Code repository analysis (large codebases)

2. **RAG (Retrieval-Augmented Generation)**
   - Processing many retrieved documents in single context
   - Reducing need for multiple API calls
   - Better context coherence across retrieved chunks

3. **Complex Instruction Following**
   - Multi-step task completion
   - Structured output generation
   - API and tool use scenarios

4. **Balanced Quality and Performance**
   - When you need better quality than 8B models
   - When you can't fit 70B models in memory budget
   - Sweet spot between Qwen3-8B and Qwen3-32B

5. **Multilingual Tasks**
   - Tekken tokenizer provides better multilingual support
   - Competitive with Qwen on non-English languages

## When NOT to Use This Model

**Consider alternatives when you need:**

- **Maximum Speed:** Use Qwen3-8B-FP8 or Llama-3.1-8B-FP8 for fastest throughput
- **Maximum Quality:** Use Llama-3.3-70B-FP8 for highest reasoning capability
- **Standard Context Only:** 8B models may be more efficient if you don't need long context
- **Extreme Concurrency:** Smaller models can handle more concurrent short requests

**Trade-offs:**
- Slightly slower than 8B models due to larger size
- Lower batched throughput vs 8B models
- Long context configuration reduces max concurrent requests
- Still limited to 65K context on DGX Spark (vs native 128K)

## Model Specifications

- **Model:** neuralmagic/Mistral-Nemo-Instruct-2407-FP8 (vLLM-optimized)
- **Base Model:** mistralai/Mistral-Nemo-Instruct-2407
- **Parameters:** 12.0 billion
- **Quantization:** FP8 (8-bit floating point, vLLM-optimized)
- **Architecture:** Dense Transformer with Grouped Query Attention
- **Context Length:** 128,000 tokens (native), configured to 65,536 for DGX Spark
- **Tokenizer:** Tekken (efficient, multilingual-optimized)
- **Memory Requirements:**
  - Model: ~12 GB
  - KV Cache: ~75-80 GB (with 65K context, 64 concurrent sequences)
  - Total: ~87-92 GB peak GPU memory usage

## Hardware: NVIDIA DGX Spark

For detailed hardware specifications, see [docs/nvidia-spark.md](../nvidia-spark.md).

### Key Performance Characteristics
- **Primary Bottleneck:** Memory bandwidth (273 GB/s)
- **Optimization Strategy:** FP8 quantization + long-context batching + prefix caching
- **Measured Performance:**
  - Single request: ~8-9 tokens/sec generation
  - Batched (64 concurrent): ~400-450 tokens/sec aggregate (estimated)
  - Time to first token: 150-350ms (depends on prompt length)
- **FP8 Advantages:**
  - Model memory: ~12 GB (50% reduction vs BF16)
  - KV cache: ~75-80 GB (enables 65K context with good concurrency)
  - Context length: 65K configured (128K native max)
  - Concurrent sequences: 64 with 65K context support
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
  - Long-context optimization

## Docker Compose Service Configuration

```yaml
services:
  vllm-mistral-nemo-12b-fp8:
    image: nvcr.io/nvidia/vllm:25.09-py3
    container_name: vllm-mistral-nemo-12b-fp8
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
      - neuralmagic/Mistral-Nemo-Instruct-2407-FP8
      - --download-dir
      - /root/.cache/huggingface
      - --max-model-len
      - "65536"
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
| `--model` | `neuralmagic/Mistral-Nemo-Instruct-2407-FP8` | vLLM-optimized pre-quantized version |
| `--max-model-len` | `65536` | Half of native 128K (balanced for DGX Spark memory) |
| `--gpu-memory-utilization` | `0.90` | Aggressive utilization for long-context support |
| `--max-num-batched-tokens` | `16384` | High value to maximize bandwidth utilization |
| `--max-num-seqs` | `64` | Good concurrency even with long context |
| `--enable-prefix-caching` | enabled | Critical for long-context efficiency |
| `--trust-remote-code` | enabled | May be required for model features |

**Performance Benefits:**
- Model memory: ~12 GB (vLLM-optimized)
- KV cache: ~75-80 GB (enables long context)
- Context length: 65K (2x larger than 8B models)
- Concurrent sequences: 64 with long context
- Single-request TPS: ~8-9 (competitive)
- Batched TPS: ~400-450 (strong for 12B size)

**Long-Context Benefits:**
- Can process documents up to 65K tokens in single request
- Reduces need for chunking and multiple API calls
- Better coherence for long-form content
- Ideal for RAG with many retrieved documents

## Deployment

### Prerequisites

1. **Environment Variables:**
   Create `.env` file:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```

2. **Model Cache:**
   Ensure `/opt/hf` directory exists with proper permissions

3. **Disk Space:**
   Need ~15GB free space for model storage

### Start Service

```bash
# Stop any running vLLM services (all use port 8000)
docker compose stop vllm-qwen3-32b-fp8 vllm-qwen3-30b-a3b-fp8 vllm-llama33-70b-fp8 vllm-qwen3-8b-fp8 vllm-llama31-8b-fp8

# Start vLLM Mistral-NeMo-12B-FP8 service
docker compose up -d vllm-mistral-nemo-12b-fp8
```

### Monitor Loading

```bash
# View logs
docker compose logs -f vllm-mistral-nemo-12b-fp8

# Filter for key loading events
docker compose logs -f vllm-mistral-nemo-12b-fp8 | grep -E "(Loading|INFO|Ready|Serving)"
```

**Model loading takes ~7-10 minutes on first run:**
1. Model download (~5 min if not cached - ~12GB)
2. Loading safetensors shards (~2.5 min)
3. torch.compile (~20 sec)
4. KV cache initialization and CUDA graph capture (~10 sec for long context)

**Subsequent startups:** ~3-4 minutes (model already cached)

### Health Check

```bash
# API health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test inference (short)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/Mistral-Nemo-Instruct-2407-FP8",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test long-context capability
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/Mistral-Nemo-Instruct-2407-FP8",
    "messages": [{"role": "user", "content": "Summarize this long document: [insert 10K+ token document]"}],
    "max_tokens": 500,
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

# Long-context document analysis
with open('long_document.txt', 'r') as f:
    document = f.read()  # Can be up to ~65K tokens

response = client.chat.completions.create(
    model="neuralmagic/Mistral-Nemo-Instruct-2407-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful document analysis assistant."},
        {"role": "user", "content": f"Analyze this document and provide key insights:\n\n{document}"}
    ],
    max_tokens=1000,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### cURL (Chat Completions)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neuralmagic/Mistral-Nemo-Instruct-2407-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What are the main themes in this text?"}
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
    "model": "neuralmagic/Mistral-Nemo-Instruct-2407-FP8",
    "messages": [{"role": "user", "content": "Tell me about machine learning"}],
    "max_tokens": 500,
    "stream": true
  }'
```

## Performance Optimization

### For Long-Context Processing (Recommended)

```yaml
--max-model-len 65536              # Extended context (half of native 128K)
--gpu-memory-utilization 0.90      # Maximum memory usage for long context
--max-num-batched-tokens 16384     # Large batches
--max-num-seqs 64                  # Good concurrency with long context
--enable-prefix-caching            # Critical for long-context efficiency
```

**Expected Performance:**
- Context: 65K tokens (best in platform for 12B class)
- Concurrent requests: 64
- Batched throughput: ~400-450 tokens/sec
- Best for: Long document analysis, RAG with many documents

### For Maximum Throughput

```yaml
--max-model-len 32768              # Standard context
--gpu-memory-utilization 0.90      # Maximum memory usage
--max-num-batched-tokens 20480     # Larger batches
--max-num-seqs 96                  # Higher concurrency
--enable-prefix-caching            # Enable caching
```

**Expected Performance:**
- Context: 32K tokens
- Concurrent requests: 96+
- Batched throughput: ~450-500 tokens/sec
- Trade-off: Shorter context for higher throughput

### For Extreme Long Context (Lower Concurrency)

```yaml
--max-model-len 98304              # 96K tokens (3/4 of native 128K)
--gpu-memory-utilization 0.90      # Maximum memory usage
--max-num-batched-tokens 16384     # Standard batches
--max-num-seqs 32                  # Reduced concurrency
```

**Expected Performance:**
- Context: 96K tokens (approaching native limit)
- Concurrent requests: 32 (vs 64 standard)
- Best for: Very long document analysis
- Note: May be unstable - 65K recommended for production

## Monitoring

### GPU Metrics

```bash
# Check GPU utilization
docker exec vllm-mistral-nemo-12b-fp8 nvidia-smi

# Watch GPU status (updates every 2 seconds)
watch -n 2 docker exec vllm-mistral-nemo-12b-fp8 nvidia-smi

# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# Container stats
docker stats vllm-mistral-nemo-12b-fp8
```

### Key Metrics to Watch

- **Memory Usage:** ~87-92 GB total (~12 GB model + 75-80 GB KV cache)
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **Throughput:** Check `vllm:num_generation_tokens_total` in metrics
- **Queue Depth:** Monitor running vs waiting requests (max 64 concurrent)
- **KV Cache Usage:** ~280,000+ tokens capacity at 65K context (check `vllm:gpu_cache_usage_perc`)
- **Prefix Cache Hit Rate:** Critical for long-context efficiency

## Comparison with Other Models

| Model | Parameters | Model Memory | KV Cache | Context | Single TPS | Batched TPS | Concurrency |
|-------|-----------|--------------|----------|---------|------------|-------------|-------------|
| Qwen3-8B-FP8 | 8B | ~8 GB | ~80-85 GB | 32K | ~9-10 | ~450-500 | 64+ |
| Llama-3.1-8B-FP8 | 8B | ~8 GB | ~80-85 GB | 32K | ~9-10 | ~450-500 | 64+ |
| **Mistral-NeMo-12B-FP8** | 12B | ~12 GB | ~75-80 GB | **65K** | ~8-9 | ~400-450 | 64 |
| Qwen3-32B-FP8 | 32B | ~32 GB | ~66 GB | 32K | ~6-7 | ~300-400 | 64 |
| Qwen3-30B-A3B-FP8 | 30B (3B active) | ~30 GB | ~55-70 GB | 32K | ~7-9 | ~200-350 | 64 |
| Llama-3.3-70B-FP8 | 70B | ~35 GB | ~40-60 GB | 65K | ~5-7 | ~80-150 | 32 |

**Mistral-NeMo-12B-FP8 Advantages:**
- **Longest context in 12B class** (65K configured, 128K native)
- **Best for long-document analysis** at this size
- **Balanced performance** between 8B and 32B models
- **Tekken tokenizer** for better multilingual support
- **NVIDIA collaboration** - optimized for NVIDIA hardware
- **Apache 2.0 license** - fully open for commercial use

**Use Case Positioning:**
- **Choose 8B models** for maximum speed/throughput
- **Choose Mistral-NeMo-12B** for long-context needs (best choice)
- **Choose 32B models** for better quality on standard context
- **Choose 70B models** for maximum quality (long context also available)

## Troubleshooting

### Issue: Container Fails with Memory Errors

**Symptom:**
```
ValueError: Free memory on device (XX GiB) is less than desired GPU memory utilization
```

**Solution:**
1. Stop all containers: `docker compose down`
2. Clean Docker resources: `docker system prune -f`
3. Reduce `--max-model-len` to 32768 (from 65536)
4. Reduce `--gpu-memory-utilization` to 0.85
5. Reduce `--max-num-seqs` to 32

### Issue: FP8 Quantization Fails

**Symptom:**
```
RuntimeError: Error Internal
cutlass_scaled_mm failed
```

**Solution:**
- **Recommended:** Use pre-quantized model `neuralmagic/Mistral-Nemo-Instruct-2407-FP8` (default)
- vLLM-optimized version works reliably on NVIDIA hardware
- Avoid using `--quantization fp8` flag with base model

### Issue: Model Download Incomplete

**Symptom:**
```
Loading safetensors checkpoint shards: X/Y
[hangs or crashes]
```

**Solution:**
1. Verify `HF_TOKEN` is set correctly in `.env`
2. Check disk space: `df -h /opt/hf` (need ~15GB free)
3. Clean incomplete downloads:
   ```bash
   rm -rf /opt/hf/models--neuralmagic--Mistral-Nemo-Instruct-2407-FP8/*.incomplete
   ```
4. Restart container

### Issue: Long-Context Requests Timeout or OOM

**Symptom:**
```
CUDA out of memory
or requests timeout with very long contexts
```

**Solution:**
1. Reduce `--max-model-len` to 32768 or 49152
2. Reduce `--max-num-seqs` to 32 for very long contexts
3. Ensure you're not sending multiple full-length (65K) contexts simultaneously
4. Monitor KV cache usage - should not exceed 90%

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
3. For long contexts, reduce `--max-num-seqs` to avoid memory pressure
4. Verify prefix caching is enabled and working (critical for long contexts)
5. Monitor KV cache usage - should be 80-95% utilized

## Important Notes

1. **All vLLM services use port 8000** - Only one vLLM model can run at a time
2. **Long-context champion** - 65K configured context (128K native)
3. **vLLM-optimized** - Neural Magic version optimized for vLLM performance
4. **Tekken tokenizer** - Different from standard tokenizers, more efficient
5. **Batching essential** - Especially for long-context workloads
6. **First load slow** - Expect 7-10 minutes for initial download and compilation
7. **Subsequent starts faster** - Model cached at `/opt/hf` after first load
8. **Memory bandwidth bottleneck** - 273 GB/s affects all models
9. **Context vs Concurrency trade-off** - Longer context = fewer concurrent requests

## Known Limitations

1. **Memory Bandwidth Bottleneck:** 273 GB/s limits single-request token generation speed
2. **Context Length on DGX Spark:** Native 128K limited to 65K by memory constraints
3. **CUDA Graphs:** Limited capture sizes (max 96 vs 512 on datacenter GPUs)
4. **Long Context Trade-off:** 65K context reduces max concurrent sequences vs standard 32K
5. **Throughput vs Context:** Optimized for long-context quality over maximum throughput

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

- **Model:** https://huggingface.co/neuralmagic/Mistral-Nemo-Instruct-2407-FP8
- **Base Model:** https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
- **Mistral-NeMo Announcement:** https://mistral.ai/news/mistral-nemo/
- **NVIDIA Collaboration:** https://developer.nvidia.com/blog/mistral-nemo-12b-minitron-llm-family/
- **vLLM Documentation:** https://docs.vllm.ai
- **NVIDIA vLLM Container:** https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html
- **DGX Spark Hardware:** [docs/nvidia-spark.md](../nvidia-spark.md)
- **Related Models:**
  - [Qwen3-8B-FP8](qwen3-8b-fp8.md) - Faster 8B alternative
  - [Llama-3.1-8B-FP8](llama31-8b-fp8.md) - Another 8B option
  - [Qwen3-32B-FP8](qwen3-32b-fp8.md) - Larger dense model
  - [Llama-3.3-70B-FP8](llama33-70b-fp8.md) - Highest quality with long context

---

**Last Updated:** 2025-11-10
**Model:** neuralmagic/Mistral-Nemo-Instruct-2407-FP8
**vLLM Version:** 0.10.1.1+381074ae.nv25.09
**Container:** nvcr.io/nvidia/vllm:25.09-py3
