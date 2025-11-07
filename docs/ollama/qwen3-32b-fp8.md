# Ollama: Qwen3-32B-FP8 on NVIDIA DGX Spark

## Overview

Configuration and deployment guide for running Qwen3-32B with Q8_0 quantization using Ollama on NVIDIA DGX Spark (GB10 Grace Blackwell).

**Model:** `qwen3:32b-q8_0` (8-bit quantization, comparable to FP8)

## Service Specifications

- **Service Name:** `ollama-qwen3-32b-fp8`
- **Container:** `ollama/ollama:latest`
- **Port:** 11434
- **Model Storage:** `/opt/ollama`
- **API Style:** Ollama native (simpler than OpenAI)

## Configuration

### Docker Compose Service

```yaml
ollama-qwen3-32b-fp8:
  image: ollama/ollama:latest
  container_name: ollama-qwen3-32b-fp8
  restart: unless-stopped
  ports:
    - "11434:11434"
  environment:
    OLLAMA_KEEP_ALIVE: "-1"              # Never unload (match vLLM behavior)
    OLLAMA_HOST: "0.0.0.0:11434"
    OLLAMA_NUM_PARALLEL: "8"             # Concurrent requests
    OLLAMA_MAX_LOADED_MODELS: "1"        # Only Qwen3-32B
    OLLAMA_NUM_THREADS: "20"             # Match ARM cores
  volumes:
    - /opt/ollama:/root/.ollama
    - ./models/ollama/Modelfile-qwen3-32b-fp8:/root/Modelfile-qwen3-32b-fp8:ro
  entrypoint: ["/bin/sh", "-c"]
  command:
    - |
      ollama serve &
      sleep 5
      ollama pull qwen3:32b-q8_0
      ollama create qwen3-32b-fp8 -f /root/Modelfile-qwen3-32b-fp8
      wait
```

### Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `OLLAMA_KEEP_ALIVE` | `-1` | Never unload model from memory (matches vLLM behavior) |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Bind to all interfaces on port 11434 |
| `OLLAMA_NUM_PARALLEL` | `8` | Process up to 8 concurrent requests |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Keep only one model loaded (dedicated service) |
| `OLLAMA_NUM_THREADS` | `20` | CPU threads (matches DGX Spark 20 ARM cores) |

### Modelfile Configuration

Location: `models/ollama/Modelfile-qwen3-32b-fp8`

```
FROM qwen3:32b-q8_0

PARAMETER num_ctx 32768
PARAMETER num_batch 512
PARAMETER num_gpu 99
PARAMETER num_thread 20
PARAMETER temperature 0.7

SYSTEM "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
```

#### Modelfile Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `FROM` | `qwen3:32b-q8_0` | Base model with Q8_0 quantization (~35-40GB) |
| `num_ctx` | `32768` | Context window size (32K tokens, matches vLLM) |
| `num_batch` | `512` | Batch size for throughput optimization |
| `num_gpu` | `99` | Use all GPU layers (entire model on GPU) |
| `num_thread` | `20` | CPU threads for parallel processing |
| `temperature` | `0.7` | Default sampling temperature |

## Deployment

### Start Service

```bash
# Start Ollama Qwen3-32B-FP8 service
docker compose up -d ollama-qwen3-32b-fp8

# Monitor startup (model pull + load takes 10-15 minutes first time)
docker compose logs -f ollama-qwen3-32b-fp8
```

### First-Time Startup

**Expected Timeline:**
1. Container starts and launches Ollama server (~5 seconds)
2. Pulls `qwen3:32b-q8_0` model (~10-12 minutes, ~35-40GB download)
3. Creates custom model `qwen3-32b-fp8` with Modelfile (~30 seconds)
4. Model loaded and ready for inference

**Subsequent Startups:**
- Model already cached, starts in ~30 seconds
- Model stays loaded (OLLAMA_KEEP_ALIVE: "-1")

### Health Check

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# List loaded models
curl http://localhost:11434/api/tags

# Check running models
docker exec ollama-qwen3-32b-fp8 ollama ps
```

## Usage Examples

### Generate Text

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-32b-fp8",
  "prompt": "Explain quantum computing in simple terms:",
  "stream": false
}'
```

### Chat Completion

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3-32b-fp8",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "What is the NVIDIA Grace Blackwell architecture?"
    }
  ],
  "stream": false
}'
```

### Streaming Response

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-32b-fp8",
  "prompt": "Write a Python function to calculate fibonacci numbers:",
  "stream": true
}'
```

### Custom Parameters

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-32b-fp8",
  "prompt": "Explain the theory of relativity:",
  "stream": false,
  "options": {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "num_predict": 500
  }
}'
```

## Performance Characteristics

### Memory Usage

- **Model Size:** ~35-40 GB (Q8_0 quantization)
- **KV Cache:** ~20-25 GB (with 32K context at 8 concurrent requests)
- **Total Memory:** ~55-65 GB peak usage
- **Available for OS/Other:** ~63-73 GB remaining (128GB total)

### Throughput & Latency

**Expected Performance:**

| Metric | Performance |
|--------|-------------|
| **Single Request** | ~5-8 tokens/sec |
| **2 Concurrent** | ~10-15 tokens/sec aggregate |
| **4 Concurrent** | ~20-35 tokens/sec aggregate |
| **8 Concurrent** | ~40-80 tokens/sec aggregate |
| **Time to First Token** | 200-800ms (varies with prompt length) |
| **Max Context** | 32,768 tokens |

**Performance Notes:**
- Single-request performance similar to vLLM (~5-8 tok/s vs ~6-7 tok/s)
- Aggregate throughput lower than vLLM due to simpler batching
- Latency competitive for low-concurrency workloads (<8 requests)
- Memory bandwidth (273 GB/s) is the bottleneck, not compute

### Context Size Impact

```
Context Length × Concurrent Requests = Effective Memory Usage

Examples:
- 32K context × 1 request = 32K effective context
- 32K context × 4 requests = 128K effective context
- 32K context × 8 requests = 256K effective context
```

**KV Cache Memory Scales Linearly:**
- Each additional concurrent request adds ~3-4 GB KV cache memory
- Monitor with `docker stats` and `nvidia-smi`

## Performance Tuning

### For Maximum Throughput

Edit Modelfile:
```
PARAMETER num_ctx 32768
PARAMETER num_batch 1024
PARAMETER num_gpu 99
```

Edit docker-compose.yml environment:
```yaml
OLLAMA_NUM_PARALLEL: "8"
```

### For Minimum Latency (Single Request)

Edit Modelfile:
```
PARAMETER num_ctx 16384
PARAMETER num_batch 256
PARAMETER num_gpu 99
```

Edit docker-compose.yml environment:
```yaml
OLLAMA_NUM_PARALLEL: "2"
```

### For Maximum Context (Lower Concurrency)

Edit Modelfile:
```
PARAMETER num_ctx 32768
PARAMETER num_batch 512
```

Edit docker-compose.yml environment:
```yaml
OLLAMA_NUM_PARALLEL: "4"
```

## Monitoring

### Check Model Status

```bash
# List running models
docker exec ollama-qwen3-32b-fp8 ollama ps

# Show loaded models
docker exec ollama-qwen3-32b-fp8 ollama list
```

### GPU Utilization

```bash
# Check GPU from inside container
docker exec ollama-qwen3-32b-fp8 nvidia-smi

# Watch GPU usage (update every 2 seconds)
watch -n 2 docker exec ollama-qwen3-32b-fp8 nvidia-smi
```

### Container Metrics

```bash
# Real-time container stats
docker stats ollama-qwen3-32b-fp8

# View logs
docker compose logs -f ollama-qwen3-32b-fp8

# Check container health
docker compose ps
```

### Key Metrics to Watch

- **GPU Memory Usage:** ~55-65 GB during inference
- **GPU Utilization:** 50-90% typical (memory-bandwidth bound)
- **CPU Usage:** 5-15% (mostly for request handling)
- **Concurrent Requests:** Monitor with `ollama ps` inside container

## Troubleshooting

### Model Not Loading

**Symptom:**
```
Error: model 'qwen3-32b-fp8' not found
```

**Solution:**
```bash
# Check if base model exists
docker exec ollama-qwen3-32b-fp8 ollama list

# Manually pull and create
docker exec -it ollama-qwen3-32b-fp8 ollama pull qwen3:32b-q8_0
docker exec ollama-qwen3-32b-fp8 ollama create qwen3-32b-fp8 -f /root/Modelfile-qwen3-32b-fp8
```

### Out of Memory (OOM)

**Symptom:**
```
Error: failed to allocate memory
```

**Solution:**
```bash
# Stop other GPU services
docker compose stop vllm-qwen3-32b-fp8

# Reduce concurrent requests in docker-compose.yml
OLLAMA_NUM_PARALLEL: "4"

# Or reduce context length in Modelfile
PARAMETER num_ctx 16384

# Restart service
docker compose down ollama-qwen3-32b-fp8
docker compose up -d ollama-qwen3-32b-fp8
```

### Slow Response Times

**Symptom:**
Very slow token generation (<2 tokens/sec)

**Solution:**
```bash
# Verify model is on GPU, not CPU
docker exec ollama-qwen3-32b-fp8 ollama ps
# Should show GPU memory usage

# Check GPU utilization
docker exec ollama-qwen3-32b-fp8 nvidia-smi
# Should show high GPU usage during generation

# If offloading to CPU, increase GPU memory
OLLAMA_NUM_PARALLEL: "4"  # Reduce concurrency
```

### Model Keeps Unloading

**Symptom:**
Model unloads after requests, causing slow cold starts

**Solution:**
```bash
# Verify OLLAMA_KEEP_ALIVE is set to -1 in docker-compose.yml
OLLAMA_KEEP_ALIVE: "-1"

# Restart service
docker compose restart ollama-qwen3-32b-fp8
```

### Connection Refused

**Symptom:**
```
curl: (7) Failed to connect to localhost port 11434: Connection refused
```

**Solution:**
```bash
# Check if container is running
docker compose ps

# Check logs for startup errors
docker compose logs ollama-qwen3-32b-fp8

# Restart service
docker compose restart ollama-qwen3-32b-fp8
```

## Comparison: Ollama vs vLLM

### When to Use Ollama

✓ **Simpler API** - Native Ollama API is easier than OpenAI format
✓ **Development/Testing** - Quick iteration and experimentation
✓ **Low Concurrency** - <8 concurrent requests performs well
✓ **Model Management** - Easy to switch models with `ollama pull`
✓ **Single Requests** - Competitive latency for individual queries

### When to Use vLLM

✓ **High Throughput** - 5-10x better aggregate throughput
✓ **High Concurrency** - Handles 64+ concurrent requests efficiently
✓ **Production Workloads** - Better scaling and resource utilization
✓ **OpenAI API** - Drop-in replacement for OpenAI endpoints
✓ **Advanced Features** - PagedAttention, prefix caching, CUDA graphs

### Performance Comparison Table

| Metric | Ollama FP8 | vLLM FP8 |
|--------|------------|----------|
| **Model** | qwen3:32b-q8_0 | Qwen3-32B-FP8 |
| **Context** | 32K tokens | 32K tokens |
| **Memory** | ~35-40GB model | ~32GB model |
| **KV Cache** | ~20-25GB (8 req) | ~66GB (64 req) |
| **Concurrency** | 8 requests | 64 requests |
| **Single Request** | ~5-8 tok/s | ~6-7 tok/s |
| **Max Throughput** | ~40-80 tok/s | ~300-400 tok/s |
| **API Style** | Ollama native | OpenAI compatible |
| **Setup Complexity** | Lower | Higher |
| **Production Ready** | Development/Testing | Production |

## API Reference

### Ollama API Endpoints

- **Generate:** `POST http://localhost:11434/api/generate`
- **Chat:** `POST http://localhost:11434/api/chat`
- **List Models:** `GET http://localhost:11434/api/tags`
- **Show Model:** `POST http://localhost:11434/api/show`
- **Copy Model:** `POST http://localhost:11434/api/copy`
- **Delete Model:** `DELETE http://localhost:11434/api/delete`
- **Version:** `GET http://localhost:11434/api/version`

Full API documentation: https://docs.ollama.com/api

## References

- **Ollama Documentation:** https://docs.ollama.com
- **Ollama API Reference:** https://docs.ollama.com/api
- **Qwen3 Model Card:** https://huggingface.co/Qwen/Qwen3-32B
- **DGX Spark Hardware:** [docs/nvidia-spark.md](../nvidia-spark.md)
- **vLLM Comparison:** [docs/vllm/qwen3-32b-fp8.md](../vllm/qwen3-32b-fp8.md)

---

**Last Updated:** 2025-11-07
**Tested On:** NVIDIA DGX Spark (GB10), Ollama latest
**Model:** qwen3:32b-q8_0 (Q8_0 quantization)
