# GPU Inference Platform for NVIDIA DGX Spark

Multi-provider LLM inference platform optimized for NVIDIA DGX Spark (GB10 Grace Blackwell). Deploy production-ready models using vLLM or Ollama with unified memory architecture optimization.

**Currently Supported:**
- **Model:** Qwen3-32B-FP8 (32.8B parameters, 8-bit quantization)
- **Providers:** vLLM (high-throughput) and Ollama (development-friendly)
- **Hardware:** NVIDIA DGX Spark GB10 (128GB unified memory, 273 GB/s bandwidth)

---

## Quick Start

### 1. Prerequisites

```bash
# Verify GPU
nvidia-smi

# Create directories
sudo mkdir -p /opt/hf /opt/ollama
sudo chown -R $(id -u):$(id -g) /opt/hf /opt/ollama

# Create environment file
echo "HF_TOKEN=hf_your_token_here" > .env
```

### 2. Start Services

```bash
# vLLM (recommended for production/high-throughput)
docker compose up -d vllm-qwen3-32b-fp8

# Ollama (recommended for development/simplicity)
docker compose up -d ollama-qwen3-32b-fp8

# Both services (for comparison/testing)
docker compose up -d vllm-qwen3-32b-fp8 ollama-qwen3-32b-fp8
```

### 3. Test Inference

**vLLM (OpenAI-compatible API):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Ollama (Native API):**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3-32b-fp8",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

---

## Available Services

### vLLM: Qwen3-32B-FP8 (Port 8000)

High-performance inference server with OpenAI-compatible API.

| Feature | Value |
|---------|-------|
| **Container** | `nvcr.io/nvidia/vllm:25.09-py3` |
| **Model** | Qwen/Qwen3-32B-FP8 (official pre-quantized) |
| **Memory** | ~32 GB model + 66 GB KV cache |
| **Context** | 32,000 tokens |
| **Concurrency** | 64 concurrent requests |
| **Throughput** | ~6-7 tok/s (single), ~300-400 tok/s (batched) |
| **API** | OpenAI-compatible |
| **Use Case** | Production, high concurrency |

**Documentation:** [docs/vllm/qwen3-32b-fp8.md](docs/vllm/qwen3-32b-fp8.md)

### Ollama: Qwen3-32B-FP8 (Port 11434)

Simple, developer-friendly inference server with native API.

| Feature | Value |
|---------|-------|
| **Container** | `ollama/ollama:latest` |
| **Model** | qwen3:32b-q8_0 (Q8_0 quantization) |
| **Memory** | ~35-40 GB model + 20-25 GB KV cache |
| **Context** | 32,768 tokens |
| **Concurrency** | 8 concurrent requests |
| **Throughput** | ~5-8 tok/s (single), ~40-80 tok/s (batched) |
| **API** | Ollama native (simpler) |
| **Use Case** | Development, testing, low concurrency |

**Documentation:** [docs/ollama/qwen3-32b-fp8.md](docs/ollama/qwen3-32b-fp8.md)

---

## Provider Comparison

| Metric | vLLM | Ollama |
|--------|------|--------|
| **Setup Complexity** | Higher | Lower |
| **API Style** | OpenAI compatible | Ollama native |
| **Single Request** | ~6-7 tok/s | ~5-8 tok/s |
| **Max Throughput** | ~300-400 tok/s | ~40-80 tok/s |
| **Concurrency** | 64 requests | 8 requests |
| **Memory Usage** | ~98 GB | ~55-65 GB |
| **Production Ready** | ✓ Yes | Development/Testing |
| **Model Management** | Manual | Easy (`ollama pull`) |

**When to use vLLM:**
- Production workloads
- High concurrent load (>8 requests)
- Maximum throughput needed
- OpenAI API compatibility required

**When to use Ollama:**
- Development and testing
- Low concurrency (<8 requests)
- Simpler API preferred
- Easy model switching needed

---

## Service Management

### Starting/Stopping

```bash
# Start specific service
docker compose up -d <service-name>

# Stop specific service
docker compose stop <service-name>

# Restart service
docker compose restart <service-name>

# View logs
docker compose logs -f <service-name>

# Check status
docker compose ps
```

### Service Names

- `vllm-qwen3-32b-fp8` - vLLM FP8 service
- `ollama-qwen3-32b-fp8` - Ollama FP8 service

### First-Time Loading

**vLLM:** ~9-10 minutes (downloads ~32GB model)
```bash
docker compose logs -f vllm-qwen3-32b-fp8
# Wait for: "INFO: Application startup complete."
```

**Ollama:** ~10-15 minutes (downloads ~35-40GB model)
```bash
docker compose logs -f ollama-qwen3-32b-fp8
# Wait for model pull and creation to complete
```

---

## Hardware

**NVIDIA DGX Spark - GB10 Grace Blackwell**

- **CPU:** 20-core ARM (10 Cortex-X925 + 10 Cortex-A725)
- **GPU:** 6,144 CUDA cores, 5th Gen Tensor Cores
- **Memory:** 128 GB unified LPDDR5x (273 GB/s bandwidth)
- **Key Feature:** Coherent unified memory (no separate VRAM)

**Performance Characteristics:**
- Memory bandwidth (273 GB/s) is the primary bottleneck
- Optimized for batch processing over single-request latency
- FP8 quantization essential for optimal performance

**Full hardware documentation:** [docs/nvidia-spark.md](docs/nvidia-spark.md)

---

## Configuration

### Environment Variables

Create `.env` file in repository root:

```bash
HF_TOKEN=hf_your_huggingface_token_here
```

**Get token:** https://huggingface.co/settings/tokens (read permission required)

### Storage Directories

- `/opt/hf` - vLLM model cache (~32GB)
- `/opt/ollama` - Ollama model storage (~35-40GB)

Ensure at least 100GB free space.

### Customizing Services

Edit `docker-compose.yml` to modify:
- Port mappings
- Memory utilization
- Context length
- Concurrent sequences
- Environment variables

See provider-specific docs for detailed tuning guidance.

---

## Monitoring

### GPU Utilization

```bash
# Host-level monitoring
nvidia-smi

# vLLM container
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# Ollama container
docker exec ollama-qwen3-32b-fp8 nvidia-smi
```

### Metrics

```bash
# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# Ollama model status
docker exec ollama-qwen3-32b-fp8 ollama ps

# Container stats
docker stats <service-name>
```

### Key Metrics

- **GPU Memory:** 80-95% usage typical
- **GPU Utilization:** 50-90% (memory-bandwidth bound)
- **Throughput:** Check provider-specific metrics endpoints
- **Queue Depth:** Monitor concurrent vs queued requests

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs <service-name>

# Free up memory
docker compose down
docker system prune -f

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Out of Memory

```bash
# Stop other services
docker compose stop <other-service>

# Reduce memory settings in docker-compose.yml
# For vLLM: lower --gpu-memory-utilization
# For Ollama: reduce OLLAMA_NUM_PARALLEL
```

### Slow Performance

```bash
# Verify GPU usage
nvidia-smi

# Check if model is on GPU
docker exec <service-name> nvidia-smi

# For vLLM: increase batching (send concurrent requests)
# For Ollama: verify num_gpu=99 in Modelfile
```

### Port Conflicts

```bash
# Check what's using the port
sudo lsof -i :8000   # vLLM
sudo lsof -i :11434  # Ollama

# Change port in docker-compose.yml if needed
```

**Full troubleshooting:** See provider-specific documentation

---

## API Examples

### vLLM (OpenAI-Compatible)

**Chat Completion:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**Text Completion:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "prompt": "Write a Python function to",
    "max_tokens": 150
  }'
```

### Ollama (Native)

**Generate:**
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-32b-fp8",
  "prompt": "Explain quantum computing:",
  "stream": false
}'
```

**Chat:**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3-32b-fp8",
  "messages": [
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "stream": false
}'
```

---

## Documentation

### Provider-Specific Guides

- **vLLM Configuration:** [docs/vllm/qwen3-32b-fp8.md](docs/vllm/qwen3-32b-fp8.md)
- **Ollama Configuration:** [docs/ollama/qwen3-32b-fp8.md](docs/ollama/qwen3-32b-fp8.md)

### Hardware & Setup

- **NVIDIA DGX Spark:** [docs/nvidia-spark.md](docs/nvidia-spark.md)
- **AI Assistant Guide:** [CLAUDE.md](CLAUDE.md)

### External Resources

- **vLLM Documentation:** https://docs.vllm.ai
- **Ollama Documentation:** https://docs.ollama.com
- **Qwen3 Model Card:** https://huggingface.co/Qwen/Qwen3-32B-FP8
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

---

## Repository Structure

```
.
├── README.md                           # This file (platform overview)
├── CLAUDE.md                           # AI assistant guidance
├── docker-compose.yml                  # Service definitions
├── .env                                # Environment variables (create this)
├── docs/
│   ├── nvidia-spark.md                # Hardware specs & Docker setup
│   ├── vllm/
│   │   └── qwen3-32b-fp8.md          # vLLM-specific configuration
│   └── ollama/
│       └── qwen3-32b-fp8.md          # Ollama-specific configuration
└── models/
    └── ollama/
        └── Modelfile-qwen3-32b-fp8   # Ollama model configuration
```

---

## Performance Notes

### DGX Spark Optimization

The DGX Spark's 273 GB/s memory bandwidth (vs 900+ GB/s on datacenter GPUs) requires specific optimization:

1. **Use FP8/Q8 quantization** - Reduces memory bandwidth requirements
2. **Maximize batching** - Send concurrent requests for better aggregate throughput
3. **Enable caching** - Prefix caching reduces redundant computation
4. **Right-size context** - Only use needed context length

### Expected Performance

**vLLM Qwen3-32B-FP8:**
- Single request: ~6-7 tokens/sec
- Batched (16-32 concurrent): ~100-200 tokens/sec
- Batched (48-64 concurrent): ~300-400 tokens/sec

**Ollama Qwen3-32B-FP8:**
- Single request: ~5-8 tokens/sec
- Batched (4 concurrent): ~20-35 tokens/sec
- Batched (8 concurrent): ~40-80 tokens/sec

**Note:** Single-request latency is limited by memory bandwidth. Aggregate throughput scales with concurrency.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly on DGX Spark hardware
4. Submit a pull request

---

## License

This repository is provided as-is for use with NVIDIA DGX Spark systems. Model licenses apply:
- **Qwen3-32B-FP8:** [Tongyi Qianwen License](https://huggingface.co/Qwen/Qwen3-32B-FP8)

---

## Changelog

### 2025-11-07 - Multi-Provider Platform

- Restructure as multi-provider inference platform
- Add Ollama Qwen3-32B-FP8 service (qwen3:32b-q8_0)
- Remove BF16 variants (focus on FP8 only)
- Reorganize documentation by provider
- Remove Docker Compose profiles
- Add comprehensive comparison tables

### 2025-11-07 - FP8 as Primary

- Promote FP8 quantized model as primary recommendation
- Add detailed performance comparisons
- Update all documentation to prioritize FP8

### 2025-11-06 - Initial Release

- Docker Compose setup for DGX Spark
- vLLM service with Qwen3-32B-FP8
- Verified working configuration on GB10 hardware

---

**Built for NVIDIA DGX Spark GB10 Grace Blackwell** | Optimized for unified memory architecture
