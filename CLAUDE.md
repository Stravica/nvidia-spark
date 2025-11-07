# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Purpose:** Multi-provider GPU inference platform for NVIDIA DGX Spark (GB10 Grace Blackwell)

**Current Focus:** Qwen3-32B-FP8 model deployment using vLLM and Ollama providers

**Hardware:** NVIDIA DGX Spark with 128GB unified memory, 273 GB/s bandwidth

---

## Hardware

**NVIDIA DGX Spark - GB10 Grace Blackwell Superchip**

**Key Specifications:**
- **Hostname:** spark.thefootonline.local
- **CPU:** 20-core Arm (10 Cortex-X925 + 10 Cortex-A725)
- **GPU:** NVIDIA Blackwell (6,144 CUDA cores, 5th Gen Tensor Cores)
- **Memory:** 128 GB LPDDR5x unified (273 GB/s bandwidth)
- **Key Feature:** Coherent unified memory - no separate VRAM

**Performance Characteristics:**
- **Bottleneck:** Memory bandwidth (273 GB/s) is the primary limiting factor
- **Optimization:** FP8 quantization + batching + caching essential
- **Best Use Case:** Prototyping, experimentation, model serving with batching

**Full hardware documentation:** `docs/nvidia-spark.md`

---

## Available Services

### 1. vLLM: Qwen3-32B-FP8 (Port 8000)

High-performance inference with OpenAI-compatible API.

**Configuration:**
- **Model:** `Qwen/Qwen3-32B-FP8` (official pre-quantized)
- **Memory:** ~32 GB model, ~66 GB KV cache
- **Context:** 32,000 tokens
- **Concurrency:** 64 concurrent requests
- **Performance:** ~6-7 tok/s (single), ~300-400 tok/s (batched)

**Documentation:** `docs/vllm/qwen3-32b-fp8.md`

### 2. Ollama: Qwen3-32B-FP8 (Port 11434)

Simple inference with native Ollama API.

**Configuration:**
- **Model:** `qwen3:32b-q8_0` (Q8_0 quantization)
- **Memory:** ~35-40 GB model, ~20-25 GB KV cache
- **Context:** 32,768 tokens
- **Concurrency:** 8 concurrent requests
- **Performance:** ~5-8 tok/s (single), ~40-80 tok/s (batched)

**Documentation:** `docs/ollama/qwen3-32b-fp8.md`

---

## Common Commands

### Service Management

```bash
# Start services
docker compose up -d vllm-qwen3-32b-fp8       # vLLM
docker compose up -d ollama-qwen3-32b-fp8     # Ollama

# View logs
docker compose logs -f <service-name>

# Stop services
docker compose stop <service-name>

# Restart
docker compose restart <service-name>

# Check status
docker compose ps
```

### Monitoring

```bash
# GPU status (host)
nvidia-smi

# GPU status (vLLM container)
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# GPU status (Ollama container)
docker exec ollama-qwen3-32b-fp8 nvidia-smi

# vLLM metrics
curl http://localhost:8000/metrics

# Ollama model status
docker exec ollama-qwen3-32b-fp8 ollama ps

# Container stats
docker stats <service-name>
```

### Testing

```bash
# Test vLLM (OpenAI-compatible)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test Ollama (native)
curl http://localhost:11434/api/chat \
  -d '{"model": "qwen3-32b-fp8", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Service Naming Pattern

All services follow: `{provider}-{model}-{quantization}`

**Examples:**
- `vllm-qwen3-32b-fp8` - vLLM with Qwen3-32B FP8
- `ollama-qwen3-32b-fp8` - Ollama with Qwen3-32B Q8_0

**Note:** Quantization indicated by service name only (no Docker Compose profiles used)

---

## Configuration Files

### Docker Compose Services

**Location:** `docker-compose.yml`

Both services configured with:
- GPU resource allocation (all GPUs)
- Environment variables
- Volume mounts for model storage
- Restart policies

### Environment Variables

**Location:** `.env` (create if not exists)

```bash
HF_TOKEN=hf_your_token_here  # Required for vLLM
```

### Ollama Modelfile

**Location:** `models/ollama/Modelfile-qwen3-32b-fp8`

Defines:
- Base model (`qwen3:32b-q8_0`)
- Context length (32,768 tokens)
- Batch size (512)
- GPU allocation (99 layers)
- System prompt

---

## Model-Specific Documentation

Detailed configuration guides for deployed models:

- **vLLM Qwen3-32B-FP8:** `docs/vllm/qwen3-32b-fp8.md`
  - Service configuration
  - Parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **Ollama Qwen3-32B-FP8:** `docs/ollama/qwen3-32b-fp8.md`
  - Service configuration
  - Modelfile parameters
  - Native API examples
  - Performance tuning
  - Troubleshooting

- **Hardware & Docker:** `docs/nvidia-spark.md`
  - DGX Spark specifications
  - Docker setup
  - GPU drivers
  - Storage configuration
  - General troubleshooting

---

## Provider Comparison

| Feature | vLLM | Ollama |
|---------|------|--------|
| **Port** | 8000 | 11434 |
| **API** | OpenAI-compatible | Native Ollama |
| **Concurrency** | 64 requests | 8 requests |
| **Throughput** | ~300-400 tok/s | ~40-80 tok/s |
| **Memory** | ~98 GB | ~55-65 GB |
| **Use Case** | Production, high-concurrency | Development, testing |

**When to use vLLM:**
- Production workloads
- High concurrent load (>8 requests)
- Maximum throughput required
- OpenAI API compatibility needed

**When to use Ollama:**
- Development and testing
- Low concurrency (<8 requests)
- Simpler API preferred
- Easy model switching needed

---

## Performance Expectations

### vLLM (Qwen3-32B-FP8)

- **Single Request:** ~6-7 tokens/sec
- **Batched (16-32 concurrent):** ~100-200 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~300-400 tokens/sec aggregate
- **Context:** Full 32K tokens supported
- **KV Cache:** 66 GB, 271,360 tokens capacity

### Ollama (Qwen3-32B-FP8)

- **Single Request:** ~5-8 tokens/sec
- **Batched (4 concurrent):** ~20-35 tokens/sec aggregate
- **Batched (8 concurrent):** ~40-80 tokens/sec aggregate
- **Context:** Full 32,768 tokens supported
- **KV Cache:** ~20-25 GB

### Performance Notes

- Memory bandwidth (273 GB/s) is the primary bottleneck
- Single-request performance limited by hardware, not software
- Aggregate throughput scales with concurrent requests
- Batching is essential for optimal performance
- FP8/Q8 quantization provides ~99% quality of full precision

---

## Optimization Strategy

For maximum performance on DGX Spark:

1. **Use FP8/Q8 quantization** - Reduces memory bandwidth requirements
2. **Maximize batching** - Send concurrent requests for better aggregate throughput
3. **Enable prefix caching** - Reduces redundant computation (vLLM)
4. **Right-size context** - Only use needed context length
5. **Monitor memory usage** - Keep GPU memory 80-95% utilized

---

## Directory Structure

```
/opt/inference/
├── docker-compose.yml              # Service definitions
├── .env                           # Environment variables (HF_TOKEN)
├── README.md                      # Platform overview
├── CLAUDE.md                      # This file (AI assistant guide)
├── docs/
│   ├── nvidia-spark.md           # Hardware & Docker setup
│   ├── vllm/
│   │   └── qwen3-32b-fp8.md     # vLLM configuration
│   └── ollama/
│       └── qwen3-32b-fp8.md     # Ollama configuration
└── models/
    └── ollama/
        └── Modelfile-qwen3-32b-fp8  # Ollama model config
```

**Storage Locations:**
- `/opt/hf` - vLLM model cache (~32GB)
- `/opt/ollama` - Ollama model storage (~35-40GB)

---

## Common Troubleshooting

### Memory Issues

```bash
# Check GPU memory
nvidia-smi

# Stop conflicting services
docker compose stop <other-service>

# Clean up Docker resources
docker system prune -f

# Restart service
docker compose down && docker compose up -d <service-name>
```

### Slow Performance

```bash
# Verify GPU usage
nvidia-smi

# Check model is on GPU
docker exec <service-name> nvidia-smi

# For vLLM: send concurrent requests for batching
# For Ollama: verify num_gpu=99 in Modelfile
```

### Service Won't Start

```bash
# Check logs
docker compose logs -f <service-name>

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check environment variables
cat .env

# Verify disk space
df -h /opt
```

**Full troubleshooting:** See provider-specific documentation in `docs/`

---

## Important Notes

- **Never substitute the 240W PSU** - DGX Spark requires specific power supply
- **No Docker Compose profiles** - Services differentiated by name only
- **Both services can't run simultaneously** - vLLM and Ollama both need significant GPU memory
- **First-time loading is slow** - vLLM: ~9-10 min, Ollama: ~10-15 min
- **Subsequent starts are faster** - Models cached locally
- **Memory bandwidth is the bottleneck** - Not compute capacity
- **Batching is essential** - Single-request performance is hardware-limited

---

## References

- **README.md** - Platform overview and quick start
- **docs/nvidia-spark.md** - Hardware specifications and Docker setup
- **docs/vllm/qwen3-32b-fp8.md** - vLLM configuration and tuning
- **docs/ollama/qwen3-32b-fp8.md** - Ollama configuration and tuning
- **vLLM Documentation:** https://docs.vllm.ai
- **Ollama Documentation:** https://docs.ollama.com
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/

---

**Last Updated:** 2025-11-07
**Repository Purpose:** Multi-provider inference platform for DGX Spark
**Current Model:** Qwen3-32B-FP8 (vLLM + Ollama)
