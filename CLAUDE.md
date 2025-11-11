# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Purpose:** Multi-provider GPU inference platform for NVIDIA DGX Spark (GB10 Grace Blackwell)

**Current Focus:** Multiple LLM model deployments using vLLM and Ollama providers
- vLLM: Qwen3-8B-FP8, Llama-3.1-8B-FP8, Mistral-NeMo-12B-FP8, Qwen3-32B-FP8, Qwen3-30B-A3B-FP8, Llama 3.3 70B-FP8
- Ollama: Qwen3-32B-FP8

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

**Note:** All vLLM services use port 8000 - only run one vLLM model at a time.

### vLLM Models (Port 8000)

#### 1. vLLM: Qwen3-8B-FP8

High-performance inference with OpenAI-compatible API. Compact 8B model (fastest).

**Configuration:**
- **Model:** `Qwen/Qwen3-8B-FP8` (official pre-quantized)
- **Memory:** ~8 GB model, ~80-85 GB KV cache
- **Context:** 32,768 tokens
- **Concurrency:** 64+ concurrent requests
- **Performance:** ~9-10 tok/s (single), ~450-500 tok/s (batched)
- **Best For:** Maximum speed and throughput

**Documentation:** `docs/vllm/qwen3-8b-fp8.md`

#### 2. vLLM: Llama-3.1-8B-Instruct-FP8

High-performance inference with OpenAI-compatible API. NVIDIA-optimized 8B model.

**Configuration:**
- **Model:** `nvidia/Llama-3.1-8B-Instruct-FP8` (NVIDIA pre-quantized)
- **Memory:** ~8 GB model, ~80-85 GB KV cache
- **Context:** 32,768 tokens (128K native)
- **Concurrency:** 64+ concurrent requests
- **Performance:** ~9-10 tok/s (single), ~450-500 tok/s (batched)
- **Best For:** Fast inference, instruction following, NVIDIA optimization

**Documentation:** `docs/vllm/llama31-8b-fp8.md`

#### 3. vLLM: Mistral-NeMo-12B-Instruct-FP8

High-performance inference with OpenAI-compatible API. 12B model with long-context (128K native).

**Configuration:**
- **Model:** `neuralmagic/Mistral-Nemo-Instruct-2407-FP8` (vLLM-optimized)
- **Memory:** ~12 GB model, ~75-80 GB KV cache
- **Context:** 65,536 tokens (128K native)
- **Concurrency:** 64 concurrent requests
- **Performance:** ~8-9 tok/s (single), ~400-450 tok/s (batched)
- **Best For:** Long-context analysis, document processing, balanced quality/speed

**Documentation:** `docs/vllm/mistral-nemo-12b-fp8.md`

#### 4. vLLM: Qwen3-32B-FP8

High-performance inference with OpenAI-compatible API. Dense 32B model (baseline).

**Configuration:**
- **Model:** `Qwen/Qwen3-32B-FP8` (official pre-quantized)
- **Memory:** ~32 GB model, ~66 GB KV cache
- **Context:** 32,000 tokens
- **Concurrency:** 64 concurrent requests
- **Performance:** ~6-7 tok/s (single), ~300-400 tok/s (batched)
- **Best For:** Maximum throughput with batching

**Documentation:** `docs/vllm/qwen3-32b-fp8.md`

#### 5. vLLM: Qwen3-30B-A3B-FP8

High-performance inference with OpenAI-compatible API. MoE model (30B total, 3B active).

**Configuration:**
- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (official pre-quantized)
- **Memory:** ~30 GB model, ~55-70 GB KV cache
- **Context:** 32,768 tokens
- **Concurrency:** 64 concurrent requests
- **Performance:** ~7-9 tok/s (single), ~200-350 tok/s (batched)
- **Best For:** Efficiency, mixed workloads, better single-request latency

**Documentation:** `docs/vllm/qwen3-30b-a3b-fp8.md`

#### 6. vLLM: Llama 3.3 70B-FP8

High-performance inference with OpenAI-compatible API. Dense 70B model (highest quality).

**Configuration:**
- **Model:** `nvidia/Llama-3.3-70B-Instruct-FP8` (NVIDIA pre-quantized)
- **Memory:** ~35 GB model, ~40-60 GB KV cache
- **Context:** 65,536 tokens (128K max)
- **Concurrency:** 32 concurrent requests
- **Performance:** ~5-7 tok/s (single), ~80-150 tok/s (batched)
- **Best For:** Highest quality, long-context analysis, complex reasoning

**Documentation:** `docs/vllm/llama33-70b-fp8.md`

### Ollama Models (Port 11434)

#### 7. Ollama: Qwen3-32B-FP8

Simple inference with native Ollama API.

**Configuration:**
- **Model:** `qwen3:32b-q8_0` (Q8_0 quantization)
- **Memory:** ~35-40 GB model, ~20-25 GB KV cache
- **Context:** 32,768 tokens
- **Concurrency:** 8 concurrent requests
- **Performance:** ~5-8 tok/s (single), ~40-80 tok/s (batched)
- **Best For:** Development, testing, simple workflows

**Documentation:** `docs/ollama/qwen3-32b-fp8.md`

---

## Common Commands

### Service Management

```bash
# Start vLLM services (only run ONE at a time - all use port 8000)
docker compose up -d vllm-qwen3-8b-fp8             # Dense 8B (fastest)
docker compose up -d vllm-llama31-8b-fp8           # Llama-3.1 8B (NVIDIA-optimized)
docker compose up -d vllm-mistral-nemo-12b-fp8     # Mistral-NeMo 12B (long-context)
docker compose up -d vllm-qwen3-32b-fp8            # Dense 32B (baseline)
docker compose up -d vllm-qwen3-30b-a3b-fp8        # MoE 30B (efficient)
docker compose up -d vllm-llama33-70b-fp8          # Dense 70B (high quality)

# Start Ollama service
docker compose up -d ollama-qwen3-32b-fp8          # Ollama (port 11434)

# View logs
docker compose logs -f <service-name>

# Stop services
docker compose stop <service-name>

# Stop all vLLM services
docker compose stop vllm-qwen3-8b-fp8 vllm-llama31-8b-fp8 vllm-mistral-nemo-12b-fp8 vllm-qwen3-32b-fp8 vllm-qwen3-30b-a3b-fp8 vllm-llama33-70b-fp8

# Restart
docker compose restart <service-name>

# Check status
docker compose ps
```

### Monitoring

```bash
# GPU status (host)
nvidia-smi

# GPU status (vLLM containers)
docker exec vllm-qwen3-8b-fp8 nvidia-smi
docker exec vllm-llama31-8b-fp8 nvidia-smi
docker exec vllm-mistral-nemo-12b-fp8 nvidia-smi
docker exec vllm-qwen3-32b-fp8 nvidia-smi
docker exec vllm-qwen3-30b-a3b-fp8 nvidia-smi
docker exec vllm-llama33-70b-fp8 nvidia-smi

# GPU status (Ollama container)
docker exec ollama-qwen3-32b-fp8 nvidia-smi

# vLLM metrics (whichever service is running on port 8000)
curl http://localhost:8000/metrics

# Ollama model status
docker exec ollama-qwen3-32b-fp8 ollama ps

# Container stats
docker stats <service-name>
```

### Testing

```bash
# Test vLLM models (OpenAI-compatible, port 8000)
# Qwen3-8B-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-8B-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Llama-3.1-8B-Instruct-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Mistral-NeMo-12B-Instruct-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "neuralmagic/Mistral-Nemo-Instruct-2407-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Qwen3-32B-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Qwen3-30B-A3B-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Llama 3.3 70B-FP8
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.3-70B-Instruct-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test Ollama (native, port 11434)
curl http://localhost:11434/api/chat \
  -d '{"model": "qwen3-32b-fp8", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Performance Testing

```bash
# Model-vs-model comparison (all vLLM models)
./tools/perftest-models.js

# Test specific models only
./tools/perftest-models.js --skip qwen3-32b-fp8
./tools/perftest-models.js --skip llama33-70b-fp8

# Custom iterations
./tools/perftest-models.js --iterations 5

# Provider comparison (Ollama vs vLLM for Qwen3-32B)
./tools/perftest.js qwen3-32b-fp8
./tools/perftest.js qwen3-32b-fp8 ollama    # Ollama only
./tools/perftest.js qwen3-32b-fp8 vllm      # vLLM only
```

---

## Service Naming Pattern

All services follow: `{provider}-{model}-{quantization}`

**Examples:**
- `vllm-qwen3-8b-fp8` - vLLM with Qwen3-8B FP8
- `vllm-llama31-8b-fp8` - vLLM with Llama-3.1-8B FP8
- `vllm-mistral-nemo-12b-fp8` - vLLM with Mistral-NeMo-12B FP8
- `vllm-qwen3-32b-fp8` - vLLM with Qwen3-32B FP8
- `vllm-qwen3-30b-a3b-fp8` - vLLM with Qwen3-30B-A3B FP8
- `vllm-llama33-70b-fp8` - vLLM with Llama 3.3 70B FP8
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

### vLLM Models

- **vLLM Qwen3-8B-FP8:** `docs/vllm/qwen3-8b-fp8.md`
  - Compact 8B model (fastest)
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **vLLM Llama-3.1-8B-FP8:** `docs/vllm/llama31-8b-fp8.md`
  - NVIDIA-optimized 8B model
  - Instruction following and tool use
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **vLLM Mistral-NeMo-12B-FP8:** `docs/vllm/mistral-nemo-12b-fp8.md`
  - 12B model with long-context (128K native)
  - What is Mistral-NeMo and good use cases
  - Long-context configuration (65K configured)
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **vLLM Qwen3-32B-FP8:** `docs/vllm/qwen3-32b-fp8.md`
  - Dense 32B model (baseline)
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **vLLM Qwen3-30B-A3B-FP8:** `docs/vllm/qwen3-30b-a3b-fp8.md`
  - MoE 30B model (3B active per token)
  - What is MoE and good use cases
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

- **vLLM Llama 3.3 70B-FP8:** `docs/vllm/llama33-70b-fp8.md`
  - Dense 70B model (highest quality)
  - What is Llama 3.3 and good use cases
  - Long-context configuration (128K max)
  - Service configuration and parameter tuning
  - OpenAI API examples
  - Performance optimization
  - Troubleshooting

### Ollama Models

- **Ollama Qwen3-32B-FP8:** `docs/ollama/qwen3-32b-fp8.md`
  - Service configuration
  - Modelfile parameters
  - Native API examples
  - Performance tuning
  - Troubleshooting

### Infrastructure

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

## Model Comparison (vLLM)

| Feature | Qwen3-8B | Llama-3.1-8B | Mistral-NeMo-12B | Qwen3-32B | Qwen3-30B-A3B | Llama-3.3-70B |
|---------|----------|--------------|------------------|-----------|---------------|---------------|
| **Architecture** | Dense 8B | Dense 8B | Dense 12B | Dense 32B | MoE 30B (3B) | Dense 70B |
| **Model Memory** | ~8 GB | ~8 GB | ~12 GB | ~32 GB | ~30 GB | ~35 GB |
| **KV Cache** | ~80-85 GB | ~80-85 GB | ~75-80 GB | ~66 GB | ~55-70 GB | ~40-60 GB |
| **Total Memory** | ~88-93 GB | ~88-93 GB | ~87-92 GB | ~98 GB | ~85-100 GB | ~75-95 GB |
| **Context Length** | 32K | 32K (128K native) | 65K (128K native) | 32K | 32K | 65K (128K) |
| **Max Concurrency** | 64+ | 64+ | 64 | 64 | 64 | 32 |
| **Single Request TPS** | ~9-10 | ~9-10 | ~8-9 | ~6-7 | ~7-9 | ~5-7 |
| **Batched TPS** | ~450-500 | ~450-500 | ~400-450 | ~300-400 | ~200-350 | ~80-150 |
| **Best For** | Max speed | NVIDIA-optimized | Long-context | High throughput | Efficiency | Max quality |

**Model Selection Guide:**
- **Qwen3-8B-FP8:** Choose for maximum speed, highest batched throughput, fastest single-request performance
- **Llama-3.1-8B-FP8:** Choose for NVIDIA-optimized performance, instruction following, proven Meta architecture
- **Mistral-NeMo-12B-FP8:** Choose for long-context tasks (65K), document analysis, balanced quality/speed
- **Qwen3-32B-FP8:** Choose for higher quality than 8B models, proven baseline performance
- **Qwen3-30B-A3B-FP8:** Choose for better single-request latency (MoE), more memory headroom, mixed workloads
- **Llama 3.3 70B-FP8:** Choose for highest quality, complex reasoning, long-context analysis

---

## Performance Expectations

### vLLM: Qwen3-8B-FP8 (Compact 8B, Fastest)

- **Single Request:** ~9-10 tokens/sec (fastest in platform)
- **Batched (16-32 concurrent):** ~150-250 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~450-500 tokens/sec aggregate
- **Context:** Full 32K tokens supported
- **KV Cache:** ~80-85 GB, ~350,000+ tokens capacity

### vLLM: Llama-3.1-8B-FP8 (NVIDIA-Optimized 8B)

- **Single Request:** ~9-10 tokens/sec
- **Batched (16-32 concurrent):** ~150-250 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~450-500 tokens/sec aggregate
- **Context:** Full 32K tokens supported (128K native)
- **KV Cache:** ~80-85 GB, ~350,000+ tokens capacity

### vLLM: Mistral-NeMo-12B-FP8 (Long-Context 12B)

- **Single Request:** ~8-9 tokens/sec
- **Batched (16-32 concurrent):** ~150-230 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~400-450 tokens/sec aggregate
- **Context:** 65K tokens configured (128K native max)
- **KV Cache:** ~75-80 GB, ~280,000+ tokens capacity at 65K context

### vLLM: Qwen3-32B-FP8 (Baseline Dense 32B)

- **Single Request:** ~6-7 tokens/sec
- **Batched (16-32 concurrent):** ~100-200 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~300-400 tokens/sec aggregate
- **Context:** Full 32K tokens supported
- **KV Cache:** 66 GB, 271,360 tokens capacity

### vLLM: Qwen3-30B-A3B-FP8 (MoE 30B)

- **Single Request:** ~7-9 tokens/sec (better than dense due to MoE)
- **Batched (16-32 concurrent):** ~100-200 tokens/sec aggregate
- **Batched (48-64 concurrent):** ~200-350 tokens/sec aggregate
- **Context:** Full 32K tokens supported
- **KV Cache:** ~55-70 GB (configurable via gpu_memory_utilization)

### vLLM: Llama 3.3 70B-FP8 (Dense 70B, Long-Context)

- **Single Request:** ~5-7 tokens/sec
- **Batched (16-32 concurrent):** ~80-150 tokens/sec aggregate
- **Context:** 65K tokens configured (128K native max)
- **KV Cache:** ~40-60 GB (varies with context length)

### Ollama: Qwen3-32B-FP8

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
│   ├── nvidia-spark.md               # Hardware & Docker setup
│   ├── vllm/
│   │   ├── qwen3-8b-fp8.md           # vLLM Qwen3-8B configuration
│   │   ├── llama31-8b-fp8.md         # vLLM Llama-3.1-8B configuration
│   │   ├── mistral-nemo-12b-fp8.md   # vLLM Mistral-NeMo-12B configuration
│   │   ├── qwen3-32b-fp8.md          # vLLM Qwen3-32B configuration
│   │   ├── qwen3-30b-a3b-fp8.md      # vLLM Qwen3-30B-A3B configuration
│   │   └── llama33-70b-fp8.md        # vLLM Llama 3.3 70B configuration
│   └── ollama/
│       └── qwen3-32b-fp8.md          # Ollama configuration
├── tools/
│   ├── perftest-models.js        # Model-vs-model comparison
│   └── perftest.js               # Provider comparison (Ollama vs vLLM)
└── models/
    └── ollama/
        └── Modelfile-qwen3-32b-fp8  # Ollama model config
```

**Storage Locations:**
- `/opt/hf` - vLLM model cache (~8GB Qwen3-8B, ~8GB Llama-3.1-8B, ~12GB Mistral-NeMo-12B, ~32GB Qwen3-32B, ~30GB Qwen3-30B-A3B, ~35GB Llama 3.3 70B)
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
- **Only one vLLM model at a time** - All vLLM services use port 8000, cannot run simultaneously
- **vLLM and Ollama can't run simultaneously** - Both need significant GPU memory
- **First-time loading is slow** - vLLM models: ~8-12 min, Ollama: ~10-15 min (depending on model size)
- **Subsequent starts are faster** - Models cached locally at `/opt/hf` (vLLM) and `/opt/ollama` (Ollama)
- **Memory bandwidth is the bottleneck** - Not compute capacity (273 GB/s limitation)
- **Batching is essential** - Single-request performance is hardware-limited
- **Model selection matters** - Choose Qwen3-8B/Llama-3.1-8B for speed, Mistral-NeMo-12B for long-context, Qwen3-30B-A3B for efficiency, Llama 3.3 70B for quality

---

## References

### Documentation

- **README.md** - Platform overview and quick start
- **docs/nvidia-spark.md** - Hardware specifications and Docker setup

### Model Documentation

- **docs/vllm/qwen3-8b-fp8.md** - vLLM Qwen3-8B-FP8 configuration and tuning
- **docs/vllm/llama31-8b-fp8.md** - vLLM Llama-3.1-8B-FP8 configuration and tuning
- **docs/vllm/mistral-nemo-12b-fp8.md** - vLLM Mistral-NeMo-12B-FP8 configuration and tuning
- **docs/vllm/qwen3-32b-fp8.md** - vLLM Qwen3-32B-FP8 configuration and tuning
- **docs/vllm/qwen3-30b-a3b-fp8.md** - vLLM Qwen3-30B-A3B-FP8 configuration and tuning
- **docs/vllm/llama33-70b-fp8.md** - vLLM Llama 3.3 70B-FP8 configuration and tuning
- **docs/ollama/qwen3-32b-fp8.md** - Ollama Qwen3-32B-FP8 configuration and tuning

### Performance Testing

- **tools/perftest-models.js** - Model-vs-model comparison (vLLM models)
- **tools/perftest.js** - Provider comparison (Ollama vs vLLM)

### External Resources

- **vLLM Documentation:** https://docs.vllm.ai
- **Ollama Documentation:** https://docs.ollama.com
- **DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/
- **Qwen Models:** https://huggingface.co/Qwen
- **Llama Models:** https://huggingface.co/meta-llama

---

**Last Updated:** 2025-11-10
**Repository Purpose:** Multi-provider inference platform for DGX Spark
**Current Models:** Qwen3-8B-FP8, Llama-3.1-8B-FP8, Mistral-NeMo-12B-FP8, Qwen3-32B-FP8, Qwen3-30B-A3B-FP8, Llama 3.3 70B-FP8 (vLLM); Qwen3-32B-FP8 (Ollama)
