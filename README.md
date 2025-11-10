# NVIDIA Spark Model Testing Platform

**Starter platform for testing and evaluating open source LLMs on NVIDIA DGX Spark (GB10 Grace Blackwell)**

Optimized vLLM configurations for multiple models with automated performance testing. Designed to run directly on the Spark device (`/opt/inference` recommended) with AI-assisted management via Claude Code.

---

## Quick Start

```bash
# 1. Clone to Spark device
cd /opt
git clone <repo-url> inference
cd inference

# 2. Setup environment
echo "HF_TOKEN=hf_your_token_here" > .env
sudo mkdir -p /opt/hf /opt/ollama
sudo chown -R $(id -u):$(id -g) /opt/hf /opt/ollama

# 3. Start a model (only one vLLM service at a time)
docker compose up -d vllm-qwen3-30b-a3b-fp8  # Recommended: Fastest

# 4. Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Available Models

All models use vLLM on port 8000 (run one at a time):

| Service | Model | Type | Context | Best For |
|---------|-------|------|---------|----------|
| `vllm-qwen3-30b-a3b-fp8` | Qwen3-30B-A3B-FP8 | MoE (3B active) | 32K | **Speed** (42 tok/s) ⚡ |
| `vllm-qwen3-32b-fp8` | Qwen3-32B-FP8 | Dense 32B | 32K | Baseline (6.3 tok/s) |
| `vllm-llama33-70b-fp8` | Llama 3.3 70B-FP8 | Dense 70B | 65K | Quality (2.7 tok/s) |
| `ollama-qwen3-32b-fp8` | Qwen3:32b-q8_0 | Dense 32B | 32K | vLLM comparison |

**Performance tested on DGX Spark GB10 (see below)**

### Commands

```bash
# Start model
docker compose up -d <service-name>

# View logs
docker compose logs -f <service-name>

# Stop model
docker compose stop <service-name>

# GPU status
nvidia-smi
docker exec <service-name> nvidia-smi
```

---

## Performance Testing

Two automated CLI testing tools included:

### 1. Model-vs-Model Comparison (vLLM)

```bash
# Test all vLLM models
./tools/perftest-models.js

# Test specific models
./tools/perftest-models.js --skip qwen3-32b-fp8

# Custom iterations
./tools/perftest-models.js --iterations 5
```

**Tests:** Single-request latency, long-context handling (1K/8K/16K/32K tokens)

### 2. Provider Comparison (vLLM vs Ollama)

```bash
# Compare vLLM vs Ollama for Qwen3-32B
./tools/perftest.js qwen3-32b-fp8

# Test specific provider
./tools/perftest.js qwen3-32b-fp8 vllm
./tools/perftest.js qwen3-32b-fp8 ollama
```

**Tests:** Throughput comparison between inference providers

---

## Latest Performance Results

**Test Date:** 2025-11-09 | **Hardware:** DGX Spark GB10 (128GB unified memory)

### Single-Request Latency (500 tokens output)

| Model | Architecture | TTFT | Tokens/sec | Winner |
|-------|--------------|------|------------|--------|
| **Qwen3-30B-A3B** | MoE (3B active) | 78ms | **42.02** | ⚡ **FASTEST** |
| Qwen3-32B | Dense 32B | 205ms | 6.25 | Baseline |
| Llama 3.3 70B | Dense 70B | 396ms | 2.74 | Highest quality |

### Long-Context Performance (32K tokens input)

| Model | TTFT | Total Time |
|-------|------|------------|
| **Qwen3-30B-A3B** | **881ms** | **4.1s** ⚡ |
| Qwen3-32B | 5,798ms | 23.8s |
| Llama 3.3 70B | 10,333ms | 49.0s |

**Winner:** Qwen3-30B-A3B-FP8 (MoE) delivers **6.7x faster** single-request performance than Dense 32B baseline

**Full report:** `docs/reports/model-comparison-2025-11-09T17-11-44-162Z.txt`

---

## Documentation

### Quick Reference
- **[CLAUDE.md](CLAUDE.md)** - AI assistant guide (for Claude Code)
- **[Hardware Specs](docs/nvidia-spark.md)** - DGX Spark GB10 specifications

### Model Configuration Guides
- **[Qwen3-30B-A3B-FP8](docs/vllm/qwen3-30b-a3b-fp8.md)** - MoE model (recommended)
- **[Qwen3-32B-FP8](docs/vllm/qwen3-32b-fp8.md)** - Dense baseline
- **[Llama 3.3 70B-FP8](docs/vllm/llama33-70b-fp8.md)** - Long-context specialist
- **[Ollama Qwen3-32B](docs/ollama/qwen3-32b-fp8.md)** - Provider comparison

### External Resources
- **vLLM Docs:** https://docs.vllm.ai
- **Ollama Docs:** https://docs.ollama.com
- **DGX Spark:** https://docs.nvidia.com/dgx/dgx-spark/
- **Qwen Models:** https://huggingface.co/Qwen
- **Llama Models:** https://huggingface.co/meta-llama

---

## AI-Assisted Management

This repository is optimized for [Claude Code](https://claude.ai/code) with:

- **CLAUDE.md** - Comprehensive guidance for AI assistant
- **docs/** - Technical documentation for hardware, vLLM, and per-model configs
- Enables Claude to configure, manage, and troubleshoot the Spark device

Simply open this repository in Claude Code to get intelligent assistance with model deployment, performance tuning, and troubleshooting.

---

## Hardware

**NVIDIA DGX Spark - GB10 Grace Blackwell Superchip**

- **CPU:** 20-core Arm (10x Cortex-X925 + 10x Cortex-A725)
- **GPU:** NVIDIA Blackwell (6,144 CUDA cores, 5th Gen Tensor Cores)
- **Memory:** 128 GB LPDDR5x unified (273 GB/s bandwidth)
- **Key Feature:** Coherent unified memory (no separate VRAM)

**Performance Characteristics:**
- Memory bandwidth (273 GB/s) is primary bottleneck
- FP8 quantization essential for optimal performance
- Single-request latency limited by hardware
- Batching and MoE architectures maximize throughput

**Full specs:** [docs/nvidia-spark.md](docs/nvidia-spark.md)

---

## Repository Structure

```
.
├── README.md                      # This file
├── CLAUDE.md                      # AI assistant guidance (Claude Code)
├── docker-compose.yml             # vLLM & Ollama service definitions
├── .env                           # Environment variables (create this)
├── tools/
│   ├── perftest-models.js         # Model-vs-model comparison
│   └── perftest.js                # vLLM vs Ollama comparison
├── docs/
│   ├── nvidia-spark.md           # Hardware specs & setup
│   ├── vllm/                     # vLLM model configurations
│   │   ├── qwen3-30b-a3b-fp8.md # MoE model (fastest)
│   │   ├── qwen3-32b-fp8.md     # Dense baseline
│   │   └── llama33-70b-fp8.md   # Long-context specialist
│   ├── ollama/                   # Ollama configurations
│   │   └── qwen3-32b-fp8.md     # Provider comparison
│   └── reports/                  # Performance test results
└── models/
    └── ollama/
        └── Modelfile-qwen3-32b-fp8
```

---

## Important Notes

- **One vLLM model at a time** - All use port 8000
- **First load is slow** - Model downloads: 8-15 minutes (cached locally afterward)
- **Memory bandwidth limited** - 273 GB/s vs 900+ GB/s on datacenter GPUs
- **MoE architecture wins** - Qwen3-30B-A3B significantly outperforms dense models
- **FP8 quantization required** - Essential for Spark's unified memory architecture
- **Batching increases throughput** - Single-request performance is hardware-limited

---

## License

Provided as-is for NVIDIA DGX Spark systems. Model licenses apply:
- **Qwen Models:** [Tongyi Qianwen License](https://huggingface.co/Qwen)
- **Llama Models:** [Llama 3 Community License](https://huggingface.co/meta-llama)

---

**Built for NVIDIA DGX Spark GB10** | Optimized for unified memory architecture | Powered by vLLM
