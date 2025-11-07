# NVIDIA DGX Spark Inference Setup

GPU-accelerated inference services for NVIDIA DGX Spark (GB10 Grace Blackwell). This repository provides Docker Compose configurations for running production-ready LLM inference with optimized settings for the DGX Spark's unified memory architecture.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Stravica/nvidia-spark.git
cd nvidia-spark

# Create environment file
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Start vLLM service with Qwen3-32B-FP8 (RECOMMENDED)
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8

# Monitor loading (takes ~9-10 minutes on first run)
docker compose logs -f vllm-qwen3-32b-fp8

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "prompt": "Explain the NVIDIA Grace Blackwell architecture:",
    "max_tokens": 100
  }'
```

## Hardware

**NVIDIA DGX Spark - GB10 Grace Blackwell Superchip**

- **CPU:** 20-core Arm (10 Cortex-X925 + 10 Cortex-A725)
- **GPU:** NVIDIA Blackwell Architecture
  - 6,144 CUDA cores
  - 5th Gen Tensor Cores
  - 4th Gen RT Cores
  - Compute Capability 12.1
- **Memory:** 128 GB LPDDR5x unified memory (273 GB/s bandwidth)
- **Key Feature:** Coherent unified memory - entire 128GB pool available to GPU without system-to-VRAM transfer overhead
- **AI Performance:** Up to 1 PFLOP sparse FP4, supports models up to 200B parameters

## Available Services

### 1. vLLM + Qwen3-32B-FP8 (Port 8000) - RECOMMENDED

High-performance inference server running Alibaba's Qwen3-32B model (32.8B parameters) with FP8 quantization.

**Features:**
- OpenAI-compatible API
- FP8 quantization (official pre-quantized model from Qwen team)
- 32K token context length (full native support)
- Up to 64 concurrent requests
- 2.4x more KV cache than BF16
- Better single-request performance than BF16 (~6-7 tokens/sec vs ~3-4)
- Minimal quality degradation (<1% on benchmarks)
- Prefix caching enabled
- Optimized for DGX Spark GB10 architecture

**Configuration:**
```yaml
Model: Qwen/Qwen3-32B-FP8
Precision: FP8 (8-bit)
Memory: ~32 GB
GPU Utilization: 90% (~115 GB)
KV Cache: 66.25 GB (271,360 tokens)
Max Sequences: 64
Context Length: 32,000 tokens
```

**Start FP8 service:**
```bash
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8

# Monitor loading
docker compose logs -f vllm-qwen3-32b-fp8

# Check health
curl http://localhost:8000/health
```

**API Endpoints:**
- Health: `http://localhost:8000/health`
- Chat: `http://localhost:8000/v1/chat/completions`
- Completions: `http://localhost:8000/v1/completions`
- Models: `http://localhost:8000/v1/models`
- Docs: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`

### 2. vLLM + Qwen3-32B (Port 8000) - Alternative BF16

BF16 full-precision version for extremely quality-sensitive workloads (requires stopping FP8 service first).

**Features:**
- Full BF16 precision (marginal quality improvement over FP8)
- 24K token context length
- Up to 48 concurrent requests
- Lower throughput than FP8
- Larger memory footprint

**Configuration:**
```yaml
Model: Qwen/Qwen3-32B
Precision: BF16 (16-bit)
Memory: ~65 GB
GPU Utilization: 85% (~109 GB)
KV Cache: ~28 GB (smaller than FP8)
Max Sequences: 48
Context Length: 24,000 tokens
```

**Start BF16 service:**
```bash
# Stop FP8 service first (both can't run simultaneously on port 8000)
docker compose down vllm-qwen3-32b-fp8

# Start BF16 service
docker compose up -d vllm-qwen3-32b

# Monitor loading
docker compose logs -f vllm-qwen3-32b
```

**When to use BF16:**
- Extremely quality-sensitive applications (marginal gain)
- Lower concurrency requirements acceptable
- Willing to trade throughput for potential quality improvement

### 3. Ollama (Port 11434)

General-purpose LLM inference server with support for multiple models.

**Features:**
- Simple model management
- 30-minute keep-alive
- Persistent model storage in `/opt/ollama`

**Usage:**
```bash
# Start Ollama
docker compose up -d ollama

# Pull a model
docker exec -it ollama ollama pull llama3.2

# Run inference
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

## Prerequisites

### 1. NVIDIA GPU Drivers & Container Toolkit

```bash
# Verify GPU is detected
nvidia-smi

# Install NVIDIA Container Toolkit (if not already installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Docker & Docker Compose

```bash
# Verify installation
docker --version
docker compose version
```

### 3. Storage & Directories

```bash
# Create model cache directories
sudo mkdir -p /opt/hf
sudo mkdir -p /opt/ollama

# Set permissions (replace 1000:1000 with your user:group)
sudo chown -R 1000:1000 /opt/hf
sudo chown -R 1000:1000 /opt/ollama

# Check available space (models require significant storage)
df -h /opt
```

### 4. Hugging Face Token

Required for downloading gated models like Qwen3-32B.

1. Create account at https://huggingface.co
2. Generate token: Settings → Access Tokens → New Token (read permission)
3. Create `.env` file:
   ```bash
   echo "HF_TOKEN=hf_your_token_here" > .env
   ```

## Deployment

### Starting Services

**Recommended: FP8 Service**
```bash
# Start Qwen3-32B-FP8 (port 8000) - RECOMMENDED
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8

# View logs
docker compose logs -f vllm-qwen3-32b-fp8

# Check status
docker compose ps

# Start Ollama (optional, port 11434)
docker compose up -d ollama
```

**Alternative: BF16 Service**
```bash
# IMPORTANT: Stop FP8 service first (both use port 8000)
docker compose down vllm-qwen3-32b-fp8

# Start Qwen3-32B in BF16 (port 8000)
docker compose up -d vllm-qwen3-32b

# Monitor loading
docker compose logs -f vllm-qwen3-32b

# Switch back to FP8 (recommended)
docker compose down vllm-qwen3-32b
docker compose --profile fp8 up -d vllm-qwen3-32b-fp8
```

### First-Time Model Loading

**FP8 Model (Qwen3-32B-FP8) - RECOMMENDED:** Expect 9-10 minutes for initialization:
1. Model Download: ~6 min (~32 GB, 7 safetensors shards)
2. Model Loading: ~3.4 min
3. torch.compile: ~17 sec
4. KV cache + CUDA graphs: ~6 sec

**BF16 Model (Qwen3-32B) - Alternative:** Expect 7-8 minutes for initialization:
1. Model Download: ~3-4 min (~65 GB, 17 safetensors shards)
2. Model Loading: ~5.8 min
3. torch.compile: ~19 sec
4. CUDA compilation: ~42 sec (first run only)
5. KV cache + CUDA graphs: ~6 sec

Watch logs for progress:
```bash
# FP8
docker compose logs -f vllm-qwen3-32b-fp8 | grep -E "(Loading|INFO|Ready|Serving)"

# BF16
docker compose logs -f vllm-qwen3-32b | grep -E "(Loading|INFO|Ready|Serving)"
```

Look for: `INFO:     Application startup complete.`

### Health Checks

```bash
# vLLM health (FP8 or BF16, depending on which is running)
curl http://localhost:8000/health

# Ollama health
curl http://localhost:11434/api/version

# GPU status (FP8)
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# GPU status (BF16)
docker exec vllm-qwen3-32b nvidia-smi
```

## Usage Examples

### vLLM (Qwen3-32B-FP8 Recommended)

**Text Completion (FP8 - Recommended):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Chat Completion (FP8 - Recommended):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum entanglement in simple terms"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Streaming Response (FP8):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "prompt": "Write a story about a robot:",
    "max_tokens": 300,
    "stream": true
  }'
```

**Using BF16 Model (when BF16 service is running):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Ollama

```bash
# List available models
docker exec ollama ollama list

# Pull a model
docker exec ollama ollama pull llama3.2:3b

# Generate text
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "What are the benefits of edge AI?"
}'
```

## Configuration

### docker-compose.yml

Edit `docker-compose.yml` to customize:

**vLLM FP8 parameters (Recommended):**
- `--max-model-len`: Context length (32000 default for FP8, max 32768)
- `--gpu-memory-utilization`: GPU memory usage (0.90 = 90%)
- `--max-num-seqs`: Max concurrent requests (64 default for FP8)
- `--max-num-batched-tokens`: Batch size for throughput (16384 default)

**vLLM BF16 parameters (Alternative):**
- `--max-model-len`: Context length (24000 default for BF16)
- `--gpu-memory-utilization`: GPU memory usage (0.85 = 85%)
- `--max-num-seqs`: Max concurrent requests (48 default for BF16)
- `--max-num-batched-tokens`: Batch size for throughput (16384 default)

**Ollama settings:**
- `OLLAMA_KEEP_ALIVE`: Model unload timeout (30m default)

### Performance Tuning

See [docs/models/qwen3-32b.md](docs/models/qwen3-32b.md) for detailed tuning guide.

**For maximum throughput (FP8 recommended):**
```yaml
--max-model-len 32000
--gpu-memory-utilization 0.90
--max-num-batched-tokens 20480
--max-num-seqs 64
```

**For minimum latency (FP8):**
```yaml
--max-model-len 16384
--gpu-memory-utilization 0.85
--max-num-batched-tokens 8192
--max-num-seqs 16
```

## Monitoring

### Metrics

```bash
# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# GPU utilization (FP8)
docker exec vllm-qwen3-32b-fp8 nvidia-smi

# GPU utilization (BF16)
docker exec vllm-qwen3-32b nvidia-smi

# Container stats (FP8)
docker stats vllm-qwen3-32b-fp8

# Container stats (BF16)
docker stats vllm-qwen3-32b

# vLLM logs with throughput (FP8)
docker compose logs vllm-qwen3-32b-fp8 | grep "throughput"
```

### Key Metrics to Watch

- **GPU Memory Usage:** Should be ~80-95% of allocated (check `nvidia-smi`)
- **KV Cache Usage:** Monitor in logs (0-100%)
- **Throughput:** Tokens/sec (varies with batch size)
- **Queue Depth:** Running/waiting requests
- **Prefix Cache Hit Rate:** Higher = better efficiency

## Troubleshooting

### Container Won't Start - Memory Error

**Symptom:**
```
ValueError: Free memory on device (XX GiB) is less than desired GPU memory utilization
```

**Solution:**
```bash
# Stop all containers and clean up
docker compose down
docker system prune -f

# Reduce memory utilization in docker-compose.yml
--gpu-memory-utilization 0.70

# Or reduce context length
--max-model-len 16384

# Restart
docker compose up -d vllm-qwen3-32b
```

### Model Download Fails

**Symptom:**
```
Failed to download model / 401 Unauthorized
```

**Solution:**
```bash
# Verify HF_TOKEN in .env
cat .env

# Get new token from https://huggingface.co/settings/tokens
echo "HF_TOKEN=hf_new_token_here" > .env

# Restart container
docker compose restart vllm-qwen3-32b
```

### Slow Inference / Low Throughput

**Causes & Solutions:**

1. **Single request latency is normal (~3-4 tokens/sec)**
   - DGX Spark is memory-bandwidth limited (273 GB/s)
   - Solution: Use batching for better aggregate throughput

2. **Not using batching**
   - Send multiple concurrent requests
   - Enable prefix caching (already enabled in config)

3. **Context length too long**
   - Reduce `--max-model-len` to free KV cache memory
   - More KV cache = more concurrent requests

### Container Logs Show Warnings

**"Compute Capability 12.1" Warning:**
```
WARNING: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1
Maximum cuda capability supported is 12.0
```
**Status:** Cosmetic warning, safely ignore. GB10 is fully supported.

**"Not enough SMs to use max_autotune_gemm mode":**
**Status:** Expected on GB10. Container uses appropriate fallback.

### Port Already in Use

**Symptom:**
```
Error: bind: address already in use
```

**Solution:**
```bash
# Check what's using the port
sudo lsof -i :8000

# Stop conflicting service or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 externally
```

## Documentation

- **[Qwen3-32B Deployment Guide](docs/models/qwen3-32b.md)** - Detailed configuration, performance tuning, troubleshooting
- **[CLAUDE.md](CLAUDE.md)** - Hardware specifications and technical details for AI assistants

## Performance Notes

### Expected Performance on DGX Spark

**FP8 Model (Qwen3-32B-FP8) - RECOMMENDED:**
- **Single Request:** ~6-7 tokens/sec generation
- **Batched (16-32 concurrent):** ~100-200 tokens/sec aggregate (estimated)
- **Batched (48-64 concurrent):** ~300-400 tokens/sec aggregate (estimated)
- **Max Throughput:** Scales with concurrent requests up to 64 sequences
- **Context Length:** Full 32K tokens supported
- **Quality:** ~99% of BF16 performance on benchmarks
- **Memory Efficiency:** 50% less model memory, 2.4x more KV cache

**BF16 Model (Qwen3-32B) - Alternative:**
- **Single Request:** ~3-4 tokens/sec generation
- **Batched (8-16 concurrent):** ~30-60 tokens/sec aggregate
- **Batched (32-48 concurrent):** ~100-180 tokens/sec aggregate
- **First Token Latency:** 200-500ms (depends on prompt length)
- **Max Throughput:** Scales with concurrent requests up to 48 sequences
- **Context Length:** 24K tokens configured (memory limited)

### Optimization Strategy

The DGX Spark's 273 GB/s memory bandwidth is the primary bottleneck (vs 900+ GB/s on datacenter GPUs). To maximize performance:

1. **Use FP8 quantization** - Pre-quantized Qwen/Qwen3-32B-FP8 model (primary recommendation)
2. **Use batching** - Send multiple requests concurrently
3. **Enable prefix caching** - Reuse common prompt prefixes (already enabled)
4. **Leverage large KV cache** - FP8's 66GB KV cache enables more concurrency
5. **Reduce context length if needed** - If you don't need 32K tokens, reduce to 16K or 20K for even more concurrency

## Model Comparison

| Feature | Qwen3-32B-FP8 (RECOMMENDED) | Qwen3-32B (BF16) |
|---------|----------------------------|------------------|
| **Port** | 8000 | 8000 |
| **Precision** | FP8 (8-bit) | BF16 (16-bit) |
| **Model Size** | ~32 GB | ~65 GB |
| **KV Cache** | 66.25 GB (271K tokens) | ~28 GB (smaller) |
| **Context Length** | 32K tokens (native) | 24K tokens (configured) |
| **Max Concurrent** | 64 requests | 48 requests |
| **GPU Memory Usage** | 90% (~115 GB) | 85% (~109 GB) |
| **Single Request** | ~6-7 tokens/sec | ~3-4 tokens/sec |
| **Loading Time** | 9-10 minutes | 7-8 minutes |
| **Model Shards** | 7 safetensors | 17 safetensors |
| **Quality** | ~99% of BF16 | 100% (baseline) |
| **Use Case** | **Best for DGX Spark** | Quality-sensitive edge cases |
| **Profile Flag** | `--profile fp8` required | Default (no flag) |
| **Recommendation** | **PRIMARY** | Alternative |

**When to use FP8 (RECOMMENDED):**
- General use (default recommendation)
- Maximum throughput needed
- High concurrent load (> 32 requests)
- Full 32K context windows
- Memory efficient operation
- Better single-request performance

**When to use BF16:**
- Extremely quality-sensitive applications (marginal gain)
- Lower concurrent load acceptable (< 32 requests)
- Willing to trade performance for potential quality improvement
- Testing/comparison purposes

## Repository Structure

```
.
├── README.md                    # This file
├── CLAUDE.md                    # Hardware specs and AI assistant instructions
├── docker-compose.yml           # Service definitions (BF16 + FP8)
├── .env                         # Environment variables (create this)
├── docs/
│   └── models/
│       └── qwen3-32b.md        # Detailed Qwen3-32B guide
└── .gitignore                   # Git ignore patterns
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Test thoroughly on DGX Spark hardware
4. Commit changes (`git commit -am 'Add feature'`)
5. Push to branch (`git push origin feature/improvement`)
6. Open a Pull Request

## License

This repository is provided as-is for use with NVIDIA DGX Spark systems. Model licenses apply:
- Qwen3-32B: [Tongyi Qianwen License](https://huggingface.co/Qwen/Qwen3-32B)
- Ollama models: Check individual model licenses

## Support & Resources

- **Issues:** https://github.com/Stravica/nvidia-spark/issues
- **vLLM Docs:** https://docs.vllm.ai
- **NVIDIA DGX Spark:** https://docs.nvidia.com/dgx/dgx-spark/
- **Qwen3 Model:** https://huggingface.co/Qwen/Qwen3-32B

## Changelog

### 2025-11-07 - FP8 as Primary Recommendation

- Promote FP8 quantized model as primary recommendation
- Update all documentation to prioritize FP8 configuration
- Add detailed performance comparisons (FP8 vs BF16)
- Clarify FP8 uses official pre-quantized model from Qwen team
- BF16 remains available as alternative for edge cases

### 2025-11-06 - Initial Release

- Docker Compose setup for DGX Spark
- vLLM service with Qwen3-32B in BF16 and FP8 variants
- Ollama service for multi-model support
- Comprehensive documentation
- Verified working configuration on GB10 hardware

---

**Built for NVIDIA DGX Spark GB10 Grace Blackwell** | Optimized for unified memory architecture
