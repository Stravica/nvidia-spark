# NVIDIA DGX Spark Inference Setup

GPU-accelerated inference services for NVIDIA DGX Spark (GB10 Grace Blackwell). This repository provides Docker Compose configurations for running production-ready LLM inference with optimized settings for the DGX Spark's unified memory architecture.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Stravica/nvidia-spark.git
cd nvidia-spark

# Create environment file
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Start vLLM service with Qwen3-32B
docker compose up -d vllm-qwen3-32b

# Monitor loading (takes ~7-8 minutes on first run)
docker compose logs -f vllm-qwen3-32b

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
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

### 1. vLLM + Qwen3-32B (Port 8000)

High-performance inference server running Alibaba's Qwen3-32B model (32.8B parameters).

**Features:**
- OpenAI-compatible API
- 24K token context length
- Prefix caching enabled
- Optimized for DGX Spark GB10 architecture
- Up to 48 concurrent requests

**Configuration:**
```yaml
Model: Qwen/Qwen3-32B
Memory: ~61 GB (BF16)
GPU Utilization: 85% (~109 GB)
KV Cache: 35.3 GB (144,624 tokens)
Max Sequences: 48
```

**API Endpoints:**
- Health: `http://localhost:8000/health`
- Chat: `http://localhost:8000/v1/chat/completions`
- Completions: `http://localhost:8000/v1/completions`
- Models: `http://localhost:8000/v1/models`
- Docs: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`

### 2. Ollama (Port 11434)

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

```bash
# Start specific service
docker compose up -d vllm-qwen3-32b
# or
docker compose up -d ollama

# Start all services
docker compose up -d

# View logs
docker compose logs -f vllm-qwen3-32b

# Check status
docker compose ps
```

### First-Time Model Loading

The first time you start vLLM, expect 7-8 minutes for initialization:

1. **Model Download** (~2-3 min, ~63 GB)
   - Downloads 17 safetensors shards from Hugging Face
   - Cached in `/opt/hf` for future runs

2. **Model Loading** (~5.8 min)
   - Loads all 17 shards into unified memory

3. **Compilation** (~1 min)
   - torch.compile optimization
   - CUDA kernel compilation (first run only)

4. **Initialization** (~6 sec)
   - KV cache allocation
   - CUDA graph capture

Watch logs for progress:
```bash
docker compose logs -f vllm-qwen3-32b | grep -E "(Loading|INFO|Ready|Serving)"
```

Look for: `INFO:     Application startup complete.`

### Health Checks

```bash
# vLLM health
curl http://localhost:8000/health

# Ollama health
curl http://localhost:11434/api/version

# GPU status
docker exec vllm-qwen3-32b nvidia-smi
```

## Usage Examples

### vLLM (Qwen3-32B)

**Text Completion:**
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

**Chat Completion:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum entanglement in simple terms"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Streaming Response:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "prompt": "Write a story about a robot:",
    "max_tokens": 300,
    "stream": true
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

**vLLM parameters:**
- `--max-model-len`: Context length (24000 default, max 32768)
- `--gpu-memory-utilization`: GPU memory usage (0.85 = 85%)
- `--max-num-seqs`: Max concurrent requests (48 default)
- `--max-num-batched-tokens`: Batch size for throughput (16384 default)

**Ollama settings:**
- `OLLAMA_KEEP_ALIVE`: Model unload timeout (30m default)

### Performance Tuning

See [docs/models/qwen3-32b.md](docs/models/qwen3-32b.md) for detailed tuning guide.

**For maximum throughput:**
```yaml
--max-model-len 20000
--gpu-memory-utilization 0.90
--max-num-batched-tokens 20480
--max-num-seqs 64
```

**For minimum latency:**
```yaml
--max-model-len 16384
--gpu-memory-utilization 0.75
--max-num-batched-tokens 8192
--max-num-seqs 16
```

## Monitoring

### Metrics

```bash
# vLLM Prometheus metrics
curl http://localhost:8000/metrics

# GPU utilization
docker exec vllm-qwen3-32b nvidia-smi

# Container stats
docker stats vllm-qwen3-32b

# vLLM logs with throughput
docker compose logs vllm-qwen3-32b | grep "throughput"
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

### Expected Performance (Qwen3-32B on DGX Spark)

- **Single Request:** ~3-4 tokens/sec generation
- **Batched (8-16 concurrent):** ~30-60 tokens/sec aggregate
- **Batched (32-48 concurrent):** ~100-180 tokens/sec aggregate
- **First Token Latency:** 200-500ms (depends on prompt length)
- **Max Throughput:** Scales with concurrent requests up to max_num_seqs

### Optimization Strategy

The DGX Spark's 273 GB/s memory bandwidth is the primary bottleneck (vs 900+ GB/s on datacenter GPUs). To maximize performance:

1. **Use batching** - Send multiple requests concurrently
2. **Enable prefix caching** - Reuse common prompt prefixes (already enabled)
3. **Reduce context length** - If you don't need 24K tokens, reduce to 16K or 20K
4. **Use FP8 quantization** - For even more concurrent requests (see docs)

## Repository Structure

```
.
├── README.md                    # This file
├── CLAUDE.md                    # Hardware specs and AI assistant instructions
├── docker-compose.yml           # Service definitions
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

### 2025-11-06 - Initial Release

- Docker Compose setup for DGX Spark
- vLLM service with Qwen3-32B (optimized config)
- Ollama service for multi-model support
- Comprehensive documentation
- Verified working configuration on GB10 hardware

---

**Built for NVIDIA DGX Spark GB10 Grace Blackwell** | Optimized for unified memory architecture
