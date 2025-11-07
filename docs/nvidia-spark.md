# NVIDIA DGX Spark - Hardware & Docker Setup

## Hardware Specifications

### NVIDIA DGX Spark - GB10 Grace Blackwell Superchip

**Hostname:** spark.thefootonline.local

#### Processor
- **CPU:** 20-core Arm (10 Cortex-X925 + 10 Cortex-A725)
- **GPU:** NVIDIA Blackwell Architecture
  - 6,144 CUDA cores
  - 5th Gen Tensor Cores
  - 4th Gen RT Cores
  - 2 Copy Engines (simultaneous data transfers)
  - Compute Capability: 12.1

#### Memory Architecture
- **Total Memory:** 128 GB LPDDR5x unified system memory
- **Memory Bandwidth:** 273 GB/s (shared between CPU and GPU)
- **Interface:** 256-bit, 4266 MHz
- **Channels:** 16 channels LPDDR5X 8533
- **Key Feature:** Coherent unified memory - no separate VRAM, entire 128GB pool available to GPU without system-to-VRAM transfer overhead

#### AI Performance
- Up to 1 PFLOP sparse FP4 tensor performance
- Up to 1,000 TOPS (trillion operations per second) inference
- Supports AI models up to 200 billion parameters (single device)
- Can cluster 2 units for up to 405 billion parameter models

#### Storage
- 1 TB or 4 TB NVMe M.2 with self-encryption

#### Networking
- 1x RJ-45 (10 GbE)
- ConnectX-7 Smart NIC
- 2x QSFP Network connectors (ConnectX-7) - 200 Gbps aggregate
- Wi-Fi 7, Bluetooth 5.4

#### Power
- TDP: 140W (GB10 SOC)
- Total Power: 240W external PSU (required - do not substitute)
- Operating temp: 0°C to 35°C (32°F to 95°F)

#### Performance Characteristics
- **Bottleneck:** Memory bandwidth (273 GB/s) is the primary limiting factor for inference
- **Thermal:** Excellent sustained performance without throttling due to external PSU design
- **Best Use Case:** Prototyping, experimentation, smaller model serving with batching

## Docker & GPU Setup

### Prerequisites

#### 1. NVIDIA GPU Drivers & Container Toolkit

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

#### 2. Docker & Docker Compose

```bash
# Verify installation
docker --version
docker compose version
```

#### 3. Storage & Directories

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

**Storage Requirements:**
- vLLM Qwen3-32B-FP8: ~32 GB
- Ollama Qwen3:32b-q8_0: ~35-40 GB
- Recommend at least 100 GB free space for models and Docker images

#### 4. Hugging Face Token (for vLLM)

Required for downloading gated models like Qwen3-32B.

1. Create account at https://huggingface.co
2. Generate token: Settings → Access Tokens → New Token (read permission)
3. Create `.env` file in repository root:
   ```bash
   echo "HF_TOKEN=hf_your_token_here" > .env
   ```

### Environment Configuration

The `.env` file stores environment variables:

```bash
# .env file
HF_TOKEN=hf_your_huggingface_token_here
```

**Do not commit this file to version control** - it's already in `.gitignore`.

## Docker Compose Services

All services are defined in `docker-compose.yml` with GPU resource reservations configured for the unified memory architecture.

### GPU Resource Configuration

Each service uses:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

This allocates all available GPU resources to the container, appropriate for the DGX Spark's unified memory model.

### Service Management

```bash
# Start specific service
docker compose up -d <service-name>

# Start all services
docker compose up -d

# View logs
docker compose logs -f <service-name>

# Check status
docker compose ps

# Stop service
docker compose stop <service-name>

# Stop all services
docker compose down

# Restart service
docker compose restart <service-name>
```

## Monitoring

### GPU Utilization

```bash
# Host-level GPU monitoring
nvidia-smi

# Watch GPU status (updates every 2 seconds)
watch -n 2 nvidia-smi

# Check GPU from inside container
docker exec <container-name> nvidia-smi
```

### Container Stats

```bash
# Monitor container resource usage
docker stats <container-name>

# View logs
docker compose logs -f <container-name>

# Check container health
docker compose ps
```

### Key Metrics

- **GPU Memory Usage:** Should be 80-95% of allocated for optimal performance
- **GPU Utilization:** 50-90% typical during inference (memory-bandwidth bound)
- **Temperature:** Monitor with `nvidia-smi`, should stay below 85°C
- **Power Draw:** Check with `nvidia-smi`, typical 100-140W under load

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

# Reduce memory utilization in service configuration
# Or reduce context length parameters

# Restart
docker compose up -d <service-name>
```

### GPU Not Detected in Container

**Symptom:**
```
RuntimeError: No CUDA GPUs are available
```

**Solution:**
```bash
# Verify GPU visible on host
nvidia-smi

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker

# Recreate containers
docker compose down
docker compose up -d
```

### Port Already in Use

**Symptom:**
```
Error: bind: address already in use
```

**Solution:**
```bash
# Check what's using the port
sudo lsof -i :<port-number>

# Stop conflicting service or change port in docker-compose.yml
ports:
  - "<new-port>:<container-port>"
```

### Disk Space Issues

**Symptom:**
```
no space left on device
```

**Solution:**
```bash
# Check disk usage
df -h

# Clean up Docker resources
docker system prune -a -f
docker volume prune -f

# Remove old/unused models
rm -rf /opt/hf/models--*/.cache
rm -rf /opt/ollama/models/*
```

### Permission Issues

**Symptom:**
```
Permission denied: '/root/.cache/huggingface'
```

**Solution:**
```bash
# Fix directory permissions
sudo chown -R 1000:1000 /opt/hf
sudo chown -R 1000:1000 /opt/ollama
sudo chmod -R 755 /opt/hf
sudo chmod -R 755 /opt/ollama
```

## Performance Optimization

### Memory Bandwidth Considerations

The DGX Spark's 273 GB/s memory bandwidth is the primary bottleneck for LLM inference (vs 900+ GB/s on datacenter GPUs like A100/H100).

**Optimization Strategies:**

1. **Use Quantized Models:** FP8/Q8 quantization reduces memory bandwidth requirements
2. **Maximize Batching:** Send multiple concurrent requests for better aggregate throughput
3. **Enable Caching:** Use prefix caching to reduce redundant computation
4. **Right-size Context:** Only use the context length you need; smaller contexts = more memory for concurrency

### Unified Memory Advantages

- No PCIe transfer overhead between CPU and GPU memory
- Entire 128GB pool available for models, KV cache, and computations
- Simplified memory management vs discrete GPU systems

### Recommended Configurations

**For Maximum Throughput:**
- Use FP8/Q8 quantized models
- High GPU memory utilization (0.85-0.90)
- Large batch sizes
- Enable prefix caching
- Maximize concurrent sequences

**For Minimum Latency:**
- Reduce context length
- Lower concurrent sequences
- Smaller batch sizes
- Prioritize single-request performance

## References

- **NVIDIA DGX Spark Docs:** https://docs.nvidia.com/dgx/dgx-spark/
- **NVIDIA Container Toolkit:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **Docker Compose Documentation:** https://docs.docker.com/compose/
- **vLLM Documentation:** https://docs.vllm.ai
- **Ollama Documentation:** https://docs.ollama.com

---

**Last Updated:** 2025-11-07
**Hardware:** NVIDIA DGX Spark GB10 Grace Blackwell
