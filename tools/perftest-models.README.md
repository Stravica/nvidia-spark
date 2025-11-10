# perftest-models.js - vLLM Model Performance Testing Tool

**Purpose:** Compare inference performance across multiple vLLM models on NVIDIA DGX Spark.

**Location:** `/opt/inference/tools/perftest-models.js`

---

## Overview

This tool performs comprehensive performance testing of vLLM models to measure:
- **Single-request latency** - Time to first token (TTFT) and throughput (tokens/second)
- **Long-context handling** - Performance with varying input sizes (1K, 8K, 16K, 32K tokens)

The tool automatically manages Docker containers, runs tests with multiple iterations for statistical accuracy, and generates performance reports.

---

## Models Tested

### 1. Qwen3-32B-FP8 (Dense 32B - Baseline)
- **Service:** `vllm-qwen3-32b-fp8`
- **Model ID:** `Qwen/Qwen3-32B-FP8`
- **Max Context:** 32,000 tokens
- **Architecture:** Dense 32B parameter model
- **Use Case:** Baseline reference performance

### 2. Qwen3-30B-A3B-FP8 (MoE 30B - Efficiency)
- **Service:** `vllm-qwen3-30b-a3b-fp8`
- **Model ID:** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- **Max Context:** 32,768 tokens
- **Architecture:** Mixture of Experts (MoE) - 30B total, 3B active per token
- **Use Case:** High-efficiency inference, best throughput

### 3. Llama 3.3 70B-FP8 (Dense 70B - Quality)
- **Service:** `vllm-llama33-70b-fp8`
- **Model ID:** `nvidia/Llama-3.3-70B-Instruct-FP8`
- **Max Context:** 65,536 tokens
- **Architecture:** Dense 70B parameter model
- **Use Case:** Highest quality outputs, long-context analysis

---

## Test Configuration

### Global Settings

```javascript
VLLM_PORT: 8000                    // All vLLM models use this port
HEALTH_CHECK_TIMEOUT: 1800000      // 30 minutes (for first-time model downloads)
HEALTH_CHECK_INTERVAL: 3000        // 3 seconds between health checks
WARMUP_TOKENS: 50                  // Pre-warm model before testing
LATENCY_TEST_TOKENS: 500           // Output tokens for latency test
ITERATIONS: 3                      // Test runs per scenario (configurable via CLI)
```

### Test Scenarios

#### 1. Single-Request Latency Test

**Purpose:** Measure throughput and time-to-first-token for single requests.

**Configuration:**
- **Input:** Minimal prompt (~50 tokens): "Write a detailed technical explanation of how neural network backpropagation works..."
- **Output:** 500 tokens (fixed)
- **Temperature:** 0.7
- **Iterations:** 3 (default)

**Metrics Measured:**
- **TTFT (Time to First Token):** Latency in milliseconds
- **TPS (Tokens Per Second):** Generation throughput
- **Tokens In/Out:** Input and output token counts
- **Total Time:** End-to-end request duration

#### 2. Long-Context Handling Tests

**Purpose:** Measure input processing performance with varying context sizes.

**Configuration:**
- **Input Sizes:** 1K, 8K, 16K, 32K tokens (generated repetitive text)
- **Output:** 100 tokens (fixed)
- **Temperature:** 0.7
- **Iterations:** 3 per context size

**Metrics Measured:**
- **TTFT (Time to First Token):** Context processing latency
- **Tokens In:** Actual input token count
- **Total Time:** End-to-end request duration

**Context Generation:** Uses `generateContextPrompt()` to create repetitive text at ~4 characters per token.

---

## Understanding Test Results: Why Latency Tests Take Longer

### The Counterintuitive Result

You may notice that the **Single-Request Latency test takes 3-4x LONGER** than even the **Very Long Context (32K tokens) test**:

**Example Results:**
- Latency test (50 input, 500 output): 80.7s total
- 32K context test (18K input, 100 output): 24.3s total

**Why?** The latency test generates **5x more output tokens** (500 vs 100).

### The Root Cause: Output Generation is the Bottleneck

On the NVIDIA DGX Spark (and most inference hardware), **output generation is significantly slower than input processing**:

#### Input Processing (Prefill Phase) - FAST ⚡
- **Parallel processing:** All input tokens processed simultaneously
- **32K tokens can be processed in ~10-30 seconds** (see TTFT times in results)
- Limited by memory bandwidth but highly parallelizable

#### Output Generation (Decode Phase) - SLOW 🐌
- **Sequential processing:** One token generated at a time
- **Memory bandwidth bottleneck:** DGX Spark's 273 GB/s bandwidth is the limiting factor
- **500 tokens takes 3-4x longer than 100 tokens**

### Performance Breakdown

| Model | Latency (500 out) | 32K Context (100 out) | Extra Time for 400 Tokens |
|-------|-------------------|----------------------|---------------------------|
| Qwen3-32B | 80.7s | 24.3s | +56.4s (400 tokens) |
| Qwen3-30B-A3B | 12.2s | 4.2s | +8.0s (400 tokens) |
| Llama 3.3 70B | 178.9s | 48.7s | +130.2s (400 tokens) |

### Why This Test Design Makes Sense

- **Latency test (500 output):** Measures generation throughput accurately - need many tokens to calculate reliable tokens/second
- **Context tests (100 output):** Measures input processing speed (TTFT) - output is kept short to isolate input processing performance

**Conclusion:** A test with 50 input tokens takes longer than a test with 32,000 input tokens because it generates 5x more output. This is **working as designed**.

---

## Usage

### Basic Usage

```bash
# Test all three models (default 3 iterations each)
./perftest-models.js

# Run with 5 iterations per test
./perftest-models.js --iterations 5

# Skip specific models
./perftest-models.js --skip qwen3-32b-fp8
./perftest-models.js --skip llama33-70b-fp8

# Skip multiple models
./perftest-models.js --skip qwen3-32b-fp8 --skip qwen3-30b-a3b-fp8

# Show help
./perftest-models.js --help
```

### CLI Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--iterations <n>` | Number of test runs per scenario | 3 | `--iterations 5` |
| `--skip <model>` | Skip testing a specific model | None | `--skip qwen3-32b-fp8` |
| `--help` | Show help message | - | `--help` |

**Available model names for `--skip`:**
- `qwen3-32b-fp8`
- `qwen3-30b-a3b-fp8`
- `llama33-70b-fp8`

---

## Output and Reports

### Terminal Output

During test execution, the tool displays:
- Service discovery and validation
- Real-time test progress with status indicators
- Test results in formatted ASCII tables
- Winner identification for each metric (🏆)

### Report Files

Reports are automatically saved to: `docs/reports/model-comparison-{timestamp}.txt`

**Filename format:** `model-comparison-2025-11-09T14-30-45.txt`

**Report contents:**
- Complete test output (stdout capture)
- Latency test results table
- Long-context test results tables
- Performance comparison across all models

---

## Test Execution Flow

1. **Service Discovery:** Validates all model services exist in docker-compose.yml
2. **Container Management:** Stops all running inference containers
3. **Per-Model Testing:**
   - Start model container
   - Wait for health check (up to 30 minutes for first load)
   - Pre-warm model with small request
   - Run latency tests (3 iterations)
   - Run context tests (3 iterations per context size)
   - Stop model container
4. **Results Analysis:** Calculate statistics (mean, stddev, min, max)
5. **Report Generation:** Display tables and save report file

---

## Performance Metrics Explained

### TTFT (Time to First Token)
- **Definition:** Time from request submission to receiving the first token
- **Measures:** Input processing speed + model overhead
- **Unit:** Milliseconds (ms)
- **Lower is better** ✓

### TPS (Tokens Per Second)
- **Definition:** Output generation throughput
- **Measures:** How fast the model generates tokens
- **Unit:** Tokens per second (tok/s)
- **Higher is better** ✓
- **Note:** Only measured in latency test (requires many output tokens)

### Tokens In
- **Definition:** Number of input tokens in the prompt
- **Measured:** Via vLLM's usage statistics API
- **Note:** Actual tokenized count, not estimated

### Tokens Out
- **Definition:** Number of output tokens generated
- **Measured:** Via vLLM's usage statistics API
- **Fixed:** 500 (latency test), 100 (context tests)

### Total Time
- **Definition:** End-to-end request duration
- **Includes:** TTFT + output generation time
- **Unit:** Seconds (s)
- **Lower is better** ✓

---

## Hardware Environment

### NVIDIA DGX Spark GB10 (Grace Blackwell Superchip)

**CPU:**
- 20-core Arm (10x Cortex-X925, 10x Cortex-A725)

**GPU:**
- NVIDIA Blackwell
- 6,144 CUDA cores
- 5th Gen Tensor Cores

**Memory:**
- 128 GB LPDDR5x unified memory
- 273 GB/s bandwidth
- **Coherent unified memory** (no separate VRAM)

**Key Characteristic:**
- Memory bandwidth (273 GB/s) is the primary bottleneck
- FP8 quantization essential for optimal performance
- Batching and caching critical for throughput

---

## Troubleshooting

### Models Not Found
- **Error:** "Service {name} not found in docker-compose.yml"
- **Solution:** Verify service names in `docker-compose.yml` match MODELS configuration

### Health Check Timeout
- **Error:** "{model} health check timeout after 1800s"
- **Cause:** First-time model download taking longer than 30 minutes
- **Solution:** Model may be very large (70B+), check container logs: `docker compose logs -f {service}`

### Port Conflicts
- **Error:** "Connection refused" on port 8000
- **Cause:** Another service already using port 8000
- **Solution:** Stop conflicting containers: `docker compose stop`

### Out of Memory
- **Error:** Container crashes or OOM errors
- **Cause:** Insufficient GPU memory for model
- **Solution:** Only one vLLM model can run at a time; ensure others are stopped

---

## Statistical Analysis

Results are aggregated across iterations to provide:
- **Mean:** Average value
- **Standard Deviation (±):** Variability measure
- **Min/Max:** Range of observed values

**Example:** `TTFT: 190 ± 26 ms` means:
- Average: 190 ms
- Standard deviation: 26 ms
- Indicates consistent performance (low variability)

---

## Integration with CI/CD

### Capture Full Output

```bash
# Save complete test output to file
./perftest-models.js | tee test-output.txt

# Run in headless environment
./perftest-models.js --iterations 1 > results.txt 2>&1
```

### Parse Results

Reports are saved to `docs/reports/` automatically. Parse these files for:
- Performance regression detection
- Trend analysis over time
- Comparison between hardware configurations

---

## Related Documentation

- **CLAUDE.md** - Main repository guide with model overview
- **docs/vllm/qwen3-32b-fp8.md** - Qwen3-32B-FP8 configuration details
- **docs/vllm/qwen3-30b-a3b-fp8.md** - Qwen3-30B-A3B-FP8 configuration details
- **docs/vllm/llama33-70b-fp8.md** - Llama 3.3 70B-FP8 configuration details
- **docs/nvidia-spark.md** - Hardware specifications and Docker setup

---

## Version History

- **2025-11-09:** Initial version with automatic report generation
  - Added stream_options for usage statistics
  - Increased health check timeout to 30 minutes
  - Fixed context test token counting

---

**Last Updated:** 2025-11-09
**Tool Version:** 1.0
**Maintainer:** Stravica
