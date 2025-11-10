# perftest-models.js - vLLM Model Performance Testing Tool

**Purpose:** Automatically discover and test all vLLM models in docker-compose.yml

**Location:** `/opt/inference/tools/perftest-models.js`

---

## Overview

This tool performs comprehensive performance testing of vLLM models to measure:
- **Single-request latency** - Time to first token (TTFT) and throughput (tokens/second)
- **Long-context handling** - Performance with varying input sizes (1K, 8K, 16K, 32K tokens)

The tool automatically discovers all vLLM services in docker-compose.yml, manages Docker containers, runs tests with multiple iterations for statistical accuracy, and generates performance reports.

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
- **Input:** Minimal prompt (~50 tokens)
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

---

## Understanding Test Results: Why Latency Tests Take Longer

### The Counterintuitive Result

You may notice that the **Single-Request Latency test takes 3-4x LONGER** than even the **Very Long Context (32K tokens) test**.

**Why?** The latency test generates **5x more output tokens** (500 vs 100).

### The Root Cause: Output Generation is the Bottleneck

On most inference hardware, **output generation is significantly slower than input processing**:

#### Input Processing (Prefill Phase) - FAST ⚡
- **Parallel processing:** All input tokens processed simultaneously
- **32K tokens can be processed in ~10-30 seconds** (see TTFT times in results)
- Limited by memory bandwidth but highly parallelizable

#### Output Generation (Decode Phase) - SLOW 🐌
- **Sequential processing:** One token generated at a time
- **Memory bandwidth bottleneck** is the limiting factor
- **500 tokens takes 3-4x longer than 100 tokens**

### Why This Test Design Makes Sense

- **Latency test (500 output):** Measures generation throughput accurately - need many tokens to calculate reliable tokens/second
- **Context tests (100 output):** Measures input processing speed (TTFT) - output is kept short to isolate input processing performance

**Conclusion:** A test with 50 input tokens takes longer than a test with 32,000 input tokens because it generates 5x more output. This is **working as designed**.

---

## Usage

### Basic Usage

```bash
# Test all discovered vLLM models (default 3 iterations each)
./perftest-models.js

# Test only specific models (comma-separated, no spaces)
./perftest-models.js model1,model2

# Run with 5 iterations per test
./perftest-models.js --iterations 5

# Test specific models with custom iterations
./perftest-models.js model1,model2 --iterations 10

# Show help
./perftest-models.js --help
```

### CLI Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `models` | Comma-separated list of model names (positional) | All discovered | `model1,model2` |
| `--iterations <n>` | Number of test runs per scenario | 3 | `--iterations 5` |
| `--help` | Show help message | - | `--help` |

**Model names:** Derived from service names by removing the `vllm-` prefix
- Service: `vllm-qwen3-32b-fp8` → Model name: `qwen3-32b-fp8`
- Service: `vllm-llama33-70b-fp8` → Model name: `llama33-70b-fp8`

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
- Exact stdout capture (everything you see in the terminal)
- Latency test results table
- Long-context test results tables
- Performance comparison across all tested models

---

## Model Discovery

Models are **automatically discovered** from `docker-compose.yml`:

1. Finds all services matching pattern `vllm-*`
2. Extracts model ID from `command:` array (after "serve")
3. Extracts max context length from `--max-model-len` argument
4. Builds model objects dynamically

**No hardcoded model list** - works with any vLLM services you add to docker-compose.yml

---

## Test Execution Flow

1. **CLI Parsing:** Parse arguments and determine which models to test
2. **Service Discovery:** Scan docker-compose.yml for all vLLM services
3. **Model Filtering:** If specific models requested, filter the list
4. **Container Management:** Stop all running inference containers
5. **Per-Model Testing:**
   - Start model container
   - Wait for health check (up to 30 minutes for first load)
   - Pre-warm model with small request
   - Run latency tests (3 iterations)
   - Run context tests (3 iterations per context size)
   - Stop model container
6. **Results Analysis:** Calculate statistics (mean, stddev, min, max)
7. **Report Generation:** Save exact stdout to report file

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
- **Error:** "Models not found: {names}"
- **Solution:** Check service names in docker-compose.yml match requested models

### Health Check Timeout
- **Error:** "{model} health check timeout after 1800s"
- **Cause:** First-time model download taking longer than 30 minutes
- **Solution:** Check container logs: `docker compose logs -f {service}`

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
# Output is already saved to file automatically
./perftest-models.js

# Additionally capture to custom location
./perftest-models.js | tee custom-output.txt

# Run in headless environment
./perftest-models.js --iterations 1
```

### Parse Results

Reports are saved to `docs/reports/` automatically. Parse these files for:
- Performance regression detection
- Trend analysis over time
- Comparison between hardware configurations

---

## Adding New Models

To add a new model for testing:

1. Add vLLM service to `docker-compose.yml`:
```yaml
vllm-your-model:
  image: nvcr.io/nvidia/vllm:25.09-py3
  command:
    - vllm
    - serve
    - your/model-id        # Model will be auto-detected
    - --max-model-len
    - "32000"              # Context length will be extracted
    # ... other vLLM args
```

2. Run the test tool:
```bash
./perftest-models.js                    # Tests all including new model
./perftest-models.js your-model         # Test only new model
```

**No code changes needed!** The tool automatically discovers and tests any vLLM service.

---

## Version History

- **2025-11-09:** Dynamic service discovery
  - Removed hardcoded model list
  - Added automatic service discovery from docker-compose.yml
  - Simplified CLI (comma-separated models, removed --skip)
  - Changed to exact stdout capture for reports

- **2025-11-09:** Initial version
  - Added stream_options for usage statistics
  - Increased health check timeout to 30 minutes
  - Fixed context test token counting

---

**Last Updated:** 2025-11-09
**Tool Version:** 2.0
**Maintainer:** Stravica
