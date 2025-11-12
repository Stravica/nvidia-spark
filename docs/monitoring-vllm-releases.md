# Monitoring NVIDIA vLLM Container Releases

This guide explains how to monitor for new NVIDIA vLLM container releases, specifically to check for GPT-OSS-20B compatibility.

---

## Current Status

**Your Version:** `nvcr.io/nvidia/vllm:25.09-py3` (September 2025)
**Required for GPT-OSS:** `25.10-py3` or later
**Why:** GPT-OSS-20B requires mature attention backend support for the `sinks` parameter

---

## Method 1: NGC Catalog (Web Interface) ⭐ RECOMMENDED

**URL:** https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

**Steps:**
1. Visit the URL above
2. Click on the "Tags" tab
3. Look for newer versions than `25.09-py3`:
   - `25.10-py3` (October 2025)
   - `25.11-py3` (November 2025)
   - `25.12-py3` (December 2025)
4. Click on a tag to see:
   - Release date
   - vLLM version included
   - Release notes
   - Size and architecture support

**What to Look For:**
- vLLM version 0.10.2+ or 0.11.0+
- Release notes mentioning "GPT-OSS" or "attention sinks"
- Support for Blackwell (GB10) architecture

---

## Method 2: Command-Line Script

**Quick Check:**
```bash
./tools/check-vllm-version.sh
```

**What It Shows:**
- Your current vLLM container version and creation date
- Link to NGC catalog
- Expected newer version tags based on current date

---

## Method 3: Claude Code Slash Command

**Usage:**
```
/check-vllm
```

This will:
- Run the version checking script
- Provide guidance on checking for updates
- Give commands to pull and test new versions

---

## Method 4: Docker Command-Line Tools

### Option A: Using `crane` (if installed)

```bash
# Install crane (one-time)
# Visit: https://github.com/google/go-containerregistry/releases

# List all available tags
crane ls nvcr.io/nvidia/vllm
```

### Option B: Using `skopeo` (if installed)

```bash
# Install skopeo (one-time)
sudo apt-get install skopeo

# List all available tags
skopeo list-tags docker://nvcr.io/nvidia/vllm
```

### Option C: Using Docker directly

```bash
# Check for a specific newer version
docker manifest inspect nvcr.io/nvidia/vllm:25.10-py3 2>&1 | grep -q "no such manifest" && echo "Version not available yet" || echo "Version available!"

# Try pulling (will fail if doesn't exist)
docker pull nvcr.io/nvidia/vllm:25.10-py3
```

---

## Method 5: Subscribe to Updates

### NVIDIA Developer Forum
- **URL:** https://forums.developer.nvidia.com/c/accelerated-computing-cloud-edge/54
- Subscribe to the "Containers & Cloud Native" section
- Set up email notifications for new posts

### GitHub Watch
- **vLLM Repo:** https://github.com/vllm-project/vllm
- Watch for releases (click "Watch" → "Custom" → "Releases")
- NVIDIA containers typically follow vLLM releases by 2-4 weeks

### NVIDIA NGC Changelog
- **URL:** https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
- Check the "Deep Learning Frameworks Support Matrix"
- Updated quarterly with container release schedules

---

## When a New Version is Available

### Step 1: Check Release Notes

Visit NGC catalog and read the release notes for:
- vLLM version included (should be 0.10.2+ or 0.11.0+)
- Support for new models (look for GPT-OSS mentions)
- Bug fixes related to attention backends
- Blackwell GPU support

### Step 2: Pull the New Image

```bash
# Example for version 25.10
docker pull nvcr.io/nvidia/vllm:25.10-py3

# Verify it downloaded
docker images nvcr.io/nvidia/vllm
```

### Step 3: Update docker-compose.yml

```bash
# Edit the GPT-OSS-20B service
# Change: image: nvcr.io/nvidia/vllm:25.09-py3
# To:     image: nvcr.io/nvidia/vllm:25.10-py3
```

Or update all vLLM services at once:

```bash
# Using sed (backup first!)
cp docker-compose.yml docker-compose.yml.backup
sed -i 's/nvcr.io\/nvidia\/vllm:25.09-py3/nvcr.io\/nvidia\/vllm:25.10-py3/g' docker-compose.yml
```

### Step 4: Test GPT-OSS-20B

```bash
# Stop all other vLLM services
docker compose stop vllm-qwen3-8b-fp8 vllm-llama31-8b-fp8 vllm-mistral-nemo-12b-fp8 \
  vllm-qwen3-32b-fp8 vllm-qwen3-30b-a3b-fp8 vllm-llama33-70b-fp8

# Start GPT-OSS-20B
docker compose up -d vllm-gpt-oss-20b-mxfp4

# Monitor logs (should see successful startup without crashes)
docker compose logs -f vllm-gpt-oss-20b-mxfp4

# Look for these success indicators:
# - "Loading model weights took X seconds"
# - "Available KV cache memory: XX GiB"
# - "Uvicorn running on http://0.0.0.0:8000"
# - NO "TypeError: ...unexpected keyword argument 'sinks'"
# - NO restart loops
```

### Step 5: Test Inference

```bash
# Wait for startup to complete (~2-3 minutes)
# Then test with a simple request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello! Please introduce yourself."}],
    "max_tokens": 100
  }'
```

**Success Indicators:**
- Gets a response (not an error)
- Response contains generated text
- No crashes in logs

### Step 6: Run Performance Tests

```bash
# Add GPT-OSS-20B to perftest-models.js (if not already added)
# Then run performance comparison
./tools/perftest-models.js
```

---

## Troubleshooting New Versions

### Issue: New container version still crashes

**Possible Causes:**
- GPT-OSS support still incomplete
- Need even newer vLLM version
- Different configuration needed

**Actions:**
1. Check vLLM GitHub issues: https://github.com/vllm-project/vllm/issues
2. Search for "gpt-oss" or "attention sinks"
3. Check NGC release notes for known issues
4. Revert to previous container version

### Issue: Can't pull new image

**Possible Causes:**
- Version doesn't exist yet
- Need to authenticate with NGC
- Network issues

**Actions:**
```bash
# Check if version exists
docker manifest inspect nvcr.io/nvidia/vllm:25.10-py3

# If authentication needed (usually not required for public containers)
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### Issue: Performance worse than older version

**Possible Causes:**
- Configuration changes between versions
- Need to adjust parameters for new engine

**Actions:**
1. Check NGC release notes for breaking changes
2. Compare your configuration with new defaults
3. Monitor memory usage: `nvidia-smi`
4. Try different `gpu-memory-utilization` values

---

## Checking Schedule

**Recommended Check Frequency:**
- **Monthly:** Check NGC catalog on the 1st of each month
- **After vLLM Release:** Check NGC 2-3 weeks after major vLLM releases
- **When Needed:** Before deploying new models

**Setting Up Reminders:**

```bash
# Add to your crontab for monthly email reminder
# crontab -e
# 0 9 1 * * /opt/inference/tools/check-vllm-version.sh | mail -s "vLLM Version Check" your@email.com
```

---

## Version History Reference

| NVIDIA Container | vLLM Version | Release Date | GPT-OSS Support |
|-----------------|--------------|--------------|-----------------|
| 25.09-py3 | 0.10.1.1 | Sep 2025 | ❌ Incomplete |
| 25.10-py3 | 0.10.2+ | Oct 2025 | ⏳ Check release notes |
| 25.11-py3 | 0.11.0+ | Nov 2025 | ⏳ Check release notes |

---

## Quick Reference Commands

```bash
# Check current version
docker images nvcr.io/nvidia/vllm

# Run version check script
./tools/check-vllm-version.sh

# Use Claude Code command
/check-vllm

# Check NGC catalog
# Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

# Pull new version (example)
docker pull nvcr.io/nvidia/vllm:25.10-py3

# Test GPT-OSS-20B
docker compose up -d vllm-gpt-oss-20b-mxfp4
docker compose logs -f vllm-gpt-oss-20b-mxfp4

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hi!"}]}'
```

---

## Additional Resources

- **NGC Catalog:** https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
- **vLLM GitHub:** https://github.com/vllm-project/vllm
- **vLLM Docs:** https://docs.vllm.ai
- **NVIDIA DGX Docs:** https://docs.nvidia.com/dgx/
- **GPT-OSS Paper:** https://openai.com/research/gpt-oss

---

**Last Updated:** 2025-11-12
**Current Container:** 25.09-py3
**Awaiting:** 25.10-py3 or later for GPT-OSS-20B support
