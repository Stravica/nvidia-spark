# Check vLLM Version

Check the current NVIDIA vLLM container version and provide guidance on checking for updates.

## Task

Run the version checking script and provide information about:

1. Current vLLM container version (shown in local Docker images)
2. How to check NGC catalog for newer versions
3. What version we need for GPT-OSS-20B support (25.10 or later)
4. Instructions to pull and test a new version if available

Execute: `./tools/check-vllm-version.sh`

Then guide the user on:
- Visiting https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
- Looking for versions 25.10-py3, 25.11-py3, or newer
- How to pull and test a new version safely

If a newer version exists, provide the exact commands to:
```bash
# Pull new version (example for 25.10)
docker pull nvcr.io/nvidia/vllm:25.10-py3

# Update docker-compose.yml image tag
# Test GPT-OSS-20B service
docker compose up -d vllm-gpt-oss-20b-mxfp4
docker compose logs -f vllm-gpt-oss-20b-mxfp4
```
