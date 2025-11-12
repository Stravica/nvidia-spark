#!/bin/bash
# Check for new NVIDIA vLLM container versions

echo "=== Current vLLM Version ==="
docker images nvcr.io/nvidia/vllm --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}"

echo ""
echo "=== Checking NGC Catalog for Latest Versions ==="
echo "Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm"
echo ""
echo "Or use crane/skopeo to query available tags:"
echo "  crane ls nvcr.io/nvidia/vllm"
echo ""
echo "Current date: $(date +'%Y-%m')"
echo "Look for tags: 25.10-py3, 25.11-py3, 25.12-py3, etc."
