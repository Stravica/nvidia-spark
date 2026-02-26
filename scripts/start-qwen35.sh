#!/bin/bash
# Startup script for Qwen3.5-35B-A3B-FP8
# Upgrades transformers to v5 for native qwen3_5_moe support
# Patches list-vs-set compat issue between vLLM config and transformers v5
# Uses pip cache volume for fast subsequent starts

set -e

MARKER="/pip-cache/.qwen35-transformers5-ok"

if [ ! -f "$MARKER" ]; then
    echo "[qwen35] Upgrading transformers to v5 for native Qwen3.5 support..."
    pip install --cache-dir /pip-cache "transformers>=5.0" --quiet 2>&1 | tail -5
    touch "$MARKER"
    echo "[qwen35] Upgrade complete."
else
    echo "[qwen35] Applying cached transformers v5 upgrade..."
    pip install --cache-dir /pip-cache "transformers>=5.0" --quiet 2>&1 | tail -3
fi

# Fix list-vs-set compat: vLLM config uses list for ignore_keys_at_rope_validation
# but transformers v5 expects a set (uses | operator for set union)
python3 << 'PATCH'
import re

config_path = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"
with open(config_path) as f:
    content = f.read()

# Replace the specific list assignment with a set assignment
old = '''kwargs["ignore_keys_at_rope_validation"] = [
            "mrope_section",
            "mrope_interleaved",
        ]'''
new = '''kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }'''

if old in content:
    content = content.replace(old, new)
    with open(config_path, 'w') as f:
        f.write(content)
    print("[qwen35] Patched rope validation list->set compat fix")
elif '{' in content.split('ignore_keys_at_rope_validation')[1][:50]:
    print("[qwen35] Rope validation already patched")
else:
    print("[qwen35] WARNING: Could not find expected pattern to patch")
PATCH

python3 -c "
import transformers, huggingface_hub, vllm
print(f'[qwen35] transformers={transformers.__version__} huggingface_hub={huggingface_hub.__version__} vllm={vllm.__version__}')
"

echo "[qwen35] Starting vLLM server..."
exec vllm serve "$@"
