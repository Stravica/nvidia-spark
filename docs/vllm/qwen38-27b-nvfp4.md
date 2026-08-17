# vLLM: Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark

## Overview

Deployment guide for running `unsloth/Qwen3.8-27B-NVFP4` under vLLM on a DGX Spark (GB10 Grace Blackwell). Dense 27B reasoning model, pre-quantized to NVFP4, with MTP speculative decoding enabled for roughly 2x single-stream decode over the MTP-off baseline.

The service is defined in `docker-compose.yml` as `vllm-qwen38-27b-nvfp4` and serves an OpenAI-compatible endpoint on port 8000.

---

## Why this configuration exists

Stock `vllm/vllm-openai` builds crash at CUDA init on GB10 because bundled PyTorch only ships kernels through sm_120; the GB10 GPU is sm_121. This is tracked at [vllm-project/vllm#36821](https://github.com/vllm-project/vllm/issues/36821), still open at the time of writing. Two practical routes exist:

1. Pin a vLLM-published image that carries sm_121-native kernels, or
2. Build from source with `TORCH_CUDA_ARCH_LIST=12.1a`.

This service uses route 1 with an immutable version tag (not `:latest`), so restarts are reproducible.

**Reference:** the vLLM Recipes DGX Spark NVFP4 walkthrough, [docs.vllm.ai/projects/recipes](https://docs.vllm.ai/projects/recipes/en/latest/).

---

## Image choice

**`vllm/vllm-openai:v0.24.0-ubuntu2404`** is vLLM's DGX Spark recipe build. First candidate in the ladder, and it works on the first attempt on GB10:

- CUDA init succeeds (no sm_121 crash).
- Kernel selection is sm_121-native at boot (`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`, `Using Triton/FLA GDN prefill kernel`).
- Weight loading via `fastsafetensors` completes cleanly.

**Never use `:latest`.** Floating tags on this hardware are a rollback landmine because the image ladder (stock crash vs vendor recipe vs community sm_121 build) is not obvious from logs. Pin the exact tag.

**Backup candidates if v0.24.0 stops matching your driver:**

- `timothystewart6/vllm-gb10:v0.24.0-gb10.0` (or a sha-pinned tag from that repo). Community-maintained, sm_121-native, arm64.
- `nvcr.io/nvidia/vllm:26.02-py3` or newer (NGC).
- Source build with `TORCH_CUDA_ARCH_LIST=12.1a` per the vLLM issue.

Image size is around 21 GB. First pull on Spark can take 15 to 20 minutes because aarch64 layer extraction is not fast; a `docker system prune` before the pull is worth doing.

---

## Checkpoint choice

**`unsloth/Qwen3.8-27B-NVFP4`** is a community NVFP4 requant of `Qwen/Qwen3.8-27B`, roughly 22 GB of safetensors on disk (13 files including `model.safetensors`, `model_mtp.safetensors`, tokenizer, configs).

Chosen because:

- The nvidia-namespace candidate `nvidia/Qwen3.8-27B-NVFP4` returns HTTP 404 even with an authenticated `HF_TOKEN`. It is absent, not gated.
- Unsloth's namespace is public and non-gated.
- Ships `model_mtp.safetensors` (the MTP head), which is what enables the speculative decoding uplift below.
- Architecture is `Qwen3_5ForConditionalGeneration` (multimodal-capable), disabled here via `--limit-mm-per-prompt '{"image":0,"video":0}'` for text-only serving.

---

## Unified memory: `--gpu-memory-utilization 0.60`

DGX Spark has one 128 GB pool shared across CPU, GPU, OS, container runtime, model weights, and KV cache. There is no separate VRAM. `--gpu-memory-utilization` therefore needs deliberate headroom rather than the near-1.0 defaults used on discrete GPUs. Documented working values across published sources fall in the 0.5 to 0.9 range depending on model size and workload.

For this service:

- Model weights (NVFP4): about 21 GB in GPU memory.
- MTP head: adds about 0.8 GB (weights are shared with the target model's embedding and lm_head, not duplicated).
- KV cache headroom at 0.60 utilization: about 22 GB with bf16 KV cache, comfortably above the `max-num-seqs=8 x max-model-len=131072 = ~1.05M` worst case.

If you raise `--max-num-seqs` past 8, or you serve two 131K contexts back-to-back with prefix caching disabled, revisit the utilization number.

---

## MTP speculative decoding

The checkpoint ships a Multi-Token-Prediction head (`model_mtp.safetensors`). Enabling it delivers substantial single-stream and moderate concurrent uplift on GB10.

```
--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'
```

Notes on this flag:

- vLLM emits `WARNING [speculative.py:590] method 'qwen3_5_mtp' is deprecated and replaced with mtp.` The two names resolve to the same code path. The explicit `qwen3_5_mtp` string is kept here as documentation of intent.
- `mtp_num_hidden_layers` in the checkpoint is 1. Setting `num_speculative_tokens=3` iterates that single head three times per step; vLLM warns "may result in lower acceptance rate," and in practice acceptance held up in the benchmarks below.
- `min_p` and `logit_bias` sampling parameters do not work with speculative decoding in this vLLM build. If any caller starts passing them, disable MTP for that path.
- MTP loading is logged as `Detected MTP model. Sharing target model embedding weights with the draft model`. This is why the head only costs about 0.8 GB extra.

A `num_speculative_tokens` sweep (n=1, 2, 4, 5) was not run; n=3 was the vendor recipe default and the measured uplift was already sizeable. If TTFT becomes a concern for chat use, try n=2 or n=1.

---

## Tool-call parser: `qwen3_xml`

This model emits tool calls in Qwen3.5 XML form:

```
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

vLLM's `hermes` parser expects JSON inside `<tool_call>` and will silently leave `msg.tool_calls[]` empty on every request. Use one of the qwen3 parsers instead. All three (`qwen3_xml`, `qwen3_coder`, `mimo`) resolve to the same `Qwen3EngineToolParser` class in `vllm/parser/qwen3.py`. This service picks `qwen3_xml` as the most descriptive alias for the format.

Reasoning output uses `--reasoning-parser qwen3`, matching the model's `<think>...</think>` framing.

---

## KV cache dtype

`--kv-cache-dtype` is deliberately not set. vLLM defaults to the model dtype (bf16 here). Setting `--kv-cache-dtype fp8` doubles KV headroom but costs a per-decode-step packing overhead that showed up as 4 to 12 percent single-stream loss and about 18 percent loss at c=4 aggregate on this workload (see the tuning table below).

If you need the extra KV headroom (higher `--max-num-seqs`, longer live contexts under prefix caching), re-enable fp8 and accept the throughput hit.

---

## Other flags, quickly

| Flag | Why |
|---|---|
| `--max-model-len 131072` | Model native. Vendor recipe uses 262144; halved here because 131K is enough for realistic workloads and keeps KV pressure lower. |
| `--max-num-seqs 8` | Vendor recipe. Larger values increase per-step scheduling overhead without helping single-stream. |
| `--max-num-batched-tokens 8192` | Vendor recipe. Prefill throughput ceiling; irrelevant to decode. |
| `--enable-chunked-prefill` | Keeps TTFT under concurrency roughly linear rather than quadratic. Confirmed working (TTFT scales at about 40 ms per additional concurrent stream up to c=8). |
| `--async-scheduling` | Vendor recipe. Tested off in cycle 3 and lost 3 percent at c=8, 8 percent at c=4. Keep it on. |
| `--enable-prefix-caching` | Standard vLLM win for repeated system prompts. |
| `--load-format fastsafetensors` | Vendor recipe. Emits a benign GDS warning on GB10 (GDS is not supported; falls back to nogds automatically). |
| `--trust-remote-code` | Required by the checkpoint. |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | Text-only serving. Saves the vision tower load, adds about 11 seconds at startup registering text-only mode. |

`--quantization` is unset. vLLM auto-detects `compressed-tensors` (the checkpoint's format) at boot. The vLLM docs also mention `--quantization modelopt` as required for ModelOpt-served NVFP4 checkpoints via the OpenAI server; this is an unsettled area of the docs. For the Unsloth requant, unset works.

---

## Benchmarks

Measured on a single Spark, LAN client, node.js streaming client against `/v1/completions`, `temperature=0.1`, `max_tokens=256`, `stream_options.include_usage=true`. Warmup was one full 256-token prose completion before measurement.

### Batch-1 decode by content type (tok/s, higher is better)

| Configuration | prose | code | math | TTFT prose (ms) |
|---|---:|---:|---:|---:|
| Baseline (MTP off, kv-fp8, async on) | 11.66 | 11.65 | 11.66 | 138 |
| MTP=3 added | 20.25 | 26.36 | 26.27 | 264 |
| MTP=3 + kv auto (this service) | 21.17 | 29.51 | 27.53 | 265 |
| MTP=3 + kv auto + async off | 21.04 | 29.40 | 27.40 | 267 |
| This service (re-measured after locking in) | 20.36 | 28.38 | 26.07 | 273 |

Content dependence appears exactly once speculative decoding is on (math > code > prose). In the MTP-off baseline the three content types are indistinguishable because decode is one token per step regardless of content; with speculation, the model's confidence in the next-3 draft varies by content type and accepted-token count spreads accordingly.

### Concurrent aggregate (tok/s)

| Configuration | prose c=4 | prose c=8 |
|---|---:|---:|
| Baseline | 43.41 | 82.82 |
| MTP=3 | 64.84 | 121.23 |
| MTP=3 + kv auto (this service) | 76.41 | 127.32 |
| MTP=3 + kv auto + async off | 70.14 | 123.80 |
| This service (re-measured) | 62.29 | 129.16 |

The c=4 numbers show meaningful run-to-run noise (about 20 percent), driven by TTFT variance immediately after container recreate (CUDA-graph capture cache not fully warm). The c=1 and c=8 numbers were stable across the re-run.

### Before vs after

| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| batch-1 decode prose | 11.66 | 20.36 | +75% |
| batch-1 decode code | 11.65 | 28.38 | +144% |
| batch-1 decode math | 11.66 | 26.07 | +124% |
| c=8 aggregate | 82.82 | 129.16 | +56% |
| TTFT prose (ms) | 138 | 273 | +98% (spec-decoding overhead) |

The single-stream code and math figures sit at or above the published vLLM Recipes DGX Spark NVFP4 headline (about 24.5 tok/s prose-weighted). Prose sits slightly below because prose has lower spec-decoding acceptance than structured content.

TTFT roughly doubled versus the baseline. This is expected: MTP now runs the target model plus one draft step before the first token can be emitted. If TTFT becomes a UX concern for chat, drop `num_speculative_tokens` to 2 or 1.

---

## Startup timings (first boot on this image)

- Image pull: about 20 min (5 to 6 layers, 21.4 GB, slow aarch64 extraction).
- Model xet re-fetch on first run: about 11.5 min (about 686 seconds downloading weights).
- Weight load via fastsafetensors: about 11 seconds.
- Model to GPU: 21.26 GiB memory (with MTP head), total 700 seconds dominated by the xet fetch.
- Torch inductor compile: about 34 seconds (single compile range 1 to 8192).
- AOT compile save: about 6 seconds.
- Torch.compile total: about 52 seconds.
- Initial profiling and warmup: about 44 seconds.
- FlashInfer autotune (fp4_gemm, 21 profiles): about 35 seconds.
- API up (Uvicorn ready): roughly 15 minutes from container start, minus the xet fetch on subsequent starts (typically 2 to 3 min).

### Torch compile cache caveat

The torch-compile cache lives at `/root/.cache/vllm/torch_compile_cache/...` inside the container's writable layer. It is not bind-mounted here.

- `docker compose restart vllm-qwen38-27b-nvfp4` preserves the cache; the service returns in about 2 to 3 min.
- `docker compose down && up -d` (recreation) discards it and pays the 52-second compile cost again.

If cold-start latency matters to you, bind `~/.cache/vllm` to a host path.

---

## Benign warnings during startup

None of these need action:

- `Prefix caching in Mamba cache 'align' mode is currently enabled. Its support for Mamba layers is experimental.` Qwen3.5-family uses Mamba/GDN hybrid attention.
- `GDS is not supported in this platform but nogds is False. use nogds=True`. Fastsafetensors GPUDirect Storage; GB10 does not support it, falls back automatically.
- `Not enough SMs to use max_autotune_gemm mode`. Inductor cosmetic.
- `Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT / PIPELINE / URL / IMAGE_TAG`. Image metadata set by the vendor image.
- `CUDA graph memory profiling is enabled ... equivalent to --gpu-memory-utilization=0.5946`. Informational.
- `method 'qwen3_5_mtp' is deprecated and replaced with mtp`. Cosmetic; auto-resolved.
- `num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate`. Architectural warning; acceptance held up in measurement.
- `min_p and logit_bias parameters won't work with speculative decoding.` Enforce this in your client if you rely on either.

---

## Verify it works

Once the container is up:

```bash
# Health
curl -s http://localhost:8000/health

# Plain completion
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38-27b-nvfp4","prompt":"The capital of France is","max_tokens":8,"temperature":0.1}'

# Tool call (should return a populated tool_calls[] with valid JSON args)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b-nvfp4",
    "messages": [{"role":"user","content":"What is the weather in Lisbon right now? Use the tool."}],
    "tools": [{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
```

Tool-call success looks like `tool_calls: [{id:..., type:"function", function:{name:"get_weather", arguments:'{"city":"Lisbon"}'}}]` with `content` empty. If `tool_calls` is empty and the raw XML appears in `content`, the parser flag is wrong (probably `hermes` instead of `qwen3_xml`).

---

## References

- vLLM Recipes DGX Spark: https://docs.vllm.ai/projects/recipes/en/latest/
- vLLM issue #36821 (GB10 sm_121 support): https://github.com/vllm-project/vllm/issues/36821
- Checkpoint: https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
- Base model: https://huggingface.co/Qwen/Qwen3.8-27B
- Hardware notes: [../nvidia-spark.md](../nvidia-spark.md)
