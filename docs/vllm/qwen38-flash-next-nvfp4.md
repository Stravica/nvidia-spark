# vLLM: Qwen3.8-Flash-Next-NVFP4 on NVIDIA DGX Spark

## Overview

Deployment guide for running `RadixArk/Qwen3.8-Flash-Next-NVFP4` under vLLM on a DGX Spark (GB10 Grace Blackwell). Qwen3.8-Flash-Next is a ~176B-parameter MoE model (125B main + 51B n-gram embedding table, 6B active per token) released 2026-08-26 as a preview of the Qwen4 architecture. NVFP4 pre-quantised. Native context 262K, extensible to 1M with YaRN.

The service is defined in `docker-compose.yml` as `vllm-qwen38-flash-next-nvfp4` and serves an OpenAI-compatible endpoint on port 8000.

---

## Why this configuration exists

The NVFP4 checkpoint is 122 GiB on disk. Loaded resident, it does not fit next to a usable KV cache in the Spark's 128 GB unified pool. 44 GiB of that is the model's n-gram embedding table (a Positional Learned Embedding, "PLE"), which is a pure lookup: each token only touches 16 rows of it (about 2.5 KB). The blazux/qwen3.8-Flash-DGX recipe adds one patch to the vendor-published vLLM Flash-Next image that serves the PLE table via `mmap` from NVMe instead of keeping it resident. Weights drop to ~76 GiB, the rest of the pool goes to KV cache and activations, and the model serves on a single Spark.

The recipe is a community adaptation of the vLLM `release/qwen38next` branch (PR #53896). The base image and model checkpoint are official; the mmap patch is the community add.

**Sources:**

- Recipe: [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
- vLLM release branch: `release/qwen38next` (base image `vllm/vllm-openai:qwen38-flash-next`, pinned by digest)
- Model card: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)

---

## Image choice

The service uses a locally-built image `qwen38-flash-dgx:2026-08-27`, which is `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` with one Python patch applied on top. The base image tag is not floating; the sha256 pin makes rebuilds reproducible.

The compose service references the dated tag rather than `:latest` so a future rebuild cannot silently swap what the running service resolves to. Bump both the tag on disk and the compose reference in the same commit when the recipe advances.

The patch (`src/vllm_ple_mmap.py` from the recipe repo) hooks the `Qwen3_8FlashNextNGramEmbedding` class. It is a no-op unless `VLLM_PLE_MMAP=1` is set at runtime, so the image behaves identically to upstream when the flag is off.

Build steps (one-time on the Spark host):

```bash
sudo mkdir -p /usr/local/apps/vllm-flash-next && sudo chown "$USER:$USER" /usr/local/apps/vllm-flash-next
cd /usr/local/apps/vllm-flash-next
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX
docker build -t qwen38-flash-dgx:2026-08-27 -t qwen38-flash-dgx:latest .
```

The base image is multi-arch and the build works on arm64 (Spark) and x86 Blackwell.

---

## Checkpoint choice

`RadixArk/Qwen3.8-Flash-Next-NVFP4` is the vLLM team's referenced NVFP4 requant. About 122 GiB on disk across multiple safetensors shards including the 44 GiB `model-plefp8-*.safetensors` PLE shards (mmapped) and the main weight shards (resident). One-time download (~130 GiB required free on the mount):

> **The compose service passes a local snapshot path, not the HF repo id.** The community PLE mmap patch resolves the n-gram (PLE) shards through the filesystem: passing the repo id (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) makes weight loading run to about 93% and then die with `RuntimeError: PLE mmap: model path '...' is not a local directory; point --model at the downloaded snapshot` from `src/vllm_ple_mmap.py::_setup_table`. The recipe's own `scripts/serve.sh` resolves the snapshot dir into `SNAP_HOST` / `SNAP_IN` before it invokes `docker run`. In compose, the equivalent is to hardcode the in-container snapshot path (`/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<hash>`) as the model positional arg and bump the hash in-place whenever the checkpoint is re-downloaded to a new one. `HF_HUB_OFFLINE=1` is set on the service so vLLM cannot try to reach the hub for a re-resolve.

```bash
docker run --rm --name qwen38-flash-dl \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -v /opt/hf:/hf --entrypoint bash qwen38-flash-dgx \
  -c "hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --max-workers 8"
```

`HF_HUB_DISABLE_XET=1` disables the Xet backend, which has stalled on some Spark setups. Plain HTTPS is reliable and saturates the link.

---

## Unified memory: `--gpu-memory-utilization 0.78`

The Spark's 128 GB pool is shared across OS, container runtime, model weights, KV cache, and the CPU. There is no separate VRAM. At 0.78:

- Resident weights (mmap patch active): ~76 GiB.
- PLE table: served from disk (page cache warms as tokens hit it; hot rows stay resident cheaply). Prewarm optional via `VLLM_PLE_MMAP_PREWARM=1` for steadier first-request latency at ~10s extra startup.
- KV cache and activations: the remaining budget under 0.78.

Do not raise this above 0.80 on the Spark. The recipe README is explicit: an OOM in this pool can freeze the box, because the same LPDDR5x backs the OS.

---

## Mandatory GB10 (sm_121) workarounds

Two flags are load-bearing on this hardware and non-obvious. Both are set in the compose service.

**`--no-enable-prefix-caching`.** The GB10 GDN (Gated Delta Net) kernel has a bug that corrupts on the cached-block path. Enabling prefix caching produces silent output corruption, not a crash. Estate default across other vLLM services is prefix caching on; on this service it must stay off until vLLM ships a fix.

**`torch.compile` off (via `-cc.cudagraph_mode=PIECEWISE` with the full splitting-ops list, plus `--no-enable-flashinfer-autotune`).** Full `torch.compile` triggers an Inductor int64-indexing assert on sm_121. The recipe's approach: run in PIECEWISE cudagraph mode with every custom op declared as a splitting op, which keeps CUDA graphs but bypasses the failing Inductor path. `--enforce-eager` also works but is slower.

The `-cc.splitting_ops=[...]` list is exhaustive and matches the recipe upstream:

```
vllm::unified_attention_with_output
vllm::unified_mla_attention_with_output
vllm::mamba_mixer2
vllm::mamba_mixer
vllm::short_conv
vllm::qwen3_8_flash_next_ple_short_conv
vllm::qwen3_8_flash_next_qsa_with_output
vllm::linear_attention
vllm::qwen_gdn_attention_core
vllm::qwen_gdn_attention_core_fused_norm_packed
vllm::sparse_attn_indexer
vllm::ple_mmap_lookup   # from the mmap patch itself
```

Adding or removing entries here is not free; the PLE gather is a CPU op followed by a pageable host-to-device copy and it must run outside CUDA graphs.

---

## MTP speculative decoding

The recipe defaults MTP off; opt-in via `MTP=2` at serve time. This service's compose default enables MTP=2:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Recipe benchmarks give ~1.6x decode uplift over MTP-off at ~67% acceptance. MTP=3 is possible but was not the default upstream. To disable MTP for a specific run, edit the compose command or override.

Caveat: `min_p` and `logit_bias` sampling parameters do not compose with speculative decoding in this vLLM build. If a caller needs them, disable MTP for that path.

---

## Tool-call and reasoning parsers

- `--tool-call-parser qwen3_coder` matches Qwen3.8-Flash-Next's tool-call format.
- `--reasoning-parser qwen3` matches the model's `<think>...</think>` framing.

Both are the recipe defaults; other parsers were not tested against this model on this platform.

---

## Context length: `--max-model-len 32768`

Native context is 262,144 tokens (extensible to 1M with YaRN). The service is configured at 32K because KV grows linearly with context and the whole KV budget shares the pool with 76 GiB of resident weights. Raising it costs KV cache capacity for concurrent sequences and can push the pool into the OOM danger zone; do not raise it without redoing the memory math and lowering `--gpu-memory-utilization` accordingly.

---

## Concurrency: `--max-num-seqs 2`

The recipe treats this as a large-model prototype-serving profile: two concurrent sequences at 32K each. Aggregate throughput scales weakly beyond that on Spark memory bandwidth, and the resident-weight ceiling is the binding constraint.

---

## Performance

Measured on the estate's Spark (GB10, 128 GB, arm64), single request, ctx 32K, MTP=2, verified 2026-08-27:

- **Prefill:** 1,725 tok/s on an 8,001-token prompt (recipe reference on ASUS GX10 is ~2,400-2,660 tok/s; our Spark unit runs about 65-72% of that on this synthetic prompt).
- **Decode (MTP=2):** 28.5 tok/s over a 256-token completion (recipe reference ~27 tok/s at ~67% MTP acceptance; matches).
- **Decode (MTP off):** not re-measured on our unit; recipe reference ~17 tok/s. To measure locally, override the `--speculative-config` in the compose command and recreate the service (adds ~15 min for a full reload).
- **Coherence:** correct (`17 * 23` returns `391.` in the visible completion; the reasoning parser leaves `reasoning_content: null` on plain-arithmetic prompts).
- **Load time:** about 15 min from `docker compose up` to `Application startup complete` on a warm HF cache — weight-shard loading is roughly 10 min at 2.7 s/shard for 206 shards, CUDA graph capture and startup add about 5 min.
- **Memory footprint at rest:** 102 GB of the 128 GB pool in use, 16 GB available. Container user-space RSS is only 5.5 GiB; the balance is kernel page cache backing the mmapped PLE shards plus vLLM's own resident weights and KV cache. Headroom above the 0.78 utilisation budget is real but not generous — this is close to what the box can serve without freeze risk, and dial back before raising `--max-model-len` or running anything else.

Compared to the only prior working single-box option (llama.cpp GGUF IQ4_XS), this recipe is roughly 5x faster on prefill and adds MTP support. Dense reasoning quality is Flash-Next's headline; expect single-stream decode below the estate's 30B-A3B MoE options because the 6B active path plus the on-demand PLE gather cannot beat a hot in-memory MoE on this bandwidth-bound hardware.

If measured tok/s falls substantially below these targets, first suspect: prefix caching quietly turned on (check the container `command:`), then the splitting-ops list drifting.

---

## Files

- Compose service: `docker-compose.yml` (`vllm-qwen38-flash-next-nvfp4`)
- Recipe repo (host, clone target): `/usr/local/apps/vllm-flash-next/qwen3.8-Flash-DGX`
- Model cache (host, bind-mounted to `/root/.cache/huggingface`): `/opt/hf`

---

## Troubleshooting

**Container refuses to start with "image not found" for `qwen38-flash-dgx:latest`.** The image is built locally, not pulled. Run the build step above on the Spark host before compose-up.

**Load stalls partway through weight ingest.** The PLE mmap patch requires the PLE shards to be present on disk. Confirm with `du -sh /opt/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4` around 122 GiB. If the download aborted, re-run `hf download` (resumable).

**EngineCore dies at ~93% with `PLE mmap: model path '...' is not a local directory`.** The compose service's model positional arg is the HF repo id instead of the local snapshot path. See the checkpoint-choice section above. Fix by editing the compose command to point at the current snapshot directory.

**Nonsense output.** The GDN prefix-caching bug corrupts silently. Verify the compose `command:` still contains `--no-enable-prefix-caching`.

**Immediate assert on start, message about int64 indexing.** The `torch.compile` workaround has slipped. Check the compose `command:` still has `-cc.cudagraph_mode=PIECEWISE` and the full `-cc.splitting_ops` list.

**Box freezes during load.** The 0.78 headroom got eroded (context raised, other tenants on the pool, prewarm colliding with a large model). Reduce `--max-model-len` first, then `--gpu-memory-utilization`.
