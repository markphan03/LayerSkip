# Early-exit speculative decoding optimization

This change reduces GPU memory consumption and decoding overhead in
`early_exit_speculative`. The default CUDA backend now uses ordinary CUDA
graphs that share one memory pool and capture stream. The GRU still selects an
exit depth independently for each draft token, and the full model still
verifies the speculative block.

The measurements below were collected on November 11, 2026, using an NVIDIA
RTX 4090, PyTorch `2.14.1+cu130`, and Transformers `4.50.0`.

## Why the previous backend used so much memory

The previous default used conditional CUDA graphs. It captured separate draft
and verification graphs for each possible block length. Within each graph,
each decoder layer had a conditional branch for every possible missing suffix
length.

For six speculative tokens and unrestricted draft depth, this creates 13
top-level graphs and 49 captured decoder bodies per layer. The number of
captured decoder bodies grows quadratically with the speculation count.

Two allocation patterns made this expensive:

- Each top-level capture used a separate private memory pool.
- Conditional branches introduced CUDA streams and associated BLAS workspaces.
  The allocator probe showed repeated 8.125 MiB workspace allocations, along
  with substantial unused reserved space.

In the installed PyTorch build, most conditional-graph memory also remained
reserved after deleting the engine, collecting garbage, and calling
`torch.cuda.empty_cache()`. This behavior matches the reported
[conditional-graph pool-release issue](https://github.com/pytorch/pytorch/issues/201567).
Rebuilding graphs for different prompt capacities or decoding settings could
therefore accumulate memory across requests.

PyTorch documents the relevant
[CUDA graph memory pools and cuBLAS workspaces](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## What changed

### Shared-pool graphs with grouped exit checks

The new `GroupedDraftGraphs` implementation captures decoder groups between
scheduled exit checkpoints. All captures use one memory pool and capture
stream. There are no conditional-node branches in this backend.

Graph inputs and outputs live in persistent buffers allocated before capture.
Intermediate tensors remain local to each graph execution. Results are copied
into those persistent buffers before the next graph reuses the shared pool.
Graphs execute sequentially, allowing the decoder to select different groups
and suffix lengths without retaining a separate workspace pool for each one.

The host tracks how many positions each group has completed. A group processes
only its missing suffix, both during drafting and verification. Raw hidden
states are retained at group boundaries; intermediate layer states used by the
GRU do not need persistent storage between calls.

After a scheduled group, the CPU reads the classifier score and decides whether
to exit. This introduces one host decision per checked group, but avoids the
conditional backend's branch-stream allocations. It does not fix draft depth
for an entire block: every draft token still gets its own exit decision.

### Separate prompt prefill in the portable path

When CUDA graphs are disabled or unavailable on CPU, the model now prefills
the prompt except for its final token using the normal model forward pass.
The adaptive cache then starts with that final token.

Previously, the first adaptive round retained prompt activations for every
layer it processed. Now its activation storage covers only the speculative
suffix. The normal KV cache still holds the verified prompt context.

### Batched greedy verification

For greedy decoding without logits processors, the implementation computes
the whole verification block's token IDs together and transfers them to the
host once. It also avoids an unnecessary float conversion and clone when
scores do not need processing.

Prefix-dependent logits processors keep their sequential handling. Sampling
continues to use the existing acceptance-ratio and residual-distribution
correction logic. EOS handling, stopping criteria, and verified-token
streaming remain in place.

## Backend selection

| `--ee_gpu_branching` | Behavior |
|---|---|
| `auto` | New shared-pool CUDA graphs on CUDA; portable grouped checks on CPU. This is the default. |
| `off` | Portable grouped checks without CUDA graphs. |
| `on` | Existing experimental conditional graphs with GPU exit decisions. Requires the conditional-node API and retains the known memory limitation on the tested build. |

The change avoids the PyTorch conditional-graph issue in the default path; it
does not patch PyTorch or repair the explicit `on` backend. Restart a process
that has already accumulated conditional-graph pools before comparing memory.

## Measurements

### Small-model memory and latency probe

The synthetic probe used an eight-layer Llama with hidden size 512, vocabulary
size 1,024, FP16 weights, SDPA attention, a 32-token prompt, cache capacity 512,
and six speculative tokens. A threshold of zero selected the first scheduled
exit at layer 2. It measured drafting plus verification for a block, excluding
graph construction and acceptance processing.

| Measurement | Previous conditional backend | New grouped backend |
|---|---:|---:|
| Reserved memory before graph construction | 102 MiB | 102 MiB |
| Reserved memory after graph construction | 8,370 MiB | 150 MiB |
| Round latency | 5.108 ms | 4.301 ms |
| Reserved memory after rounds and engine cleanup | 8,346 MiB | 106 MiB |

These are PyTorch allocator measurements, not total device memory reported by
`nvidia-smi`. Reserved memory includes both allocated tensors and cached space.
Allocator behavior can vary between runs. Matching verification logits were
checked against the portable and fixed-depth implementations for the new
backend.

### Cached 7B model benchmark

This table records the November 11 measurements, before CUDA graphs were
added to fixed-depth `self_speculative`. Fixed-depth decoding now shares the
optimized backend; use [the comparison tool](SPECULATIVE_COMPARISON.md) to
measure the current implementations under matching conditions.

The real-model check used `facebook/layerskip-llama2-7B`, the local trained GRU
checkpoint, FP16 weights, SDPA attention, six speculative tokens, and 64 greedy
output tokens. The prompt was:

```text
Question: Why is the sky blue?
Answer:
```

The model was warmed before measurement. CUDA synchronization bracketed timed
generation; graph setup was measured separately. No streaming or logits
processors were used.

| Decoder | Draft-depth cap | Tokens/s | Acceptance rate | Graph/backend setup | Reserved GPU memory after generation |
|---|---:|---:|---:|---:|---:|
| Fixed-depth `self_speculative` | 8 | 51.24 | 43.4% | 0.000 s | 12,974 MiB |
| Adaptive, graphs `off` | 8 | 53.35 | 43.4% | 0.019 s | 12,924 MiB |
| New backend, `auto` | 8 | 67.98 | 43.4% | 0.338 s | 12,968 MiB |
| New backend, `auto` | All layers allowed | 70.75 | 51.1% | 0.730 s | 12,980 MiB |

All four runs produced the same 64 tokens as full-model greedy generation.
Reserved memory returned to 12,874 MiB after each engine was released and the
allocator cache was cleared.

For unrestricted adaptive drafting, the observed exits were layer 8 for 87
drafts, layer 12 for four drafts, layer 14 for one draft, and layer 32 for two
drafts. This explains why adaptive drafting remained relatively inexpensive
on this prompt.

This is a single-prompt diagnostic benchmark, not a dataset-wide speedup
claim. Performance depends on prompt length, acceptance rate, exit depths,
sampling settings, and hardware. The layer-8 rows provide the comparison with
matching draft-depth caps. Graph setup is excluded from tokens/s and still
contributes to first-request wall-clock latency.

## Usage

```bash
python generate.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 128 \
  --sample False
```

Enter a prompt and press Ctrl+D twice to generate. Add `--exit_layer 8` to cap
draft depth at layer 8; full-model verification still uses every layer.

The model must reside on one device and use eager or SDPA attention. The
checkpoint must match the base model. No cuBLAS workspace environment-variable
override is needed for the new backend.

Static KV capacity still covers the prompt plus the requested output budget,
rounded up to a 128-token bucket. Attention receives that reserved capacity,
with future positions masked. The change does not eliminate that overhead.
Graph instances are intended for one request/stream at a time, not concurrent
replay. Larger speculation counts and deeper draft exits can still increase
cost.

## Validation and implementation files

Validation completed with 104 passing tests in the focused CPU run, seven
passing tests for the new CUDA backend, and four passing tests for the
existing conditional backend: 115 passing tests in total. Coverage includes
changing exit depths, rejection rollback, eager and SDPA attention, zero-draft
generation, sampling with `top_k=1`, prompt-state storage, shared-pool reuse,
and memory release after repeated engine reconstruction.

Run the dedicated CUDA regressions on a CUDA-enabled machine:

```bash
python -m pytest -q tests/test_early_exit_grouped_graphs.py
python -m pytest -q tests/test_early_exit_latency.py -k cuda
```

The principal files are:

- [early_exit_cuda_graphs.py](self_speculation/early_exit_cuda_graphs.py): new
  `GroupedDraftGraphs` backend and existing experimental conditional backend.
- [early_exit_speculation_generator.py](self_speculation/early_exit_speculation_generator.py):
  backend selection, separate portable prefill, and batched greedy verification.
- [generate.py](generate.py): updated backend option help.
- [test_early_exit_grouped_graphs.py](tests/test_early_exit_grouped_graphs.py):
  CUDA correctness and memory-release regressions.
- [test_early_exit_speculation_generator.py](tests/test_early_exit_speculation_generator.py):
  regression coverage for keeping prompt activations out of adaptive storage.
