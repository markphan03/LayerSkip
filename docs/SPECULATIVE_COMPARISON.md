# Comparing fixed-depth and adaptive speculative decoding

Both `self_speculative` and `early_exit_speculative` now use the same shared
cache, exact verification loop, and ordinary CUDA graph backend. This makes
it possible to compare their draft policies with similar implementation
optimizations.

Fixed-depth drafting runs to `exit_layer` every time. It does not load a GRU
checkpoint, allocate GRU state, or read classifier scores on the host. Adaptive
drafting uses the trained classifier to choose a depth for each draft token.
The full model verifies both strategies' drafts.

## Run a repeated comparison

```bash
python compare_speculative.py \
  --model facebook/layerskip-llama2-7B \
  --ee-checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --exit-layer 8 \
  --num-speculations 6 \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 5 \
  --local-files-only \
  --output /tmp/speculative-comparison.json
```

Remove `--local-files-only` if the model and tokenizer are not already cached.
The classifier checkpoint must be available locally and match the model.
The default prompt is `Question: Why is the sky blue?` followed by a newline
and `Answer:`. Repeat `--prompt` to measure additional prompts.

The tool measures four configurations on CUDA:

| Draft policy | Graph mode | Exit behavior |
|---|---|---|
| Fixed | `off` | Always uses the fixed depth |
| Fixed | `auto` | Same fixed depth, with CUDA graphs |
| Adaptive | `off` | GRU-selected depths with portable execution |
| Adaptive | `auto` | GRU-selected depths with shared-pool CUDA graphs |

Both `off` and `auto` use the shared verification and cache implementation, so
their comparison measures the benefit of CUDA graphs within the current code.
The older `single_step_speculation` method remains available as a compatibility
API; normal generation uses the shared implementation.

Adaptive drafting allows all model layers by default. Set
`--adaptive-exit-layer 8` to match a fixed depth cap of 8. On a 32-layer model
with the default initial fraction of 0.25, the first adaptive check is already
at layer 8, so this particular cap removes adaptive depth selection. Use a
larger common cap or a smaller `--ee-initial-fraction` to compare policies that
can choose among several depths under the same cap.

## Timing and correctness

- Graph/backend setup is measured separately from generation. Model and
  tokenizer loading, tokenization, and classifier checkpoint loading are
  outside both reported measurements.
- Each configuration receives the requested warmup generations. The report
  shows median generation time and median tokens per second over the timed
  repetitions, plus the mean acceptance rate.
- CUDA is synchronized before and after timed generation. Generation includes
  prompt prefill, drafting, verification, and token selection, with no text
  streamer. Graph setup contributes to first-request latency even though it
  is excluded from generation throughput.
- Greedy decoding is the default. Every timed greedy output is checked against
  full-model greedy generation. EOS may cause fewer tokens than the requested
  maximum; per-run token counts are included in the JSON report.
- Add `--sample`, `--temperature`, `--top-k`, and `--top-p` for sampling. Each
  configuration uses the same seed schedule, but sampled token sequences can
  differ because draft policies consume randomness differently. No equality
  assertion is made for sampled outputs.
- CUDA memory columns report PyTorch peak allocated and reserved memory,
  including the loaded model. They are not total device usage from
  `nvidia-smi`. Engines are released between configurations.

Use an otherwise idle GPU for timing. Compare several prompts, acceptance
rates, and speculation counts before drawing a general speedup conclusion.
Different depth policies can produce different accepted tokens per round,
even when the final greedy text is identical.

On CPU, `--device cpu` runs only the two portable configurations. CUDA graph
acceleration requires one CUDA device and eager or SDPA attention; the script
loads the model with SDPA.

## Interactive fixed-depth generation

```bash
python generate.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy self_speculative \
  --ss_cuda_graphs auto \
  --exit_layer 8 \
  --num_speculations 6 \
  --max_steps 512 \
  --sample False
```

## Early-Exit generation

```bash
python generate.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample False
```

Use `--ss_cuda_graphs off` for portable fixed-depth execution. The equivalent
adaptive graph option is `--ee_gpu_branching auto`. The comparison tool uses
the ordinary shared-pool backend, not experimental conditional graphs.

## Measured example

On November 12, 2026, the command above was run on an RTX 4090 with PyTorch
2.14.1+cu130, Transformers 4.50.0, the cached LayerSkip Llama-2 7B model, and
the trained GRU checkpoint. It used the default sky-blue prompt, 64 greedy
output tokens, six speculative tokens, one warmup, and five timed repetitions.

| Strategy | Graphs | Draft cap | Setup (s) | Median generation (s) | Median tokens/s | Acceptance | Reserved MiB |
|---|---|---|---:|---:|---:|---:|---:|
| Fixed | Off | 8 | 0.002 | 1.147 | 55.78 | 43.4% | 12,922 |
| Fixed | Auto | 8 | 0.335 | 0.901 | 71.02 | 43.4% | 12,968 |
| Adaptive | Off | All layers | 0.024 | 1.172 | 54.59 | 51.1% | 12,936 |
| Adaptive | Auto | All layers | 0.705 | 0.891 | 71.82 | 51.1% | 12,980 |

All timed greedy outputs matched full-model generation. Fixed-depth CUDA
graphs improved throughput by about 27% over the current portable fixed-depth
path on this prompt. With both strategies using graphs, their throughput was
close: the approximately 1% difference is too small to establish a general
advantage from this single-prompt measurement. The adaptive strategy also had
a larger allowed depth and a higher acceptance rate.

These results supersede comparisons in which only adaptive decoding received
CUDA graph acceleration. Use additional prompts and repeated runs to compare
the policies, and include setup time when first-request latency matters.

## Implementation and regression checks

- [self_speculation_generator.py](self_speculation/self_speculation_generator.py)
  selects fixed-depth drafting with no classifier.
- [early_exit_speculation_generator.py](self_speculation/early_exit_speculation_generator.py)
  contains the shared `VerifiedSpeculativeGenerationStrategy` and adaptive
  classifier setup.
- [early_exit_cuda_graphs.py](self_speculation/early_exit_cuda_graphs.py) supports
  fixed-depth graphs with a single draft exit and no GRU state.
- [generator_base.py](self_speculation/generator_base.py) measures completed
  GPU work using synchronization and a monotonic timer.
- [compare_speculative.py](compare_speculative.py) produces the repeated
  comparison and optional JSON report.
- [test_fixed_speculative_graphs.py](tests/test_fixed_speculative_graphs.py)
  checks greedy and top-k-one sampled outputs, rejection, logits processors,
  full-depth drafting, short budgets, CUDA replay without classifier checks,
  and timing boundaries.

```bash
python -m pytest -q tests/test_fixed_speculative_graphs.py \
  tests/test_early_exit_grouped_graphs.py
```
