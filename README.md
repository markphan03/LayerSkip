# LayerSkip
<a href='https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![License: CC BY-NC](https://img.shields.io/badge/License-CC_BY--NC-lightgrey.svg)](./LICENSE) [![Slides](https://img.shields.io/badge/Slides-PPT-orange)](https://docs.google.com/presentation/d/1r33OxlXsv-eI__9Jx9w2vQNYYzuV0bng2IJHKbKo3WE/edit?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=oPxdfVVmLP8) [![arXiv](https://img.shields.io/badge/arXiv-2404.16710-b31b1b.svg)](https://arxiv.org/abs/2404.16710) [![alphaXiv](https://img.shields.io/badge/alphaXiv-2404.16710-9a2037.svg)](https://www.alphaxiv.org/abs/2404.16710)

This code base is the implementation of [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710).

<div align="center">
  <img src="https://github.com/user-attachments/assets/1fdd91d9-37ea-4b42-b5be-579fb5e1f2f2" width="500">
</div>

## News
- [2024/11] 🤗 LayerSkip inference has been integrated into [Hugging Face `transformers`](https://huggingface.co/blog/layerskip).
- [2024/12] 🔥 LayerSkip training recipe has been integrated into [PyTorch `torchtune`](https://github.com/pytorch/torchtune/pull/1076).
- [2025/03] 🤗 LayerSkip training recipe implemented in [Hugging Face `trl`](https://github.com/huggingface/trl/pull/3111)

## Getting Started
- Clone repo:
```console
$ git clone git@github.com:facebookresearch/LayerSkip.git
$ cd LayerSkip
```

- Setup environment:
```console
$ conda create --name layer_skip python=3.10
$ conda activate layer_skip

$ pip install -r requirements.txt
```

- Access models:
In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe. We provide 6 checkpoints on [HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a) of different Llama models continually pretrained using the LayerSkip recipe:

    - [`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
    - [`facebook/layerskip-llama2-13B`](https://huggingface.co/facebook/layerskip-llama2-13B)
    - [`facebook/layerskip-llama2-70B`](https://huggingface.co/facebook/layerskip-llama2-70B)
    - [`facebook/layerskip-codellama-7B`](https://huggingface.co/facebook/layerskip-codellama-7B)
    - [`facebook/layerskip-codellama-34B`](https://huggingface.co/facebook/layerskip-codellama-34B)
    - [`facebook/layerskip-llama3-8B`](https://huggingface.co/facebook/layerskip-llama3-8B)
    - [`facebook/layerskip-llama3.2-1B`](https://huggingface.co/facebook/layerskip-llama3.2-1B)

In order to access each model:

1. Visit the model's corresponding link above, make sure you are logged on the HuggingFace website with your account.
2. Fill the request form and submit it. Approval may take a while and you should receive an email notification to notify you that permission to the model is granted.
3. Follow the steps [here](https://huggingface.co/docs/hub/en/security-tokens) to obtain a user access token.
4. In the command-line run `huggingface-cli login`, and you will be prompted to provide the token you have obtained in Step 3.

Once you run those steps, the commands below to run the LayerSkip checkpoints should work.

## Generate

To run one of our models in interactive mode using regular autoregressive decoding:
```console
$ python generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512
```

In order to observe speedup, you need to use self-speculative decoding to generate tokens, and specify `--exit_layer`, the layer the draft stage to exit at, and `--num_speculations`, the number of draft tokens:
```console
$ python generate.py --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6
```

Tips:
- `self_speculative` now uses shared-pool CUDA graphs by default on CUDA. Add
  `--ss_cuda_graphs off` to disable graphs. Fixed-depth drafting does not load
  a GRU checkpoint or perform classifier exit checks. Both modes require the
  model on one device with eager or SDPA attention.
- You may change `--model` to any HuggingFace model but in order to observe speedup with self-speculative decoding, use a model trained using the LayerSkip recipe, such as those we have [open sourced on HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a).
- By default we enable sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python generate.py --help` for details on different command-line arguments.

To choose each draft token's exit layer using the trained early-exit GRU:

```console
$ python generate.py \
    --model facebook/layerskip-llama2-7B \
    --sample True \
    --max_steps 512 \
    --generation_strategy early_exit_speculative \
    --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
    --ee_threshold 0.5 \
    --ee_initial_fraction 0.25 \
    --ee_gpu_branching auto \
    --num_speculations 6
```

No distributed environment variables are required for `python generate.py`.
Launching with `torchrun` is also supported; only global rank 0 generates text.

Enter a question or completion prompt using the same interactive input as the
fixed-layer command. For this base language model, a prompt such as
`Question: Why is the sky blue?` followed by a newline and `Answer:` is useful.
The checkpoint above was trained with `facebook/layerskip-llama2-7B` on
CNN/DailyMail summaries; use the matching base model.

Exit checks begin after `ceil(total_layers * ee_initial_fraction)` layers.
Each subsequent increment halves, rounded up to at least one layer; once the
increment reaches one, checks continue at every layer. With 35 layers and
`--ee_initial_fraction 0.25`, the checks are at layers 9, 14, 17, 19, 20, ..., 35.
Fractions refer to the full model depth. The argument accepts values in `(0, 1]`.
The GRU consumes every intermediate layer's raw state in grouped calls and exits
at the first scheduled score reaching `--ee_threshold`. Every draft token has
its own exit decision; depth is never fixed for a whole speculative block.

If no check passes, drafting reaches the final layer. Optionally add
`--exit_layer 8` to cap draft depth at layer 8; verification still uses all
layers. `--num_speculations` sets the maximum number of drafts per round.
The arguments shown above are the defaults except for the speculation count.

The classifier uses the same device and floating-point dtype as Llama.
`--ee_gpu_branching auto` uses ordinary CUDA graphs with one host exit check per
scheduled decoder group. The graphs share one memory pool and capture stream;
each group computes only its missing suffix. This preserves per-token adaptive
exits and cache reuse while avoiding the conditional backend's many branch
streams and BLAS workspaces. Rejected suffixes are logically rolled back.
CUDA conditional-node support is not required. On CPUs, `auto` uses the portable
implementation. Use `off` to disable CUDA graphs.

`--ee_gpu_branching on` explicitly selects the experimental conditional backend
for exit decisions on the GPU. It requires the CUDA conditional-node API and
can consume substantially more memory. On the tested PyTorch 2.14.1/CUDA 13
build, conditional graph pools remain reserved after deletion; changing prompt
capacity or settings can accumulate memory. Prefer `auto` for normal use and
restart the process after experimenting with conditional graphs to reclaim it.

CUDA graph capture/warmup is a setup cost,
performed before the generation timer by `HuggingfaceLlamaGenerator`; captured
graphs are reused when the model, capacity bucket, and decoding settings match.
It reserves KV storage for the prompt plus requested output budget, rounded up
to a 128-token bucket. Large speculation counts increase graph capture time
and memory. The portable path prefills the prompt separately, so adaptive
activation storage covers only the short speculative suffix. For latency
comparisons, match the model, prompt, draft-depth cap, speculation count, and
sampling settings; allowing adaptive drafts to reach all layers can cost more
than a fixed shallow draft.

The full model verifies each block, with acceptance/rejection correction when
sampling. KV caches and raw states are reused as draft depth changes, and
rejected positions are removed before the next round. The command reports
acceptance rate and throughput. See [Early-exit optimization](EARLY_EXIT_OPTIMIZATION.md)
for implementation details, validation, and a single-prompt 7B benchmark. This mode
requires a Llama model on one device with eager or SDPA attention. The original
`self_speculative` strategy continues to use a fixed exit layer.

### Generate the same prompt N times

`generate_n_times.py` loads the model once and generates independent completions
of the same prompt. For adaptive early-exit speculation:

```bash
python generate_n_times.py \
    --model facebook/layerskip-llama2-7B \
    --generation_strategy early_exit_speculative \
    --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
    --ee_gpu_branching auto \
    --ee_threshold 0.8 \
    --ee_initial_fraction 0.25 \
    --num_speculations 6 \
    --max_steps 512 \
    --sample True --N 100 \
    --prompt "What is the capital of France?"
```

For fixed-depth self-speculative decoding, specify the draft exit layer:

```bash
python generate_n_times.py \
    --model facebook/layerskip-llama2-7B \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --max_steps 512 \
    --sample True --N 100 \
    --prompt "What is the capital of France?"
```

Fixed-depth mode does not require an early-exit checkpoint. Both commands print
completion text only for runs 1–4 (`n < 5`, using 1-based run numbers), including
when streaming is enabled. Every run prints generation time, token count, time
per token (seconds/token), throughput, and acceptance rate, followed by aggregate
metrics for the N runs. Aggregate time per token is total generation time divided
by total generated tokens; zero-token results display `N/A`.
Add `--output logs/generations.json` to save all completions, settings,
text, token IDs, per-run seeds, and metrics. Runs use `--seed + run_index`
(zero-based; default seed 42), so sampled runs have separate reproducible random
streams. Greedy decoding (`--sample False`) normally repeats the same output.
`--max_steps` is the per-run token limit; EOS can stop a completion earlier.

Optional `--warmup 1` performs an extra unreported generation before the N runs
without changing their seeds. Reported generation timing excludes model loading,
tokenization, decoding, graph setup, and warmup runs. Aggregate throughput is
total generated tokens divided by total generation time; acceptance is the mean
of the per-run rates. Text streaming is off by default; use `--streamer standard`
or `--streamer speculative` to enable it (streaming adds overhead to timing).
The script also supports `--generation_strategy autoregressive`.

## Benchmark

To compare fixed-depth and adaptive self-speculation with the same graph/cache
backend, run:

```bash
python compare_speculative.py \
    --model facebook/layerskip-llama2-7B \
    --ee-checkpoint checkpoints/ee_cnn/best_model/final.pt \
    --exit-layer 8 --num-speculations 6 \
    --max-new-tokens 64 --warmup 1 --repeats 5 \
    --output /tmp/speculative-comparison.json
```

This measures both strategies with graphs enabled and disabled, reports setup
separately, and checks greedy outputs against full-model generation. Adaptive
drafts may use all layers by default; add `--adaptive-exit-layer 8` to match the
fixed draft-depth cap. See [Speculative decoding comparison](SPECULATIVE_COMPARISON.md)
for timing methodology and command options.

To benchmark on a dataset:

```console
$ torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 100 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- You can specify different tasks by modifying the `--dataset` argument:
    - `cnn_dm_summarization`: CNN/DM Summarization
    - `xsum_summarization`: XSUM Summarization
    - `cnn_dm_lm`: CNN/DM Language Modeling (given the first few words of an article, generate the remaining article)
    - `human_eval`: HumanEval Coding
- By default, the tasks run as 0-shot. You can change to any specified `n`-shot by specifying the `--n_shot` argument.
- By default we enable sampling, while the results reported in the paper were greedy decoding without sampling. You may change the sampling behaviour using the `--sample`, `--temperature`, `--top_p`, and `--top_k` arguments.
- You may run `python benchmark.py --help` for details on different command-line arguments.

## Evaluate

We have integrated our generation scripts with [Eleuther Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to enable a large number of tasks and properly post-process generated text.

```console
$ torchrun eval.py --model facebook/layerskip-llama2-7B \
    --tasks gsm8k \
    --limit 10 \
    --generation_strategy self_speculative \
    --exit_layer 8 \
    --num_speculations 6 \
    --output_dir ./logs
```

Tips:
- Note that with speculative decoding we can only obtain speedups from generation tasks (e.g., `gsm8k` or `cnn_dailymail`), while classificaton tasks, i.e., multiple choice question tasks (e.g., `piqa`, `social_iqa`) or True/False question tasks (e.g., `boolq`) will not lead to speedup.
- You can specify arbitrary number of tasks supported by Eleuther Evaluation Harness using the `--tasks` argument. To get a list of all of possible tasks, check this [link](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).
- Similar to the `generate.py` and `benchmark.py` scripts, you may specify different models, datasets, and sampling parameters
- You may run `python benchmark.py --help` for details on different command-line arguments.

## Sweep
Our inference hyperparameters, `exit_layer` and `num_speculations` determine the speedup during inference:
- `exit_layer`:
    - smaller means a faster but less accurate draft stage
    - larger means a more accurate but slower draft stage
- `num_speculations`:
    - smaller means higher acceptance rate but verification stage will amortize less the draft stage
    - learger means verification stage will better amortize the draft stage but acceptance rate decreases

The optimal combination of `exit_layer` and `num_speculations` may change with the model, dataset and sampling parameters. Hence, we provided a script to sweep over a grid of different `exit_layer` and `num_speculations`:

```console
$ torchrun sweep.py --model facebook/layerskip-llama2-7B \
    --dataset human_eval \
    --generation_strategy self_speculative \
    --num_samples 150 \
    --max_steps 256 \
    --output_dir ./logs/ \
    --sample False
```

This will create a CSV file in the directory specified in the `--outpu_dir` argument.

Tips:
- Similar to the `generate.py` and `benchmark.py` scripts, you may specify different models, datasets, and sampling parameters
- You may run `python sweep.py --help` for details on different command-line arguments.

## Correctness
In order to verify that the generated tokens of our self-speculative decoding algorithm are correct, we have created a script to compare the outputs of autoregressive decoding with self-speculative decoding. Note that the outputs we can only guarantee equivalence when there is no sampling (i.e., `--sample False`): 
```console
$ torchrun correctness.py --model facebook/layerskip-llama2-7B \
    --dataset human_eval \
    --generation_strategy self_speculative \
    --num_speculations 6 \
    --exit_layer 4 \
    --num_samples 10 \
    --sample False \
    --output_dir ./logs
```

## Using Docker
Kindy check [DOCKER.md](DOCKER.md) to setup the project using docker

## Other Implementations
We also have other implementations of LayerSkip inference:
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast/tree/LayerSkip?tab=readme-ov-file#self-speculative-sampling): gpt-fast is a simple and efficient pytorch-native transformer text generation. We have implemented LayerSkip in the gpt-fast codebase to enable compouding it with other optimizations such as `torch.compile()`, quantization, and tensor parallelism.
- [Native HuggingFace](https://huggingface.co/facebook/layerskip-llama2-7B#huggingface): in the model card of each of our HuggingFace models, we have provided simple code snippets that leverages HuggingFace speculative decoding capabilities using a simple trick to clone the earlier layers of the main model without cloning its weights. Although this implementation is simple and does not require implementing other functions or importing other libraries, it does not share the KV cache or execution between the draft and verification stages.

## Training
Our training implementation is work-in-progress. You can check this [pull request](https://github.com/pytorch/torchtune/pull/1076) for details and discussions.

## Tests
To run unit/integration tests:
```
$ pytest ./tests/
```

## License
LayerSkip is licensed under CC-by-NC license. Refer to the LICENSE file in the top level directory.

## Contributing
We welcome contributions to LayerSkip. If you are interested in contributing please see [this document](./CONTRIBUTING.md).

## Citation
If you use LayerSkip in your research, please use the following BibTex entry:

```bibtex
@misc{layerskip,
    title={LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
    author={Mostafa Elhoushi and Akshat Shrivastava and Diana Liskovich and Basil Hosmer and Bram Wasti and Liangzhen Lai and Anas Mahmoud and Bilge Acun and Saurabh Agarwal and Ahmed Roman and Ahmed A Aly and Beidi Chen and Carole-Jean Wu},
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.681",
    doi = "10.18653/v1/2024.acl-long.681",
    pages = "12622--12642",
}
```
