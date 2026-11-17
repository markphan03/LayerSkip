# Experiment Commands

## 1. Paris — EEC, exit layer 13

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee-classifier-start-layer 1 \
  --exit_layer 13 \
  --ee_threshold 0.8 \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "What is the capital of France?" \
  --output experiment_logs/llama2-7b-Paris-eec13-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 2. Paris — EEC, exit layer 8

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee-classifier-start-layer 1 \
  --exit_layer 8 \
  --ee_threshold 0.8 \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "What is the capital of France?" \
  --output experiment_logs/llama2-7b-Paris-eec8-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 3. Paris — EEC, maximum exit depth

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee-classifier-start-layer 1 \
  --ee_threshold 0.8 \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "What is the capital of France?" \
  --output experiment_logs/llama2-7b-Paris-eecmax-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 4. Paris — Fixed exit layer 13

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy self_speculative \
  --exit_layer 13 \
  --num_speculations 6 \
  --ss_cuda_graphs auto \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "What is the capital of France?" \
  --output experiment_logs/llama2-7b-Paris-fe13-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 5. Two Sum — EEC, exit layer 13

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee-classifier-start-layer 1 \
  --exit_layer 13 \
  --ee_threshold 0.8 \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "Write for me Two Sum solution from leetcode?" \
  --output experiment_logs/llama2-7b-twosum-eec13-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 6. Two Sum — EEC, maximum exit depth

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy early_exit_speculative \
  --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
  --ee-classifier-start-layer 1 \
  --ee_threshold 0.8 \
  --ee_gpu_branching auto \
  --num_speculations 6 \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "Write for me Two Sum solution from leetcode?" \
  --output experiment_logs/llama2-7b-twosum-eecmax-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 7. Two Sum — Fixed exit layer 13

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy self_speculative \
  --exit_layer 13 \
  --num_speculations 6 \
  --ss_cuda_graphs auto \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "Write for me Two Sum solution from leetcode?" \
  --output experiment_logs/llama2-7b-twosum-fe13-threshold08-noc6-N50-maxstep-512-sample-true.txt
```

## 8. Two Sum — Fixed exit layer 8

```bash
python generate_n_times.py \
  --model facebook/layerskip-llama2-7B \
  --generation_strategy self_speculative \
  --exit_layer 8 \
  --num_speculations 6 \
  --ss_cuda_graphs auto \
  --max_steps 512 \
  --sample True \
  --N 50 \
  --warmup 1 \
  --prompt "Write for me Two Sum solution from leetcode?" \
  --output experiment_logs/llama2-7b-twosum-fe8-threshold08-noc6-N50-maxstep-512-sample-true.txt
```
