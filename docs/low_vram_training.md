# Low-VRAM Training Playbook

This guide explains how to fine-tune `vlm-gym` vision-language models on GPUs with as little as **8 GB** of memory. The core idea is to keep the 4B foundation model frozen while training lightweight LoRA adapters and enabling activation checkpointing to reduce peak memory use.

## Step-by-step workflow

### Step 0 – Prerequisites
1. Install the [UV](https://github.com/astral-sh/uv) package manager (or use `pipx install uv`).
2. Ensure you have a Qwen3-VL 4B checkpoint synced locally under `checkpoints/qwen3vl_4b` (follow the instructions in the project README if you need to download it).
3. Verify your GPU driver and CUDA/JAX stack recognises the device (`nvidia-smi`).

### Step 1 – Create an isolated environment
```bash
uv run --python 3.10 --no-sync -- python -V
```

The first invocation creates `.venv/` with the dependencies declared in `pyproject.toml`. Subsequent `uv run` calls reuse the environment.

### Step 2 – Configure LoRA adapters

Enable lightweight adapters on the dense projections inside the Qwen3-VL text stack:

```bash
--lora_rank 8 \
--lora_targets q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--lora_freeze_base 1
```

* `lora_rank` selects the adapter rank. Ranks between 4 and 16 work well for 8 GB cards.
* `lora_targets` lists the dense layers that receive adapters. You can trim this list to save more memory or focus on specific blocks.
* `lora_freeze_base=1` keeps the frozen backbone on device while optimising only the adapters (and optionally biases / lm_head).
* Optional knobs:
  * `--lora_train_bias 1` also updates bias terms (small overhead, useful for stability).
  * `--lora_train_lm_head 1` trains the LM head alongside adapters for faster convergence.

Behind the scenes, the trainer splits the parameter tree so the optimiser tracks only adapter weights, slashing optimizer memory and eliminating gradient storage for the 4 B backbone.

### Step 3 – Turn on activation checkpointing

Set `--policy_checkpoint 1` to wrap the PPO policy forward pass in `jax.checkpoint`. The additional compute trades for dramatically lower activation memory, which is essential once the batch size is reduced to 1.

### Step 4 – Shrink rollout and PPO batches

Use tiny batches that fit on a small GPU:

```bash
--batch_size 1 \
--groups_per_batch 1 \
--group_size 1 \
--ppo_minibatch 1
```

These settings pair naturally with LoRA because only a handful of tokens need gradients. Increase `ppo_epochs` if you need more optimisation steps per rollout.

### Step 5 – Launch the training run

```bash
uv run python core/train.py \
  --model_dir checkpoints/qwen3vl_4b \
  --env_name geospot \
  --total_steps 4000 \
  --batch_size 1 \
  --ppo_minibatch 1 \
  --groups_per_batch 1 \
  --group_size 1 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_targets q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_freeze_base 1 \
  --lora_train_bias 1 \
  --policy_checkpoint 1
```

This run keeps the 4B backbone frozen, only learns ~20 M LoRA weights, and uses activation checkpointing so that peak VRAM stays under 8 GB on consumer GPUs (tested on RTX 3070 and 4060 laptop SKUs). Expect slower tokens/sec than on a large accelerator, but PPO remains stable.

### Step 6 – Monitor and iterate

* Use a slightly higher learning rate (e.g. `5e-5` to `1e-4`) because LoRA has far fewer parameters.
* Monitor KL and entropy—the adapters can overfit quickly. Consider enabling adaptive KL (`--adaptive_kl 1`).
* Save checkpoints frequently (`--save_interval 50`) so you can resume without reinitialising adapters.
* For inference, load the same checkpoint: frozen base weights plus trained adapters produce the adapted behaviour with minimal overhead.

With these switches you can iterate on environments and reward functions using affordable GPUs, while retaining compatibility with full-sized training runs on larger hardware.
