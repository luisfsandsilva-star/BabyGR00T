# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements (includes Lion optimizer)
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Logging & experiment tracking

- The pretraining script exposes a `use_wandb` flag (enabled by default) plus `log_dir`/`log_file` paths in `config/cfg_pretrain.yaml`. Override them from the command line with Hydra, for example `python pretrain.py use_wandb=false log_dir=logs/run_01`.
- When `use_wandb=true`, training behaves as before and syncs metrics to the configured W&B project/run.
- When `use_wandb=false`, W&B calls are skipped. Rank 0 creates `<log_dir>/<log_file>` (defaults to `logs/pretrain.log`) and appends flattened `train/*` and `val/*` metrics to it while mirroring the per-step values and epoch summaries to the console via `tqdm.write`, so the progress bar stays intact.
- At the end of runs with local logging enabled, the trainer loads the `.log` file with pandas/matplotlib and saves `train_vs_val_loss.png` alongside the checkpoints (or under `<log_dir>` when checkpointing is disabled). The plot and log capture the full history of epoch averages (`train_loss`/`val_loss`) so you can inspect learning progress without W&B.
- By default the trainer runs a quick sanity check on the first batch of every split (train/eval) to validate shapes, dtypes and NaN/inf issues. Disable it with `skip_sanity_checks=true` if you need to speed up start-up (e.g. when debugging custom loaders).

### Checkpointing & resuming training

- All checkpoints live under `checkpoint.path` (defaults to `checkpoints/<project>/<run>`). Each evaluation step writes a numbered file (`step_00000010.ckpt`) plus an updated `latest.txt` pointer. When the monitored metric improves the same payload is mirrored to `best.ckpt`.
- Set `checkpoint.monitor` (default `val/loss`) and `checkpoint.mode` (`min` or `max`) to control which validation metric defines “best”. The monitor string is matched against the flattened validation metrics (`val/regression_loss`, `eval/arc/accuracy`, …). Disable automatic best-tracking by setting `checkpoint.monitor=null`.
- Use `checkpoint.keep_last=<N>` to keep only the most recent `N` step checkpoints while still retaining `best.ckpt`. Leave it as `null` to keep the full history.
- To resume, point Hydra to the directory and flip `checkpoint.resume=true`. By default the trainer reloads `latest.txt`. Override the exact file with `checkpoint.resume_from=path/to/checkpoint.ckpt`.
- Examples:
  - `python pretrain.py checkpoint.path=/tmp/run_01 checkpoint.monitor=val/accuracy checkpoint.mode=max`
  - `python pretrain.py checkpoint.path=/tmp/run_01 checkpoint.resume=true`
  - `python pretrain.py checkpoint.path=/tmp/run_01 checkpoint.resume_from=best.ckpt`
- Resuming restores the model weights, every optimizer in the stack, tracked schedulers, the EMA shadow weights (when enabled), and the training counters (`step`, `epoch`, `best_val_metric`). Training picks up at the next evaluation window using the restored learning-rate schedule and progress bar state.

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

> **Regression datasets:** When exporting new data, ensure the generated `dataset.json` includes the extended metadata (`task_type`, `input_dim`, `target_dim`, `input_pad_value`, and `target_pad_value`). Regression tasks must set `task_type: "regression"`, provide float32 `targets`/`target_mask` arrays (saved as `.npy`/`.npz`), and train with the `losses@ACTRegressionLossHead` loss head (configurable via `arch.loss` in the YAML configs). The loader first looks for `.npy` files and automatically falls back to sibling `.npz` archives; when saving `.npz`, either store a single unnamed array or place the data under a key that matches the field name (e.g., `targets` or `target_mask`).

> **Latent episode datasets (.npz):** `PuzzleDataset` can also stream autoregressive targets directly from episode archives when a split directory omits `dataset.json`. Place each task under `data_path/<split>/` with two folders: `latents/episode_XXX.npz` (containing a `latents.npy` array saved via `np.savez`) and optional metadata files in `metadata/episode_XXX.json`. Metadata files may specify the `set` name, a `group`, and a `puzzle_identifier`; unspecified identifiers are auto-incremented. At load time the dataset infers the latent dimensionality, builds prefix sums across episodes, and creates lazy views for inputs, targets (the next latent for teacher forcing), and boolean masks. Multiple episode folders can be supplied through `data_paths=[...]`—puzzle identifiers are offset automatically to avoid collisions across tasks.

## Experiments

### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
