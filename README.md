# ActionUNet: Improving Robustness of VLA Models with Efficient Multi-scale Fine-tuning

<p align="center">
  <strong>Official PyTorch implementation for NeurIPS 2026</strong>
</p>

ActionUNet fine-tunes a π0.5 vision-language-action model for LIBERO, LIBERO-Plus, and RoboTwin. This README is written as a reproducible, copy-and-run guide. Follow the numbered sections in order.

## Published training settings

`--batch-size` always means the **global batch size across all DDP processes**.

| Benchmark | GPUs | Global batch size | Batch size per GPU | Training steps |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | **8** | **256** | 32 | 30,000 |
| RoboTwin | **2** | **64** | 32 | 8,000 |

The exact training commands are in [Step 4: Train on LIBERO](#step-4-train-on-libero) and [Step 5: Train on RoboTwin](#step-5-train-on-robotwin). Do not change `--nproc-per-node` without also checking that the global batch size is divisible by the number of processes.

## Before you start

You need:

- Linux;
- NVIDIA GPUs and a working CUDA driver;
- Git and `curl`;
- enough disk space for the π0.5 checkpoint, datasets, and training checkpoints;
- Python 3.11 for ActionUNet training and serving.

All commands below assume Bash. Run each code block from top to bottom. Replace only paths beginning with `/absolute/path/to/...`.

## Step 1: Clone and install ActionUNet

Choose where to install the repository:

```bash
export ACTIONUNET_ROOT="$HOME/ActionUNet"

git clone https://github.com/Di-Zhu123/ActionUNet.git "$ACTIONUNET_ROOT"
cd "$ACTIONUNET_ROOT"
```

Install `uv`, Python 3.11, and the project dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version
uv python install 3.11
uv sync
```

Copy the bundled transformer replacements into the installed `transformers` package:

```bash
cd "$ACTIONUNET_ROOT"

TRANSFORMERS_DIR="$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
cp -a src/openpi/models_pytorch/transformers_replace/. "$TRANSFORMERS_DIR/"
```

Verify the installation before downloading data:

```bash
uv run python -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())'
uv run python -m actionunet.train --help
uv run python -m actionunet.serve --help
```

The vendored `src/openpi/` source is based on [OpenPI commit `c23745b`](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79). Use that same revision when converting an official JAX π0.5 checkpoint.

## Step 2: Prepare the π0.5 base checkpoint

ActionUNet training requires a PyTorch π0.5 checkpoint directory containing `model.safetensors`.

If you already have the converted checkpoint, set its absolute path and verify it:

```bash
export BASE_WEIGHTS="/absolute/path/to/pytorch/pi05_base"
test -f "$BASE_WEIGHTS/model.safetensors"
```

The `test` command prints nothing when the file exists. Check the exit code if needed:

```bash
echo $?
```

An output of `0` means the checkpoint is ready. If you only have the official JAX checkpoint, use OpenPI's converter at the pinned revision:

```bash
export OPENPI_CONVERTER_ROOT="/absolute/path/to/openpi-c23745b"
export JAX_CHECKPOINT="/absolute/path/to/jax/pi05_base"
export BASE_WEIGHTS="/absolute/path/to/pytorch/pi05_base"

git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_CONVERTER_ROOT"
git -C "$OPENPI_CONVERTER_ROOT" checkout c23745b5ad24e98f66967ea795a07b2588ed6c79

cd "$ACTIONUNET_ROOT"
uv run python "$OPENPI_CONVERTER_ROOT/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "$JAX_CHECKPOINT" \
  --config-name pi05_aloha \
  --output-path "$BASE_WEIGHTS" \
  --precision bfloat16

test -f "$BASE_WEIGHTS/model.safetensors"
```

## Step 3: Choose the data and checkpoint directories

Set these variables once in every new shell:

```bash
export ACTIONUNET_ROOT="$HOME/ActionUNet"
export HF_LEROBOT_HOME="/absolute/path/to/lerobot"
export BASE_WEIGHTS="/absolute/path/to/pytorch/pi05_base"
export CHECKPOINT_ROOT="/absolute/path/to/actionunet_checkpoints"

mkdir -p "$HF_LEROBOT_HOME" "$CHECKPOINT_ROOT"
cd "$ACTIONUNET_ROOT"
```

The examples below use these variables instead of repeating long paths.

## Step 4: Train on LIBERO

The published LIBERO model is trained on LIBERO demonstrations only. LIBERO-Plus is used only for zero-shot evaluation and must not be mixed into training.

### 4.1 Download the converted LIBERO dataset

```bash
cd "$ACTIONUNET_ROOT"

uv run hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir "$HF_LEROBOT_HOME/physical-intelligence/libero"
```

Check that the dataset directory is not empty:

```bash
test -n "$(find "$HF_LEROBOT_HOME/physical-intelligence/libero" -mindepth 1 -maxdepth 1 -print -quit)"
```

### 4.2 Compute LIBERO normalization statistics

This is a one-process preprocessing step. Run it once before training:

```bash
cd "$ACTIONUNET_ROOT"
export LIBERO_EXP="libero_actionunet"

uv run python -m actionunet.train pi05_libero \
  --exp-name "$LIBERO_EXP" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --batch-size 256 \
  --compute-norm-stats
```

If you use a locally converted LIBERO dataset instead of `physical-intelligence/libero`, add `--data-repo-id your_name/libero` to this command and the training command below.

### 4.3 Start the published LIBERO training run

This is the official **8-GPU, global batch size 256** setup. Each GPU receives a local batch size of 32.

```bash
cd "$ACTIONUNET_ROOT"
export LIBERO_EXP="libero_actionunet"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m actionunet.train pi05_libero \
  --exp-name "$LIBERO_EXP" \
  --base-weights "$BASE_WEIGHTS" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --batch-size 256 \
  --num-train-steps 30000 \
  --save-interval 5000
```

Checkpoints are written under:

```text
<CHECKPOINT_ROOT>/pi05_libero__actionunet/libero_actionunet/<step>/
```

To resume the same run, repeat the command with `--resume`. Use `--overwrite` only when you intentionally want to remove the existing experiment and restart it. Never combine `--resume` and `--overwrite`.

## Step 5: Train on RoboTwin

RoboTwin training is single-task: convert one `demo_clean` dataset, train one checkpoint, and evaluate that checkpoint on the same task.

### 5.1 Download RoboTwin and its demonstrations

```bash
export ROBOTWIN_ROOT="/absolute/path/to/RoboTwin"

git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT"
cd "$ROBOTWIN_ROOT"
bash scripts/download_xpolicylab_data.sh
```

Use the official RoboTwin environment for the conversion command below.

### 5.2 Convert one RoboTwin task to LeRobot format

Choose one task. `move_can_pot` is used only as an example:

```bash
export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"
export HF_LEROBOT_HOME="/absolute/path/to/lerobot"

cd "$ROBOTWIN_ROOT"
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.${TASK}.aloha_agilex" \
  --repo_id "$DATA_REPO" \
  --max_episode 50
```

Confirm that the converted dataset exists:

```bash
test -n "$(find "$HF_LEROBOT_HOME/$DATA_REPO" -mindepth 1 -maxdepth 1 -print -quit)"
```

The official converter creates the following mapping:

- head camera → `cam_high`;
- left wrist camera → `cam_left_wrist`;
- right wrist camera → `cam_right_wrist`;
- state/action order → left arm, left gripper, right arm, right gripper.

The converter must use RoboTwin's `decode_image_bit`. Replacing it with a direct `cv2.imdecode` can produce an incorrect RGB/BGR channel order for some trajectories.

### 5.3 Compute normalization statistics for this task

Return to the ActionUNet Python 3.11 environment:

```bash
cd "$ACTIONUNET_ROOT"
export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"

uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --batch-size 64 \
  --compute-norm-stats
```

### 5.4 Start the published RoboTwin training run

This is the official **2-GPU, global batch size 64** setup. Each GPU receives a local batch size of 32.

```bash
cd "$ACTIONUNET_ROOT"
export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"

CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --base-weights "$BASE_WEIGHTS" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --batch-size 64 \
  --num-train-steps 8000 \
  --save-interval 2000
```

Checkpoints are written under:

```text
<CHECKPOINT_ROOT>/pi05_aloha__actionunet/demo_clean_<task>_robotwin/<step>/
```

To validate all resolved paths and configuration values without starting training, run:

```bash
uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --batch-size 64 \
  --dry-run
```

## Step 6: Start an inference server

The server performs image resizing, prompt tokenization, input normalization, model inference, and action unnormalization. Keep the server running while the simulator evaluates the policy.

### 6.1 Serve a LIBERO checkpoint

Set the checkpoint to an existing step directory containing `model.safetensors` and `metadata.pt`:

```bash
cd "$ACTIONUNET_ROOT"
export LIBERO_CHECKPOINT="$CHECKPOINT_ROOT/pi05_libero__actionunet/libero_actionunet/29999"

test -f "$LIBERO_CHECKPOINT/model.safetensors"
test -f "$LIBERO_CHECKPOINT/metadata.pt"

uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint "$LIBERO_CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

### 6.2 Serve a RoboTwin checkpoint

```bash
cd "$ACTIONUNET_ROOT"
export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"
export ROBOTWIN_CHECKPOINT="$CHECKPOINT_ROOT/pi05_aloha__actionunet/${DATA_REPO}/7999"

test -f "$ROBOTWIN_CHECKPOINT/model.safetensors"
test -f "$ROBOTWIN_CHECKPOINT/metadata.pt"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "$ROBOTWIN_CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

If training used an explicit `--asset-id`, pass the same `--asset-id` to the server.

## Step 7: Evaluate on LIBERO and LIBERO-Plus

The same checkpoint trained only on LIBERO is evaluated on both benchmarks.

| Benchmark | Evaluation | Seed | Training data |
| --- | --- | ---: | --- |
| LIBERO | 4 suites × 10 tasks × 50 episodes = 2,000 episodes | 7 | LIBERO only |
| LIBERO-Plus | 10,030 perturbation cases × 1 episode | 7 | LIBERO only |

### 7.1 Create the standard LIBERO simulator environment

```bash
export LIBERO_ROOT="/absolute/path/to/LIBERO"
export LIBERO_VENV="/absolute/path/to/venvs/libero"

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
git -C "$LIBERO_ROOT" checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c

python3.8 -m venv "$LIBERO_VENV"
source "$LIBERO_VENV/bin/activate"

cd "$ACTIONUNET_ROOT"
uv pip sync examples/libero/requirements.txt "$LIBERO_ROOT/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e "$LIBERO_ROOT"

deactivate
```

Generate the LIBERO configuration once. Answering `n` keeps the default paths:

```bash
printf 'n\n' | LIBERO_CONFIG_PATH="$LIBERO_ROOT/.libero" \
  "$LIBERO_VENV/bin/python" -c 'import libero.libero'
```

### 7.2 Create the separate LIBERO-Plus environment

LIBERO-Plus replaces the Python `libero` package, so do not install it into the standard LIBERO environment.

```bash
export LIBERO_PLUS_ROOT="/absolute/path/to/LIBERO-plus"
export LIBERO_PLUS_VENV="/absolute/path/to/venvs/libero-plus"

git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"

python3.8 -m venv "$LIBERO_PLUS_VENV"
source "$LIBERO_PLUS_VENV/bin/activate"

pip install -r "$LIBERO_PLUS_ROOT/requirements.txt"
pip install -r "$LIBERO_PLUS_ROOT/extra_requirements.txt"
pip install -e "$LIBERO_PLUS_ROOT"
pip install -e "$ACTIONUNET_ROOT/packages/openpi-client"
pip install 'tyro==0.9.2' 'imageio==2.35.1' 'imageio-ffmpeg==0.5.1'

deactivate
```

Download `assets.zip` from [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main), then extract it so this directory exists:

```text
<LIBERO_PLUS_ROOT>/libero/libero/assets/
```

Generate the LIBERO-Plus configuration once:

```bash
printf 'n\n' | LIBERO_CONFIG_PATH="$LIBERO_PLUS_ROOT/.libero" \
  "$LIBERO_PLUS_VENV/bin/python" -c 'import libero.libero'
```

### 7.3 Start the ActionUNet server

In terminal A:

```bash
cd "$ACTIONUNET_ROOT"
export LIBERO_CHECKPOINT="$CHECKPOINT_ROOT/pi05_libero__actionunet/libero_actionunet/29999"

uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint "$LIBERO_CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

Leave terminal A running. Run the evaluation commands in terminal B.

### 7.4 Evaluate standard LIBERO

```bash
cd "$ACTIONUNET_ROOT"
mkdir -p eval/libero

"$LIBERO_VENV/bin/python" examples/libero/eval_queue.py init \
  --db eval/libero/jobs.sqlite3 \
  --benchmark libero \
  --seed 7

NO_COLOR=1 \
TQDM_DISABLE=1 \
MUJOCO_GL=egl \
LIBERO_CONFIG_PATH="$LIBERO_ROOT/.libero" \
"$LIBERO_VENV/bin/python" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.job-db eval/libero/jobs.sqlite3 \
  --args.worker-id worker-0 \
  --args.no-save-videos \
  --args.seed 7

"$LIBERO_VENV/bin/python" examples/libero/eval_queue.py summary \
  --db eval/libero/jobs.sqlite3 \
  --output eval/libero/summary.tsv
```

The final results are written to `eval/libero/summary.tsv`.

### 7.5 Evaluate LIBERO-Plus

```bash
cd "$ACTIONUNET_ROOT"
mkdir -p eval/libero-plus

export LIBERO_EVAL_CATEGORY_CLASSIFICATION="$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"

"$LIBERO_PLUS_VENV/bin/python" examples/libero/eval_queue.py init \
  --db eval/libero-plus/jobs.sqlite3 \
  --benchmark libero-plus \
  --seed 7

NO_COLOR=1 \
TQDM_DISABLE=1 \
MUJOCO_GL=egl \
LIBERO_CONFIG_PATH="$LIBERO_PLUS_ROOT/.libero" \
LIBERO_EVAL_CATEGORY_CLASSIFICATION="$LIBERO_EVAL_CATEGORY_CLASSIFICATION" \
"$LIBERO_PLUS_VENV/bin/python" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.job-db eval/libero-plus/jobs.sqlite3 \
  --args.worker-id worker-0 \
  --args.no-save-videos \
  --args.seed 7

"$LIBERO_PLUS_VENV/bin/python" examples/libero/eval_queue.py summary \
  --db eval/libero-plus/jobs.sqlite3 \
  --output eval/libero-plus/summary.tsv
```

The final results are written to `eval/libero-plus/summary.tsv`. The SQLite queues are resumable: rerunning `init` returns stale `running` jobs to `pending`.

## Step 8: Evaluate on RoboTwin

Evaluate each single-task checkpoint on the same task in both `demo_clean` and `demo_randomized` modes.

### 8.1 Put ActionUNet under `RoboTwin/policy/`

RoboTwin loads policies from its `policy` directory. If ActionUNet is installed elsewhere, create a symbolic link:

```bash
mkdir -p "$ROBOTWIN_ROOT/policy"
ln -s "$ACTIONUNET_ROOT" "$ROBOTWIN_ROOT/policy/actionunet"
```

If `$ROBOTWIN_ROOT/policy/actionunet` already exists, do not run the `ln -s` command again.

### 8.2 Start the checkpoint server

In terminal A, using the ActionUNet environment:

```bash
cd "$ACTIONUNET_ROOT"
export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"
export ROBOTWIN_CHECKPOINT="$CHECKPOINT_ROOT/pi05_aloha__actionunet/${DATA_REPO}/7999"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "$ROBOTWIN_CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

Leave terminal A running.

### 8.3 Run both RoboTwin evaluation modes

In terminal B, activate the official RoboTwin/SAPIEN environment, then run:

```bash
cd "$ROBOTWIN_ROOT"
python -c 'import msgpack, websockets'

export TASK="move_can_pot"
export DATA_REPO="demo_clean_${TASK}_robotwin"

for MODE in demo_clean demo_randomized; do
  python script/eval_policy.py \
    --config policy/actionunet/robotwin_deploy.yml \
    --overrides \
    --task_name "$TASK" \
    --task_config "$MODE" \
    --ckpt_setting "${DATA_REPO}_step7999" \
    --seed 0 \
    --host 127.0.0.1 \
    --port 8000 \
    --action_chunk 50
done
```

RoboTwin evaluates 100 episodes per mode by default. The adapter sends head, left-wrist, and right-wrist RGB observations plus the 14-dimensional dual-arm state, then executes the returned action chunk.

## Optional: Convert LIBERO from the source RLDS dataset

Most users should download `physical-intelligence/libero` in Step 4. Use this section only when reproducing the conversion itself.

```bash
cd "$ACTIONUNET_ROOT"

uv run hf download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir /absolute/path/to/modified_libero_rlds

uv sync --group rlds
```

Set `REPO_NAME` in `examples/libero/convert_libero_data_to_lerobot.py`, then run:

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir /absolute/path/to/modified_libero_rlds
```

The converter merges the `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` `no_noops/1.0.0` subsets. It writes third-person RGB, wrist RGB, an 8-dimensional state, a 7-dimensional action, and the language instruction at 10 FPS.

**Warning:** the converter removes an existing output directory with the same repository ID. Back up any data that must be retained before running it.

## Common problems

### The number of GPUs does not match the command

`CUDA_VISIBLE_DEVICES` and `--nproc-per-node` must expose the same number of GPUs:

- LIBERO: eight GPU IDs and `--nproc-per-node=8`;
- RoboTwin: two GPU IDs and `--nproc-per-node=2`.

### The batch size is wrong

`--batch-size` is global, not per GPU:

- LIBERO: `--batch-size 256` with 8 GPUs;
- RoboTwin: `--batch-size 64` with 2 GPUs.

Both official settings produce a local batch size of 32 per GPU.

### Training cannot find normalization statistics

Run the matching `--compute-norm-stats` command first. The base config, `--data-repo-id`, and `--asset-id` must be identical between normalization, training, and serving.

### RoboTwin uses the wrong task

Keep `TASK`, `DATA_REPO`, the training checkpoint, and the evaluation `--task_name` synchronized. One checkpoint is trained for one RoboTwin task.

### The server starts but evaluation cannot connect

Make sure the server and evaluator use the same `--host` and `--port`. If they run on different machines, bind the server to `0.0.0.0`, use the server machine's reachable IP in the client, and allow the selected port through the firewall.

## Checkpoint layout

```text
<checkpoint-root>/<base-config>__actionunet/<experiment>/<step>/
  model.safetensors
  optimizer.pt
  metadata.pt
  assets/<asset-id>/norm_stats.json
```

## License

This project is released under the [Apache License 2.0](LICENSE). The Gemma components are subject to [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).

## Acknowledgements

This implementation builds on [OpenPI](https://github.com/Physical-Intelligence/openpi), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).
