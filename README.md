# ActionUNet: Improving Robustness of VLA Models with Efficient Multi-scale Fine-tuning

<p align="center">
  <strong>Official PyTorch implementation for NeurIPS 2026</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#training">Training</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#evaluation">Evaluation</a>
</p>

This repository provides the training, inference, and evaluation code for ActionUNet on LIBERO, LIBERO-Plus, and RoboTwin.

## Installation

ActionUNet requires Linux, Python 3.11, an NVIDIA GPU with a compatible CUDA driver, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

TRANSFORMERS_DIR=$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"

uv run python -m actionunet.train --help
uv run python -m actionunet.serve --help
```

The vendored `src/openpi/` tree is kept identical to [OpenPI commit c23745b](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79). The PyTorch trainer in `actionunet/openpi_train_pytorch.py` is also based on that revision.

### π0.5 base checkpoint

Training expects a PyTorch checkpoint containing `model.safetensors`. If you only have the official JAX checkpoint, follow the [OpenPI JAX-to-PyTorch conversion instructions](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79#converting-jax-models-to-pytorch) with the same OpenPI revision:

```bash
uv run python /path/to/openpi/examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/jax/pi05_base \
  --config-name pi05_aloha \
  --output-path /path/to/pytorch/pi05_base \
  --precision bfloat16

test -f /path/to/pytorch/pi05_base/model.safetensors
```

Use `/path/to/pytorch/pi05_base` as `--base-weights` in the training commands below.

## Datasets

Store all converted LeRobot datasets under one root directory:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot
mkdir -p "$HF_LEROBOT_HOME"
```

### LIBERO

The LIBERO checkpoint is trained only on LIBERO demonstrations. LIBERO-Plus is used only for zero-shot robustness evaluation and is never mixed into the training data.

#### Download the converted dataset

OpenPI provides a converted LeRobot dataset at [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero):

```bash
uv run hf download physical-intelligence/libero --repo-type dataset \
  --local-dir "$HF_LEROBOT_HOME/physical-intelligence/libero"
```

#### Convert the RLDS dataset

To reproduce the conversion from the source data, download [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds):

```bash
uv run hf download openvla/modified_libero_rlds --repo-type dataset \
  --local-dir /path/to/modified_libero_rlds
uv sync --group rlds
```

Set `REPO_NAME` in `examples/libero/convert_libero_data_to_lerobot.py` to the desired output repository ID, then run:

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir /path/to/modified_libero_rlds
```

The converter merges the `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` `no_noops/1.0.0` subsets. It writes third-person RGB, wrist RGB, an 8-dimensional state, a 7-dimensional action, and the language instruction at 10 FPS. The output is saved to `$HF_LEROBOT_HOME/<REPO_NAME>`.

The converter removes an existing output directory with the same repository ID. Back up any data that must be retained before running it.

### RoboTwin

RoboTwin training and evaluation are single-task. Convert one `demo_clean` dataset per task and train one checkpoint for that task.

The original trajectories are available from the [RoboTwin 2.0 Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). The converter is provided by the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) XPolicyLab submodule:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
cd /path/to/RoboTwin
bash scripts/download_xpolicylab_data.sh
```

Convert one task at a time:

```bash
export TASK=move_can_pot
export HF_LEROBOT_HOME=/path/to/lerobot

python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.${TASK}.aloha_agilex" \
  --repo_id "demo_clean_${TASK}_robotwin" \
  --max_episode 50
```

The output is saved to `$HF_LEROBOT_HOME/demo_clean_<task>_robotwin`. Repeat the command for each task, and pass only one resulting repository ID to each training run.

The official converter:

- decodes both image byte formats used by RoboTwin with `decode_image_bit`;
- maps the head, left-wrist, and right-wrist cameras to `cam_high`, `cam_left_wrist`, and `cam_right_wrist`;
- writes `observation.state` and `action` in left arm, left gripper, right arm, right gripper order;
- selects a language instruction for each episode and uses the sampling rate from the task configuration;
- detects the common `240×320` and `480×640` image resolutions.

Use `decode_image_bit` rather than a direct `cv2.imdecode` call. Some official trajectories otherwise receive an incorrect RGB/BGR channel order.

## Training

Training consists of computing normalization statistics and then optimizing the model. `--batch-size` is the global batch size and must be divisible by the number of DDP processes.

### LIBERO

Compute normalization statistics:

```bash
uv run python -m actionunet.train pi05_libero \
  --exp-name libero_actionunet \
  --checkpoint-root ./checkpoints \
  --batch-size 256 \
  --compute-norm-stats
```

Train with four GPUs:

```bash
uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m actionunet.train pi05_libero \
  --exp-name libero_actionunet \
  --base-weights /path/to/pytorch/pi05_base \
  --checkpoint-root ./checkpoints \
  --batch-size 256 \
  --num-train-steps 30000 \
  --save-interval 5000
```

When using a locally converted LIBERO dataset, add `--data-repo-id your_name/libero` to both commands.

### RoboTwin single-task training

Use the same task for normalization, training, and evaluation:

```bash
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"
```

Compute normalization statistics:

```bash
uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --checkpoint-root ./checkpoints \
  --batch-size 64 \
  --compute-norm-stats
```

Train with four GPUs:

```bash
uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --base-weights /path/to/pytorch/pi05_base \
  --checkpoint-root ./checkpoints \
  --batch-size 128 \
  --num-train-steps 8000 \
  --save-interval 2000
```

RoboTwin uses the three-camera ALOHA mapping with `adapt_to_pi=False`. OpenPI converts dual-arm joint actions to deltas from the current state at the model input and restores absolute actions at the output.

Validate the configuration without starting training:

```bash
uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name check \
  --checkpoint-root ./checkpoints \
  --dry-run
```

Checkpoints use the following layout:

```text
<checkpoint-root>/<base-config>__actionunet/<exp-name>/<step>/
  model.safetensors
  optimizer.pt
  metadata.pt
  assets/<asset-id>/norm_stats.json
```

Add `--resume` to continue from the latest step of the same experiment. Add `--overwrite` to remove an existing experiment and train it again. These options are mutually exclusive.

## Inference

ActionUNet exposes an OpenPI-compatible WebSocket policy server. The server performs image resizing, prompt tokenization, input normalization, and action unnormalization.

### LIBERO

```bash
uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint ./checkpoints/pi05_libero__actionunet/libero_actionunet/29999 \
  --device cuda:0 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8000
```

### RoboTwin

```bash
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "./checkpoints/pi05_aloha__actionunet/${DATA_REPO}/7999" \
  --device cuda:0 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8000
```

If training used an explicit `--asset-id`, pass the same value to the inference server.

## Evaluation

### LIBERO and LIBERO-Plus

Both benchmarks use the same checkpoint trained only on LIBERO:

| Benchmark | Suites | Episodes | Seed | Training data |
| --- | --- | ---: | ---: | --- |
| LIBERO | Spatial, Object, Goal, 10 | `4 × 10 × 50 = 2000` | 7 | LIBERO |
| LIBERO-Plus | Seven perturbation categories over the same four suites | `10030 × 1 = 10030` | 7 | LIBERO |

LIBERO-Plus is a zero-shot out-of-distribution evaluation. Do not use LIBERO-Plus trajectories to train this checkpoint.

#### Simulator environments

Standard LIBERO uses the revision pinned by OpenPI:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c

python3.8 -m venv /path/to/venvs/libero
source /path/to/venvs/libero/bin/activate
uv pip sync examples/libero/requirements.txt /path/to/LIBERO/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e /path/to/LIBERO
deactivate
```

Install LIBERO-Plus in a separate environment because it replaces the Python `libero` package:

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git /path/to/LIBERO-plus

python3.8 -m venv /path/to/venvs/libero-plus
source /path/to/venvs/libero-plus/bin/activate
pip install -r /path/to/LIBERO-plus/requirements.txt
pip install -r /path/to/LIBERO-plus/extra_requirements.txt
pip install -e /path/to/LIBERO-plus
pip install -e packages/openpi-client
pip install 'tyro==0.9.2' 'imageio==2.35.1' 'imageio-ffmpeg==0.5.1'
deactivate
```

Download `assets.zip` from [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and extract it into `/path/to/LIBERO-plus/libero/libero/`. The resulting directory must contain `/path/to/LIBERO-plus/libero/libero/assets/`.

Generate the simulator `config.yaml` files before the first run. Answer `n` when prompted:

```bash
printf 'n\n' | LIBERO_CONFIG_PATH=/path/to/LIBERO/.libero \
  /path/to/venvs/libero/bin/python -c 'import libero.libero'

printf 'n\n' | LIBERO_CONFIG_PATH=/path/to/LIBERO-plus/.libero \
  /path/to/venvs/libero-plus/bin/python -c 'import libero.libero'
```

#### Start the policy server

```bash
export CHECKPOINT=/absolute/path/to/checkpoints/pi05_libero__actionunet/libero_actionunet/29999

uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

Keep the server running while executing both evaluation queues. Do not change the checkpoint between the two evaluations.

#### Standard LIBERO

```bash
/path/to/venvs/libero/bin/python examples/libero/eval_queue.py init \
  --db eval/libero/jobs.sqlite3 \
  --benchmark libero \
  --seed 7

NO_COLOR=1 TQDM_DISABLE=1 MUJOCO_GL=egl \
LIBERO_CONFIG_PATH=/path/to/LIBERO/.libero \
/path/to/venvs/libero/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.job-db eval/libero/jobs.sqlite3 \
  --args.worker-id worker-0 \
  --args.no-save-videos \
  --args.seed 7

/path/to/venvs/libero/bin/python examples/libero/eval_queue.py summary \
  --db eval/libero/jobs.sqlite3 \
  --output eval/libero/summary.tsv
```

#### LIBERO-Plus

```bash
export LIBERO_EVAL_CATEGORY_CLASSIFICATION=/path/to/LIBERO-plus/libero/libero/benchmark/task_classification.json

/path/to/venvs/libero-plus/bin/python examples/libero/eval_queue.py init \
  --db eval/libero-plus/jobs.sqlite3 \
  --benchmark libero-plus \
  --seed 7

NO_COLOR=1 TQDM_DISABLE=1 MUJOCO_GL=egl \
LIBERO_CONFIG_PATH=/path/to/LIBERO-plus/.libero \
LIBERO_EVAL_CATEGORY_CLASSIFICATION="$LIBERO_EVAL_CATEGORY_CLASSIFICATION" \
/path/to/venvs/libero-plus/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.job-db eval/libero-plus/jobs.sqlite3 \
  --args.worker-id worker-0 \
  --args.no-save-videos \
  --args.seed 7

/path/to/venvs/libero-plus/bin/python examples/libero/eval_queue.py summary \
  --db eval/libero-plus/jobs.sqlite3 \
  --output eval/libero-plus/summary.tsv
```

The queues are resumable. Initialization returns stale `running` jobs to `pending`. For parallel evaluation, run one ActionUNet server and one worker per GPU, assign a unique `worker-id` to every worker, and share the same SQLite database.

### RoboTwin single-task evaluation

Evaluate `demo_clean_<task>_robotwin` only on the same task used for training. Run both the `demo_clean` and `demo_randomized` settings with that checkpoint.

RoboTwin uses its `script/eval_policy.py` interface. Place this repository under `RoboTwin/policy/` with the directory name `actionunet`:

```text
/path/to/RoboTwin/
  script/eval_policy.py
  policy/
    actionunet/
```

Start the checkpoint server in the ActionUNet Python 3.11 environment:

```bash
cd /path/to/RoboTwin/policy/actionunet
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"
export CHECKPOINT="/absolute/path/to/checkpoints/pi05_aloha__actionunet/${DATA_REPO}/7999"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --num-steps 10 \
  --host 127.0.0.1 \
  --port 8000
```

Run the simulator in the RoboTwin/SAPIEN environment. The root package adds the bundled `openpi-client` to the import path, so the simulator environment only needs `msgpack` and `websockets>=11`:

```bash
cd /path/to/RoboTwin
python -c 'import msgpack, websockets'
export TASK=move_can_pot
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

RoboTwin evaluates 100 episodes per setting by default. The adapter sends head, left-wrist, and right-wrist RGB observations together with the 14-dimensional dual-arm joint state, then executes the returned action chunk.

## License

This project is released under the [Apache License 2.0](LICENSE). The Gemma components are subject to the terms in [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).

## Acknowledgements

This implementation builds on [OpenPI](https://github.com/Physical-Intelligence/openpi), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).
