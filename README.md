# ActionUNet（UNet + nearest + FC + Matérn 2.5）

这是 ActionUNet 固定版本的最小训练、推理与评测仓库。模型固定为 π0.5 主干、时间维 1D UNet、stride-2 下采样、nearest 上采样、FC bottleneck、两层 direct SIREN、逐层时间坐标注入和 γ=2.5 的 Matérn-index 聚合。

`src/openpi/` 的 56 个文件保持 [OpenPI c23745b](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79) 原样；`actionunet/openpi_train_pytorch.py` 是同一提交的官方 PyTorch trainer。仓库不包含权重、实验结果、消融分支、标注工具或其他策略。

## 目录

```text
actionunet/
  config.py                 # 固定模型配置
  model_actionunet.py       # ActionUNet 模型
  release.py                # LIBERO/RoboTwin 数据与训练配置
  train.py                  # 归一化统计和训练入口
  serve.py                  # WebSocket 推理服务
  robotwin_client.py        # RoboTwin 单任务评测客户端
  openpi_train_pytorch.py   # OpenPI 官方 PyTorch trainer
src/openpi/                 # 未修改的 OpenPI 源码
packages/openpi-client/     # 未修改的 OpenPI 客户端
examples/libero/
  convert_libero_data_to_lerobot.py
  main.py                   # LIBERO/LIBERO-Plus rollout worker
  eval_queue.py             # 可恢复的完整评测队列
```

## 1. 安装

要求 Linux、Python 3.11、NVIDIA GPU、CUDA 驱动和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync

TRANSFORMERS_DIR=$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')
cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"

uv run python -m actionunet.train --help
uv run python -m actionunet.serve --help
```

### π0.5 基座权重

训练入口读取 PyTorch `model.safetensors`。若只有官方 JAX 权重，按 [OpenPI 的转换说明](https://github.com/Physical-Intelligence/openpi/tree/c23745b5ad24e98f66967ea795a07b2588ed6c79#converting-jax-models-to-pytorch) 使用同一 OpenPI 提交的转换脚本：

```bash
uv run python /path/to/openpi/examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/jax/pi05_base \
  --config-name pi05_aloha \
  --output-path /path/to/pytorch/pi05_base \
  --precision bfloat16

test -f /path/to/pytorch/pi05_base/model.safetensors
```

后文的 `--base-weights` 指向 `/path/to/pytorch/pi05_base` 目录。

## 2. 数据下载与转换

训练数据统一写到同一个 LeRobot 根目录：

```bash
export HF_LEROBOT_HOME=/path/to/lerobot
mkdir -p "$HF_LEROBOT_HOME"
```

### 2.1 LIBERO

本项目的 LIBERO 模型只使用 LIBERO 数据训练。LIBERO-Plus 只用于训练完成后的零样本鲁棒性评测，不把 LIBERO-Plus 数据混入训练。

OpenPI 已转换的数据是 [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)：

```bash
uv run hf download physical-intelligence/libero --repo-type dataset \
  --local-dir "$HF_LEROBOT_HOME/physical-intelligence/libero"
```

也可以从 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 重新转换：

```bash
uv run hf download openvla/modified_libero_rlds --repo-type dataset \
  --local-dir /path/to/modified_libero_rlds
uv sync --group rlds
```

先把 `examples/libero/convert_libero_data_to_lerobot.py` 中的 `REPO_NAME` 改为自己的 repo ID，再运行：

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir /path/to/modified_libero_rlds
```

该 OpenPI 脚本合并 `libero_spatial/object/goal/10_no_noops/1.0.0` 四个 RLDS 子集，以 10 FPS 写入第三视角 RGB、腕部 RGB、8 维状态、7 维动作和语言指令。输出位于 `$HF_LEROBOT_HOME/<REPO_NAME>`。脚本会删除同名旧输出，运行前应保存需要保留的数据。

### 2.2 RoboTwin：每个任务单独转换

原始轨迹来自 [RoboTwin 2.0 Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)，转换器来自 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 的 XPolicyLab 子模块。先下载官方仓库和轨迹：

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
cd /path/to/RoboTwin
bash scripts/download_xpolicylab_data.sh
```

设定一个任务名并只转换它的 `demo_clean` 轨迹：

```bash
export TASK=move_can_pot
export HF_LEROBOT_HOME=/path/to/lerobot

python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.${TASK}.aloha_agilex" \
  --repo_id "demo_clean_${TASK}_robotwin" \
  --max_episode 50
```

输出是 `$HF_LEROBOT_HOME/demo_clean_<task>_robotwin`。每个任务重复一次该命令，得到彼此独立的数据集；一次训练只传其中一个 repo ID。

官方转换器完成以下处理：

- 使用 `decode_image_bit` 解码 RoboTwin 中并存的两种图像字节格式；
- 将 head、left wrist、right wrist 写为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`；
- 按左臂、左夹爪、右臂、右夹爪的顺序写入 `observation.state` 和 `action`；
- 从该 episode 的指令中选择 prompt，并使用任务配置中的采样频率；
- 自动检测常见的 `240×320` 或 `480×640` 图像分辨率。

不要用裸 `cv2.imdecode` 代替 `decode_image_bit`，否则部分官方轨迹会发生 RGB/BGR 通道错误。

## 3. 训练

训练分为归一化统计和正式训练。`--batch-size` 是全局 batch size，DDP 时必须能被进程数整除。

### 3.1 LIBERO

```bash
uv run python -m actionunet.train pi05_libero \
  --exp-name libero_actionunet \
  --checkpoint-root ./checkpoints \
  --batch-size 256 \
  --compute-norm-stats

uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m actionunet.train pi05_libero \
  --exp-name libero_actionunet \
  --base-weights /path/to/pytorch/pi05_base \
  --checkpoint-root ./checkpoints \
  --batch-size 256 \
  --num-train-steps 30000 \
  --save-interval 5000
```

使用自己转换的 LIBERO 数据时，统计和训练命令都添加 `--data-repo-id your_name/libero`。

### 3.2 RoboTwin 单任务

下面的统计、训练和后续评测始终使用同一个 `$TASK`：

```bash
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"

uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name "$DATA_REPO" \
  --checkpoint-root ./checkpoints \
  --batch-size 64 \
  --compute-norm-stats

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

RoboTwin 使用三相机 ALOHA 映射和 `adapt_to_pi=False`。OpenPI 在模型输入侧把双臂关节动作转换为相对当前状态的 delta，在输出侧恢复绝对动作。

配置检查不会启动训练：

```bash
uv run python -m actionunet.train pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --exp-name check \
  --checkpoint-root ./checkpoints \
  --dry-run
```

检查点位于：

```text
<checkpoint-root>/<base-config>__actionunet/<exp-name>/<step>/
  model.safetensors
  optimizer.pt
  metadata.pt
  assets/<asset-id>/norm_stats.json
```

在原训练命令后加 `--resume` 可从同一实验的最新 step 恢复；加 `--overwrite` 会删除同名旧实验后重训，两者不能同时使用。

## 4. 推理服务

LIBERO：

```bash
uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint ./checkpoints/pi05_libero__actionunet/libero_actionunet/29999 \
  --device cuda:0 --num-steps 10 --host 0.0.0.0 --port 8000
```

RoboTwin 单任务：

```bash
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "./checkpoints/pi05_aloha__actionunet/${DATA_REPO}/7999" \
  --device cuda:0 --num-steps 10 --host 0.0.0.0 --port 8000
```

若训练时显式传过 `--asset-id`，推理时必须传相同值。服务端负责图像缩放、prompt 分词、输入归一化和动作反归一化。

## 5. LIBERO 与 LIBERO-Plus 评测

两项评测加载同一个只在 LIBERO 上训练的 checkpoint：

| 评测 | suites | episode 数 | seed | 训练数据 |
| --- | ---: | ---: | ---: | --- |
| LIBERO | spatial、object、goal、10 | 4 × 10 × 50 = 2000 | 7 | LIBERO |
| LIBERO-Plus | 同四个 suite 的七类扰动 | 10030 × 1 = 10030 | 7 | LIBERO |

LIBERO-Plus 是零样本 OOD 评测。不要下载其 RLDS 或 LeRobot 训练集用于这里的 checkpoint。

### 5.1 分别安装两个模拟器环境

标准 LIBERO 使用 OpenPI 当时固定的 LIBERO commit：

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

LIBERO-Plus 使用独立环境，因为它会替换 Python 中的 `libero` 包：

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

从 [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) 下载 `assets.zip`，解压到 `/path/to/LIBERO-plus/libero/libero/`。最终应存在 `/path/to/LIBERO-plus/libero/libero/assets/`。

两个环境第一次运行前分别生成各自的 `config.yaml`；提示时回答 `n` 即可：

```bash
printf 'n\n' | LIBERO_CONFIG_PATH=/path/to/LIBERO/.libero \
  /path/to/venvs/libero/bin/python -c 'import libero.libero'

printf 'n\n' | LIBERO_CONFIG_PATH=/path/to/LIBERO-plus/.libero \
  /path/to/venvs/libero-plus/bin/python -c 'import libero.libero'
```

### 5.2 启动同一个 checkpoint

```bash
export CHECKPOINT=/absolute/path/to/checkpoints/pi05_libero__actionunet/libero_actionunet/29999

uv run python -m actionunet.serve \
  --base-config pi05_libero \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 --num-steps 10 --host 127.0.0.1 --port 8000
```

保持服务运行。下面两个队列顺序执行，期间不更换 checkpoint。

### 5.3 标准 LIBERO：2000 episodes

```bash
/path/to/venvs/libero/bin/python examples/libero/eval_queue.py init \
  --db eval/libero/jobs.sqlite3 --benchmark libero --seed 7

NO_COLOR=1 TQDM_DISABLE=1 MUJOCO_GL=egl \
LIBERO_CONFIG_PATH=/path/to/LIBERO/.libero \
/path/to/venvs/libero/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.job-db eval/libero/jobs.sqlite3 \
  --args.worker-id worker-0 --args.no-save-videos --args.seed 7

/path/to/venvs/libero/bin/python examples/libero/eval_queue.py summary \
  --db eval/libero/jobs.sqlite3 --output eval/libero/summary.tsv
```

### 5.4 LIBERO-Plus：10030 episodes

```bash
export LIBERO_EVAL_CATEGORY_CLASSIFICATION=/path/to/LIBERO-plus/libero/libero/benchmark/task_classification.json

/path/to/venvs/libero-plus/bin/python examples/libero/eval_queue.py init \
  --db eval/libero-plus/jobs.sqlite3 --benchmark libero-plus --seed 7

NO_COLOR=1 TQDM_DISABLE=1 MUJOCO_GL=egl \
LIBERO_CONFIG_PATH=/path/to/LIBERO-plus/.libero \
LIBERO_EVAL_CATEGORY_CLASSIFICATION="$LIBERO_EVAL_CATEGORY_CLASSIFICATION" \
/path/to/venvs/libero-plus/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.job-db eval/libero-plus/jobs.sqlite3 \
  --args.worker-id worker-0 --args.no-save-videos --args.seed 7

/path/to/venvs/libero-plus/bin/python examples/libero/eval_queue.py summary \
  --db eval/libero-plus/jobs.sqlite3 --output eval/libero-plus/summary.tsv
```

队列可以中断后用同一命令恢复；初始化时会把遗留的 `running` 项重新放回 `pending`。需要并行时，每个 GPU 启动一个 ActionUNet 服务和一个不同 `worker-id` 的 worker，所有 worker 共用同一个 SQLite 文件。

## 6. RoboTwin 单任务评测

RoboTwin 的训练和评测单位都是一个任务。`demo_clean_<task>_robotwin` 训练出的 checkpoint 只在同一个 `<task>` 上评测 `demo_clean` 和 `demo_randomized`，不把一个任务的 checkpoint 用到另一个任务。

历史实验使用 RoboTwin 的 `script/eval_policy.py` 接口。本仓库根目录的 `__init__.py`、`robotwin_deploy.yml` 和 `actionunet/robotwin_client.py` 保留了该接口。模型和模拟器使用两个独立环境，通过 WebSocket 通信。把本仓库以 `actionunet` 为目录名放到 RoboTwin 的 `policy/` 下：

```text
/path/to/RoboTwin/
  script/eval_policy.py
  policy/
    actionunet/              # 本仓库
```

先在 ActionUNet 的 Python 3.11 环境启动该任务的 checkpoint：

```bash
cd /path/to/RoboTwin/policy/actionunet
export TASK=move_can_pot
export DATA_REPO="demo_clean_${TASK}_robotwin"
export CHECKPOINT="/absolute/path/to/checkpoints/pi05_aloha__actionunet/${DATA_REPO}/7999"

uv run python -m actionunet.serve \
  --base-config pi05_aloha \
  --data-repo-id "$DATA_REPO" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 --num-steps 10 --host 127.0.0.1 --port 8000
```

然后在 RoboTwin/SAPIEN 环境启动评测。仓库根入口会直接加入随仓库发布的 `openpi-client`，RoboTwin 环境只需已有 `msgpack` 和 `websockets>=11`：

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

RoboTwin 默认每种模式评测 100 episodes。适配器读取 head、left wrist、right wrist 三路 RGB 和 14 维双臂关节状态，并执行同一 checkpoint 输出的动作 chunk。

## 7. 与整理前实现的一致性

使用整理前 RoboDojo 15,000-step checkpoint 和相同输入完成 CUDA 对比：

| 项目 | 最大绝对差 |
| --- | ---: |
| ActionUNet decoder 输出 | `0` |
| decoder 输入梯度 | `0` |
| Matérn GP noise（同一随机种子） | `0` |
| 训练 forward loss | `0` |
| 固定噪声推理动作 | `0` |

旧 checkpoint 中从未参与该固定版本前向计算的兼容参数不再注册或保存；加载旧 checkpoint 时只忽略这些已验证无计算作用的键。训练和推理计算保持一致。
