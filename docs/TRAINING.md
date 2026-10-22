# Training and evaluation

All commands below start from the repository root and assume an activated CUDA environment. The shell parentheses keep each command's working-directory change local. YAML paths are relative to the corresponding training directory, not to `conf/`.

## CIFAR-10 and CIFAR-100

Prepare ImageFolder data and the original QKFormer checkpoint as described in [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md). For CIFAR-10:

```bash
(cd MaxFormer-main/cifar10-100 && python train.py -c cifar10_max_temporal.yaml)
(cd QKFormer-master/cifar10 && python train_qk_temporal.py -c cifar10_qk_temporal.yml)
(cd MaxFormer-main/cifar10-100 && python train.py -c cifar10_max_qk_temporal_router_v2.yaml)
```

For CIFAR-100:

```bash
(cd MaxFormer-main/cifar10-100 && python train.py -c cifar100_max_temporal.yaml)
(cd QKFormer-master/cifar100 && python train_qk_temporal.py -c cifar100_qk_temporal.yml)
(cd MaxFormer-main/cifar10-100 && python train.py -c cifar100_max_qk_temporal_router_v2.yaml)
```

Max-T checkpoints are saved under `MaxFormer-main/cifar10-100/output/train/<experiment>/model_best.pth.tar`. QK-T checkpoints are saved under `QKFormer-master/<dataset>/output/train/<experiment>/model_best.pth.tar`. The default router configurations refer to these paths.

The original expert training scripts remain available, e.g. `(cd QKFormer-master/cifar10 && python train.py -c cifar10.yml)`. If training an original expert yourself, update the temporal configuration's `initial_checkpoint` to its actual saved checkpoint. The temporal-from-scratch configurations are separate ablations.

Evaluate a static router using the exporter and evaluation helper:

```bash
python tnnls_experiments/export_static_logits.py \
  --dataset cifar100 --split test \
  --router-checkpoint MaxFormer-main/cifar10-100/output/train/cifar100_max_qk_temporal_router_v2/model_best.pth.tar \
  --output outputs/cifar100_test_logits.npz --device cuda:0
python scripts/evaluate_logits.py --npz outputs/cifar100_test_logits.npz
```

## CIFAR10-DVS

```bash
(cd MaxFormer-main/event && python train.py -c cifar10dvs_max_temporal_error_gate.yaml)
(cd QKFormer-master/cifar10-dvs && python train_qk_temporal.py -c cifar10dvs_qk_temporal.yaml)
(cd MaxFormer-main/event && python train.py -c cifar10dvs_max_qk_temporal_hard_router_v2.yaml)
```

Before the last command, verify `max_checkpoint` and `qk_checkpoint` in the router YAML. Event-run checkpoint directories are constructed from the model, batch size, T, weight decay, optimizer, experiment name, and learning rate; changing these fields changes the output location.

```bash
python tnnls_experiments/export_event_logits.py \
  --dataset cifar10dvs --split test \
  --router-checkpoint /absolute/path/to/trained_router.pth \
  --output outputs/cifar10dvs_test_logits.npz --device cuda:0
python scripts/evaluate_logits.py --npz outputs/cifar10dvs_test_logits.npz
```

The `/absolute/path/to/trained_router.pth` argument is a placeholder for the actual `checkpoint_max_test_acc1.pth` saved by the router run.

## DVS128-Gesture

```bash
(cd MaxFormer-main/event && python train.py -c dvsgesture_max_temporal_error_gate.yaml)
(cd QKFormer-master/dvs128-gesture && python train_qk_temporal.py -c dvsgesture_qk_temporal.yaml)
(cd MaxFormer-main/event && python train.py -c dvsgesture_max_qk_temporal_hard_router_v2.yaml)

python tnnls_experiments/export_event_logits.py \
  --dataset dvsgesture --split test \
  --router-checkpoint /absolute/path/to/trained_router.pth \
  --output outputs/dvsgesture_test_logits.npz --device cuda:0
python scripts/evaluate_logits.py --npz outputs/dvsgesture_test_logits.npz
```

For direct event evaluation, `train.py` also accepts `--test-only --resume /absolute/path/to/trained_router.pth` alongside the matching `-c` configuration. Expert checkpoints are still required during model construction.

## ImageNet-1K

The default sequence is **Max-T + QK-T**, with 384-dimensional backbones and T=4. Prepare the baseline Max-Former and QKFormer checkpoints listed in [DATA_AND_CHECKPOINTS.md](DATA_AND_CHECKPOINTS.md). Run these commands in order:

```bash
cd MaxFormer-main/imagenet_linux_2

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train_hard_router.py -c conf/imagenet_max_temporal_384_t4.yml

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train_hard_router.py -c conf/imagenet_qk_temporal_384_t4.yml

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train_hard_router.py -c conf/imagenet_max_qk_temporal_router_384_t4.yml
```

The first two stages are independent. The final stage uses `output/max_temporal_384_t4/checkpoint_best.pth` and `output/qk_temporal_384_t4/checkpoint_best.pth`. The external QKFormer module is already specified by relative path in the YAML files. Use `--data_path /path/to/imagenet` to override the default dataset root, and set `--nproc_per_node` to the number of GPUs available.

Evaluate the saved router from the same directory:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train_hard_router.py -c conf/imagenet_max_qk_temporal_router_384_t4.yml \
  --eval_only true \
  --resume output/max_qk_temporal_router_384_t4/checkpoint_best.pth
```

The ImageNet router pipeline reserves a configured fraction of the training set for checkpoint selection. Its official validation split is separate. Match checkpoint architecture, embedding width, class count, and simulation steps; the loader checks compatibility.

## Configuration reference

| Purpose | Field |
|---|---|
| Dataset location | `data_path` or `data_dir` |
| Original QK checkpoint for adaptation | `initial_checkpoint` (static), `init_checkpoint` (event), `qk_checkpoint` (ImageNet) |
| Max-T / QK-T expert weights | `max_checkpoint`, `qk_checkpoint`; ImageNet additionally uses `max_temporal_checkpoint`, `qk_temporal_checkpoint` |
| QK Python implementation | `qk_model_file` |
| Experiment naming | `experiment` or `output_dir` |
| Simulation steps | `time_step` (static/ImageNet), `T` (event) |

Do not infer the epoch count for one stage from another. The YAML files retain the supplied experiment schedules; the manuscript's reported protocols and saved experiment metadata are the reference for exact reproduction.

