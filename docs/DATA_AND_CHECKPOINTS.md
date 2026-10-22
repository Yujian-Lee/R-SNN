# Data and checkpoints

Data and model files are intentionally excluded. Paths below are relative to the repository root. YAML paths are resolved from each training script's working directory.

## Dataset layouts

| Dataset | Default root | Loader / split |
|---|---|---|
| CIFAR-10 | `data/cifar10` | ImageFolder: `train/<class>` and `test/<class>` |
| CIFAR-100 | `data/cifar100` | ImageFolder: `train/<class>` and `test/<class>` |
| CIFAR10-DVS | `data/cifar10-dvs` | SpikingJelly `CIFAR10DVS`, 16 frames, number-based integration |
| DVS128-Gesture | `data/dvs128` | SpikingJelly `DVS128Gesture`, official train/test split, 16 frames |
| ImageNet-1K | `data/imagenet` | ImageFolder: `train/<class>` and `val/<class>` |

CIFAR training uses timm's folder dataset loader with the supplied dataset names. Exporters also support raw torchvision CIFAR archives as a fallback, but do not change layout or class-index mapping between expert training and evaluation. Reordering class labels invalidates expert combination even when individual models can load successfully.

CIFAR10-DVS uses the source implementation's deterministic class-wise 90%/10% partition. Preserve the same dataset ordering and integration settings for both experts and all exports. DVS128-Gesture uses the dataset's official split. ImageNet router training uses a held-out training subset for checkpoint selection, controlled by `selection_val_fraction`.

## Original expert weights

The primary paths referenced by this release are:

| Weight | Expected local path |
|---|---|
| CIFAR-10 original QKFormer | `checkpoints/qkformer/cifar10.pth.tar` |
| CIFAR-100 original QKFormer | `checkpoints/qkformer/cifar100.pth.tar` |
| CIFAR10-DVS original QKFormer | `checkpoints/qkformer/cifar10dvs.pth` |
| DVS128-Gesture original QKFormer | `checkpoints/qkformer/dvsgesture.pth` |
| ImageNet Max-Former, 384/T4 | `checkpoints/imagenet/10-384-T4.pth.tar` |
| ImageNet QKFormer, 384/T4 | `checkpoints/qkformer/imagenet_10_384_t4.pth` |

Obtain compatible weights from the upstream projects or train the corresponding original expert, then use these filenames or update the YAML paths. A matching filename alone does not guarantee a compatible state dict.

MS-QKFormer ablations additionally reference `checkpoints/ms_qkformer/cifar10dvs.pth`, `checkpoints/ms_qkformer/dvsgesture.pth`, and `checkpoints/imagenet/ms_qk_10-384_T4.pth.tar`.

## Trained temporal experts and routers

Temporal experts and routers are written into their experiment-specific `output/` or `logs/` directories. The router loads the temporal experts before loading its own state, so keep all required checkpoints locally. Update `max_checkpoint`, `qk_checkpoint`, and temporal-checkpoint fields if run names or output directories change.

No model-download link for R-SNN is claimed in this release. Checkpoints are not present in the source archive.

