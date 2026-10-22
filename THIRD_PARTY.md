# Third-party code and attribution

This release contains local research modifications of the following projects:

| Project | Source | Included role |
|---|---|---|
| Max-Former | https://github.com/bic-L/MaxFormer | Spiking backbones, embedding/token-mixing modules, training utilities |
| QKFormer | https://github.com/zhouchenlin2096/QKFormer | QK attention backbones and supporting training code |
| timm | https://github.com/huggingface/pytorch-image-models | Training, augmentation, and loader patterns; attribution remains in source headers |

SpikingJelly, PyTorch, torchvision, CuPy, and the other packages in the installation instructions are external dependencies, not vendored distributions. Some inherited utilities also credit the PyTorch examples, NVIDIA Apex, DeiT, BEiT, TensorFlow TPU, or AugMix projects in their original headers.

Upstream attribution and copyright headers are preserved. The supplied Max-Former and QKFormer snapshots did not contain a project-level LICENSE file among the selected source files. This packaging step does not assign a new blanket license or replace any existing notices. A project license for the R-SNN additions has not yet been selected.

