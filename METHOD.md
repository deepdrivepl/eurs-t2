# Method description

Team: deepdrivepl   Track(s): both (this repository: Track 2)   Reported testing-phase submission(s): Track 2 #19256 (`e35-all`, composite 0.397)

## Approach (one page at most)

One static ONNX graph, input `(1, 3, 768, 1024)` RGB in [0, 1], outputs `restored_image` and `seg_logits` at the same size.
0.967 M parameters; 6.87 GFLOPs by `check_submission_flops.py` (about 7.1 counting the unsharp depthwise convolution it skips); opset 17.
Every operator is in the checker's vocabulary: hardswish and hardsigmoid are rewritten with `Clip`, global average pooling as `Reshape` + `MatMul` with a constant 1/HW vector.
The graph sees pixels only. It does not read the file name (no Meta).

**Segmentation.**
- Encoder: ImageNet MobileNetV3-Small `features[:12]`, dilated (output stride 16).
- Head: LR-ASPP. A 1x1 conv 96→128 + BN + ReLU is gated by a sigmoid of the pooled feature, upsampled to stride 8 and added to a 1x1 classifier on the 24-ch stride-8 feature, then bilinear to full size.
- Training: distilled from a frozen DINOv3 ViT-B/16 + conv-decoder teacher that runs live on the student's crop. Loss is KL with T = 2 on every pixel, plus cross-entropy on the labels.
- Export time: +1.5 is added to the bias of the class-4 ("other") logit. The value was chosen on a 5-fold scene split of the EURS training set.

**Restoration** (inside the same graph):
1. Per-image sRGB affine. The pooled 96-d backbone feature, the per-channel mean and mean-square, and an 8-threshold soft histogram of a 36x48 downsample feed an MLP (126→128→128→6). It outputs a per-channel gain `clip(u, 0, 12)^2` and an offset; `restored = clip(gain*x + offset, 0, 1)`. During training the MLP is supervised by the true water-type/depth cell's least-squares affine (training labels only).
2. Residual U-Net refiner (10/20/40 ch, 42k params) on `[affine output, input]` at half resolution. The residual is upsampled x2 and added.
3. Fixed unsharp mask: 7x7 Gaussian (sigma 1) depthwise conv, amount 0.5.

`export_onnx.py` builds the network from `weights/e33c-all.pt` (the trained E33c checkpoint). It then applies the two export-time constants (unsharp 0.5, class-4 bias +1.5) and exports with `torch.onnx.export` (TorchScript exporter, opset 17, constant folding). No simplifier, no quantisation.

## Training data

- EURS training split, all 20 scenes and all 200 `(A, B)` variants each, for the final weights.
- Model selection: a 5-fold split by scene of the 20 training scenes, plus the development-phase leaderboard (public Dev split, used only through leaderboard scores).
- The released testing-phase images were not used for training, tuning, model selection or pseudo-labelling.

External data and pretrained models (declared "Other" on every submission):
- ImageNet-1k MobileNetV3-Small weights (torchvision `IMAGENET1K_V1`, BSD-3-Clause), as the encoder initialisation.
- DINOv3 ViT-B/16 LVD-1689M (Meta, DINOv3 License), frozen backbone of the distillation teacher, used at training time only.
- Public underwater segmentation sets, labels remapped to the five EURS classes and re-degraded on the fly with the EURS forward model fitted on the training split. They were de-duplicated by perceptual hash against EURS train and Dev. Total 33,548 images:
  - AquaOV255
  - UIIS10K
  - MAS3K
  - Coralscapes
  - SUIM

## Anything non-standard at inference

None outside the graph. The unsharp mask and the class-4 logit offset are part of the ONNX graph.

## Reproduction notes

- `python repro_check.py . --export-onnx` regenerates `onnx/submitted_model.onnx`.
  - With PyTorch 2.9.1 (the version used for the upload) the file is byte-identical: SHA-256 `2c7b483985a8247dd7a05586f64dcb488856b4035a3f76f3cdfe17282233f18e`.
  - The export takes about 20 s on a CPU and needs no GPU.
- On Linux, in the organizers' base image (PyTorch 2.4.0, `--network none`) and equally in a PyTorch 2.9.1 image, the export runs and gives the same graph: same 565 nodes in the same order, same tensor names.
  - 65 of the 136 constant-folded tensors (convolutions with BatchNorm folded in) differ in the last float bits, max relative difference 1.4e-7. The checker therefore reports "differs".
  - The cause is the platform, not the PyTorch version: the upload was exported on Windows, and Windows and Linux CPU kernels round the float32 BatchNorm folding differently.
  - On 12 testing-phase images (onnxruntime CPU), the two files give identical masks on 99.99997% of pixels and restored images within one grey level.
- The checkpoint loads with `weights_only=True`.
- Training code and the full experiment record are in the team's main repository (link in the factsheet).
