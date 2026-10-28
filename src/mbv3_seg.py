"""deepdrivepl Track 2 network (E33c + E35), inference / export only.

ImageNet MobileNetV3-Small `features[:12]` (dilated, OS 16) + LR-ASPP seg head, and an in-graph
restorer: per-image affine (gain = clip(u, 0, 12)^2, offset) from the pooled feature + input
statistics, a half-resolution residual U-Net refiner, and a fixed unsharp mask (E35).
Classes copied verbatim from `eurs/scripts/mbv3_seg.py` of our training repository (training code
removed) so that the export reproduces the submitted graph tensor-name for tensor-name.

Op vocabulary of the organizers' FLOPs checker has no Hardswish / Hardsigmoid / pooling /
ReduceMean, so: hardswish = x * Clip(x + 3, 0, 6) / 6, hardsigmoid = Clip(x + 3, 0, 6) / 6,
global mean = Reshape + MatMul with a constant 1/HW vector (shapes are static).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

MEAN = (0.485, 0.456, 0.406)
HIST_K = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)  # E32b soft-histogram thresholds
STD = (0.229, 0.224, 0.225)


class HSwish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class HSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class GlobalMean(nn.Module):
    """(B, C, H, W) -> (B, C, 1, 1) as Reshape + MatMul; H, W static at export."""

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = (int(v) for v in x.shape)  # python ints: constants in the trace
        ones = torch.full((h * w, 1), 1.0 / (h * w), dtype=x.dtype, device=x.device)
        return torch.matmul(x.reshape(b, c, h * w), ones).reshape(b, c, 1, 1)


def _vocab_safe(m: nn.Module) -> nn.Module:
    for name, child in m.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(m, name, HSwish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(m, name, HSigmoid())
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(m, name, GlobalMean())
        else:
            _vocab_safe(child)
    return m


class Refiner(nn.Module):
    """E32c: residual U-Net on [affine-restored, input] at full res, 10/20/40 ch — ~42k params,
    ~8.9 GFLOPs at 576x768 (E31 body 1.7 → 10.6 of 12). Conv / Relu / Resize / Concat / Add only.
    Last conv zero-init, so the graph starts bit-identical to its E32b init."""

    def __init__(self, c: tuple[int, int, int] = (10, 20, 40)) -> None:
        super().__init__()
        c1, c2, c3 = c

        def blk(i: int, o: int, s: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU(inplace=True))

        self.e1 = nn.Sequential(blk(6, c1), blk(c1, c1))
        self.e2 = nn.Sequential(blk(c1, c2, 2), blk(c2, c2))
        self.e3 = nn.Sequential(blk(c2, c3, 2), blk(c3, c3))
        self.d2 = blk(c3 + c2, c2)
        self.d1 = blk(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        assert self.out.bias is not None
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        def up(t: Tensor, ref: Tensor) -> Tensor:
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        d2 = self.d2(torch.cat([up(s3, s2), s2], 1))
        return self.out(self.d1(torch.cat([up(d2, s1), s1], 1)))


class Mbv3Seg(nn.Module):
    LOW, HIGH = 3, 11  # features index: 24 ch at stride 8, 96 ch at stride 16 (dilated)

    def __init__(self, pretrained: bool = True, inter: int = 128, rest: str = "global",
                 refiner: bool = False, refiner_w: tuple[int, int, int] = (10, 20, 40),
                 refiner_scale: float = 1.0, tta: bool = False, sharpen: float = 0.0) -> None:
        super().__init__()
        weights = (torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                   if pretrained else None)
        feats = torchvision.models.mobilenet_v3_small(weights=weights, dilated=True).features
        self.body: nn.Sequential = _vocab_safe(feats[: self.HIGH + 1])  # type: ignore[assignment]
        self.cbr = nn.Sequential(nn.Conv2d(96, inter, 1, bias=False), nn.BatchNorm2d(inter),
                                 nn.ReLU(inplace=True))
        self.pool = GlobalMean()
        self.scale = nn.Sequential(nn.Conv2d(96, inter, 1), nn.Sigmoid())
        self.low_cls = nn.Conv2d(24, 5, 1)
        self.high_cls = nn.Conv2d(inter, 5, 1)
        last = nn.Linear(32, 6)
        nn.init.zeros_(last.weight)  # start at identity: gain 1, offset 0
        nn.init.zeros_(last.bias)
        self.rest_kind = rest
        self.cell_cnn: nn.Sequential | None = None
        if rest == "cnn":  # E32e: photometry CNN on the 72x96 input (seg features are KD-shaped,
            # not photometric); the E32 oracle (true cell) was +4.5 dB over the E32b estimate
            self.cell_cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 32, 3, 2, 1),
                nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
                GlobalMean())
            last = nn.Linear(128, 6)
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
            self.rest = nn.Sequential(nn.Linear(64 + 96 + 30, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 128), nn.ReLU(inplace=True), last)
        elif rest == "cell2":  # E32b: + soft-histogram stats, deeper head
            last = nn.Linear(128, 6)
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
            self.rest = nn.Sequential(nn.Linear(96 + 30, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 128), nn.ReLU(inplace=True), last)
        elif rest == "cell":
            last = nn.Linear(64, 6)
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))  # u=1 -> gain 1
            self.rest = nn.Sequential(nn.Linear(102, 64), nn.ReLU(inplace=True), last)
        else:
            self.rest = nn.Sequential(nn.Linear(96, 32), nn.ReLU(inplace=True), last)
        self.refiner: Refiner | None = Refiner(refiner_w) if refiner else None
        self.refiner_scale = refiner_scale  # E33: run the refiner at half res (FLOPs ∝ pixels)
        self.tta = tta  # E34: in-graph h-flip TTA on the seg path (body x2; Slice op), no params
        # E35: fixed unsharp mask in the graph (board sharpness: ours 0.30, UW-IPL 0.96; 0.03 of
        # the composite). 7x7 Gaussian sigma 1 = cv2.GaussianBlur((0, 0), 1.0) as in E27; zero pad.
        self.sharpen = sharpen
        g = torch.exp(-0.5 * (torch.arange(7, dtype=torch.float32) - 3.0) ** 2)
        g = g / g.sum()
        self.blur_k: Tensor
        self.register_buffer("blur_k", (g[:, None] * g[None, :]).expand(3, 1, 7, 7).clone(),
                             persistent=False)
        self.mean: Tensor
        self.std: Tensor
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def _seg(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Body + LR-ASPP head: (seg_logits at H x W, pooled 96-d feature)."""
        h = (x - self.mean) / self.std
        low = None
        for i, layer in enumerate(self.body):
            h = layer(h)
            if i == self.LOW:
                low = h
        assert low is not None
        g = self.pool(h)
        y = self.cbr(h) * self.scale(g)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.high_cls(y) + self.low_cls(low)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False), g

    def forward(self, x: Tensor, params: bool = False) -> tuple[Tensor, ...]:
        """x: (B, 3, H, W) RGB 0-1 -> (restored_image 0-1, seg_logits), both at H x W."""
        logits, g = self._seg(x)
        if self.tta:
            logits_f, g_f = self._seg(torch.flip(x, dims=[-1]))
            logits = 0.5 * (logits + torch.flip(logits_f, dims=[-1]))
            g = 0.5 * (g + g_f)
        nb = g.shape[0]
        if self.rest_kind in ("cell", "cell2", "cnn"):
            xs = F.interpolate(x, size=(36, 48), mode="bilinear", align_corners=False)
            st = [self.pool(xs).reshape(nb, 3), self.pool(xs * xs).reshape(nb, 3)]
            if self.rest_kind in ("cell2", "cnn"):  # soft CDF: mean(relu(x - k)), Sub+Relu+MatMul
                st += [self.pool(F.relu(xs - k)).reshape(nb, 3) for k in HIST_K]
            feats = [g.reshape(nb, 96), 10.0 * torch.cat(st, 1)]
            if self.cell_cnn is not None:
                x72 = F.interpolate(x, size=(72, 96), mode="bilinear", align_corners=False)
                feats.insert(0, self.cell_cnn(x72).reshape(nb, 64))
            ab = self.rest(torch.cat(feats, 1))
            u = torch.clamp(ab[:, :3], 0.0, 12.0)
            gain = (u * u).reshape(-1, 3, 1, 1)  # 0..144, Clip + Mul only
        else:
            ab = self.rest(g.reshape(nb, 96))
            u = ab[:, :3]
            gain = torch.clamp(1.0 + u, 0.2, 8.0).reshape(-1, 3, 1, 1)  # no Exp in vocab
        restored = torch.clamp(x * gain + ab[:, 3:].reshape(-1, 3, 1, 1), 0.0, 1.0)
        if self.refiner is not None:  # E32c
            inp = torch.cat([restored, x], 1)
            if self.refiner_scale != 1.0:
                # scale_factor, not size: exports Resize with constant scales, so ONNX shape
                # inference (and the organizers' FLOPs script) can see the convs behind it
                inp = F.interpolate(inp, scale_factor=self.refiner_scale, mode="bilinear",
                                    align_corners=False)
                res = F.interpolate(self.refiner(inp), scale_factor=1.0 / self.refiner_scale,
                                    mode="bilinear", align_corners=False)
            else:
                res = self.refiner(inp)
            restored = torch.clamp(restored + res, 0.0, 1.0)
        if self.sharpen > 0:
            blur = F.conv2d(restored, self.blur_k, padding=3, groups=3)
            restored = torch.clamp(restored + self.sharpen * (restored - blur), 0.0, 1.0)
        if params:
            return restored, logits, u, ab[:, 3:]
        return restored, logits


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

