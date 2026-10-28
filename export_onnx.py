"""Track 2: regenerate our submitted ONNX (testing-phase #19256, `e35-all`) from the weights.

The organizers call   main(model_dir, out_onnx).

weights/e33c-all.pt holds the trained network (E33c). Two export-time constants make the
submitted graph (E35), no training involved:
  * SHARPEN = 0.5   fixed in-graph unsharp mask on restored_image (7x7 Gaussian, sigma 1)
  * CLS_BIAS        per-class logit offsets added to the classifier bias (+1.5 on class 4 "other")
Static input (1, 3, 768, 1024) RGB in [0, 1]; outputs restored_image, seg_logits; opset 17.
"""
import inspect
import os

import torch

from src.mbv3_seg import Mbv3Seg

WEIGHTS = "e33c-all.pt"
SHARPEN = 0.5
CLS_BIAS = (0.0, 0.0, 0.0, 0.0, 1.5)


def main(model_dir, out_onnx):
    ck = torch.load(os.path.join(model_dir, WEIGHTS), map_location="cpu", weights_only=True)
    cfg = ck["config"]
    model = Mbv3Seg(pretrained=False, rest=cfg["rest"], refiner=True,
                    refiner_w=tuple(cfg["refiner_width"]), refiner_scale=cfg["refiner_scale"],
                    tta=False, sharpen=SHARPEN)
    model.load_state_dict(ck["model"])
    with torch.no_grad():
        model.high_cls.bias += torch.tensor(list(CLS_BIAS))
    model.eval()
    h, w = cfg["res"]
    os.makedirs(os.path.dirname(out_onnx) or ".", exist_ok=True)
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, (torch.zeros(1, 3, h, w),), out_onnx, input_names=["input"],
                      output_names=["restored_image", "seg_logits"], opset_version=17,
                      dynamic_axes=None, do_constant_folding=True, **kw)


if __name__ == "__main__":
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    main(os.path.join(here, "weights"), sys.argv[1] if len(sys.argv) > 1 else "export.onnx")
