# deepdrivepl — EURS @ MaCVi WACV 2027, Track 2 reproduction

Testing-phase submission **#19256** (`e35-all`, composite 0.397), team `deepdrivepl`.
Made from the organizers' [reproduction template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template).

```bash
python repro_check.py . --export-onnx      # regenerates onnx/submitted_model.onnx from weights/
```

- `export_onnx.py` — `main(model_dir, out_onnx)`, the whole checkpoint → ONNX path
- `src/mbv3_seg.py` — the network (MobileNetV3-Small + LR-ASPP + in-graph affine / refiner / unsharp)
- `weights/e33c-all.pt` — trained weights (state_dict + config, loads with `weights_only=True`)
- `onnx/submitted_model.onnx` — the uploaded file, SHA-256 `2c7b483985a8247dd7a05586f64dcb488856b4035a3f76f3cdfe17282233f18e`
- `METHOD.md` — method, data, external resources
