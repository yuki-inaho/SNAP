# Tests

Run the default suite (loader unit tests, demo integration, real off-screen VTK
plotter interactions):

```bash
pixi run python -m pytest tests -q
```

The Qt widget tests are opt-in because a Qt OpenGL widget and an off-screen
VTK render window cannot be mixed reliably in one VTK process on this stack.
Run them on their own with a virtual display:

```bash
SNAP_QT_GUI_TESTS=1 xvfb-run -a pixi run python -m pytest tests/test_gui_qt.py -q
```

Notes:

- `test_pointcloud_loader.py` reads the real files in `data_examples/` and
  synthesises malformed inputs in `tmp_path`.
- `test_gui_interaction.py` builds the real `InteractivePlotter` with a real
  off-screen VTK render window. Only `SegmentationModel` is replaced by the
  deterministic fake in `gui_test_support.py`; click handling, mask colours,
  actor bookkeeping and the NMS/voxel helpers run for real.
- `test_gui_qt.py` builds the real `MainWindow` (PyQt6 + `QtInteractor`) and
  clicks the actual buttons. It needs `DISPLAY`; `xvfb-run` supplies one.
- `test_precision.py` checks the opt-in FP16 policy: FP32 remains the default,
  autocast covers the backbone/decoder/CLIP text path only, coordinates and
  post-processing stay FP32, parameters are never converted, and FP16 is
  rejected clearly on CPU. The CUDA cases skip when no GPU is available.
- Tests skip with a message (never silently pass) when an optional dependency
  such as Open3D is unavailable.
