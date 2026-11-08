# SNAP-grounded notes

This reference maps the generic workflow in `SKILL.md` to this repository. It records
existing entry points, not a claim that a particular test currently passes.

## Environment

- `pixi.toml` + `pixi.lock` pin Python 3.10, PyTorch 2.4.1/CUDA 12.4 and the PyG/spconv
  extensions. Use `pixi run ...`; do not use an ambient interpreter for GPU/GUI checks.
- `justfile` recipes used by the workflow:
  - `just gpu` — Python, PyTorch, CUDA and device capability.
  - `just imports` — imports model, dataset and GUI modules without running the network.
  - `just validate` — CUDA matmul, `torch-scatter`, `torch-cluster` and an spconv kernel.
  - `just model-smoke` — constructs SNAP on the GPU and prints the parameter count.
  - `just test` — pytest suite.
  - `just test-gui` — Qt/VTK tests with `QT_QPA_PLATFORM=offscreen` and
    `PYVISTA_OFF_SCREEN=true`.
- A real host used for development was an RTX 2070 (sm75, 8 GB). FlashAttention-2 is not
  available there; SNAP resolves the regular PyTorch attention path. Do not assume the
  development GPU is representative of a deployment target.

## Files and interfaces

- `tests/test_gui_interaction.py` exercises the real Qt widgets and VTK plotter with a
  deterministic fake model. It shows the minimal wrapper surface:
  `intialize_pointcloud`, `extract_backbone_features`, `segment`,
  `segment_everything`, `get_text_label`, `find_best_mask_from_text`.
- The misspelling `intialize_pointcloud` is part of the current interface; preserve it in
  a fake unless the wrapper is deliberately renamed everywhere.
- `iterative_gui_demo.py` contains `SegmentationModel` and `InteractivePlotter`. Click
  handling lives in `InteractivePlotter`; real model calls live in `SegmentationModel`.
- `pointcloud_loader.py` owns point-cloud loading and color handling. The fake model in the
  GUI test divides color by `127.5` and subtracts `1.0` to mimic the Indoor normalization;
  do not apply that twice.
- `src/snap.py` exposes `SNAP(... attention_backend=...)`. `"auto"` selects FlashAttention-2
  only when the package and device support it, `"flash"` requires it, and `"normal"` forces
  the PyTorch fallback. `SNAP_ATTENTION_BACKEND` can set the default.
- A bounded real-model smoke can construct `SNAP(num_points=1, num_merge_blocks=1,
  use_pdnorm=True, return_mid_points=True).cuda().eval()` (as `just model-smoke` does) and
  then run one forward pass on a fixed synthetic batch. Loading CLIP checkpoints is a
  separate cost; do not hide it inside a GUI interaction test.

## Offscreen pitfalls

- `pytest.importorskip` keeps optional GUI dependencies from failing unrelated runs, but it
  also turns a missing display stack into a skip. A GPU GUI coverage job must inspect the
  skip reason or run with an explicit gate.
- VTK may create a render window even when `PYVISTA_OFF_SCREEN=true`; assert the plotter
  object and a non-zero render window rather than only checking imports.
- Keep `QT_QPA_PLATFORM=offscreen` in the command that actually launches pytest. A value
  set in another shell does not travel through `just` or CI.

## Minimal test matrix

| Layer | Model | Device / display | Purpose | Typical gate |
|---|---|---|---|---|
| Interaction | deterministic fake | CPU | clicks, prompt sets, mask state, buttons | every test run |
| GUI wiring/render | deterministic fake | CPU + offscreen VTK or Xvfb | widget/plotter/screenshot smoke | GUI test run |
| Real inference | real SNAP | CUDA | construction, one forward pass, shape/device | explicit slow/GPU marker |
| Real GUI integration | real SNAP + authorized data | CUDA + display | one fixed click prompt through the UI, rendered mask/screenshot | nightly or manual |

The matrix is a starting point. Add a row only when it proves a distinct failure mode.
Do not stop at the fake rows when the request is real-checkpoint GUI operation.

Offscreen Qt is not universally sufficient for Qt/VTK. On hosts where the offscreen
platform cannot create a VTK render window, start Xvfb as a separate process (or use
`xvfb-run`) and pass its `DISPLAY`; assert the render window instead of inferring success
from imports. Keep real-data screenshots and logs private; publish synthetic fixtures.
