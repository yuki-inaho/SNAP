---
name: pointcloud-ml-pixi-gpu-gui
description: Add point-cloud model inference coverage to a Pixi-managed GPU GUI test suite by separating fast fake-model interaction tests, real-checkpoint inference, and real GUI/screenshot validation, with offscreen or Xvfb Qt/VTK checks and explicit CUDA capability gates. Use for SNAP-like point-cloud GUI test work under Pixi; not for training, benchmark accuracy, or model architecture changes.
metadata:
  short-description: Point-cloud ML in Pixi GPU GUI tests
---

# Point-cloud ML in Pixi GPU GUI tests

Bring point-cloud model inference into a GUI test suite without turning every interaction
test into a slow, memory-hungry GPU job. "Pixi" here means the pixi package manager used by
the repository, not the pixi.js renderer. The target is layered:

1. **Interaction proof** — clicks, mask state and button wiring run against a deterministic
   fake model on CPU.
2. **Real-checkpoint inference proof** — the real model constructs, moves to CUDA, runs one
   bounded forward pass, and reports the selected attention backend.
3. **Real GUI integration proof** — when the request is real-checkpoint GUI operation, drive
   the real checkpoint through the actual UI and validate the rendered result/screenshot.

Keeping those layers separate is the point. A fake-model GUI test is not evidence that the
real checkpoint works; a real-model smoke test is not evidence that clicking works; and a
smoke test is not evidence that the GUI renders the real result. Use synthetic clouds for
shareable fixtures, but use authorized real data locally when the task calls for real GUI
operation.

## Invariants

- **Use the locked Pixi environment.** Run tests as `pixi run ...` (or `pixi run just ...`).
  An ambient `python` may resolve a different PyTorch/CUDA/extension ABI. If the Pixi
  environment is not installed, report `not executed`; do not substitute another interpreter
  silently.
- **Offscreen is explicit, and not universal.** Set `QT_QPA_PLATFORM=offscreen` and
  `PYVISTA_OFF_SCREEN=true` first. Some Qt/VTK builds still need a real X display: on those
  hosts start Xvfb as a separate process (or use `xvfb-run`) and set `DISPLAY`. Confirm a Qt
  application and a VTK render window were actually created; do not treat an import-only
  test as GUI coverage.
- **Skip is not pass.** Optional GUI packages may use `pytest.importorskip`. When a job is
  declared as GPU GUI coverage, a missing CUDA device, missing extension, or failed VTK
  context is a failure with a receipt, not a skipped success.
- **Gate on compute capability, not on hope.** FlashAttention-2 requires sm80+. On sm75
  (for example RTX 2070), select the regular PyTorch attention fallback and record both the
  capability and the backend actually used.
- **Bound GPU work.** Fix the point count, batch size, seed and wall-clock ceiling before
  the run. Record peak CUDA memory when available. An OOM is a failure receipt, not a
  reason to shrink the test silently.
- **Determinism is part of the contract.** Use a fixed seed, a generated point cloud rather
  than a random one, and fixed prompt coordinates. Keep the project's coordinate and color
  conventions: point coordinates are `(N, 3)`, prompt points are `(N, 3)` in the same frame,
  masks are boolean per point, and normalized colors must not be normalized twice.
- **A fake model is a fast layer, not the finish line.** The fake must implement the same
  small interface as the real wrapper (`intialize_pointcloud` in SNAP, or the project's
  equivalent, `extract_backbone_features`, `segment`, `segment_everything`, text-label
  helpers). When the request is real-checkpoint GUI operation, also run the real wrapper
  through the GUI and validate a rendered result/screenshot. Do not label a fake-only suite
  as real-model coverage.
- **Real data may be authorized.** Synthetic clouds are for shareable fixtures and
  reproducibility. If the user authorizes real data, use it for the integration layer,
  keep its logs and screenshots private, and publish only synthetic examples.

## Workflow

1. Read the repository's test entry points and GPU recipes before editing: the Pixi tasks,
   the GUI test module, the model wrapper and the point-cloud loader.
2. Run the existing GUI tests offscreen and record the baseline result. This is the
   comparison point, not a formality.
3. Identify the exact wrapper surface the GUI calls. Add or repair a fake model only for
   those methods; do not reimplement the network.
4. Add a GPU smoke test that constructs the real model, runs one forward pass with a fixed
   synthetic cloud, and asserts output shapes/device. Mark it slow or gate it on an explicit
   environment variable so ordinary GUI runs stay fast.
5. When real GUI operation is requested, add an integration test that loads the real
   checkpoint, performs one fixed click prompt, renders the mask, and checks the rendered
   result (structural assertions or a reviewed screenshot). Use authorized real data
   locally when needed; keep it out of public fixtures.
6. Make the skip/fail policy explicit, then run each layer through Pixi and save receipts.
7. If a test is slow or flaky, narrow the input before weakening the assertion.

Read [references/snap-grounded-notes.md](references/snap-grounded-notes.md) for the concrete
SNAP mapping (recipes, files, offscreen flags, fake-model pattern), and
[references/test-layer-template.md](references/test-layer-template.md) for a fake-model /
GPU-smoke test skeleton.

## Evidence and claims

Record per run: command, exit code, timeout, GPU name and compute capability, PyTorch/CUDA
versions, selected attention backend, point count/batch, display mode (offscreen or Xvfb),
data provenance (synthetic or authorized real), peak memory if available, wall time, and the
test node IDs that ran or skipped. Keep private logs and real user data out of the
repository; write them outside the working tree.

State which layer produced the evidence: fake interaction, real-checkpoint inference, or
real GUI integration/screenshot. Do not describe a fake-model pass as real-model coverage,
and do not describe a single-device smoke pass or a single screenshot as accuracy or
generalization evidence.

## Validate these assets

- Run the skill-creator validator against this directory.
- Run `python3 harness/tests/test_harness.py`; it checks the fixture, the independent
  judge and the reference planner without needing a GPU.
- Run `harness2-ordinary ... --dry-run` with a fresh output path before a real run.

The fixture tests are deterministic and synthetic. They do not replace a real
`pixi run pytest` GUI run, a real-checkpoint smoke run, or a real GUI/screenshot integration
run on the target host.

## Reusable Harness² fixture

The bundled `harness/` directory is a portable Harness² ordinary-task fixture for planning
this kind of GPU GUI test matrix. It has synthetic scenarios only, an independent judge,
and a pinned editor model. The mission asks for the **plan**; the planner is the harness
component the executor invokes. Keep that distinction when adapting it: a task that asks
the editor to "repair the planner" can turn into solver work and produce `KEEP_BASE`.
Read [references/harness2-ordinary.md](references/harness2-ordinary.md) before running it.

Selection at `k=1` is not a measured improvement. Preserve failures and ties, and claim an
improvement only from same-condition base/adopted measurements.
