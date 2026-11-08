# Test-layer template

Use these shapes as a starting point; adapt names and data fields to the repository's real
wrapper. The important part is the split, not the exact classes.

## 1. Fake model for interaction tests

The fake should be small, deterministic, and implement only what the GUI calls. Keep mask
geometry simple enough to assert exactly.

```python
class FakePointCloudModel:
    def __init__(self, domain="Indoor"):
        self.domain = domain
        self.device = torch.device("cpu")
        self.dataset = FakeDataset(labels=["wall", "floor"])
        self.point_features = None

    def initialize_pointcloud(self, data):  # keep the project's spelling if it differs
        coord = torch.as_tensor(np.asarray(data["coord"], dtype=np.float32))
        out = {"coord": coord}
        if "color" in data:
            out["color"] = torch.as_tensor(
                np.asarray(data["color"], dtype=np.float32) / 127.5 - 1.0
            )
        return out

    def extract_backbone_features(self, data):
        self.point_features = {"coord": data["coord"]}

    def segment(self, data, prompt_points, text_prompt=None, return_tokens=False):
        coord = data["coord"].numpy()
        masks, labels, ious = [], [], []
        for prompt in prompt_points:
            center = np.asarray(prompt, dtype=np.float32).reshape(-1, 3).mean(axis=0)
            distance = np.linalg.norm(coord - center, axis=1)
            masks.append(distance <= np.quantile(distance, 0.25))
            labels.append("wall")
            ious.append(np.array([0.9], dtype=np.float32))
        return masks, labels, ious
```

Do not let the fake call CUDA, CLIP, or the real checkpoint. If the GUI passes a CUDA
device, the fake may accept it but must keep the test independent of GPU availability.

## 2. GPU smoke test

Gate the real model on an explicit marker or environment variable. Assert construction,
device placement, one forward pass, and output shapes.

```python
pytestmark = pytest.mark.gpu

@pytest.mark.skipif(
    os.environ.get("SNAP_RUN_GPU_SMOKE") != "1",
    reason="set SNAP_RUN_GPU_SMOKE=1 to run the real GPU smoke",
)
def test_real_model_one_forward_pass():
    assert torch.cuda.is_available(), "GPU smoke requested but CUDA is unavailable"
    torch.manual_seed(0)
    model = build_real_model().cuda().eval()
    batch = make_fixed_synthetic_batch(points=256).to("cuda")
    with torch.no_grad():
        out = model(batch)
    assert out_is_on_cuda_and_finite(out)
    print("peak_cuda_mb", torch.cuda.max_memory_allocated() / (1024 * 1024))
```

An explicit skip is acceptable for an ordinary developer run. In a job that claims GPU
coverage, turn the same condition into a failure (or assert the marker was selected) so a
missing GPU cannot masquerade as passing coverage.

## 3. Real GUI integration and screenshot validation

When the request is real-checkpoint GUI operation, drive the real wrapper through the actual
plotter and assert on the rendered result. This is a separate layer from both the fake
interaction test and the headless forward-pass smoke.

```python
@pytest.mark.gpu
def test_real_model_click_renders_mask(qapp):
    model = build_real_model().cuda().eval()
    plotter = InteractivePlotter(args, fixed_scene(), pv.Plotter(off_screen=True))
    plotter.add_click(fixed_click_xyz)
    plotter.segment_from_points()
    assert plotter.finalized_objects, "no mask reached the plotter"
    image = plotter.plotter.screenshot(return_img=True)
    assert image.shape[0] > 0 and image.std() > 0, "blank render"
```

Use authorized real data locally when the task needs it; keep the scene, screenshots and
logs private. A synthetic scene is the shareable fixture, not a substitute for the real-data
integration run unless the user accepts that scope.

## 4. Command shape

```bash
# interaction layer, no real model
QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true pixi run pytest -q tests/test_gui_interaction.py

# real inference layer, explicit opt-in
SNAP_RUN_GPU_SMOKE=1 pixi run pytest -q tests/test_gpu_model_smoke.py

# real GUI layer; Xvfb is a separate process on hosts where offscreen VTK fails
SNAP_RUN_GPU_GUI=1 xvfb-run -a pixi run pytest -q tests/test_gui_real_model.py
```

Capture the command, exit code, wall time, GPU name/capability, selected backend, display
mode, data provenance and peak memory in the receipt. If a layer did not run, record
`not executed` rather than folding it into the other layer's pass.
