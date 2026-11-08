"""Opt-in FP16 autocast policy for the Python API and interactive GUI.

The heavy SNAP/CLIP/dataset objects are replaced with small recording fakes;
the precision policy, autocast placement and FP32 post-processing run for real.
GPU-specific checks are skipped when CUDA is unavailable.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gui_test_support import FakeModel, make_cloud  # noqa: E402


def _import_demo(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"{module_name} cannot be imported in this environment ({exc!r})")


@pytest.fixture(scope="module")
def python_demo():
    return _import_demo("python_demo")


@pytest.fixture(scope="module")
def gui_demo():
    return _import_demo("iterative_gui_demo")


# ---------------------------------------------------------------------------
# Recording fakes
# ---------------------------------------------------------------------------


def _autocast_dtype() -> torch.dtype:
    if torch.is_autocast_enabled():
        return torch.get_autocast_dtype("cuda")
    return torch.float32


class RecordingDataset:
    labels = ["floor", "wall", "chair"]

    def __init__(self, *args, **kwargs):
        pass

    def process_data(self, coord, color=None, normal=None, intensity=None):
        return {"coord": torch.as_tensor(np.asarray(coord, dtype=np.float32))}

    def process_prompts(self, data, prompt_points, text_prompt):
        out = dict(data)
        out["point"] = torch.as_tensor(np.asarray(prompt_points, dtype=np.float32))
        return out


class RecordingSNAP(torch.nn.Module):
    """Records whether each stage ran inside autocast."""

    mask_threshold = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[str, bool, torch.dtype]] = []

    def _record(self, name: str) -> None:
        self.calls.append((name, torch.is_autocast_enabled(), _autocast_dtype()))

    def run_input_encoder(self, data):
        self._record("input_encoder")
        out = dict(data)
        # Emulate the real feature concatenation: preprocessing is FP32.
        out["feat"] = torch.cat([out["coord"], out["coord"]], dim=1).float()
        out["offset"] = torch.tensor(
            [out["coord"].shape[0]], dtype=torch.long, device=out["coord"].device
        )
        return out

    def run_backbone(self, data):
        self._record("backbone")
        dtype = _autocast_dtype()
        point = {
            "coord": data["coord"],
            "feat": torch.zeros(
                (data["coord"].shape[0], 8), dtype=dtype, device=data["coord"].device
            ),
            "offset": data["offset"],
        }
        return point, None

    def run_mask_decoder(self, point, data_dict, val=False):
        self._record("mask_decoder")
        dtype = _autocast_dtype()
        device = data_dict["coord"].device
        count = data_dict["coord"].shape[0]
        # Match SNAP.run_mask_decoder: one list element per batch entry, each
        # with shape (M, N), (M, 512) and (M, 1) respectively.
        logits = torch.zeros((1, count), dtype=dtype, device=device)
        text = torch.ones((1, 512), dtype=dtype, device=device)
        iou = torch.ones((1, 1), dtype=dtype, device=device)
        return [logits], [text], [iou], [], [], [], []

    def calls_for(self, name: str):
        return [call for call in self.calls if call[0] == name]


class RecordingClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[bool, torch.dtype]] = []

    def encode_text(self, tokens):
        self.calls.append((torch.is_autocast_enabled(), _autocast_dtype()))
        dtype = _autocast_dtype()
        return torch.ones(
            (tokens.shape[0], 512), dtype=dtype, device=tokens.device
        )


def _install_fakes(module, monkeypatch):
    clip_model = RecordingClip()
    monkeypatch.setattr(module, "SNAP", RecordingSNAP)
    monkeypatch.setattr(module, "DemoDatset", RecordingDataset)
    monkeypatch.setattr(module.clip, "load", lambda *args, **kwargs: (clip_model, None))
    monkeypatch.setattr(
        module.clip, "tokenize", lambda text: torch.zeros((1, 77), dtype=torch.long)
    )
    return clip_model


def _cloud(points: int = 32) -> dict:
    coord = np.random.default_rng(0).normal(size=(points, 3)).astype(np.float32)
    return {"coord": coord}


def _assert_fp32_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, f"{name} was converted to {parameter.dtype}"


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def test_precision_names_are_strict(python_demo):
    assert python_demo.normalize_precision("fp32") == "fp32"
    assert python_demo.normalize_precision("FP16") == "fp16"
    assert python_demo.normalize_precision(None) == "fp32"
    with pytest.raises(ValueError, match="Unsupported precision"):
        python_demo.normalize_precision("bf16")


def test_fp16_is_rejected_on_cpu_and_other_backends(python_demo):
    with pytest.raises(RuntimeError, match="CUDA device"):
        python_demo.resolve_precision("fp16", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="CUDA device"):
        python_demo.resolve_precision("fp16", torch.device("mps"))


def test_fp32_context_is_a_noop(python_demo):
    with python_demo.precision_autocast(torch.device("cpu"), "fp32"):
        assert not torch.is_autocast_enabled()
        assert torch.ones(2).matmul(torch.ones(2)).dtype == torch.float32
    assert not torch.is_autocast_enabled()


def test_fp16_context_requires_cuda(python_demo):
    with pytest.raises(RuntimeError, match="CUDA"):
        with python_demo.precision_autocast(torch.device("cpu"), "fp16"):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_autocast_uses_fp16_math_without_converting_parameters(python_demo):
    device = torch.device("cuda")
    linear = torch.nn.Linear(4, 4).to(device)
    x = torch.randn(2, 4, device=device)

    with python_demo.precision_autocast(device, "fp16"):
        assert torch.is_autocast_enabled()
        out = linear(x)

    assert out.dtype == torch.float16
    assert linear.weight.dtype == torch.float32
    assert not torch.is_autocast_enabled()


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def test_python_api_defaults_to_fp32_on_cpu(python_demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    clip_model = _install_fakes(python_demo, monkeypatch)

    model = python_demo.SegmentationModel(domain="Indoor", grid_size=0.02)
    assert model.precision == "fp32"
    assert model.device.type == "cpu"

    data = model.intialize_pointcloud(_cloud())
    model.extract_backbone_features(data)
    masks, labels, ious = model.segment(data, [[[0.0, 0.0, 0.0]]], None)

    assert [call[1] for call in model.model.calls] == [False, False, False]
    assert [call[0] for call in model.model.calls] == [
        "input_encoder",
        "backbone",
        "mask_decoder",
    ]
    assert all(enabled is False for enabled, _ in clip_model.calls)
    assert len(masks) == 1 and masks[0].dtype == np.bool_
    assert ious[0].dtype == np.float32 and np.isfinite(ious[0]).all()
    assert labels[0] in RecordingDataset.labels
    _assert_fp32_parameters(model.model)


def test_python_api_rejects_fp16_on_cpu(python_demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        python_demo.SegmentationModel(domain="Indoor", precision="fp16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_python_api_fp16_autocasts_model_stages_only(python_demo, monkeypatch):
    clip_model = _install_fakes(python_demo, monkeypatch)

    model = python_demo.SegmentationModel(
        domain="Indoor", grid_size=0.02, precision="fp16"
    )
    assert model.precision == "fp16"
    assert model.device.type == "cuda"

    data = model.intialize_pointcloud(_cloud())
    model.extract_backbone_features(data)
    masks, labels, ious = model.segment(data, [[[0.0, 0.0, 0.0]]], None)

    # Preprocessing stays FP32; backbone and decoder run under FP16 autocast.
    assert model.model.calls_for("input_encoder")[0][1:] == (False, torch.float32)
    for name in ("backbone", "mask_decoder"):
        assert model.model.calls_for(name)[0][1:] == (True, torch.float16)
    assert all(enabled is True for enabled, _ in clip_model.calls)
    assert all(dtype == torch.float16 for _, dtype in clip_model.calls)
    assert len(masks) == 1 and masks[0].dtype == np.bool_
    assert ious[0].dtype == np.float32 and np.isfinite(ious[0]).all()
    assert labels[0] in RecordingDataset.labels
    _assert_fp32_parameters(model.model)


# ---------------------------------------------------------------------------
# Interactive GUI
# ---------------------------------------------------------------------------


def test_gui_parser_defaults_to_fp32_and_accepts_fp16(gui_demo):
    parser = gui_demo.get_args_parser()
    assert parser.parse_args([]).precision == "fp32"
    assert parser.parse_args(["--precision", "fp16"]).precision == "fp16"
    with pytest.raises(SystemExit):
        parser.parse_args(["--precision", "bf16"])


def test_gui_interactive_plotter_forwards_precision(gui_demo, monkeypatch):
    received = []

    class SpyModel(FakeModel):
        def __init__(self, args, *positional, **keyword):
            super().__init__(args, *positional, **keyword)
            received.append(args.precision)

    monkeypatch.setattr(gui_demo, "SegmentationModel", SpyModel)
    args = gui_demo.get_args_parser().parse_args(
        ["--domain", "Indoor", "--resume", "unused.pth", "--precision", "fp16"]
    )

    gui_demo.InteractivePlotter(args, make_cloud(), object())

    assert received == ["fp16"]


def test_gui_rejects_fp16_on_cpu(gui_demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = gui_demo.get_args_parser().parse_args(
        ["--domain", "Indoor", "--resume", "unused.pth", "--precision", "fp16"]
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        gui_demo.SegmentationModel(args, checkpoint_path=None, domain="Indoor")


def test_gui_default_fp32_does_not_enable_autocast(gui_demo, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _install_fakes(gui_demo, monkeypatch)
    args = gui_demo.get_args_parser().parse_args(
        ["--domain", "Indoor", "--resume", "unused.pth", "--num_object_chunk", "2"]
    )

    model = gui_demo.SegmentationModel(args, checkpoint_path=None, domain="Indoor")
    data = model.intialize_pointcloud(_cloud())
    model.extract_backbone_features(data)
    masks, labels, ious = model.segment(data, [[[0.0, 0.0, 0.0]]], None)

    assert model.precision == "fp32"
    assert [call[1] for call in model.model.calls] == [False, False, False]
    assert len(masks) == 1 and masks[0].dtype == np.bool_
    assert ious[0].dtype == np.float32 and np.isfinite(ious[0]).all()
    assert labels[0] in RecordingDataset.labels
    _assert_fp32_parameters(model.model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gui_fp16_autocasts_model_stages_only(gui_demo, monkeypatch):
    clip_model = _install_fakes(gui_demo, monkeypatch)
    args = gui_demo.get_args_parser().parse_args(
        [
            "--domain",
            "Indoor",
            "--resume",
            "unused.pth",
            "--precision",
            "fp16",
            "--num_object_chunk",
            "2",
        ]
    )

    model = gui_demo.SegmentationModel(args, checkpoint_path=None, domain="Indoor")
    data = model.intialize_pointcloud(_cloud())
    model.extract_backbone_features(data)
    masks, labels, ious = model.segment(data, [[[0.0, 0.0, 0.0]]], None)

    assert model.model.calls_for("input_encoder")[0][1:] == (False, torch.float32)
    for name in ("backbone", "mask_decoder"):
        assert model.model.calls_for(name)[0][1:] == (True, torch.float16)
    assert all(enabled is True for enabled, _ in clip_model.calls)
    assert len(masks) == 1 and masks[0].dtype == np.bool_
    assert ious[0].dtype == np.float32 and np.isfinite(ious[0]).all()
    assert labels[0] in RecordingDataset.labels
    _assert_fp32_parameters(model.model)
