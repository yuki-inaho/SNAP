# SNAP developer commands.
#
# Everything runs through the locked Pixi environment (CUDA 12.4 / PyTorch 2.4.1).
# Run `just` with no arguments to list the recipes.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Install the locked environment (safe on the RTX 2070 / sm75 host).
install:
    pixi install --locked

# Re-solve dependencies and refresh pixi.lock.
lock:
    pixi lock

# Show the Python, PyTorch, CUDA and GPU versions actually in use.
gpu:
    pixi run python -c 'import torch; print("python :", __import__("sys").version.split()[0]); print("torch  :", torch.__version__); print("cuda   :", torch.version.cuda); print("driver :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"); print("capab  : sm%d%d" % torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None'

# Import every model, dataset and GUI module without running the network.
imports:
    pixi run python -c 'import torch, torchvision, torch_scatter, torch_cluster; import spconv.pytorch, timm, clip, hdbscan; import open3d, pyvista, pyvistaqt, PyQt6.QtWidgets; import datasets, src.snap, main, vis_all; print("all imports OK")'

# Deep check: CUDA matmul, PyG scatter/knn kernels and a spconv SubMConv3d.
validate:
    #!/usr/bin/env bash
    set -euo pipefail
    pixi run python - <<'PY'
    import torch
    import torch_scatter
    import torch_cluster
    import spconv.pytorch as spconv

    assert torch.cuda.is_available(), "CUDA is not available"
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("gpu  ", torch.cuda.get_device_name(0),
          "sm%d%d" % torch.cuda.get_device_capability(0))

    a = torch.randn(4096, 4096, device="cuda")
    assert torch.isfinite(a @ a).all()
    print("cuda matmul        OK")

    idx = torch.zeros(4096, dtype=torch.long, device="cuda")
    out = torch_scatter.scatter_add(a[:, 0], idx, dim=0)
    assert out.shape == (1,)
    print("torch_scatter      OK", torch_scatter.__version__)

    x = torch.randn(512, 3, device="cuda")
    y = torch.randn(512, 3, device="cuda")
    dist, arg = torch_cluster.knn(x, y, 8)
    assert dist.numel() == 512 * 8
    print("torch_cluster      OK", torch_cluster.__version__)

    coords = torch.unique(
        torch.randint(0, 16, (256, 3), device="cuda", dtype=torch.int32), dim=0
    )
    batch_col = torch.zeros(coords.shape[0], 1, device="cuda", dtype=torch.int32)
    indices = torch.cat([batch_col, coords], dim=1)
    feats = torch.randn(coords.shape[0], 8, device="cuda")
    tensor = spconv.SparseConvTensor(feats, indices, [16, 16, 16], 1)
    conv = spconv.SubMConv3d(8, 16, kernel_size=3, indice_key="subm").cuda()
    result = conv(tensor)
    assert result.features.shape == (coords.shape[0], 16)
    print("spconv sm75 kernel OK")
    print("VALIDATION OK")
    PY

# Construct SNAP on the GPU with the attention backend selected automatically.
# Uses the regular-attention fallback when FlashAttention-2 is unavailable.
model-smoke:
    pixi run python -c 'import torch; from src.snap import SNAP; m = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True).cuda().eval(); n = sum(p.numel() for p in m.parameters()); print("SNAP constructed on", next(m.parameters()).device, "params", f"{n:,}")'

# Check attention backend selection and the memory-bounded implementation.
test-attention:
    pixi run python tools/check_attention_backends.py

# Run real-checkpoint inference on the labelled ScanNet sample; pass extra flags as needed.
verify-inference *args:
    pixi run python tools/verify_gpu_inference.py {{args}}

# Compare FP32 and mixed FP16 inference on the same ScanNet prompts.
benchmark-precision *args:
    pixi run python tools/benchmark_precision.py {{args}}

# Optional: build FlashAttention-2.  Refuses to run on Turing (sm75) because
# flash-attn 2.x requires sm80+; SNAP falls back to PyTorch attention there.
install-flash:
    #!/usr/bin/env bash
    set -euo pipefail
    sm=$(pixi run python -c 'import torch; a, b = torch.cuda.get_device_capability(0); print(a * 10 + b)')
    if [ "$sm" -lt 80 ]; then
        echo "FlashAttention-2 requires sm80+; detected sm${sm}. Refusing to install." >&2
        echo "SNAP uses the PyTorch attention fallback automatically on this GPU." >&2
        exit 1
    fi
    if ! pixi run bash -c 'command -v nvcc' >/dev/null 2>&1; then
        echo "nvcc not found; install a CUDA toolkit matching the PyTorch CUDA runtime." >&2
        exit 1
    fi
    pixi run pip install --no-build-isolation "flash-attn==2.6.3"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run non-GUI tests (GUI tests run separately via test-gui).
test *args:
    pixi run pytest -q tests --ignore=tests/test_gui_interaction.py --ignore=tests/test_gui_qt.py {{args}}

# Run only the Qt/VTK GUI tests with an off-screen display.
test-gui:
    xvfb-run -a env QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true pixi run pytest -q tests/test_gui_interaction.py

# Exercise real Qt widgets and VTK rendering on an isolated virtual display.
test-gui-qt:
    SNAP_QT_GUI_TESTS=1 xvfb-run -a pixi run pytest -q tests/test_gui_qt.py

# Run the suites in separate processes to isolate Qt/VTK and CUDA state.
test-all:
    just test
    just test-gui
    just test-gui-qt

# ---------------------------------------------------------------------------
# Demos / training / evaluation
# ---------------------------------------------------------------------------

# Interactive GUI; pass fp16 as the fifth argument to enable CUDA autocast.
demo pc_path domain="Indoor" grid_size="" checkpoint="checkpoints/SNAP_C.pth" precision="fp32":
    pixi run python iterative_gui_demo.py --pc_path {{quote(pc_path)}} --domain {{quote(domain)}} {{ if grid_size == "" { "" } else { "--grid_size " + quote(grid_size) } }} --precision {{quote(precision)}} --resume {{quote(checkpoint)}}

# Low-memory GUI: FP16 autocast and a 64 MiB attention budget.
demo-lowmem pc_path domain="Indoor" grid_size="" checkpoint="checkpoints/SNAP_C.pth":
    SNAP_ATTN_CHUNK_BYTES=67108864 pixi run python iterative_gui_demo.py --pc_path {{quote(pc_path)}} --domain {{quote(domain)}} {{ if grid_size == "" { "" } else { "--grid_size " + quote(grid_size) } }} --precision fp16 --resume {{quote(checkpoint)}}

# Distributed training entry point (all extra arguments are forwarded).
train *args:
    pixi run torchrun --nproc_per_node=1 main.py {{args}}

# Evaluation entry point (all extra arguments are forwarded).
eval *args:
    pixi run torchrun --nproc_per_node=1 main.py --val_only {{args}}

# Visualize predictions on a complete dataset.
vis *args:
    pixi run torchrun --nproc_per_node=1 vis_all.py {{args}}
