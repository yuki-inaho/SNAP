#!/usr/bin/env python3
"""Fast regression checks for SNAP attention backend selection and chunking.

This script lives in tools/ (tests/ is owned by another worker) and does not
need a checkpoint or dataset.  It checks:

1. device capability / package resolution (CPU, unsupported GPU, mock sm80);
2. explicit auto / flash / normal and SNAP_ATTENTION_BACKEND behaviour;
3. per-device selection including a mixed sm75 + sm80 visible-GPU mock;
4. SNAP wiring for explicit backend and environment overrides;
5. chunked vs unchunked attention produces the same output;
6. CPU execution while CUDA is visible, after constructing a flash-enabled
   module (runtime fallback must not call flash_attn on CPU).

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.ptv3_backbone_domain as ptv3  # noqa: E402
import spconv.pytorch as spconv  # noqa: E402
from src.ptv3_backbone_domain import (  # noqa: E402
    AutocastSafeSubMConv3d,
    Point,
    SerializedAttention,
    get_flash_attention_status,
    resolve_attention_backend,
)

CHECKS = []


def check(name, condition, detail=""):
    ok = bool(condition)
    CHECKS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def make_point(feat, serialized_order=None, serialized_inverse=None):
    n = feat.shape[0]
    order = torch.arange(n) if serialized_order is None else serialized_order
    inverse = torch.arange(n) if serialized_inverse is None else serialized_inverse
    return Point(
        feat=feat,
        offset=torch.tensor([n], dtype=torch.long),
        serialized_order=[order],
        serialized_inverse=[inverse],
    )


def attention_output(module, feat, chunk_bytes, fp16=False):
    old = os.environ.get("SNAP_ATTN_CHUNK_BYTES")
    os.environ["SNAP_ATTN_CHUNK_BYTES"] = str(chunk_bytes)
    try:
        point = make_point(feat.clone())
        with torch.no_grad(), torch.autocast(
            device_type=feat.device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            return module(point).feat.clone()
    finally:
        if old is None:
            os.environ.pop("SNAP_ATTN_CHUNK_BYTES", None)
        else:
            os.environ["SNAP_ATTN_CHUNK_BYTES"] = old


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_flash_available, real_reason = get_flash_attention_status(device)
    print(f"host: cuda={torch.cuda.is_available()} backend_reason={real_reason}")

    # 1. Basic resolution.
    check("CPU reports FlashAttention-2 unavailable",
          get_flash_attention_status("cpu")[0] is False)
    check("normal always resolves to normal",
          resolve_attention_backend("normal", device) == "normal")
    check("auto matches real host capability",
          resolve_attention_backend("auto", device)
          == ("flash" if real_flash_available else "normal"))
    if not real_flash_available:
        try:
            resolve_attention_backend("flash", device)
            strict_rejected = False
        except RuntimeError as exc:
            strict_rejected = "FlashAttention-2" in str(exc)
        check("explicit flash fails clearly on unsupported host", strict_rejected)
    try:
        resolve_attention_backend("not-a-backend", device)
        invalid_rejected = False
    except ValueError:
        invalid_rejected = True
    check("invalid backend raises ValueError", invalid_rejected)

    # 2. Environment variable handling.
    old_env = os.environ.get("SNAP_ATTENTION_BACKEND")
    try:
        os.environ["SNAP_ATTENTION_BACKEND"] = "normal"
        check("env normal is honoured",
              resolve_attention_backend(None, device) == "normal")
        os.environ["SNAP_ATTENTION_BACKEND"] = "flash"
        try:
            resolve_attention_backend(None, device)
            env_flash_rejected = False
        except RuntimeError:
            env_flash_rejected = True
        check("env flash fails clearly on unsupported host", env_flash_rejected)
        os.environ["SNAP_ATTENTION_BACKEND"] = "auto"
        check("env auto matches host",
              resolve_attention_backend(None, device)
              == ("flash" if real_flash_available else "normal"))
    finally:
        if old_env is None:
            os.environ.pop("SNAP_ATTENTION_BACKEND", None)
        else:
            os.environ["SNAP_ATTENTION_BACKEND"] = old_env

    # 3. Mock a supported device: auto and flash must select flash.
    real_status = ptv3.get_flash_attention_status
    ptv3.get_flash_attention_status = lambda device=None: (True, "mock sm80")
    try:
        check("mock sm80 auto selects flash",
              ptv3.resolve_attention_backend("auto") == "flash")
        check("mock sm80 explicit flash selects flash",
              ptv3.resolve_attention_backend("flash") == "flash")
        check("mock sm80 normal stays normal",
              ptv3.resolve_attention_backend("normal") == "normal")
    finally:
        ptv3.get_flash_attention_status = real_status

    # 4. Mixed visible GPUs: conservative auto, explicit cuda:1 can use flash.
    real_count = torch.cuda.device_count
    real_cap = torch.cuda.get_device_capability
    real_name = torch.cuda.get_device_name
    real_flash = ptv3.flash_attn
    torch.cuda.device_count = lambda: 2
    torch.cuda.get_device_capability = lambda index=0: (7, 5) if index == 0 else (8, 0)
    torch.cuda.get_device_name = lambda index=0: f"mock-cuda-{index}"
    ptv3.flash_attn = type(
        "MockFlashAttn",
        (),
        {
            "__version__": "2.6.3",
            "flash_attn_varlen_qkvpacked_func": staticmethod(lambda *a, **k: None),
        },
    )()
    try:
        check("mixed visible GPUs make auto conservative",
              resolve_attention_backend("auto", None) == "normal")
        check("attention_device=cuda:1 pins flash on supported GPU",
              resolve_attention_backend("auto", "cuda:1") == "flash")
        check("explicit flash on cuda:1 succeeds",
              resolve_attention_backend("flash", "cuda:1") == "flash")
    finally:
        torch.cuda.device_count = real_count
        torch.cuda.get_device_capability = real_cap
        torch.cuda.get_device_name = real_name
        ptv3.flash_attn = real_flash

    # 5. SNAP wiring (small config test; no checkpoint or data needed).
    from src.snap import SNAP
    model = SNAP(
        num_points=1,
        num_merge_blocks=1,
        use_pdnorm=True,
        return_mid_points=True,
        attention_backend="normal",
    )
    attention_layers = [
        module for module in model.modules() if isinstance(module, SerializedAttention)
    ]
    check("SNAP explicit normal wires every attention layer",
          model.attention_backend == "normal"
          and len(attention_layers) > 0
          and all(not layer.enable_flash for layer in attention_layers),
          f"layers={len(attention_layers)}")
    try:
        SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True,
             attention_backend="flash")
        snap_flash_rejected = False
    except RuntimeError:
        snap_flash_rejected = True
    check("SNAP explicit flash fails clearly on unsupported host",
          snap_flash_rejected)
    del model, attention_layers

    # 6. Chunked vs unchunked attention (random weights are fine here: this is
    #    a math-equivalence test, not a pretrained-inference claim).
    torch.manual_seed(0)
    module = SerializedAttention(
        channels=64,
        num_heads=4,
        patch_size=64,
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    ).eval()
    feat = torch.randn(256, 64)
    chunked = attention_output(module, feat, 64 * 1024)  # one patch at a time
    unchunked = attention_output(module, feat, 10**12)   # all patches at once
    diff = (chunked - unchunked).abs()
    check("chunked attention equals unchunked attention",
          torch.allclose(chunked, unchunked, atol=1e-6, rtol=1e-6),
          f"max_abs_diff={diff.max().item():.3e}")

    # 7. CPU execution with CUDA visible, after a flash-enabled construction.
    #    The runtime check must choose the normal path on CPU.
    if torch.cuda.is_available():
        real_flash = ptv3.flash_attn
        ptv3.get_flash_attention_status = lambda device=None: (True, "mock sm80")
        ptv3.flash_attn = object()  # satisfy the constructor's optional-package check
        try:
            flash_layout = SerializedAttention(
                channels=64,
                num_heads=4,
                patch_size=64,
                enable_flash=True,
                enable_rpe=False,
                upcast_attention=False,
                upcast_softmax=False,
            ).eval()
        finally:
            ptv3.get_flash_attention_status = real_status
            ptv3.flash_attn = real_flash
        flash_layout = flash_layout.cpu()
        normal_layout = SerializedAttention(
            channels=64,
            num_heads=4,
            patch_size=64,
            enable_flash=False,
            enable_rpe=False,
            upcast_attention=False,
            upcast_softmax=False,
        ).eval()
        normal_layout.load_state_dict(flash_layout.state_dict())
        cpu_feat = torch.randn(256, 64)
        flash_cpu = attention_output(flash_layout, cpu_feat, 10**12)
        normal_cpu = attention_output(normal_layout, cpu_feat, 10**12)
        check("flash layout runs on CPU with visible CUDA via runtime fallback",
              flash_cpu.device.type == "cpu"
              and bool(torch.isfinite(flash_cpu).all())
              and torch.allclose(flash_cpu, normal_cpu, atol=1e-6, rtol=1e-6),
              f"device={flash_cpu.device} max_abs_diff="
              f"{(flash_cpu - normal_cpu).abs().max().item():.3e}")
    else:
        check("CPU runtime fallback check skipped (CUDA not visible)", True)

    # 8. FP16 autocast path (attention + spconv wrapper).
    if torch.cuda.is_available():
        cuda_module = SerializedAttention(
            channels=64,
            num_heads=4,
            patch_size=64,
            enable_flash=False,
            enable_rpe=False,
            upcast_attention=False,
            upcast_softmax=False,
        ).cuda().eval()
        cuda_feat = torch.randn(256, 64, device="cuda")
        fp32_out = attention_output(cuda_module, cuda_feat, 10**12)
        fp16_out = attention_output(cuda_module, cuda_feat, 10**12, fp16=True)
        fp16_diff = (fp16_out.float() - fp32_out).abs()
        check("FP16 autocast attention is finite and close to FP32",
              bool(torch.isfinite(fp16_out).all())
              and torch.allclose(
                  fp16_out.float(), fp32_out, atol=5e-2, rtol=5e-2
              ),
              f"max_abs_diff={fp16_diff.max().item():.3e}")

        torch.manual_seed(0)
        n = 32
        features = torch.randn(n, 16, device="cuda")
        coords = torch.unique(
            torch.randint(0, 16, (n, 3), device="cuda"), dim=0
        )[:n]
        n = coords.shape[0]
        indices = torch.cat(
            [
                torch.zeros(n, 1, dtype=torch.int32, device="cuda"),
                coords.int(),
            ],
            dim=1,
        )
        sparse = spconv.SparseConvTensor(features[:n], indices, [16, 16, 16], 1)
        conv = AutocastSafeSubMConv3d(16, 16, kernel_size=3, bias=True).cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            sparse_out = conv(sparse)
        check("spconv autocast wrapper keeps FP32 features and is finite",
              sparse_out.features.dtype == torch.float32
              and bool(torch.isfinite(sparse_out.features).all())
              and sparse_out.features.shape[0] == n,
              f"dtype={sparse_out.features.dtype}")

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ATTENTION_BACKEND_REGRESSION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
