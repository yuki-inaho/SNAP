#!/usr/bin/env python3
"""End-to-end validation of the real Studio backend.

Runs only against a local server (default 127.0.0.1:9502), uploads the bundled
ScanNet sample, performs a positive-only and a positive+negative inference with
the same real scene coordinates, downloads both export packages and records
evidence.  It never reads or prints credentials and makes no external calls.

Usage:
    env/.pixi/envs/default/bin/python scripts/validate_backend.py \
        --base-url http://127.0.0.1:9502 \
        --ply outputs/ply/scene0025_00.ply \
        --out outputs/api_validation
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import requests

# Real coordinates from the bundled ScanNet scene0025_00 chair.
POSITIVE = [2.8424025, 1.3025417, 0.5630025]
NEGATIVE = [2.897325277328491, 1.533718466758728, 0.19096481800079346]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post_infer(base: str, clicks: dict, object_names: list[str]) -> dict:
    payload = {
        "clickData": {
            "clickIdx": {},
            "clickTimeIdx": {k: {p: [i for i in range(len(v))] for p, v in parts.items()} for k, parts in clicks.items()},
            "clickPositions": clicks,
            "targetId": 1,
        },
        "cubeSize": 0.02,
        "objectNames": object_names,
    }
    r = requests.post(f"{base}/api/infer/", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def check_export_metadata(name: str, export_info: dict, mask: np.ndarray, out: Path) -> dict:
    """Assert the downloaded metadata is self-consistent with mask and PLY."""
    md = export_info["metadata"]
    object_ids = {int(obj["id"]) for obj in md["objects"]}
    count_ids = {int(k) for k in md["object_counts"]}
    assert object_ids == count_ids, f"{name}: objects {object_ids} != counts {count_ids}"

    mask_labels = {int(k): int(v) for k, v in md["mask_label_counts"].items()}
    for obj in md["objects"]:
        label = int(obj["mask_label"])
        expected_id = label % 100 if label >= 100 else label
        assert int(obj["points"]) == mask_labels[label], f"{name}: bad points for {obj}"
        assert int(obj["id"]) == expected_id, f"{name}: object id {obj['id']} != {expected_id}"

    clicks = md.get("click_data", [])
    negatives = [c for c in clicks if int(c["obj_idx"]) == 0]
    positives = [c for c in clicks if int(c["obj_idx"]) != 0]
    assert all(c["is_positive"] is False for c in negatives), f"{name}: negative flag"
    assert positives and all(c["is_positive"] is True for c in positives), f"{name}: positive flag"
    assert all(int(c["part_idx"]) == int(c["obj_idx"]) for c in clicks), f"{name}: part_idx lost"

    pc = o3d.io.read_point_cloud(str(out / f"{name}_extracted" / "scene_with_colored_objects.ply"))
    colors = np.asarray(pc.colors)
    colored = ~np.all(np.abs(colors - 0.7) < 0.01, axis=1)
    assert int(colored.sum()) == int((mask > 0).sum()), f"{name}: PLY/mask count mismatch"
    first_color = colors[np.flatnonzero(colored)[0]]
    md_color = np.asarray(md["objects"][0]["color"], dtype=float)
    assert np.allclose(first_color, md_color, atol=1.0 / 255.0 + 1e-6), (
        f"{name}: PLY colour {first_color} != metadata {md_color}"
    )
    return {
        "object_ids": sorted(object_ids),
        "object_counts": md["object_counts"],
        "mask_label_counts": md["mask_label_counts"],
        "object_part_counts": md["object_part_counts"],
        "negative_clicks": len(negatives),
        "positive_clicks": len(positives),
        "ply_colored_points": int(colored.sum()),
        "color_match": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9502")
    ap.add_argument("--ply", default="outputs/ply/scene0025_00.ply")
    ap.add_argument("--out", default="outputs/api_validation")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ply = Path(args.ply).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"base_url": base, "ply": str(ply), "runs": {}}

    log(f"health {base}/health")
    h = requests.get(f"{base}/health", timeout=10)
    h.raise_for_status()
    evidence["health"] = h.json()
    log(f"health -> {evidence['health']['status']}")

    def upload() -> dict:
        with ply.open("rb") as fh:
            t0 = time.time()
            r = requests.post(
                f"{base}/api/upload/",
                files={"file": (ply.name, fh, "application/octet-stream")},
                timeout=300,
            )
            elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        data["elapsed_s"] = round(elapsed, 3)
        log(f"upload -> {data['point_count']} pts in {elapsed:.2f}s")
        return data

    def export(name: str) -> dict:
        r = requests.get(f"{base}/api/download/results", timeout=300)
        r.raise_for_status()
        zpath = out / f"{name}_results.zip"
        zpath.write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            for member in names:
                zf.extract(member, out / f"{name}_extracted")
            metadata = json.loads(zf.read("metadata.json"))
        log(f"export -> {zpath.name} {len(r.content)} bytes, members={names}")
        return {
            "zip": str(zpath),
            "bytes": len(r.content),
            "members": names,
            "metadata": metadata,
        }

    def run(name: str, clicks: dict, names: list[str]) -> dict:
        t0 = time.time()
        resp = post_infer(base, clicks, names)
        elapsed = time.time() - t0
        seg = resp["segmentedPointCloud"]
        mask = np.asarray(seg["segmentation"], dtype=np.int32)
        np.save(out / f"{name}_mask.npy", mask)
        info = {
            "elapsed_s": round(elapsed, 3),
            "object_counts": seg["object_counts"],
            "metadata": seg.get("metadata"),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()[:16],
            "foreground_points": int((mask > 0).sum()),
        }
        evidence["runs"][name] = info
        log(
            f"infer[{name}] -> {info['elapsed_s']}s, counts={info['object_counts']}, "
            f"mask={info['mask_sha256']}"
        )
        return info

    evidence["upload_positive_only"] = upload()
    pos_only = run("positive_only", {"1": {"1": [POSITIVE]}}, ["chair"])
    evidence["export_positive_only"] = export("positive_only")

    evidence["upload_positive_negative"] = upload()
    pos_neg = run(
        "positive_plus_negative",
        {"0": {"0": [NEGATIVE]}, "1": {"1": [POSITIVE]}},
        ["chair"],
    )
    evidence["export_positive_negative"] = export("positive_plus_negative")

    m_pos = np.load(out / "positive_only_mask.npy")
    m_pn = np.load(out / "positive_plus_negative_mask.npy")
    diff = np.flatnonzero(m_pos != m_pn)
    evidence["comparison"] = {
        "changed_points": int(diff.size),
        "positive_only_foreground": int((m_pos > 0).sum()),
        "positive_plus_negative_foreground": int((m_pn > 0).sum()),
        "labels_positive_only": sorted(int(x) for x in np.unique(m_pos) if x > 0),
        "labels_positive_plus_negative": sorted(int(x) for x in np.unique(m_pn) if x > 0),
    }
    log(f"native negative click changed {diff.size} points")

    evidence["metadata_checks"] = {
        "positive_only": check_export_metadata(
            "positive_only", evidence["export_positive_only"], m_pos, out
        ),
        "positive_plus_negative": check_export_metadata(
            "positive_plus_negative", evidence["export_positive_negative"], m_pn, out
        ),
    }
    log("export metadata checks passed (ids, counts, negative flag, colour)")

    # Optional recognition must be explicitly disabled without a key.
    r = requests.post(
        f"{base}/api/recognition/mask_obj_recognition",
        json={"mask": [0, 1, 1, 0]},
        timeout=30,
    )
    evidence["recognition_without_key"] = {"status": r.status_code, "body": r.json()}
    log(f"recognition (no key) -> HTTP {r.status_code}")

    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(f"evidence written to {out / 'evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
