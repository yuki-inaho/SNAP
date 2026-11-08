#!/usr/bin/env python3
"""Deterministic MOCK backend for interSeg3D-Studio browser verification.

*** THIS IS NOT PINPOINT3D INFERENCE. ***

It exists only so the official frontend can be exercised end-to-end (upload,
positive/negative clicks, mask rendering, save/download) before the real
PinPoint3D backend is ready. The mask is a geometric sphere/cube rule around
the submitted click positions. Every response carries `mock: true` and an
`X-Mock-Backend: true` header, and the frontend shows a banner when
VITE_MOCK_BACKEND=true.

Run:
  python3 mock/mock_backend.py --port 9500
"""
import argparse
import io
import json
import pathlib
import struct
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "mock"
UPLOAD_DIR = ROOT / "mock" / "uploads"
LOG_PATH = ROOT / "logs" / "mock-requests.jsonl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

LOCK = threading.Lock()

STATE = {
    "filename": None,
    "ply_path": None,
    "points": None,      # list[tuple[float,float,float]]
    "colors": None,      # list[tuple[int,int,int]]
    "segmentation": None,
    "objects": [],
    "clicks": {},
    "inference_count": 0,
}


def log(entry):
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# PLY parsing (ascii + binary little endian)
# ---------------------------------------------------------------------------
PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def parse_ply(path):
    raw = pathlib.Path(path).read_bytes()
    marker = b"end_header"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("PLY header missing end_header")
    header = raw[:idx].decode("ascii", "replace")
    body = raw[idx + len(marker):]
    body = body.lstrip(b"\r\n")
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]

    fmt = None
    vertex_count = 0
    props = []
    in_vertex = False
    for ln in lines:
        if ln.startswith("format "):
            fmt = ln.split()[1]
        elif ln.startswith("element "):
            parts = ln.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif ln.startswith("property ") and in_vertex:
            parts = ln.split()
            if parts[1] == "list":
                props.append(("list", parts[2], parts[3], parts[4]))
            else:
                props.append(("scalar", parts[1], parts[2]))

    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"unsupported PLY format {fmt}")

    points, colors = [], []
    if fmt == "ascii":
        rows = body.decode("ascii", "replace").splitlines()
        for row in rows[:vertex_count]:
            vals = row.split()
            if len(vals) < 3:
                continue
            points.append((float(vals[0]), float(vals[1]), float(vals[2])))
            r = g = b = 200
            for i, p in enumerate(props):
                if p[0] == "scalar" and p[2] in ("red", "green", "blue") and i < len(vals):
                    v = float(vals[i])
                    if p[1] not in ("float", "float32", "double", "float64"):
                        v = v / 255.0
                    elif v > 1.0:
                        v = v / 255.0
                    v = max(0, min(255, int(round(v * 255))))
                    if p[2] == "red":
                        r = v
                    elif p[2] == "green":
                        g = v
                    elif p[2] == "blue":
                        b = v
            colors.append((r, g, b))
    else:
        struct_fmt = "<"
        names = []
        for p in props:
            if p[0] != "scalar":
                raise ValueError("list properties in vertex element are not supported")
            struct_fmt += PLY_TYPES[p[1]]
            names.append(p[2])
        stride = struct.calcsize(struct_fmt)
        for i in range(vertex_count):
            rec = struct.unpack_from(struct_fmt, body, i * stride)
            row = dict(zip(names, rec))
            points.append((float(row["x"]), float(row["y"]), float(row["z"])))
            r = row.get("red", row.get("r", 200))
            g = row.get("green", row.get("g", 200))
            b = row.get("blue", row.get("b", 200))
            # normalise float colours to 0..255
            if isinstance(r, float):
                r, g, b = (max(0, min(255, int(round(v * 255)))) for v in (r, g, b))
            colors.append((int(r), int(g), int(b)))

    return points, colors


def parse_click_positions(click_positions):
    """Return {label: [(x,y,z)...]}, label 0 = background/negative."""
    grouped = {}
    for obj_key, parts in (click_positions or {}).items():
        try:
            label = int(obj_key)
        except (TypeError, ValueError):
            label = 1
        if not isinstance(parts, dict):
            continue
        bucket = grouped.setdefault(label, [])
        for _part_key, positions in parts.items():
            for pos in positions or []:
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    bucket.append((float(pos[0]), float(pos[1]), float(pos[2])))
    return grouped


def geometric_mask(points, grouped, cube_size):
    """Deterministic geometric stand-in for a real segmentation model."""
    positives = {k: v for k, v in grouped.items() if k != 0 and v}
    negatives = grouped.get(0, [])
    pos_radius = max(0.45, min(1.2, cube_size * 22.0))
    neg_radius = max(0.38, min(1.0, cube_size * 18.0))
    pos_r2 = pos_radius * pos_radius
    neg_r2 = neg_radius * neg_radius
    mask = [0] * len(points)
    if not positives:
        return mask
    for i, (x, y, z) in enumerate(points):
        best = 0
        best_d2 = pos_r2
        for label, plist in positives.items():
            for (px, py, pz) in plist:
                dx, dy, dz = x - px, y - py, z - pz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 <= best_d2:
                    best_d2 = d2
                    best = label
        if best == 0:
            continue
        for (nx, ny, nz) in negatives:
            dx, dy, dz = x - nx, y - ny, z - nz
            if dx * dx + dy * dy + dz * dz <= neg_r2:
                best = 0
                break
        mask[i] = best
    return mask


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MockInterSeg3D/0.1 (NOT PinPoint3D)"

    def log_message(self, fmt, *args):
        log({"event": "http", "message": fmt % args})

    def _send(self, code, payload, content_type="application/json", extra_headers=None):
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = payload
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("X-Mock-Backend", "true")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/health":
            self._send(200, {
                "status": "mock-ok",
                "mock": True,
                "warning": "deterministic geometry, NOT PinPoint3D inference",
            })
            return
        if path == "/api/download/results":
            self._handle_download()
            return
        self._send(404, {"detail": "mock backend: unknown GET " + path, "mock": True})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if path == "/api/upload":
            self._handle_upload(body)
        elif path == "/api/infer":
            self._handle_infer(body)
        elif path == "/api/recognition/mask_obj_recognition":
            self._handle_recognition(body)
        elif path == "/api/recognition/update_objects":
            self._handle_update_objects(body)
        else:
            self._send(404, {"detail": "mock backend: unknown POST " + path, "mock": True})

    # -- endpoints ---------------------------------------------------------
    def _handle_upload(self, body):
        ctype = self.headers.get("Content-Type", "")
        boundary = None
        for part in ctype.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part.split("=", 1)[1].strip('"')
        if not boundary:
            self._send(400, {"detail": "mock upload: missing multipart boundary", "mock": True})
            return
        delim = b"--" + boundary.encode()
        chunks = body.split(delim)
        file_bytes = None
        filename = "unknown.ply"
        for chunk in chunks:
            if b'name="file"' not in chunk and b"filename=" in chunk:
                pass
            head_end = chunk.find(b"\r\n\r\n")
            if head_end < 0:
                continue
            head = chunk[:head_end].decode("utf-8", "replace")
            if 'name="file"' not in head:
                continue
            for token in head.replace('"', " ").split():
                if token.startswith("filename="):
                    filename = token.split("=", 1)[1]
            payload = chunk[head_end + 4:]
            if payload.endswith(b"\r\n"):
                payload = payload[:-2]
            file_bytes = payload
            break
        if not file_bytes:
            self._send(400, {"detail": "mock upload: file part not found", "mock": True})
            return
        safe_name = pathlib.Path(filename).name or "upload.ply"
        dest = UPLOAD_DIR / safe_name
        dest.write_bytes(file_bytes)
        try:
            points, colors = parse_ply(dest)
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"detail": f"mock upload: PLY parse failed: {exc}", "mock": True})
            return
        with LOCK:
            STATE.update({
                "filename": safe_name,
                "ply_path": str(dest),
                "points": points,
                "colors": colors,
                "segmentation": None,
                "objects": [],
                "clicks": {},
                "inference_count": 0,
            })
        log({"event": "upload", "filename": safe_name, "points": len(points),
             "bytes": len(file_bytes), "path": str(dest)})
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        self._send(200, {
            "message": "File uploaded successfully (MOCK)",
            "filename": safe_name,
            "point_count": len(points),
            "bounding_box": {
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            },
            "mock": True,
        })

    def _handle_infer(self, body):
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send(400, {"detail": "mock infer: invalid JSON", "mock": True})
            return
        if STATE["points"] is None:
            self._send(400, {"detail": "No point cloud loaded (mock).", "mock": True})
            return
        click_positions = (req.get("clickData") or {}).get("clickPositions") or {}
        grouped = parse_click_positions(click_positions)
        cube_size = float(req.get("cubeSize") or 0.02)
        with LOCK:
            mask = geometric_mask(STATE["points"], grouped, cube_size)
            STATE["segmentation"] = mask
            STATE["clicks"] = grouped
            STATE["objects"] = req.get("objectNames") or []
            STATE["inference_count"] += 1
            counts = {}
            for v in mask:
                counts[str(v)] = counts.get(str(v), 0) + 1
        log({"event": "infer", "cube_size": cube_size,
             "click_positions_raw": click_positions,
             "click_idx_raw": (req.get("clickData") or {}).get("clickIdx"),
             "positive_clicks": {str(k): v for k, v in grouped.items() if k != 0},
             "negative_clicks": grouped.get(0, []),
             "mask_counts": counts})
        self._send(200, {
            "message": "Inference completed successfully (MOCK GEOMETRY, NOT PinPoint3D)",
            "segmentedPointCloud": {"segmentation": mask, "object_counts": counts},
            "mock": True,
        })

    def _handle_recognition(self, body):
        mask = []
        try:
            mask = (json.loads(body) or {}).get("mask") or []
        except json.JSONDecodeError:
            pass
        labels = sorted({v for v in mask if isinstance(v, int) and v > 0})
        result = [{
            "obj_id": label,
            "label": f"mock-object-{label}",
            "description": "Deterministic mock label; no LLM or model ran.",
            "selected_views": [],
            "cost": 0.0,
        } for label in labels]
        log({"event": "recognition", "objects": labels, "mock": True})
        self._send(200, {"message": "Mask object recognition completed (MOCK)", "result": result, "mock": True})

    def _handle_update_objects(self, body):
        try:
            req = json.loads(body) or {}
        except json.JSONDecodeError:
            req = {}
        objects = req.get("objects") or []
        with LOCK:
            STATE["objects"] = [o.get("name") for o in objects]
        log({"event": "update_objects", "count": len(objects),
             "objects": objects, "mock": True})
        self._send(200, {
            "message": f"Successfully updated information for {len(objects)} objects (MOCK)",
            "updated_count": len(objects),
            "mock": True,
        })

    def _handle_download(self):
        if STATE["points"] is None or STATE["segmentation"] is None:
            self._send(400, {"detail": "No results available (mock).", "mock": True})
            return
        points = STATE["points"]
        colors = STATE["colors"]
        mask = STATE["segmentation"]
        palette = [(0, 176, 255), (178, 255, 89), (255, 171, 0), (213, 0, 249)]
        out_colors = []
        for i, c in enumerate(colors):
            label = mask[i]
            if label:
                out_colors.append(palette[(label - 1) % len(palette)])
            else:
                out_colors.append(c)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        buf = io.BytesIO()
        buf.write(header.encode("ascii"))
        for (x, y, z), (r, g, b) in zip(points, out_colors):
            buf.write(struct.pack("<fffBBB", x, y, z, r, g, b))
        metadata = {
            "mock": True,
            "warning": "Deterministic mock output. NOT PinPoint3D inference.",
            "source_ply": STATE["filename"],
            "objects": STATE["objects"],
            "clicks": {str(k): v for k, v in STATE["clicks"].items()},
            "mask_counts": {str(v): mask.count(v) for v in sorted(set(mask))},
            "inference_count": STATE["inference_count"],
        }
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("segmentation_result.ply", buf.getvalue())
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            zf.writestr("mask.json", json.dumps({"segmentation": mask}))
        payload = zip_buf.getvalue()
        log({"event": "download", "bytes": len(payload), "mock": True})
        self._send(200, payload, content_type="application/zip", extra_headers={
            "Content-Disposition": 'attachment; filename="segmentation_results_mock.zip"',
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log({"event": "start", "host": args.host, "port": args.port, "mock": True})
    print(f"MOCK backend (NOT PinPoint3D) listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
