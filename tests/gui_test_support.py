"""Deterministic fakes shared by the GUI tests.

Only the neural network wrapper is faked.  Qt, VTK and the demo's own state
machine stay real; loading CLIP plus a SNAP checkpoint would add minutes and
gigabytes without testing any of the interaction logic.
"""

from __future__ import annotations

import numpy as np
import torch


def make_cloud(num: int = 5):
    """A small, deterministic synthetic cloud with normalised colour input."""

    axis = np.linspace(0.0, 1.0, num, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    coord = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    color = np.tile(np.array([[51.0, 102.0, 153.0]], dtype=np.float32), (len(coord), 1))
    return {"coord": coord, "color": color}


class FakeDataset:
    labels = ["wall", "chair", "floor"]


class FakeClipModel:
    def encode_text(self, tokens):
        return torch.zeros((tokens.shape[0], 512), dtype=torch.float32)


class FakeModel:
    """Minimal stand-in with the interface used by ``InteractivePlotter``."""

    def __init__(self, args, checkpoint_path=None, domain="Outdoor", grid_size=0.05):
        self.args = args
        self.device = torch.device("cpu")
        self.domain = domain
        self.dataset = FakeDataset()
        self.clip_model = FakeClipModel()
        self.point_features = None

    def intialize_pointcloud(self, data):
        out = {"coord": torch.as_tensor(np.asarray(data["coord"], dtype=np.float32))}
        if "color" in data:
            color = np.asarray(data["color"], dtype=np.float32)
            # Mimic DemoDatset's NormalizeColor for the Indoor domain.
            out["color"] = torch.as_tensor(color / 127.5 - 1.0)
        if "intensity" in data:
            out["strength"] = torch.as_tensor(
                np.asarray(data["intensity"], dtype=np.float32)
            )
        return out

    def extract_backbone_features(self, data):
        self.point_features = {"coord": data["coord"]}

    def segment(self, data, prompt_points, text_prompt, return_tokens=False):
        coord = data["coord"].cpu().numpy()
        masks, labels, ious = [], [], []
        for prompt in prompt_points:
            points = np.asarray(prompt, dtype=np.float32).reshape(-1, 3)
            center = points.mean(axis=0)
            distance = np.linalg.norm(coord - center, axis=1)
            mask = distance <= np.quantile(distance, 0.25) + 1e-6
            masks.append(mask)
            labels.append("wall")
            ious.append(np.array([0.9], dtype=np.float32))
        return masks, labels, ious

    def segment_everything(self, data, return_tokens=False):
        coord = data["coord"].cpu().numpy()
        masks, prompt_points = [], []
        for fraction in (0.0, 0.5):
            mask = coord[:, 0] >= np.quantile(coord[:, 0], fraction)
            masks.append(torch.as_tensor(mask))
            prompt_points.append(np.array([[float(coord[:, 0].mean()), 0.0, 0.0]]))
        tokens = [torch.zeros(512) for _ in masks]
        ious = [np.array([0.8], dtype=np.float32) for _ in masks]
        return masks, tokens, ious, prompt_points

    def get_text_label(self, clip_token, pred_token, labels):
        return labels[0], 0.5

    def find_best_mask_from_text(self, cached_tokens, text_prompt):
        print("Best mask #0 with score 0.5")
        return torch.tensor(0)
