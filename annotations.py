#!/usr/bin/env python3
"""Per-video annotation metadata (milestone M2).

One JSON file per video under a git-tracked ``annotations/`` mirror of ``videos/``, so the
manual work (which fish responded first, the stimulus/fish ROIs) is captured once and every
downstream script can run headless. The videos are git-ignored, but these hand-made
annotations must be version-controlled.

Schema (v1) — unknown top-level keys are preserved, so M3 can add geometry without churn::

    {
      "schema_version": 1,
      "video": "videos/Shiner_SloMo/circle/34.MP4",     # relative to repo root
      "disposition": "usable",                            # usable | no_response | bad_video
      "annotation": {"stim_roi": [x, y, w, h],
                     "fish_roi": [x, y, w, h]},           # extensible (M3: screen edges, loom side)
      "results": {"stim_idx": 380, "det_coarse": 800,     # regenerable cache
                  "det_refined": 796, "params": {...}}
    }

Storage roots can be overridden with ``FISH_VIDEOS_DIR`` / ``FISH_ANNOTATIONS_DIR`` (used by
tests); by default they sit next to this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
DISPOSITIONS = ("usable", "no_response", "bad_video")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

REPO_ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = Path(os.environ.get("FISH_VIDEOS_DIR", REPO_ROOT / "videos")).resolve()
ANNOTATIONS_DIR = Path(
    os.environ.get("FISH_ANNOTATIONS_DIR", REPO_ROOT / "annotations")
).resolve()

Roi = Tuple[int, int, int, int]

_KNOWN_KEYS = {"schema_version", "video", "disposition", "annotation", "results"}


# ----------------------- paths & discovery ----------------------------


def list_videos(folder: Path) -> List[Path]:
    """Video files directly inside ``folder`` (non-recursive), sorted."""
    return sorted(
        p
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def video_rel(video: Path) -> str:
    """Repo-relative string for a video path (absolute fallback if outside the repo)."""
    v = Path(video).resolve()
    try:
        return str(v.relative_to(REPO_ROOT))
    except ValueError:
        return str(v)


def annotation_path(video: Path) -> Path:
    """JSON sidecar path for ``video`` inside the annotations mirror."""
    v = Path(video).resolve()
    try:
        rel = v.relative_to(VIDEOS_DIR)
    except ValueError:
        rel = Path(v.name)
    return ANNOTATIONS_DIR / rel.with_suffix(".json")


# ----------------------- model ----------------------------------------


def _roi_tuple(roi) -> Optional[Roi]:
    if roi is None:
        return None
    return (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))


@dataclass
class Annotation:
    video: str
    disposition: str = "usable"
    annotation: Dict = field(default_factory=dict)
    results: Optional[Dict] = None
    schema_version: int = SCHEMA_VERSION
    extra: Dict = field(default_factory=dict)  # unknown keys, preserved on save

    # -- ROI convenience -------------------------------------------------
    @property
    def stim_roi(self) -> Optional[Roi]:
        return _roi_tuple(self.annotation.get("stim_roi"))

    @property
    def fish_roi(self) -> Optional[Roi]:
        return _roi_tuple(self.annotation.get("fish_roi"))

    @property
    def has_rois(self) -> bool:
        return self.stim_roi is not None and self.fish_roi is not None

    def set_rois(self, stim_roi: Roi, fish_roi: Roi) -> None:
        self.annotation["stim_roi"] = [int(v) for v in stim_roi]
        self.annotation["fish_roi"] = [int(v) for v in fish_roi]

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict:
        d = {
            "schema_version": self.schema_version,
            "video": self.video,
            "disposition": self.disposition,
            "annotation": self.annotation,
            "results": self.results,
        }
        d.update(self.extra)  # forward-compat keys
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Annotation":
        extra = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            video=d.get("video", ""),
            disposition=d.get("disposition", "usable"),
            annotation=dict(d.get("annotation") or {}),
            results=d.get("results"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            extra=extra,
        )


# ----------------------- load / save ----------------------------------


def load_annotation(video: Path) -> Optional[Annotation]:
    """Load the annotation for ``video``, or ``None`` if none exists."""
    p = annotation_path(video)
    if not p.exists():
        return None
    with p.open() as f:
        return Annotation.from_dict(json.load(f))


def save_annotation(video: Path, ann: Annotation) -> Path:
    """Write ``ann`` to its sidecar path (creating parent dirs). Returns the path."""
    p = annotation_path(video)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(ann.to_dict(), f, indent=2)
    return p


# ----------------------- results cache helpers ------------------------


def results_dict(result, params) -> Dict:
    """Build the regenerable ``results`` cache from an AnalysisResult + AnalysisParams."""
    return {
        "stim_idx": result.stim_idx,
        "det_coarse": result.det_coarse,
        "det_refined": result.det_refined,
        "params": asdict(params),
    }


def results_stale(ann: Annotation, params) -> bool:
    """True if ``ann`` has cached results computed with different params than ``params``."""
    if not ann.results:
        return False
    stored = ann.results.get("params")
    return stored is not None and stored != asdict(params)
