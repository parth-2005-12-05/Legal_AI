"""Configuration loader.

Reads config/settings.yaml (or the file named by the CASEWAY_CONFIG environment
variable) once, at process start, into a frozen-ish Settings object. Because the
file is only read at startup, flipping `matching.partner_review` takes effect on
the next service restart, which is exactly the behaviour the design asks for:
"turning it on is this one setting and a service restart, with no code change."
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = the folder that holds caseway/, config/ and data/.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


@dataclass
class MatchingSettings:
    partner_review: bool = False
    conflict_reviewers: list[str] = field(default_factory=lambda: ["senior_lawyer", "admin"])
    mode: str = "realtime"
    top_n: int = 3
    min_score: float = 0.45
    close_call_margin: float = 0.02
    conflict_threshold: float = 0.90
    rotation_bonus: float = 0.02
    urgent_subtypes: list[str] = field(default_factory=lambda: ["bail", "anticipatory_bail"])
    urgent_max_hours: int = 24
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FusionSettings:
    auto_accept: float = 0.95
    provisional: float = 0.75
    disagreement_penalty: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FuzzySettings:
    accept_score: float = 0.92
    accept_gap: float = 0.05
    trigram_min_overlap: float = 0.4
    bk_max_distance: int = 2


@dataclass
class ClassifierSettings:
    multi_label_threshold: float = 0.30
    bm25_k: int = 20


@dataclass
class IngestSettings:
    max_file_mb: int = 25
    max_pages_per_case: int = 200
    allow_plain_text: bool = True


@dataclass
class Settings:
    matching: MatchingSettings
    fusion: FusionSettings
    fuzzy: FuzzySettings
    classifier: ClassifierSettings
    ingest: IngestSettings
    data_dir: Path
    var_dir: Path


def _section(cls, raw: dict[str, Any] | None):
    """Build a settings dataclass, ignoring unknown keys so old configs still load."""
    raw = raw or {}
    known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
    return cls(**known)


def load_settings(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Settings:
    """Load settings from YAML. `overrides` lets tests flip a value without a file,
    e.g. load_settings(overrides={"matching": {"partner_review": True}})."""
    path = Path(path or os.environ.get("CASEWAY_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    paths = raw.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    var_dir = Path(paths.get("var_dir", "var"))
    return Settings(
        matching=_section(MatchingSettings, raw.get("matching")),
        fusion=_section(FusionSettings, raw.get("fusion")),
        fuzzy=_section(FuzzySettings, raw.get("fuzzy")),
        classifier=_section(ClassifierSettings, raw.get("classifier")),
        ingest=_section(IngestSettings, raw.get("ingest")),
        data_dir=data_dir if data_dir.is_absolute() else ROOT / data_dir,
        var_dir=var_dir if var_dir.is_absolute() else ROOT / var_dir,
    )
