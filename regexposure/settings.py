"""Configuration and project paths.

All settings live in config.yaml at the repository root. Paths are created on load.
"""
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _ns(raw)
    cfg.raw = raw
    cfg.root = ROOT
    data = ROOT / raw["project"]["data_dir"]
    cfg.paths = SimpleNamespace(
        data=data,
        fr=data / "raw" / "federal_register",          # rule metadata, HTML and text
        edgar=data / "raw" / "edgar",                  # 10-K HTML
        sections=data / "interim" / "sections",        # extracted Item 1 / Item 1A text
        interim=data / "interim",
        embeddings=data / "interim" / "embeddings",
        domains=data / "processed" / "domains",
        scores=data / "processed" / "scores",
        exposure=data / "processed" / "exposure",
        batches=data / "logs" / "batches",
        logs=data / "logs",
    )
    for p in vars(cfg.paths).values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def years(cfg):
    return list(range(cfg.project.start_year, cfg.project.end_year + 1))
