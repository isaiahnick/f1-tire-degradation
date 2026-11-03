"""Command line entry point: `python -m pitwall <stage>`.

Stages run in order: ingest -> clean -> features -> train. `all` runs every stage.
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pitwall")
    p.add_argument("stage", choices=["ingest", "clean", "features", "train", "all"])
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--source", choices=["archive", "fastf1"], default=None,
                   help="override ingest.source from the config")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    from . import clean, features, ingest, train

    if args.stage in ("ingest", "all"):
        ingest.run(cfg, source=args.source)
    if args.stage in ("clean", "all"):
        clean.run(cfg)
    if args.stage in ("features", "all"):
        features.run(cfg)
    if args.stage in ("train", "all"):
        train.run(cfg)


if __name__ == "__main__":
    main()
