"""Command line entry point: `python -m pitwall <stage>`.

Midpoint stages: ingest -> clean -> features -> train (`all` runs these).
Final stages:    tune -> lstm -> test -> update -> strategy (`final` runs these).
"""
from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pitwall")
    p.add_argument("stage", choices=["ingest", "clean", "features", "train", "all",
                                          "tune", "lstm", "test", "update", "strategy", "final"])
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
    if args.stage in ("tune", "final"):
        from . import tune
        tune.run(cfg)
    if args.stage in ("lstm", "final"):
        from . import lstm
        from .tune import final_config
        if cfg.get("lstm", {}).get("retrain", True):
            lstm.run(final_config(cfg))
        else:
            print("lstm: retrain is off in the config, skipped")
    if args.stage in ("test", "final"):
        from . import test
        test.run(cfg)
    if args.stage in ("update", "final"):
        from . import update
        update.run(cfg)
    if args.stage in ("strategy", "final"):
        from . import strategy
        strategy.run(cfg)


if __name__ == "__main__":
    main()
