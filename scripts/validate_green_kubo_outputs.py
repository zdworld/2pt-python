#!/usr/bin/env python3
"""Validate the lightweight evidence bundle produced by green_kubo_pipeline.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()

    summary = json.loads((args.outdir / "green_kubo_summary.json").read_text())
    assert summary["has_velocities"] is True
    assert summary["n_frames"] > 0
    assert summary["timestep_fs"] > 0
    assert summary["primary_block_count"] >= 2

    with (args.outdir / "green_kubo_curve.csv").open() as f:
        curve = list(csv.DictReader(f))
    with (args.outdir / "green_kubo_block_summary.csv").open() as f:
        blocks = list(csv.DictReader(f))
    assert len(curve) == summary["n_frames"]
    assert len(blocks) >= 3
    for name in ("green_kubo_diagnostic.png", "green_kubo_diagnostic.svg",
                 "green_kubo_prefix_convergence.png", "green_kubo_prefix_convergence.svg"):
        assert (args.outdir / name).stat().st_size > 0, name
    print(f"validated {args.outdir}")


if __name__ == "__main__":
    main()
