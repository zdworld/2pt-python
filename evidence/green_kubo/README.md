# Green–Kubo NaFSI/DME evidence bundle

This directory contains the result bundle for the 5,110-atom NaFSI/DME NVE trajectory supplied for the Green–Kubo audit. The source trajectory is intentionally **not** committed: it is a 9.5 GB external production asset. The bundle contains only derived, reviewable evidence.

## Reproduce from the source trajectory

Install the analysis dependencies from the repository root:

```bash
python3 -m pip install -r requirements-green-kubo.txt
```

Run the complete streaming pipeline. It reads the ASE ULM trajectory frame-by-frame, infers the saved timestep from the ULM metadata, validates the Na/FSI mapping, constructs the formal-charge ionic current, computes the FFT current autocorrelation, integrates the Green–Kubo curve, estimates block uncertainty, and renders PNG/SVG figures.

```bash
python3 scripts/green_kubo_pipeline.py \
  /path/to/md_production_nve.traj \
  --outdir /tmp/green_kubo
python3 scripts/validate_green_kubo_outputs.py /tmp/green_kubo
```

The current convention is +1e for each Na+ and −1e for each FSI−, using mass-weighted FSI center-of-mass velocities; neutral DME is excluded. Ten non-overlapping 2 ps blocks are used for the primary uncertainty diagnostic. This is a finite-trajectory diagnostic, not an automatic claim of convergence.

## Evidence files

- `green_kubo_diagnostic.png` / `.svg`: current, normalized CACF, cumulative conductivity decomposition, and block uncertainty.
- `green_kubo_prefix_convergence.png` / `.svg`: fixed-cutoff sensitivity to trajectory length.
- `green_kubo_curve.csv`: lag-resolved CACF and cumulative self/cross/total conductivity.
- `green_kubo_block_estimates.csv`: individual 2 ps block estimates.
- `green_kubo_block_summary.csv`: means and 95% Student-t intervals.
- `green_kubo_prefix_convergence.csv`: prefix analysis data.
- `green_kubo_summary.json`: machine-readable trajectory audit and analysis contract.
- `green_kubo_report.md`: manuscript-ready methods, results, discussion, and figure captions.

## Interpretation boundary

The source trajectory has 20.0 ps of 1 fs-sampled dynamics. The Green–Kubo integral does not show a stable long-time plateau and therefore does not support a single final conductivity for direct comparison with experiment. The evidence is useful for validating the pipeline and showing ion-correlation structure; a follow-up PR should add longer, independently replicated NVE trajectories and repeat the same convergence tests.
