#!/usr/bin/env python3
"""Green--Kubo diagnostic for an ASE NVE trajectory.

The production current is built from formal ionic charges and mass-weighted
center-of-mass velocities: +1e for each Na+ and -1e for each FSI- anion.
Neutral DME is excluded from the ionic current.  The script streams the
trajectory frame-by-frame, so the 9.5-GB source is never held in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import Trajectory
from ase.io.ulm import open as ulm_open
from scipy.integrate import cumulative_trapezoid
from scipy.stats import t as student_t


PALETTE = ["#6F55C5", "#2E9F6B", "#E24A3B", "#31A7C8", "#F0AF24", "#D95491"]


def apply_publication_style(font_size=9):
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.8,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def bold_tick_labels(ax):
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})


E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J K^-1
FS_TO_S = 1e-15
ANGSTROM_TO_M = 1e-10


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Unbiased time-origin averaged autocorrelation for vector data."""
    x = np.asarray(x, float)
    x = x - x.mean(axis=0, keepdims=True)
    n = len(x)
    nfft = 2 * n
    fx = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=nfft, axis=0)[:n]
    ac /= np.arange(n, 0, -1)[:, None]
    return ac


def integrate_curve(c: np.ndarray, dt_s: float, prefactor: float) -> np.ndarray:
    return prefactor * cumulative_trapezoid(c, dx=dt_s, initial=0.0)


def species_groups(atoms):
    symbols = np.array(atoms.get_chemical_symbols())
    na = np.flatnonzero(symbols == "Na")
    if len(na) == 0 or not np.all(na == np.arange(na[0], na[-1] + 1)):
        raise ValueError("Na atoms are not a contiguous block; automatic ion mapping stopped.")
    tail = symbols[na[-1] + 1 :]
    if len(tail) % 9 != 0:
        raise ValueError("The post-Na atom block is not divisible into 9-atom FSI groups.")
    groups = np.arange(na[-1] + 1, len(symbols)).reshape(-1, 9)
    expected = sorted(["O", "S", "O", "N", "S", "O", "O", "F", "F"])
    for i, group in enumerate(groups):
        if sorted(symbols[group]) != expected:
            raise ValueError(f"FSI group {i} does not contain the expected O4S2NF2 composition.")
    return na, groups


def current_from_atoms(atoms, na, fsi_groups, masses, velocity_to_mps):
    v = atoms.get_velocities()  # Å per ASE time unit
    na_sum = v[na].sum(axis=0)
    group_masses = masses[fsi_groups]
    fsi_com = (v[fsi_groups] * group_masses[:, :, None]).sum(axis=1) / group_masses.sum(axis=1)[:, None]
    fsi_sum = fsi_com.sum(axis=0)
    # Formal ionic charges: Na+ and FSI-.  The solvent is neutral.
    j_na = E_CHARGE * velocity_to_mps * na_sum
    j_fsi = -E_CHARGE * velocity_to_mps * fsi_sum
    return j_na + j_fsi, j_na, j_fsi


def segment_estimates(currents, dt_s, prefactor, cutoff_ps, segment_len):
    nseg = len(currents) // segment_len
    rows = []
    for seg in range(nseg):
        x = currents[seg * segment_len : (seg + 1) * segment_len]
        c = autocorrelation(x).sum(axis=1)
        max_cut = min(cutoff_ps.max(), (len(x) - 1) * dt_s * 1e12)
        for cutoff in cutoff_ps:
            if cutoff > max_cut:
                continue
            idx = int(round(cutoff * 1e-12 / dt_s))
            sigma = integrate_curve(c[: idx + 1], dt_s, prefactor)[-1]
            rows.append({"segment": seg + 1, "cutoff_ps": float(cutoff), "sigma_S_m": float(sigma)})
    return rows


def summarize_blocks(rows):
    out = []
    for cutoff in sorted({r["cutoff_ps"] for r in rows}):
        vals = np.array([r["sigma_S_m"] for r in rows if r["cutoff_ps"] == cutoff], float)
        n = len(vals)
        mean = float(vals.mean()) if n else float("nan")
        sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n > 1 else float("nan")
        tcrit = float(student_t.ppf(0.975, n - 1)) if n > 1 else float("nan")
        out.append({
            "cutoff_ps": cutoff,
            "n_blocks": n,
            "mean_S_m": mean,
            "sd_S_m": sd,
            "sem_S_m": sem,
            "ci95_low_S_m": mean - tcrit * sem if n > 1 else float("nan"),
            "ci95_high_S_m": mean + tcrit * sem if n > 1 else float("nan"),
        })
    return out


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ULM root metadata contains the MD integrator, internal timestep, and save interval.
    root = ulm_open(str(args.trajectory))
    description = dict(root.description)
    timestep_internal = float(description["timestep"])
    interval = int(description.get("interval", 1))
    # ASE's internal time unit is 1/units.fs femtoseconds.
    dt_fs = timestep_internal / units.fs * interval
    dt_s = dt_fs * FS_TO_S
    velocity_to_mps = ANGSTROM_TO_M / ((1.0 / units.fs) * FS_TO_S)

    tr = Trajectory(str(args.trajectory), "r")
    nframes = len(tr)
    first = tr[0]
    last = tr[-1]
    na, fsi_groups = species_groups(first)
    masses = first.get_masses()
    cell0 = first.cell.array.copy()
    cell_last = last.cell.array.copy()
    volume_m3 = abs(np.linalg.det(cell0)) * ANGSTROM_TO_M ** 3
    if not np.allclose(cell0, cell_last, rtol=0, atol=1e-10):
        raise ValueError("Cell varies between first and last frame; fixed-volume analysis is invalid.")

    currents = np.empty((nframes, 3), float)
    currents_na = np.empty_like(currents)
    currents_fsi = np.empty_like(currents)
    temperatures = np.empty(nframes, float)
    dof = 3 * len(first) - 3
    for i, atoms in enumerate(tr):
        j, j_na, j_fsi = current_from_atoms(atoms, na, fsi_groups, masses, velocity_to_mps)
        currents[i] = j
        currents_na[i] = j_na
        currents_fsi[i] = j_fsi
        temperatures[i] = 2.0 * atoms.get_kinetic_energy() / (dof * 8.617330337217213e-5)
        if (i + 1) % 2000 == 0:
            print(f"read {i + 1}/{nframes} frames", flush=True)
    tr.close()

    temp_mean = float(np.mean(temperatures))
    temp_sd = float(np.std(temperatures, ddof=1))
    # Use the measured kinetic temperature for the equilibrium GK prefactor.
    prefactor = 1.0 / (3.0 * volume_m3 * K_B * temp_mean)
    c_total = autocorrelation(currents).sum(axis=1)
    c_na = autocorrelation(currents_na).sum(axis=1)
    c_fsi = autocorrelation(currents_fsi).sum(axis=1)
    c_cross = c_total - c_na - c_fsi
    tau_ps = np.arange(nframes) * dt_s * 1e12
    sigma_total = integrate_curve(c_total, dt_s, prefactor)
    sigma_na = integrate_curve(c_na, dt_s, prefactor)
    sigma_fsi = integrate_curve(c_fsi, dt_s, prefactor)
    sigma_cross = integrate_curve(c_cross, dt_s, prefactor)

    # Ten 2-ps non-overlapping blocks provide the primary finite-trajectory uncertainty.
    cutoffs = np.array([0.05, 0.10, 0.20, 0.30, 0.50])
    block_len = max(2, int(round(2.0e-12 / dt_s)))
    block_rows = segment_estimates(currents, dt_s, prefactor, cutoffs, block_len)
    block_summary = summarize_blocks(block_rows)

    # Prefix convergence at 0.20 ps, a deliberately short cutoff that does not
    # pretend the noisy long-time tail has converged.
    prefix_cutoff_ps = 0.20
    prefix_rows = []
    for prefix_ps in np.arange(2.0, tau_ps[-1] + 1e-9, 2.0):
        n = min(nframes, int(round(prefix_ps * 1e-12 / dt_s)) + 1)
        c = autocorrelation(currents[:n]).sum(axis=1)
        idx = min(len(c) - 1, int(round(prefix_cutoff_ps * 1e-12 / dt_s)))
        prefix_rows.append({"prefix_ps": float(prefix_ps), "cutoff_ps": prefix_cutoff_ps,
                            "sigma_S_m": float(integrate_curve(c[:idx + 1], dt_s, prefactor)[-1])})

    # Write numerical deliverables.
    curve_rows = []
    for i in range(nframes):
        curve_rows.append({"lag_ps": float(tau_ps[i]), "C_total_SI": float(c_total[i]),
                           "C_Na_self_SI": float(c_na[i]), "C_FSI_self_SI": float(c_fsi[i]),
                           "C_Na_FSI_cross_SI": float(c_cross[i]), "sigma_total_S_m": float(sigma_total[i]),
                           "sigma_Na_self_S_m": float(sigma_na[i]), "sigma_FSI_self_S_m": float(sigma_fsi[i]),
                           "sigma_Na_FSI_cross_S_m": float(sigma_cross[i])})
    write_csv(args.outdir / "green_kubo_curve.csv", list(curve_rows[0]), curve_rows)
    write_csv(args.outdir / "green_kubo_block_estimates.csv", list(block_rows[0]), block_rows)
    write_csv(args.outdir / "green_kubo_block_summary.csv", list(block_summary[0]), block_summary)
    write_csv(args.outdir / "green_kubo_prefix_convergence.csv", list(prefix_rows[0]), prefix_rows)
    np.savez_compressed(args.outdir / "green_kubo_currents.npz", total=currents, na=currents_na, fsi=currents_fsi,
                        temperature_K=temperatures, dt_fs=dt_fs)

    # Plot atlas-compatible, publication-style figures.
    import matplotlib.pyplot as plt
    # The atlas's circular_science palette supplies six distinct colors for
    # the current components, self terms, cross term, and block summary.
    apply_publication_style(9)
    colors = PALETTE
    ink = "#303038"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    t_current = np.arange(nframes) * dt_s * 1e12
    show = t_current <= min(2.0, t_current[-1])
    for k, lab in enumerate([r"$J_x$", r"$J_y$", r"$J_z$"]):
        ax.plot(t_current[show], currents[show, k] * 1e12, lw=0.7, color=colors[k], label=lab)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"Current $J$ ($10^{-12}$ C m s$^{-1}$)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(0.02, 0.96, f"T = {temp_mean:.1f} ± {temp_sd:.1f} K", transform=ax.transAxes, va="top", color=ink)

    ax = axes[0, 1]
    norm = c_total[0] if c_total[0] else 1.0
    acf = c_total / norm
    show = tau_ps <= min(5.0, tau_ps[-1])
    ax.plot(tau_ps[show], acf[show], color=colors[3], lw=1.2)
    ax.axhline(0, color=ink, lw=0.6)
    ax.set_xlabel(r"Lag time $\tau$ (ps)")
    ax.set_ylabel(r"$C_J(\tau)/C_J(0)$")

    ax = axes[1, 0]
    show = tau_ps <= min(5.0, tau_ps[-1])
    ax.plot(tau_ps[show], sigma_total[show], color=colors[0], lw=1.5, label="Total")
    ax.plot(tau_ps[show], sigma_na[show], color=colors[1], lw=1.0, label="Na self")
    ax.plot(tau_ps[show], sigma_fsi[show], color=colors[2], lw=1.0, label="FSI self")
    ax.plot(tau_ps[show], sigma_cross[show], color=colors[4], lw=1.0, label="Na–FSI cross")
    ax.axhline(0, color=ink, lw=0.6)
    ax.set_xlabel(r"Integration upper limit $\tau$ (ps)")
    ax.set_ylabel(r"Cumulative $\sigma$ (S m$^{-1}$)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    ax = axes[1, 1]
    cuts = np.array([r["cutoff_ps"] for r in block_summary])
    means = np.array([r["mean_S_m"] for r in block_summary])
    lows = np.array([r["ci95_low_S_m"] for r in block_summary])
    highs = np.array([r["ci95_high_S_m"] for r in block_summary])
    ax.plot(cuts, means, "o-", color=colors[5], ms=3.5, lw=1.2)
    ax.fill_between(cuts, lows, highs, color=colors[5], alpha=0.2, linewidth=0)
    ax.axhline(0, color=ink, lw=0.6)
    ax.set_xlabel(r"Integration upper limit $\tau$ (ps)")
    ax.set_ylabel(r"Block mean $\sigma$ (S m$^{-1}$)")
    ax.text(0.04, 0.95, "95% t-interval; 2-ps blocks", transform=ax.transAxes, va="top", fontsize=7)
    for ax, label in zip(axes.flat, ["A", "B", "C", "D"]):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        clean_axes(ax)
        bold_tick_labels(ax)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.96, wspace=0.32, hspace=0.38)
    save_figure(fig, args.outdir / "green_kubo_diagnostic.png", dpi=300)
    plt.close(fig)

    # Prefix convergence figure.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    px = np.array([r["prefix_ps"] for r in prefix_rows])
    py = np.array([r["sigma_S_m"] for r in prefix_rows])
    ax.plot(px, py, "o-", color=colors[0], ms=3.5, lw=1.2)
    ax.axhline(0, color=ink, lw=0.6)
    ax.set_xlabel("Trajectory prefix (ps)")
    ax.set_ylabel(r"$\sigma$ at $\tau=0.20$ ps (S m$^{-1}$)")
    clean_axes(ax)
    bold_tick_labels(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.96)
    save_figure(fig, args.outdir / "green_kubo_prefix_convergence.png", dpi=300)
    plt.close(fig)

    summary = {
        "trajectory": args.trajectory.name,
        "file_size_bytes": args.trajectory.stat().st_size,
        "n_frames": nframes,
        "n_atoms": len(first),
        "formula": first.get_chemical_formula(),
        "composition_counts": {s: int(first.get_chemical_symbols().count(s)) for s in sorted(set(first.get_chemical_symbols()))},
        "pbc": [bool(x) for x in first.pbc],
        "cell_A": cell0.tolist(),
        "volume_A3": float(abs(np.linalg.det(cell0))),
        "volume_m3": volume_m3,
        "ase_md_description": description,
        "timestep_internal": timestep_internal,
        "sampling_interval": interval,
        "timestep_fs": dt_fs,
        "trajectory_span_ps": float((nframes - 1) * dt_fs / 1000.0),
        "velocity_source": "ASE momenta array -> Atoms.get_velocities()",
        "has_velocities": True,
        "ion_mapping": {"Na_indices": [int(na[0]), int(na[-1])], "n_Na": int(len(na)),
                        "FSI_group_atoms": 9, "n_FSI": int(len(fsi_groups)),
                        "FSI_atom_composition": "O4S2NF2", "neutral_solvent": "DME"},
        "charge_model": "formal charges +1e (Na+) and -1e (FSI-), mass-weighted ion COM velocities; DME excluded",
        "temperature_K_mean": temp_mean,
        "temperature_K_sd_over_frames": temp_sd,
        "sigma_cumulative_full_S_m": {str(x): float(sigma_total[min(len(sigma_total)-1, int(round(x * 1e-12 / dt_s)))]) for x in [0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0, 5.0, 10.0]},
        "primary_block_length_ps": block_len * dt_s * 1e12,
        "primary_block_count": len({r["segment"] for r in block_rows}) if block_rows else 0,
        "analysis_runtime_s": time.time() - t0,
        "caveat": "Short single trajectory; long-time Green-Kubo plateau must be judged from block and prefix diagnostics before formal experimental comparison.",
    }
    (args.outdir / "green_kubo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ["n_frames", "n_atoms", "timestep_fs", "trajectory_span_ps", "temperature_K_mean", "volume_A3", "primary_block_length_ps", "primary_block_count"]}, indent=2))
    print(f"finished in {time.time() - t0:.1f} s; outputs in {args.outdir}")


if __name__ == "__main__":
    main()
