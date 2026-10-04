# Green–Kubo ionic-conductivity audit and diagnostic

## Executive conclusion

This trajectory is technically sufficient for a Green–Kubo ionic-current calculation. It contains velocities (stored as ASE `momenta` arrays), atom identities, a fixed periodic cell, and explicit MD metadata from which the 1 fs sampling interval can be recovered.

The calculation was therefore completed as a diagnostic. It does **not** yet support a single converged, publication-level DC conductivity for direct comparison with experiment. The blocker is statistical convergence, not a missing trajectory field: the current autocorrelation tail remains noisy and the running integral has no stable long-time plateau within the 20 ps trajectory.

## Trajectory audit

| Quantity | Evidence and result | Assessment |
|---|---|---|
| File | ASE binary ULM trajectory, 10,214,255,792 bytes | Readable |
| Frames | 20,001 | Sufficient for a diagnostic; short for a converged electrolyte Green–Kubo value |
| Atoms | 5,110 per frame; formula C1200H3000F62N31Na31O724S62 | Consistent across the trajectory |
| Sampling | ULM metadata: `VelocityVerlet`, `timestep = 0.09822694788464063` ASE time units, `interval = 1`; this is 1.000 fs per saved frame | Available |
| Time span | 20.000 ps from first to last saved frame | Main convergence limitation |
| Velocities | Every streamed frame exposed `momenta` and `Atoms.get_velocities()` returned a `(5110, 3)` array | Available |
| Ionic identities | Na atoms 4800–4830; 31 contiguous FSI groups of 9 atoms, each with O4S2NF2; first 4800 atoms are neutral DME | Unambiguous mapping |
| Cell | Fixed cubic cell, 37.7636143 Å per side; volume 53,854.334 Å³; PBC in all directions | Available |
| Temperature | Kinetic temperature 298.17 ± 2.59 K over frames | Consistent with the intended 298 K NVE condition |

The trajectory also stores frame-level calculator partial charges. They were not used for the reported ionic current because they fluctuate atom-by-atom and do not define a unique molecular-ion charge current. The primary calculation uses the standard formal-charge ionic convention: +1e for Na+, −1e for FSI−, neutral DME excluded.

The file records a Velocity-Verlet molecular-dynamics description but does not embed a separate thermostat/control log. Thus the NVE designation is consistent with the supplied filename and the absence of thermostat metadata, but the trajectory alone cannot independently reconstruct the full production-control history.

## Calculation

For each saved frame, the charge current was constructed as

\[
\mathbf J(t)=e\left[\sum_{a\in \mathrm{Na^+}}\mathbf v_a(t)-
\sum_{m\in \mathrm{FSI^-}}\mathbf v_{m,\mathrm{COM}}(t)\right],
\]

where each FSI velocity is its mass-weighted center-of-mass velocity. The current was converted to SI units before evaluating

\[
\sigma(\tau)=\frac{1}{3Vk_\mathrm BT}
\int_0^\tau \left\langle \delta\mathbf J(0)\cdot\delta\mathbf J(t)\right\rangle\,dt.
\]

The autocorrelation was evaluated with an FFT-based, time-origin averaged estimator and an unbiased lag normalization. The current mean was removed before correlation. Uncertainty diagnostics use ten non-overlapping 2 ps blocks; the intervals are Student-t 95% intervals across blocks and should not be interpreted as ten independent replicas.

## Numerical results

| Integration cutoff τ (ps) | Full-trajectory cumulative σ (S m−1) | 2-ps block mean (S m−1) | 95% block interval (S m−1) |
|---:|---:|---:|---:|
| 0.05 | 15.48 | 15.24 | 13.62–16.85 |
| 0.10 | 8.66 | 8.11 | 6.28–9.94 |
| 0.20 | 11.95 | 10.55 | 7.33–13.78 |
| 0.30 | 6.05 | 4.14 | 1.28–7.01 |
| 0.50 | 5.29 | 2.92 | 0.68–5.15 |

The cumulative integral is not stable: the full-trajectory estimate changes from 15.48 S m−1 at 0.05 ps to 5.29 S m−1 at 0.50 ps and then drifts through the noise at longer lags. At the fixed 0.20 ps cutoff, the prefix estimate varies from 16.14 S m−1 at 4 ps to 11.95 S m−1 at 20 ps. This is evidence of estimator sensitivity, not a demonstrated physical time-dependent conductivity.

The self/cross decomposition is physically informative but not a convergence proof. At 0.20 ps, the cumulative Na self, FSI self, and Na–FSI cross contributions are 7.17, 5.51, and −0.72 S m−1, respectively. At 0.50 ps they are 2.13, 5.13, and −1.96 S m−1. The negative cross contribution is consistent with anticorrelated cation–anion motion reducing the total conductivity relative to an uncorrelated self-diffusion estimate.

## Publication status and exact blocker

**Status: diagnostic Green–Kubo result; not yet a final experimental-comparison value.**

The exact blocker is the absence of a defensible long-time plateau and the large dependence on integration cutoff/block selection within a single 20 ps trajectory. The data format, velocities, timestep, atom mapping, and fixed volume are not blockers.

A publication-grade follow-up should extend the NVE production sampling until the block estimates and running integral overlap over a prespecified plateau window, and should repeat the calculation for independent initial-condition replicas. A practical starting target is at least ~100 ps per replica and three replicas, followed by the same cutoff/block convergence test; the final duration should be set by the observed plateau rather than by a fixed number alone.

## Manuscript-ready methods text

> Ionic conductivity was evaluated from the equilibrium charge-current Green–Kubo relation, \(\sigma=(3Vk_\mathrm BT)^{-1}\int_0^\infty\langle\delta\mathbf J(0)\cdot\delta\mathbf J(t)\rangle dt\). The 298 K NVE trajectory comprised 20,001 saved configurations of a 5,110-atom periodic cell sampled every 1 fs. The ionic current was constructed using formal charges of +1e for Na+ and −1e for FSI−; FSI− velocities were calculated from mass-weighted molecular center-of-mass velocities, while neutral DME was excluded. The autocorrelation was calculated by FFT with time-origin averaging after removal of the finite-sample mean current, and the integral was converted to SI units using the instantaneous cell volume and the trajectory-averaged kinetic temperature (298.17 ± 2.59 K). Statistical sensitivity was assessed from ten non-overlapping 2 ps blocks using 95% Student-t intervals.

## Manuscript-ready results and discussion text

> The trajectory contains all fields required for a formal ionic Green–Kubo calculation, but its 20 ps duration is insufficient to establish a converged DC conductivity. The normalized current autocorrelation decays rapidly and subsequently oscillates around zero, while the cumulative conductivity varies from 15.48 S m−1 at a 0.05 ps cutoff to 5.29 S m−1 at 0.50 ps. Independent 2 ps block estimates likewise broaden from 13.62–16.85 S m−1 at 0.05 ps to 0.68–5.15 S m−1 at 0.50 ps. Consequently, no single long-time plateau can be identified without imposing an arbitrary integration window, and we do not report a final Green–Kubo conductivity for direct comparison with experiment from this trajectory alone.

> The current decomposition nevertheless shows the expected role of ion correlations. At 0.20 ps, the Na+ and FSI− self contributions are 7.17 and 5.51 S m−1, whereas the Na–FSI cross contribution is −0.72 S m−1; at 0.50 ps the cross contribution reaches −1.96 S m−1. These negative cross terms indicate anticorrelated cation–anion motion and provide a mechanistic explanation for why a conductivity inferred from independent self-diffusion coefficients can overestimate the collective charge transport. Because the correlation tail is not statistically converged here, this interpretation is qualitative and should be confirmed with longer, independently replicated trajectories.

## Figure captions

**Figure 1 | Green–Kubo ionic-current diagnostics for NaFSI/DME at 298 K.** (A) Instantaneous Cartesian components of the formal ionic charge current during the first 2 ps. (B) Normalized total current autocorrelation, \(C_J(\tau)/C_J(0)\). (C) Cumulative conductivity as a function of integration upper limit, separated into total, Na+ self, FSI− self, and Na–FSI cross terms. (D) Mean conductivity and 95% Student-t intervals from ten non-overlapping 2 ps blocks at several integration cutoffs. The ionic current uses formal ±1e charges and mass-weighted FSI− center-of-mass velocities; uncertainty bars describe block-to-block sampling variability, not independent simulation replicas. The absence of a stable long-time plateau limits the figure to diagnostic use.

**Figure 2 | Prefix sensitivity of the Green–Kubo estimate.** Cumulative conductivity evaluated at a fixed 0.20 ps integration cutoff as progressively longer prefixes of the same 20 ps trajectory are included. The persistent shift from the early prefixes to the full trajectory indicates finite-trajectory sensitivity; the points are not independent replicas.

## Deliverables

- [Green–Kubo diagnostic figure](green_kubo_diagnostic.png) and [editable SVG](green_kubo_diagnostic.svg)
- [Prefix-convergence figure](green_kubo_prefix_convergence.png) and [editable SVG](green_kubo_prefix_convergence.svg)
- [Trajectory and calculation summary](green_kubo_summary.json)
- [Full current-correlation and cumulative-conductivity curve](green_kubo_curve.csv)
- [Block-level estimates](green_kubo_block_estimates.csv) and [block summary with 95% intervals](green_kubo_block_summary.csv)
- [Prefix-convergence data](green_kubo_prefix_convergence.csv)
- [Current time series and temperature](green_kubo_currents.npz)
