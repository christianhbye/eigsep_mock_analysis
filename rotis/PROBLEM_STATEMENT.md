# rotis: Joint Beam and Sky Inference for EIGSEP
## Problem Statement

Read `memos/sample631.tex` before this document.
The memo derives the mathematical framework; this document specifies
what to implement, in what order, and to what standard.

---

## What rotis is

rotis performs joint inference of the EIGSEP antenna beam pattern and
the sky brightness temperature from rotation-diversity observations.
The measurement equation is bilinear in (beam, sky); the horizon and
known ground emission provide a linear anchor that makes joint recovery
tractable via a constructive identifiability theorem (memo Section 3).

rotis is **not** a forward simulator. It imports `eigsim` for all
data generation and knows nothing about croissant, HEALPix pixel
operations, or the EIGSEP hardware directly. Its inputs are:
- Simulated or real visibility datasets `y(tilt, az, lst)` from eigsim
- The H_lm grid and rotation matrices from eigsim
- The HFSS beam as a prior mean (optional, from eigsim)

---

## What eigsim provides

rotis imports the following from `eigsim`:

```python
from eigsim import simulate_dataset   # y[N_tilt, N_az, N_lst]
from eigsim import rotation_matrices  # R[N_tilt, N_az, N_lst, 3, 3]
from eigsim import Hlm_grid           # H_lm[N_tilt, N_az, n_alm]
from eigsim import beam_alm_hfss      # s2fft [L, 2L-1] format
from eigsim import config             # site params, lmax, T_gnd, etc.
```

All array shapes, coordinate conventions, alm formats, and rotation
conventions are defined by eigsim. rotis must not redefine them.
The key conventions to be aware of:

- **alm format:** s2fft `[L, 2L-1]` complex array, all m including negative
- **rotation matrices:** 3x3 JAX arrays, `R_total = R_LST @ R_mech`
- **H_lm:** depends on mechanical rotation (tilt, az) only — NOT LST
- **Ground term:** uses unrotated beam_alm, not rotated
- **LMAX:** set by eigsim config, typically `2 * NSIDE`

---

## Software stack

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, jacrev
import eigsim                          # forward model + data generation
from pygdsm import GlobalSkyModel2016  # sky model for priors + M3
```

No healpy, no croissant, no s2fft imports in rotis except where
format conversion is strictly necessary at the eigsim boundary.

---

## Experimental parameters

rotis inherits all experimental parameters from `eigsim.config`:

```python
from eigsim.config import (
    LAT_RAD, LON_RAD,    # site location
    LMAX_BEAM, LMAX_SKY, # harmonic band-limits
    T_GND,               # ground temperature (K)
    NSIDE,               # HEALPix resolution
    TILTS_DEG,           # mechanical tilt grid
    AZS_DEG,             # turntable azimuth grid
    LSTS_HR,             # LST sample grid
)

# Milestone 3 only
FREQ_MHZ_ARRAY     = np.linspace(50, 150, 51)   # 51 channels, 2 MHz
FREQ_MHZ_ARRAY_DEV = np.linspace(60, 120, 11)   # 11 channels for dev
```

No magic numbers in rotis. If a parameter is not in eigsim.config,
define it in `rotis/config.py` and document why it belongs to rotis
rather than eigsim.

---

## Three error terms in every Wigner projection

This is the most important fact to internalize before implementing
anything. The Wigner projection (memo Eq. 9) is never exact in
practice. Every call to `wigner_project` returns three distinct
error contributions:

**(i) Cross-ℓ leakage** — deterministic bias from finite rotation
grid. Computable from the rotation matrices before any data.
Correctable at the cost of noise amplification.

**(ii) LST sampling error** — deterministic bias from discrete
sidereal average. O(1/N_lst) for ℓ < N_lst/2; grows for higher ℓ.
Computable from the LST grid before any data.

**(iii) Noise** — random, zero mean, shrinks as 1/sqrt(N_obs).

Errors (i) and (ii) are known biases — they set the systematic floor
of the estimator regardless of integration time. The prior sweep
(Module 6) quantifies how these interact with the beam prior.

---

## Code structure

```
packages/rotis/
├── pyproject.toml
└── src/rotis/
    ├── __init__.py
    ├── fisher.py          # Module 1: FIM, coverage kernel, LST error
    ├── estimator.py       # Module 2: constructive estimator
    ├── als.py             # Module 3: ALS, Gibbs, prior sweep
    ├── multifreq.py       # Module 4: multi-frequency stacking (M3)
    ├── gsm_basis.py       # Module 5: GSM eigenmode decomposition (M3)
    ├── inference.py       # Module 6: Bayesian model comparison (M3)
    └── injection_recovery.py  # Module 7: validation (M3)
```

Tests live in `tests/rotis/` at the repo root, not inside the package.
Notebooks live in `notebooks/` at the repo root.

---

# MILESTONE 1: Fisher Analysis and Error Characterization

The goal of Milestone 1 is to characterize the information content
and systematic floor of the EIGSEP rotation grid **before doing any
inference**. All three error terms should be computable and plotted
from the rotation grid alone, without fitting any beam or sky.

## Module 1: `fisher.py`

```python
def compute_FIM(y_obs, beam_alm_true, sky_alm_true, Hlm_grid,
                rotation_matrices, T_gnd, noise_var, lmax):
    """
    Linearized Fisher information matrix at true (beam, sky).

    theta = [beam_alm (real+imag parts), sky_alm (real+imag parts)]
    J = jax.jacrev(simulator_fn)(theta_true)  shape (N_obs, N_params)
    FIM = J.T @ J / noise_var

    simulator_fn wraps eigsim.simulate_dataset as a function of theta.
    Use jacrev (reverse mode): N_obs >> N_params at NSIDE=32.

    Returns FIM shape (N_params, N_params).
    """

def coverage_kernel(rotation_matrices, weights, lmax):
    """
    K^{l0,l}_{m0,m0';m,m'} = (2*l0+1)/(8*pi^2) * sum_i w_i *
                               conj(D^l0_{m0,m0'}(R_i)) * D^l_{m,m'}(R_i)

    Error (i) from memo: K - delta_{l,l0}*I is the cross-l leakage bias.

    Computable from rotation_matrices alone — no data needed.
    Returns two arrays:
      diagonal:    K^{l0,l0} blocks — should be close to identity
      offdiagonal: sum_{l!=l0} ||K^{l0,l}||_F^2 vs l0 — the leakage power
    """

def compute_LST_sampling_error(lsts_hr, tilts_deg, azs_deg,
                                lat_rad, lmax):
    """
    Error (ii) from memo: LST quadrature error.

      eta[l, m, m'] = (1/N_t) * sum_j D^l_{mm'}(R_LST(t_j))
                      - d^l_{m0} * delta(m', 0)

    Computable from LST grid alone — no data needed.
    Returns eta shape (lmax+1, 2*lmax+1, 2*lmax+1).

    Key diagnostic: max|eta[l]| vs l.
    Should be < 0.01 for l < N_lst/2; grows for higher l.
    This is the LST-sampling systematic floor — independent of noise.
    """

def sky_asymmetry_snr(sky_alm, beam_alm, noise_var, lmax):
    """
    Condition C6 check: does the sky have enough m!=0 power at each l
    to constrain the beam shape from the LST residual?

    SNR_b_l = ||b_l|| * ||[a_{lm'}]_{m'!=0}|| / sigma_noise

    Returns SNR per l. Flag l-modes where SNR < 1 — beam shape
    unrecoverable from LST residual at those modes.
    Compute from GSM at each frequency in FREQ_MHZ_ARRAY.
    """

def plot_milestone1_summary(FIM, coverage_kernel_diag,
                             coverage_kernel_offdiag,
                             eta_lst, sky_snr, lmax):
    """
    Five-panel figure — the primary Milestone 1 output:
      1. FIM eigenvalue spectrum log10(lambda) vs mode index
         Mark knee and near-zero floor.
      2. Beam and sky marginal 1-sigma from diag(FIM^{-1})
         vs ell — Cramér-Rao bound per mode.
      3. Coverage kernel leakage power sum_{l!=l0}||K^{l0,l}||^2
         vs l0 — Error (i) systematic floor.
      4. LST sampling error max|eta[l]| vs l
         Mark l = N_lst/2 threshold — Error (ii) systematic floor.
      5. Sky asymmetry SNR per l at 100 MHz and 150 MHz
         Mark SNR=1 threshold — C6 violation boundary.
    Panels 3, 4, 5 together define the systematic floor before
    any inference is attempted.
    """
```

**Tests (`tests/rotis/test_fisher.py`):**
1. FIM is positive semidefinite
2. Number of near-zero FIM eigenvalues matches expected degeneracy
   count from memo conditions (~4–6)
3. FIM eigenvalues increase monotonically with number of tilts
4. `coverage_kernel`: diagonal blocks close to identity for uniform
   weights; off-diagonal leakage decreases with denser rotation grid
5. `compute_LST_sampling_error`: max|eta| < 0.01 for l < N_lst/2 = 12
   with 24 uniform samples; verify it grows for l > 12
6. `sky_asymmetry_snr`: SNR > 1 for l < ~20 at 100 MHz with GSM sky;
   plot the frequency dependence

**Definition of done for Milestone 1:**
- All six tests pass
- Five-panel summary plot produced from EIGSEP rotation grid
- Identify the systematic floor: which of errors (i), (ii), (iii)
  dominates at each ℓ for the EIGSEP noise budget
- Identify which ℓ-modes violate C6 at each frequency
- These results directly inform which beam modes to include in the
  Milestone 2 prior sweep

---

# MILESTONE 2: Constructive Estimator and Prior Sweep

**Do not begin until Milestone 1 definition of done is fully met.**

The constructive estimator implements the three-step proof of the
identifiability theorem (memo Section 3). The prior sweep is the
primary scientific output: it quantifies how much the rotation data
improve on the HFSS beam simulation at each angular scale.

## Module 2: `estimator.py`

```python
def compute_lst_residual(y_obs, lsts_hr):
    """
    delta_y[tilt, az, lst] = y_obs[tilt, az, lst] - mean_lst(y_obs[tilt, az])

    Removes the ground term exactly (constant in LST at fixed mech rotation).
    The m'=0 sky modes are also removed — recovered separately in
    recover_m0_column. Returns delta_y same shape as y_obs.
    """

def wigner_project(delta_y, rotation_matrices, weights, lmax,
                   K_leakage=None, eta_lst=None):
    """
    Wigner project LST residual to get X_hat[l] ≈ b_l @ a_l^T
    for m'!=0 columns (m'=0 column requires recover_m0_column).

    X_hat[l0, m0, m0'] = (2*l0+1)/(8*pi^2) * sum_i w_i *
                          conj(D^l0_{m0,m0'}(R_i)) * delta_y[i]

    Returns:
      X_hat_raw:       shape (lmax+1, 2*lmax+1, 2*lmax+1)
                       m'=0 column is unreliable — use recover_m0_column
      bias_leakage:    Error (i) estimate (requires K_leakage + beam/sky)
      bias_lst:        Error (ii) estimate (requires eta_lst + beam/sky)
    """

def recover_m0_column(X_hat_mneq0, Y_hat_raw, beam_hat,
                      G_matrix, T_gnd, lmax, n_iter=3):
    """
    Sub-step 1b (memo): recover m'=0 column of X_hat iteratively.

    Initializes beam shape from SVD of m'!=0 block, then corrects
    the raw m'=0 projection for ground contamination:
      X_hat[l0, m0, 0] = Y_hat[l0, m0, 0]
                        - T_gnd * sum_{lm} b_hat[lm] * G[l0,m0,0;lm]

    Requires C6: sky must have nonzero m'!=0 power at each l.
    Check sky_asymmetry_snr from Milestone 1 before calling this.
    Returns full X_hat with m'=0 column filled.
    """

def rank1_factorize(X_hat, lmax):
    """
    SVD of each l-block of X_hat → beam and sky shapes up to scale.
      X_hat[l] ≈ sigma_l * b_hat_l @ a_hat_l^T

    Returns:
      b_hat: shape (lmax+1, 2*lmax+1) — beam shape per l
      a_hat: shape (lmax+1, 2*lmax+1) — sky shape per l
      singular_values: shape (lmax+1,) — must be >> noise floor
      rank1_residual: ||X_hat[l] - sigma_l * outer(b,a)||_F per l
                      Large residual signals C6 violation at that l.
    """

def solve_alpha(b_hat, Hlm_grid, y_obs, rotation_matrices,
                weights, T_gnd, lmax):
    """
    Step 2 of theorem: solve M @ alpha = g for per-l amplitude scales.

    M[l0_m0_m0', l] = T_gnd * sum_m b_hat[l,m] * C[l0,m0,m0';l,m]
    g[l0_m0_m0']    = Wigner projection of ground term

    Ground projection extracted as: Y_hat_raw - X_hat (sky part).

    Returns alpha shape (lmax+1,), condition number of M.
    Large condition number signals C4 violation or poor beam estimate.
    """

def reconstruct(b_hat, a_hat, alpha):
    """
    b_lm = alpha_l * b_hat_lm,  a_lm = a_hat_lm / alpha_l.
    Returns b_alm, a_alm in s2fft [L, 2L-1] format.
    """

def run_constructive_estimator(y_obs, rotation_matrices, Hlm_grid,
                                G_matrix, weights, T_gnd, lmax,
                                K_leakage=None, eta_lst=None,
                                skip_l=None):
    """
    Full Steps 1+2 pipeline. skip_l: list of l-modes to exclude
    (use for C6-violating modes identified in Milestone 1).

    Returns (b_alm, a_alm, diagnostics).
    diagnostics: singular_values, rank1_residual per l,
                 alpha, M_condition_number, y_residual.
    """

def check_projection_errors(X_hat_raw, X_true, K_leakage, eta_lst,
                             beam_alm_true, sky_alm_true, noise_var):
    """
    Validation on synthetic data where truth is known.
    Decomposes total error into three components:
      X_hat_raw - X_true = leakage_term + LST_term + noise_term

    On noiseless data verifies leakage_term and LST_term match
    analytic predictions from K_leakage and eta_lst to < 1%.
    After correction: residual should be < noise floor.
    Run on noiseless data first to isolate the two bias terms.
    """
```

## Module 3: `als.py`

```python
def als_step_beam(a_alm, y_obs, Hlm_grid, rotation_matrices,
                  T_gnd, noise_var, beam_prior_mean, beam_prior_cov,
                  lmax):
    """
    Fix sky, solve for beam via Wiener filter.
    A(a)[i, lm] = sum_{m'} D^l_{m,m'}(R_i)*a[l,m'] + T_gnd*H_lm(R_i)
    Wiener: b = (A^T A / sigma^2 + C_b^{-1})^{-1}
               (A^T y / sigma^2 + C_b^{-1} mu_b)
    """

def als_step_sky(b_alm, y_obs, Hlm_grid, rotation_matrices,
                 T_gnd, noise_var, sky_prior_mean, sky_prior_cov,
                 lmax):
    """
    Fix beam, solve for sky via Wiener filter.
    Ground term subtracted as known offset using current b_alm.
    """

def run_als(y_obs, rotation_matrices, Hlm_grid, T_gnd, noise_var,
            beam_prior_mean, beam_prior_cov,
            sky_prior_mean, sky_prior_cov, lmax,
            init_b=None, init_a=None,
            max_iter=100, tol=1e-6):
    """
    ALS loop. Initializes from constructive estimator if init not given.
    Returns (b_alm, a_alm, convergence_history).
    convergence_history: residual and rho (canonical correlation) per iter.
    rho < 1 required for convergence; rho -> 1 at high SNR without priors.
    """

def gibbs_sample(y_obs, rotation_matrices, Hlm_grid, T_gnd,
                 noise_var, beam_prior_mean, beam_prior_cov,
                 sky_prior_mean, sky_prior_cov, lmax,
                 init_b, init_a,
                 n_samples=1000, n_burnin=200, key=None):
    """
    Gibbs sampler: alternating Gaussian conditional draws.
    Each step is a linear solve — exact for Gaussian noise + priors.
    Returns (b_samples, a_samples) shape (n_samples, n_alm).
    """

def beam_prior_from_hfss(beam_alm_hfss, sigma_fractional, lmax,
                          free_modes=None):
    """
    Gaussian prior centered on HFSS simulation.

    sigma_fractional: prior width as fraction of beam amplitude per mode.
      0.0    → rotation-free limit (simulation is truth)
      0.01   → 1% perturbative corrections
      0.05   → 5% perturbative corrections
      0.5    → weakly informative
      jnp.inf → flat prior (full rotis theorem, free beam)

    free_modes: list of (l, m) to exclude from prior (set flat).
    Use for modes identified as C6-violating in Milestone 1.

    Returns (prior_mean, prior_cov).
    prior_mean = beam_alm_hfss
    prior_cov  = diag((sigma_fractional * |b_hfss[lm]|)^2)
    """

def run_prior_sweep(y_obs, rotation_matrices, Hlm_grid, T_gnd,
                    noise_var, beam_alm_hfss, sky_alm_true, lmax,
                    sigma_grid=None, n_samples=500, n_burnin=100,
                    key=None):
    """
    THE PRIMARY SCIENTIFIC OUTPUT OF rotis.

    Run Gibbs sampler at each prior width in sigma_grid.
    Default sigma_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, jnp.inf]

    Returns posterior_widths shape (len(sigma_grid), lmax+1, 2*lmax+1):
      posterior_widths[i, l, m] = posterior std of b_lm at sigma_grid[i]

    Three regimes per l-mode:
      sigma_post << sigma_prior: data are informative, rotation helps
      sigma_post ~  sigma_prior: data add nothing at this l
      sigma_post ~  sigma_noise: noise floor reached

    Also returns sky_posterior_widths — how beam prior affects sky recovery.

    Required plot: sigma_post vs sigma_prior for each l on one figure.
    Identify:
      - Which l-modes are data-dominated (rotation improves on simulation)
      - Which are prior-dominated (rotation adds nothing)
      - The minimum sigma_prior where data dominate for each l
    """
```

**Tests (`tests/rotis/test_estimator.py`, `test_als.py`):**
1. `run_constructive_estimator` on noiseless synthetic data: y_residual < 1e-6
2. `check_projection_errors` on noiseless data:
   - leakage_term and LST_term match analytic predictions to < 1%
   - After bias correction, residual < noise floor
3. ALS initialized from constructive estimator converges in < 20
   iterations on noiseless data
4. Convergence diagnostic: residual and rho decrease monotonically
5. Gibbs posterior mean matches ALS MAP estimate
6. Gibbs posterior width consistent with FIM Cramér-Rao bound
7. `beam_prior_from_hfss` with sigma=0: posterior width ~ 0 (prior dominates)
8. `beam_prior_from_hfss` with sigma=inf: posterior width set by data alone

**Definition of done for Milestone 2:**
1. All eight tests pass
2. `check_projection_errors` decomposition correct on noiseless data
3. ALS convergence in < 20 iterations on noiseless data
4. Gibbs posterior consistent with FIM bound
5. Prior sweep `run_prior_sweep` produces clean three-regime plot for
   sigma_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, inf]
6. Prior sweep identifies which l-modes are data-dominated for the
   EIGSEP rotation grid and noise budget — this is the key result
7. `notebooks/milestone2_recovery.ipynb` shows:
   - True vs recovered beam and sky maps at each l
   - Singular value spectrum and alpha values
   - The prior sweep figure with regimes labeled

---

# MILESTONE 3: Multi-Frequency and 21-cm Signal Recovery

**Do not begin until Milestone 2 definition of done is fully met.**

## Scientific context

Multi-frequency observations improve conditioning through foreground
spectral structure and chromatic beam variation. The correct treatment
is a two-stage pipeline — not a log-polynomial basis, which absorbs
the 21-cm signal:

1. Recover `a_00(nu)` using the joint estimator across frequencies
2. Project out GSM foreground eigenmodes; search residual via
   Bayesian model comparison

Signal loss from the projection is exactly computable before taking
any data. The model-comparison framework is unbiased with respect to
this loss.

## Module 4: `multifreq.py`

```python
def run_multifreq_estimator(y_obs_multifreq, rotation_matrices,
                             Hlm_grid, T_gnd, noise_var, freq_array,
                             beam_spectral_model, lmax):
    """
    Stack alpha_l system across frequencies:
      [M(nu_1)]         [g(nu_1)]
      [M(nu_2)] alpha = [g(nu_2)]
         ...               ...

    beam_spectral_model options:
      'dipole': (nu/nu_0)^2 scaling (electrically small dipole)
      'hfss':   interpolate eigsim HFSS beam at each frequency
      'free':   independent alpha_l per frequency

    Returns a_00_hat(nu) with covariance N(nu, nu').
    N(nu, nu') is NOT diagonal — chromatic beam errors correlate channels.
    """

def beam_chromatic_model(beam_alm_ref, freq_array, freq_ref,
                          model='dipole'):
    """Returns beam_alm(nu) shape (n_freq, n_alm)."""
```

## Module 5: `gsm_basis.py`

```python
def compute_gsm_monopole_covariance(freq_array, n_realizations=100):
    """
    Covariance C_fg(nu, nu') of GSM2016 monopole across frequency.
    Perturb GSM spectral parameters for realizations.
    Returns C_fg shape (n_freq, n_freq). Must be positive definite.
    """

def gsm_eigenmodes(C_fg, K):
    """Top K eigenvectors. Shape (K, n_freq)."""

def projection_operator(eigenmodes):
    """Q_K = I - sum_k u_k u_k^T."""

def signal_loss(s_template, Q_K):
    """
    L_K = sum_k |<s, u_k>|^2 / ||s||^2.
    Computable before any data. Plot as heatmap over (nu_c, sigma_MHz).
    """

def project_residual(a00_hat, a00_cov, Q_K):
    """r_hat = Q_K @ a00_hat,  r_cov = Q_K @ a00_cov @ Q_K.T"""
```

## Module 6: `inference.py`

```python
def signal_template(A, nu_c_mhz, sigma_mhz, freq_array):
    """
    s(nu) = -A * exp(-(nu - nu_c)^2 / (2*sigma^2))
    Negative: 21-cm is an absorption feature.
    """

def log_likelihood(r_hat, r_cov, s_projected):
    """Gaussian log-likelihood for projected residual."""

def log_evidence_H0(r_hat, r_cov):
    """Evidence for null: no signal."""

def log_evidence_H1(r_hat, r_cov, Q_K, freq_array,
                    A_range=(0, 1), nu_c_range=(50, 150),
                    sigma_range=(5, 30), n_grid=50):
    """
    Evidence for signal via grid integration over (A, nu_c, sigma).
    Returns log_evidence, posterior_grid shape (n_A, n_nu_c, n_sigma).
    """

def bayes_factor(r_hat, r_cov, Q_K, freq_array, **prior_kwargs):
    """log B_10 = log_evidence_H1 - log_evidence_H0. B_10 > 10: strong evidence."""
```

## Module 7: `injection_recovery.py`

```python
def injection_recovery_curve(pipeline_fn, freq_array, Q_K,
                              A_grid, nu_c_mhz=80.0, sigma_mhz=15.0,
                              n_noise_trials=50, key=None):
    """
    For each A in A_grid: inject signal, run pipeline, record recovery.
    Returns DataFrame: A_injected, A_recovered, A_lower, A_upper,
                       signal_loss, log_bayes_factor, detection.
    Unbiased (slope=1) after signal-loss correction above min detectable A.
    """

def prior_sensitivity(r_hat, r_cov, Q_K, freq_array):
    """Vary prior ranges +-50%; verify |Delta log B_10| < 1."""
```

**Definition of done for Milestone 3:**
1. `compute_gsm_monopole_covariance` positive definite; top 5 modes
   explain > 99% of variance across 50–150 MHz
2. `signal_loss` heatmap: < 30% loss for typical signal parameters
   with K=5 modes
3. `log_evidence_H1` recovers correct (A, nu_c, sigma) from noiseless
   injected signal with no foreground
4. `injection_recovery_curve` unbiased after signal-loss correction
5. Minimum detectable amplitude (B_10 > 10) documented vs noise
   level and K
6. Prior sensitivity: Bayes factor stable under prior variation
7. `notebooks/milestone3b_inference.ipynb` produces injection-recovery
   plot and minimum detectable amplitude vs noise curve

---

## Cross-cutting rules

**rotis never imports from croissant directly.** All forward model
calls go through eigsim. If you find yourself importing croissant,
the function belongs in eigsim, not rotis.

**ALS and Gibbs do not need bias corrections.** They operate on raw
`y_obs` via the likelihood and correct for all biases automatically.
The error terms (i) and (ii) only matter for the constructive
estimator in Module 2.

**The noise covariance in Milestone 3 is not diagonal.** N(nu, nu')
from the multifreq estimator is frequency-correlated due to chromatic
beam errors. Verify its structure from Gibbs chains before trusting
Milestone 3 results.

**C6 modes from Milestone 1 must be excluded in Milestone 2.**
Pass them as `skip_l` to `run_constructive_estimator` and as
`free_modes` to `beam_prior_from_hfss`.
