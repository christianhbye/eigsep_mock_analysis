"""Tests for rotis.fisher -- Milestone 1 error characterization."""

import jax.numpy as jnp
import numpy as np
from rotis.fisher import (
    compute_fim,
    coverage_kernel,
    lst_sampling_error,
    sky_asymmetry_snr,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotmat_ZYZ(alpha, beta, gamma):
    """Build a 3x3 rotation matrix from ZYZ Euler angles."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz_g = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz_a @ Ry_b @ Rz_g


def _haar_random_rotmats(n, seed=42):
    """Generate n Haar-random SO(3) rotation matrices."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0, 2 * np.pi, n)
    beta = np.arccos(rng.uniform(-1, 1, n))
    gamma = rng.uniform(0, 2 * np.pi, n)
    return np.stack([_rotmat_ZYZ(a, b, g) for a, b, g in zip(alpha, beta, gamma)])


# ---------------------------------------------------------------------------
# coverage_kernel
# ---------------------------------------------------------------------------


def test_coverage_kernel_diagonal_near_identity():
    """Diagonal blocks of coverage kernel should be close to identity
    for Haar-distributed rotations."""
    lmax = 2
    L = lmax + 1
    N = 100

    rotmats = _haar_random_rotmats(N)
    weights = jnp.full(N, 8 * np.pi**2 / N)

    diagonal, offdiag = coverage_kernel(rotmats, weights, lmax)

    # Diagonal completeness should be close to 1 for physical modes
    for ell in range(L):
        lo = L - 1 - ell
        hi = L - 1 + ell + 1
        block = diagonal[ell, lo:hi, lo:hi]
        np.testing.assert_allclose(float(jnp.mean(block)), 1.0, atol=0.1)


def test_coverage_kernel_leakage_decreases_with_density():
    """Off-diagonal leakage should decrease with denser rotation grid."""
    lmax = 2

    _, leak_sparse = coverage_kernel(
        _haar_random_rotmats(50, seed=0),
        jnp.full(50, 8 * np.pi**2 / 50),
        lmax,
    )

    _, leak_dense = coverage_kernel(
        _haar_random_rotmats(200, seed=0),
        jnp.full(200, 8 * np.pi**2 / 200),
        lmax,
    )

    assert float(jnp.sum(leak_dense)) < float(jnp.sum(leak_sparse))


# ---------------------------------------------------------------------------
# lst_sampling_error
# ---------------------------------------------------------------------------


def test_lst_sampling_error_small_below_nyquist():
    """max|eta| < 0.01 for |m| < N_lst/2 with 24 uniform LST samples.

    Should grow for |m| > N_lst/2 = 12."""
    N_lst = 24
    lsts = jnp.linspace(0, 2 * jnp.pi, N_lst, endpoint=False)
    lmax = 30  # must be >= N_lst to test aliasing
    L = lmax + 1
    lat_rad = np.radians(39.25)

    eta = lst_sampling_error(lsts, lat_rad, lmax)

    # eta should be diagonal in (m, m')
    diag_vals = jnp.diagonal(eta[0])
    total = float(jnp.sum(jnp.abs(eta[0]) ** 2))
    offdiag_power = total - float(jnp.sum(jnp.abs(diag_vals) ** 2))
    assert offdiag_power < 1e-20

    # Below Nyquist: |psi(m)| should be tiny (float32 limits precision)
    m_vals = np.arange(-(L - 1), L)
    for idx, m in enumerate(m_vals):
        if 0 < abs(m) < N_lst // 2:
            assert float(jnp.abs(eta[0, idx, idx])) < 1e-5

    # At aliased frequency m = N_lst: psi should be ~1
    alias_idx = N_lst + (L - 1)
    assert float(jnp.abs(eta[0, alias_idx, alias_idx])) > 0.9


def test_lst_sampling_error_identical_across_ell():
    """In equatorial coords, psi(m) is independent of l."""
    lsts = jnp.linspace(0, 2 * jnp.pi, 12, endpoint=False)
    eta = lst_sampling_error(lsts, 0.0, lmax=6)

    for ell in range(1, 7):
        np.testing.assert_allclose(
            np.array(eta[ell]),
            np.array(eta[0]),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# sky_asymmetry_snr
# ---------------------------------------------------------------------------


def test_sky_asymmetry_snr_formula():
    """Verify SNR matches the analytic formula for a simple case."""
    lmax = 4
    L = lmax + 1

    beam_alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    beam_alm = beam_alm.at[1, L - 1].set(1.0)
    beam_alm = beam_alm.at[1, L].set(0.5)

    sky_alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    sky_alm = sky_alm.at[1, L - 1].set(100.0)
    sky_alm = sky_alm.at[1, L].set(50.0)
    sky_alm = sky_alm.at[1, L - 2].set(50.0)

    snr = sky_asymmetry_snr(sky_alm, beam_alm, 1.0, lmax)

    expected = float(jnp.sqrt(1.0 + 0.25) * jnp.sqrt(50.0**2 + 50.0**2))
    np.testing.assert_allclose(float(snr[1]), expected, rtol=1e-5)

    assert float(snr[0]) == 0.0
    for ell in range(2, L):
        assert float(snr[ell]) == 0.0


def test_sky_asymmetry_snr_axisymmetric_sky():
    """Axisymmetric sky (all power in m=0) should give SNR = 0."""
    lmax = 4
    L = lmax + 1

    beam_alm = jnp.ones((L, 2 * L - 1), dtype=complex)
    sky_alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    sky_alm = sky_alm.at[:, L - 1].set(100.0)

    snr = sky_asymmetry_snr(sky_alm, beam_alm, 1.0, lmax)
    np.testing.assert_allclose(snr, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# FIM tests -- stubs until compute_fim is implemented
# ---------------------------------------------------------------------------


def test_fim_positive_semidefinite():
    """FIM should be positive semidefinite."""
    lmax = 2
    L = lmax + 1

    rng = np.random.default_rng(42)
    A = jnp.array(rng.standard_normal((20, L, 2 * L - 1)))
    B = jnp.array(rng.standard_normal((20, L, 2 * L - 1)))

    def sim(beam_alm, sky_alm):
        return (
            jnp.einsum("ijk,jk->i", A, beam_alm) + jnp.einsum("ijk,jk->i", B, sky_alm)
        ).real

    beam = jnp.zeros((L, 2 * L - 1), dtype=complex)
    beam = beam.at[1, L - 1].set(1.0)  # b_{1,0} real
    beam = beam.at[1, L].set(0.5 + 0.1j)  # b_{1,1}
    beam = beam.at[1, L - 2].set(-(0.5 - 0.1j))  # b_{1,-1} = -conj(b_{1,1})

    sky = jnp.zeros((L, 2 * L - 1), dtype=complex)
    sky = sky.at[0, L - 1].set(100.0)
    sky = sky.at[1, L - 1].set(10.0)
    sky = sky.at[1, L].set(5.0 + 1.0j)
    sky = sky.at[1, L - 2].set(-(5.0 - 1.0j))

    fim, _J = compute_fim(sim, beam, sky, 1.0, lmax)
    eigvals = jnp.linalg.eigvalsh(fim)
    assert jnp.all(eigvals >= -1e-6), f"min eigenvalue = {float(jnp.min(eigvals))}"


def test_fim_nullspace_matches_degeneracy():
    """Block-diagonal bilinear model has one scale degeneracy per ell."""
    lmax = 2
    L = lmax + 1

    rng = np.random.default_rng(42)
    n_obs = 50
    # Complex M couples Re/Im parts like physical D-matrices do
    M = [
        jnp.array(
            rng.standard_normal((n_obs, 2 * L - 1, 2 * L - 1))
            + 1j * rng.standard_normal((n_obs, 2 * L - 1, 2 * L - 1))
        )
        for _ in range(L)
    ]

    def sim(beam_alm, sky_alm):
        y = jnp.zeros(n_obs, dtype=complex)
        for ell in range(L):
            y = y + jnp.einsum("imn,m,n->i", M[ell], beam_alm[ell], sky_alm[ell])
        return y.real

    # Non-trivial fiducial point (must satisfy reality condition)
    beam = jnp.zeros((L, 2 * L - 1), dtype=complex)
    beam = beam.at[0, L - 1].set(1.0)
    beam = beam.at[1, L - 1].set(0.8)
    beam = beam.at[1, L].set(0.3 + 0.2j)
    beam = beam.at[1, L - 2].set(-(0.3 - 0.2j))
    beam = beam.at[2, L - 1].set(0.5)
    beam = beam.at[2, L].set(0.2 + 0.1j)
    beam = beam.at[2, L - 2].set(-(0.2 - 0.1j))
    beam = beam.at[2, L + 1].set(0.1 + 0.05j)
    beam = beam.at[2, L - 3].set(0.1 - 0.05j)

    sky = jnp.zeros((L, 2 * L - 1), dtype=complex)
    sky = sky.at[0, L - 1].set(100.0)
    sky = sky.at[1, L - 1].set(50.0)
    sky = sky.at[1, L].set(20.0 + 5.0j)
    sky = sky.at[1, L - 2].set(-(20.0 - 5.0j))
    sky = sky.at[2, L - 1].set(30.0)
    sky = sky.at[2, L].set(10.0 + 3.0j)
    sky = sky.at[2, L - 2].set(-(10.0 - 3.0j))
    sky = sky.at[2, L + 1].set(5.0 + 1.0j)
    sky = sky.at[2, L - 3].set(5.0 - 1.0j)

    fim, J = compute_fim(sim, beam, sky, 1.0, lmax)
    s = jnp.linalg.svd(J, full_matrices=False)[1]

    tol = max(J.shape) * jnp.finfo(float).eps * s[0]
    n_degenerate = int(jnp.sum(s < tol))

    # Complex M couples Re/Im subspaces → one scale degeneracy per l
    assert n_degenerate == L, (
        f"Expected {L} degenerate modes, got {n_degenerate}. "
        f"Smallest singular values: {s[-L - 2 :]}"
    )


def test_fim_eigenvalues_increase_with_tilts():
    """FIM trace should increase with more observations."""
    lmax = 2
    L = lmax + 1

    rng = np.random.default_rng(42)
    # Pre-generate large projection matrices; subsets share the same rows
    A_full = jnp.array(rng.standard_normal((100, L, 2 * L - 1)))
    B_full = jnp.array(rng.standard_normal((100, L, 2 * L - 1)))

    beam = jnp.zeros((L, 2 * L - 1), dtype=complex)
    beam = beam.at[0, L - 1].set(1.0)
    sky = jnp.zeros((L, 2 * L - 1), dtype=complex)
    sky = sky.at[0, L - 1].set(100.0)

    def make_fim(n_obs):
        A = A_full[:n_obs]
        B = B_full[:n_obs]

        def sim(beam_alm, sky_alm):
            return (
                jnp.einsum("ijk,jk->i", A, beam_alm)
                + jnp.einsum("ijk,jk->i", B, sky_alm)
            ).real

        return compute_fim(sim, beam, sky, 1.0, lmax)[0]

    trace_small = float(jnp.trace(make_fim(10)))
    trace_large = float(jnp.trace(make_fim(50)))
    assert trace_large > trace_small, (
        f"trace(FIM_50)={trace_large} should exceed trace(FIM_10)={trace_small}"
    )
