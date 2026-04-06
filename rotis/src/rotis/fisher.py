"""Fisher information and error characterization for the EIGSEP rotation grid.

Milestone 1: characterize the information content and systematic floor
of the rotation grid before doing any inference. All three error terms
(cross-ell leakage, LST sampling error, noise) are computable from the
rotation grid alone, without fitting any beam or sky.

Mathematical reference: memo/sample631.tex, Sections 3--4.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import s2fft
from croissant.rotations import rotmat_to_eulerZYZ


def _wigner_D_matrices(alpha, beta, gamma, lmax):
    """Compute full Wigner D-matrices for a single ZYZ rotation.

    D^l_{mm'}(alpha, beta, gamma) = exp(-i*m*alpha) * d^l_{mm'}(beta) * exp(-i*m'*gamma)

    Parameters
    ----------
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    D : jnp.ndarray
        Shape ``(L, 2L-1, 2L-1)`` where ``L = lmax + 1``.
        ``D[l, m + L-1, m' + L-1] = D^l_{mm'}(alpha, beta, gamma)``.
    """
    L = lmax + 1
    dl = s2fft.generate_rotate_dls(L, beta)  # (L, 2L-1, 2L-1), real
    m = jnp.arange(-(L - 1), L)  # (2L-1,)
    return (
        dl.astype(complex)
        * jnp.exp(-1j * m * alpha)[None, :, None]
        * jnp.exp(-1j * m * gamma)[None, None, :]
    )


def _rotmats_to_D(rotation_matrices, lmax):
    """Compute D-matrices for a batch of rotation matrices.

    Parameters
    ----------
    rotation_matrices : array_like
        Shape ``(N, 3, 3)``.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    D_all : jnp.ndarray
        Shape ``(N, L, 2L-1, 2L-1)``.
    """
    N = rotation_matrices.shape[0]
    D_list = []
    for i in range(N):
        alpha, beta, gamma = rotmat_to_eulerZYZ(np.asarray(rotation_matrices[i]))
        D_list.append(_wigner_D_matrices(float(alpha), float(beta), float(gamma), lmax))
    return jnp.stack(D_list)


def _pack_alm(alm, L):
    """Pack complex alm into independent real parameters respecting reality.

    For a real-valued field, a_{l,-m} = (-1)^m conj(a_{l,m}) and a_{l,0}
    is real.  The independent parameters per degree l are:

        Re(a_{l,0}), Re(a_{l,1}), Im(a_{l,1}), ..., Re(a_{l,l}), Im(a_{l,l})

    giving 2l+1 parameters per l and L^2 total.

    Parameters
    ----------
    alm : jnp.ndarray
        Complex alm array, shape ``(L, 2L-1)`` in s2fft convention.
    L : int
        ``lmax + 1``.

    Returns
    -------
    params : jnp.ndarray
        Real parameter vector of length ``L**2``.
    """
    params = jnp.zeros(L * L)
    idx = 0
    for ell in range(L):
        # m = 0: purely real
        params = params.at[idx].set(alm[ell, L - 1].real)
        idx += 1
        # m = 1 .. ell
        for m in range(1, ell + 1):
            col = L - 1 + m
            params = params.at[idx].set(alm[ell, col].real)
            params = params.at[idx + 1].set(alm[ell, col].imag)
            idx += 2
    return params


def _unpack_alm(params, L):
    """Reconstruct complex alm from independent real parameters.

    Inverse of :func:`_pack_alm`.  Sets m > 0 from params, m = 0 from the
    real parameter, and m < 0 via (-1)^m conj(m > 0).

    Parameters
    ----------
    params : jnp.ndarray
        Real parameter vector of length ``L**2``.
    L : int
        ``lmax + 1``.

    Returns
    -------
    alm : jnp.ndarray
        Complex alm array, shape ``(L, 2L-1)``.
    """
    alm = jnp.zeros((L, 2 * L - 1), dtype=complex)
    idx = 0
    for ell in range(L):
        # m = 0: purely real
        alm = alm.at[ell, L - 1].set(params[idx] + 0j)
        idx += 1
        # m = 1 .. ell
        for m in range(1, ell + 1):
            re_val = params[idx]
            im_val = params[idx + 1]
            idx += 2
            val = re_val + 1j * im_val
            alm = alm.at[ell, L - 1 + m].set(val)
            alm = alm.at[ell, L - 1 - m].set((-1) ** m * jnp.conj(val))
    return alm


def compute_fim(
    simulator_fn,
    beam_alm,
    sky_alm,
    noise_var,
    lmax,
):
    """Linearized Fisher information matrix at a fiducial (beam, sky).

    Packs beam and sky alm into a single real parameter vector of
    independent degrees of freedom (respecting the reality condition
    a_{l,-m} = (-1)^m conj(a_{l,m})), computes the Jacobian
    J = d(simulator_fn)/d(theta) via ``jax.jacrev``, and returns
    FIM = J^T J / noise_var.

    Parameters
    ----------
    simulator_fn : callable
        Function mapping ``(beam_alm, sky_alm) -> y_obs``, where y_obs
        is the flattened observation vector. This should wrap
        eigsim.simulate as a differentiable function of the alm
        parameters.
    beam_alm : jnp.ndarray
        Fiducial beam alm, shape ``(L, 2L-1)`` in s2fft convention.
    sky_alm : jnp.ndarray
        Fiducial sky alm, shape ``(L, 2L-1)`` in s2fft convention.
    noise_var : float
        Noise variance per observation (scalar, assumes white noise).
    lmax : int
        Maximum multipole order. L = lmax + 1.

    Returns
    -------
    fim : jnp.ndarray
        Fisher information matrix, shape ``(N_params, N_params)`` where
        ``N_params = 2 * L**2`` (L^2 independent real parameters for each
        of beam and sky, after enforcing the reality condition).
    jacobian : jnp.ndarray
        Raw Jacobian matrix, shape ``(N_obs, N_params)``.  Use ``SVD(J)``
        instead of ``eigvalsh(J^T J)`` to avoid squaring the condition
        number and losing float64 precision.
    """
    L = lmax + 1
    n_beam = L * L

    beam_alm = jnp.asarray(beam_alm)
    sky_alm = jnp.asarray(sky_alm)

    def forward(theta):
        b = _unpack_alm(theta[:n_beam], L)
        s = _unpack_alm(theta[n_beam:], L)
        return jnp.ravel(simulator_fn(b, s).real)

    theta0 = jnp.concatenate([_pack_alm(beam_alm, L), _pack_alm(sky_alm, L)])
    J = jax.jacrev(forward)(theta0)
    return J.T @ J / noise_var, J


def coverage_kernel(rotation_matrices, weights, lmax):
    """Coverage kernel quantifying cross-ell leakage from the rotation grid.

    Computes the coverage kernel (memo Eq. in Sec. 4.1):

        K^{l0,l}_{m0,m0';m,m'} = (2*l0+1)/(8*pi^2) * sum_i w_i *
            conj(D^l0_{m0,m0'}(R_i)) * D^l_{m,m'}(R_i)

    Under Haar measure, K^{l0,l} = delta_{l,l0} * I (no leakage).
    Deviation from this identity is the cross-ell leakage bias (error
    term (i) from the memo).

    This is computable from the rotation grid alone -- no data needed.

    Parameters
    ----------
    rotation_matrices : array_like
        Rotation matrices for all observations, shape ``(N_obs, 3, 3)``.
        Each R_i encodes the full rotation (mechanical + LST).
    weights : array_like
        Quadrature weights, shape ``(N_obs,)``. Should approximate
        the Haar measure, i.e., ``sum(w_i * f(R_i)) ~ integral f dR``
        where the Haar volume is ``8 * pi^2``. For uniform weights:
        ``w_i = 8 * pi^2 / N_obs``.
    lmax : int
        Maximum multipole order to compute.

    Returns
    -------
    diagonal : jnp.ndarray
        Diagonal of the ``K^{l0,l0}`` blocks, shape
        ``(lmax+1, 2*lmax+1, 2*lmax+1)``.
        ``diagonal[l, m+L-1, m'+L-1]`` is the self-response of mode
        ``(l, m, m')``. Should be 1.0 for perfect Haar coverage.
    offdiagonal_power : jnp.ndarray
        Leakage power ``sum_{l != l0} ||K^{l0,l}||_F^2`` vs l0,
        shape ``(lmax+1,)``. Measures the cross-ell systematic floor.
    """
    L = lmax + 1
    weights = jnp.asarray(weights)

    # Compute D-matrices for all rotations: (N, L, 2L-1, 2L-1)
    D_all = _rotmats_to_D(rotation_matrices, lmax)

    # Normalization: (2l+1) / (8pi^2)
    ell = jnp.arange(L)
    norm = (2 * ell + 1) / (8 * jnp.pi**2)

    # --- Diagonal completeness ---
    # K^{l,l}_{m,m';m,m'} = norm[l] * sum_i w_i * |D^l_{mm'}(R_i)|^2
    diagonal = jnp.einsum("i,iklm->klm", weights, jnp.abs(D_all) ** 2)
    diagonal = diagonal * norm[:, None, None]

    # --- Off-diagonal leakage power via Gram matrices ---
    # G[l, i, j] = sum_{m,m'} D^l_{mm'}(R_i) * conj(D^l_{mm'}(R_j))
    N = D_all.shape[0]
    D_flat = D_all.reshape(N, L, -1)  # (N, L, (2L-1)^2)
    G = jnp.einsum("ilk,jlk->lij", D_flat, D_flat.conj())  # (L, N, N)

    W = weights[:, None] * weights[None, :]  # (N, N)
    G_total = jnp.sum(G, axis=0)  # (N, N)

    # offdiag[l0] = norm[l0]^2 * sum_{l!=l0} ||K^{l0,l}||_F^2
    #             = norm[l0]^2 * [sum_{ij} W * conj(G[l0]) * G_total
    #                             - sum_{ij} W * |G[l0]|^2]
    WG_total = W * G_total  # (N, N)
    term_total = jnp.einsum("lij,ij->l", G.conj(), WG_total).real
    G_abs_sq = (G * G.conj()).real  # (L, N, N)
    term_self = jnp.einsum("lij,ij->l", G_abs_sq, W)
    offdiagonal_power = norm**2 * (term_total - term_self)

    return diagonal, offdiagonal_power


def lst_sampling_error(lsts_rad, lat_rad, lmax):
    """LST quadrature error from discrete sidereal sampling.

    In equatorial coordinates (as used by eigsim), the LST rotation is
    a pure z-axis rotation, so D^l_{mm'}(R_LST(t)) = exp(-i*m*t) * delta_{m,m'}.

    The continuous sidereal average gives delta_{m,0} * delta_{m',0},
    so the sampling error reduces to a scalar per m:

        psi(m) = (1/N_t) * sum_j exp(-i*m*lst_j) - delta_{m,0}

    and eta[l, m, m'] = delta_{m,m'} * psi(m) for all l.

    For uniformly spaced LST samples, |psi(m)| < machine epsilon for
    |m| < N_lst/2 (below the Nyquist limit) and equals 1.0 for aliased
    modes m = k * N_lst.

    Parameters
    ----------
    lsts_rad : array_like
        LST samples in radians, shape ``(N_lst,)``.
    lat_rad : float
        Observatory latitude in radians. Unused in the equatorial frame
        (kept for API compatibility with a future topocentric variant).
    lmax : int
        Maximum multipole order.

    Returns
    -------
    eta : jnp.ndarray
        LST sampling error, shape ``(lmax+1, 2*lmax+1, 2*lmax+1)``.
        Nonzero only on the diagonal (m = m').
        Key diagnostic: ``max|eta[l]|`` vs l should be < 0.01 for
        l < N_lst/2 and grows for higher l.
    """
    L = lmax + 1
    lsts = jnp.asarray(lsts_rad)
    m = jnp.arange(-(L - 1), L)  # (2L-1,)

    # psi[m] = mean_j exp(-i*m*lst_j) - delta_{m,0}
    phases = jnp.exp(-1j * jnp.outer(m, lsts))  # (2L-1, N_t)
    psi = jnp.mean(phases, axis=1) - (m == 0).astype(complex)  # (2L-1,)

    # Build diagonal tensor: eta[l, k, k] = psi[k] for all l
    eta = jnp.zeros((L, 2 * L - 1, 2 * L - 1), dtype=complex)
    idx = jnp.arange(2 * L - 1)
    eta = eta.at[:, idx, idx].set(jnp.broadcast_to(psi, (L, 2 * L - 1)))

    return eta


def sky_asymmetry_snr(sky_alm, beam_alm, noise_var, lmax):
    """Check condition C6: sky asymmetry SNR per multipole.

    Condition C6 requires the sky to have nonzero m' != 0 power at each
    l for the beam shape to be recoverable from the LST residual.

    Computes:
        SNR_b_l = ||b_l|| * ||[a_{lm'}]_{m'!=0}|| / sigma_noise

    Parameters
    ----------
    sky_alm : jnp.ndarray
        Sky alm, shape ``(L, 2L-1)`` in s2fft convention.
    beam_alm : jnp.ndarray
        Beam alm, shape ``(L, 2L-1)`` in s2fft convention.
    noise_var : float
        Noise variance per observation.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    snr : jnp.ndarray
        Sky asymmetry SNR per l, shape ``(lmax+1,)``.
        Flag l-modes where SNR < 1 -- beam shape is unrecoverable
        from the LST residual at those modes.
    """
    L = lmax + 1
    sigma = jnp.sqrt(noise_var)

    # Beam norm per ell: ||b_l||
    b_norm = jnp.sqrt(jnp.sum(jnp.abs(beam_alm) ** 2, axis=1))  # (L,)

    # Sky with m=0 zeroed: m=0 is at column index L-1 in s2fft convention
    sky_mneq0 = sky_alm.at[:, L - 1].set(0.0)
    a_mneq0_norm = jnp.sqrt(jnp.sum(jnp.abs(sky_mneq0) ** 2, axis=1))  # (L,)

    return b_norm * a_mneq0_norm / sigma


def plot_milestone1_summary(
    jacobian,
    noise_var,
    beam_alm,
    sky_alm,
    kernel_diagonal,
    kernel_offdiag_power,
    eta_lst,
    sky_snr,
    lmax,
):
    """Five-panel summary figure for Milestone 1.

    Panels:
        1. FIM singular-value spectrum: s_i^2/noise_var vs mode index.
           Uses SVD of J to avoid squaring the condition number.
        2. Beam and sky marginal 1-sigma CRB vs ell.
        3. Statistical (CRB) vs systematic (leakage bias) floor per ell.
        4. LST sampling error max|eta[l]| vs l (error term (ii)).
           Mark l = N_lst/2 threshold.
        5. Sky asymmetry SNR per l. Mark SNR=1 threshold.

    Panels 3, 4, 5 together define the systematic floor before any
    inference is attempted.

    Parameters
    ----------
    jacobian : jnp.ndarray
        Jacobian matrix from ``compute_fim``, shape ``(N_obs, N_params)``.
    noise_var : float
        Noise variance per observation.
    beam_alm : jnp.ndarray
        Fiducial beam alm, shape ``(L, 2L-1)`` in s2fft convention.
    sky_alm : jnp.ndarray
        Fiducial sky alm, shape ``(L, 2L-1)`` in s2fft convention.
    kernel_diagonal : jnp.ndarray
        Diagonal blocks from ``coverage_kernel``.
    kernel_offdiag_power : jnp.ndarray
        Off-diagonal leakage power from ``coverage_kernel``.
    eta_lst : jnp.ndarray
        LST sampling error from ``lst_sampling_error``.
    sky_snr : jnp.ndarray
        Sky asymmetry SNR from ``sky_asymmetry_snr``.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The five-panel summary figure.
    """
    L = lmax + 1
    n_beam = L * L
    ell = np.arange(L)

    # SVD of Jacobian (avoids squaring the condition number)
    J = np.array(jacobian)
    _U, s, Vt = np.linalg.svd(J, full_matrices=False)
    eigvals = s**2 / noise_var  # FIM eigenvalues
    tol = max(J.shape) * np.finfo(float).eps * s[0]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # --- Panel 1: FIM eigenvalue spectrum (from SVD) ---
    ax = axes[0]
    ax.semilogy(np.arange(len(eigvals)), np.maximum(eigvals, 1e-30), ".-", ms=2)
    n_zero = int(np.sum(s < tol))
    if n_zero > 0:
        ax.axvline(
            len(eigvals) - n_zero,
            color="gray",
            ls=":",
            alpha=0.7,
            label=f"{n_zero} near-zero",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("FIM eigenvalue spectrum")

    # --- Panel 2: CRB per ell (SVD-based pseudo-inverse) ---
    ax = axes[1]
    mask = s > tol
    inv_s2 = np.zeros_like(s)
    inv_s2[mask] = 1.0 / s[mask] ** 2
    crb = noise_var * np.sum(Vt**2 * inv_s2[:, None], axis=0)
    beam_sigma = np.zeros(L)
    sky_sigma = np.zeros(L)
    for l_idx in range(L):
        start = l_idx * l_idx
        width = 2 * l_idx + 1
        n_modes = max(width, 1)
        beam_sigma[l_idx] = np.sqrt(
            np.sum(np.maximum(crb[start : start + width], 0)) / n_modes
        )
        sky_sigma[l_idx] = np.sqrt(
            np.sum(np.maximum(crb[n_beam + start : n_beam + start + width], 0))
            / n_modes
        )
    ax.semilogy(ell, np.maximum(beam_sigma, 1e-30), "o-", label="Beam", ms=4)
    ax.semilogy(ell, np.maximum(sky_sigma, 1e-30), "s-", label="Sky", ms=4)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma$ (CRB)")
    ax.set_title("CRB per mode")
    ax.legend(fontsize=8)

    # --- Panel 3: Statistical vs systematic floor ---
    ax = axes[2]
    offdiag = np.array(kernel_offdiag_power)
    sqrt_leak = np.sqrt(np.maximum(offdiag, 0))
    beam_arr = np.array(beam_alm)
    sky_arr = np.array(sky_alm)
    beam_rms = np.array(
        [np.sqrt(np.mean(np.abs(beam_arr[ell]) ** 2)) for ell in range(L)]
    )
    sky_rms = np.array(
        [np.sqrt(np.mean(np.abs(sky_arr[ell]) ** 2)) for ell in range(L)]
    )
    ax.semilogy(ell, np.maximum(beam_sigma, 1e-30), "o-", label="Beam CRB", ms=4)
    ax.semilogy(ell, np.maximum(sky_sigma, 1e-30), "s-", label="Sky CRB", ms=4)
    ax.semilogy(
        ell,
        np.maximum(sqrt_leak * beam_rms, 1e-30),
        "o--",
        label="Beam leakage",
        ms=4,
        alpha=0.7,
    )
    ax.semilogy(
        ell,
        np.maximum(sqrt_leak * sky_rms, 1e-30),
        "s--",
        label="Sky leakage",
        ms=4,
        alpha=0.7,
    )
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma$ or bias (alm units)")
    ax.set_title("Statistical vs systematic floor")
    ax.legend(fontsize=7)

    # --- Panel 4: LST sampling error ---
    ax = axes[3]
    eta = np.array(eta_lst)
    max_eta = np.array([np.max(np.abs(eta[l_idx])) for l_idx in range(L)])
    ax.semilogy(ell, np.maximum(max_eta, 1e-30), "o-", ms=4)
    ax.axhline(0.01, color="r", ls="--", alpha=0.5, label="0.01 threshold")
    above = np.where(max_eta > 0.01)[0]
    if len(above) > 0:
        l_nyq = above[0]
        ax.axvline(l_nyq, color="gray", ls=":", alpha=0.5, label=rf"$\ell$ = {l_nyq}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\max|\eta[\ell]|$")
    ax.set_title("LST sampling error (ii)")
    ax.legend(fontsize=8)

    # --- Panel 5: Sky asymmetry SNR ---
    ax = axes[4]
    snr = np.array(sky_snr)
    ax.semilogy(ell, np.maximum(snr, 1e-30), "o-", ms=4)
    ax.axhline(1.0, color="r", ls="--", alpha=0.5, label="SNR = 1")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("SNR")
    ax.set_title("Sky asymmetry SNR (C6)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
