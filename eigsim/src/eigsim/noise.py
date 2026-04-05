"""Radiometer noise model for EIGSEP simulations."""

import numpy as np


def radiometer_noise(t_sys, delta_freq_hz, delta_time_s, rng=None):
    """Generate radiometer noise.

    The noise standard deviation per sample is given by the radiometer
    equation:

        sigma = T_sys / sqrt(delta_freq * delta_time)

    Parameters
    ----------
    t_sys : array_like
        System temperature in Kelvin.  This sets the shape of the
        returned noise array.
    delta_freq_hz : float
        Channel bandwidth in Hz.
    delta_time_s : float
        Integration time in seconds.
    rng : numpy.random.Generator or None
        Random number generator.  If None, a new default generator is
        created.

    Returns
    -------
    noise : numpy.ndarray
        Gaussian noise with shape matching *t_sys*.

    """
    if rng is None:
        rng = np.random.default_rng()
    t_sys = np.asarray(t_sys)
    sigma = t_sys / np.sqrt(delta_freq_hz * delta_time_s)
    return rng.normal(0.0, sigma)
