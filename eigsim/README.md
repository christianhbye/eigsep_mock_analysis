# eigsim

Simulation code for the EIGSEP experiment.

## Simulation entry points

| Function | Output | Receiver term |
|---|---|---|
| `simulate()` | `(n_orientations, n_times, n_freqs)` system temperature | adds `receiver.temperature` from the config |
| `simulate_path()` | `(n_times, n_freqs)` antenna temperature, one orientation per time sample | none |

`simulate_path()` returns the plane eigsep_cal expects (`SkyTemperature.t_ant_k`): sky, horizon and ground, but no receiver, balun or coax. When correcting its output for ground loss, pass `t_rcvr=0.0`, because `correct_ground_loss()` otherwise subtracts a receiver temperature that was never added.

The receiver term in `simulate()` is a placeholder that eigsep_cal's `ReceiverModel` supersedes. It stays for now because the `horizon_position` and `horizon_chromaticity` pipelines save `t_sys` with it included. Removing it is a follow-up, planned for after the instrument paper is accepted.

## Position-dependent horizons

`PositionHorizonGrid(axes_m, azimuth_rad, alpha_rad, method="cubic")`
interpolates precomputed horizon curves over a fixed, uniform rectilinear
East/North/Up grid. The caller states the spatial frame; each axis is in
metres, and its spacing may differ from the other axes. Horizon samples have
shape `(n_e, n_n, n_u, n_az)`. The fixed azimuths are true compass angles in
radians, sorted on `[0, 2*pi)`; elevations are also in radians.

```python
from eigsim import PositionHorizonGrid, open_sky_weight
import jax
import jax.numpy as jnp

grid = PositionHorizonGrid((east_m, north_m, up_m), azimuth_rad, alpha_rad)
p = jnp.array([east, north, up])
alpha = grid(p)
dalpha_dp = jax.jacfwd(grid)(p)  # (n_az, 3), radians per metre
W = open_sky_weight(alpha, grid.azimuth_rad, lmax)
```

`grid` supports `jit`, `vmap` and position derivatives. Cubic Catmull-Rom
interpolation has continuous first derivatives between spatial cells and
uses linearly extrapolated ghost samples at the boundaries. With
`method="linear"`, interpolation is trilinear and its derivative jumps at
cell boundaries. Both need at least two nodes per spatial axis. The grid
copies and validates its inputs at construction; it performs no file I/O.

The training box includes its boundaries. Outside it, concrete calls raise
`ValueError`; traced calls return NaNs. `grid.contains(p)` is a JAX validity
predicate. Cubic overshoots are returned without clipping, including values
outside the physical horizon range, so check the curve before using it.
A smooth interpolator does not establish agreement with retraced terrain,
and a C1 horizon does not make the full `open_sky_weight`/temperature chain
C1: the cell-weight calculation can still have kinks.
