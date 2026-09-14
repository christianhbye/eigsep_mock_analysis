# eigsim

Simulation code for the EIGSEP experiment.

## Simulation entry points

| Function | Output | Receiver term |
|---|---|---|
| `simulate()` | `(n_orientations, n_times, n_freqs)` system temperature | adds `receiver.temperature` from the config |
| `simulate_path()` | `(n_times, n_freqs)` antenna temperature, one orientation per time sample | none |

`simulate_path()` returns the plane eigsep_cal expects (`SkyTemperature.t_ant_k`): sky, horizon and ground, but no receiver, balun or coax. When correcting its output for ground loss, pass `t_rcvr=0.0`, because `correct_ground_loss()` otherwise subtracts a receiver temperature that was never added.

The receiver term in `simulate()` is a placeholder that eigsep_cal's `ReceiverModel` supersedes. It stays for now because the `horizon_position` and `horizon_chromaticity` pipelines save `t_sys` with it included. Removing it is a follow-up, planned for after the instrument paper is accepted.
