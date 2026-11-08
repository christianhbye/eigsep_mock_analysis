# 063 model runtime and frequency chunks

The standalone project's default dependencies and `dev` extra support its
NumPy/SciPy operators and complete lightweight test directory. They do not
provision the real sky/SHT producer's runtime. Do not interpret a successful
default `uv sync` as a validated model environment.

Real model execution requires separately provisioned eigsim, croissant-sim,
s2fft, CPU JAX/jaxlib, Astropy, healpy and pygdsm. Each run manifest pins
the exact versions and executed operator modules; the external run preflight
also verifies the actual CPU binaries and all imported dependency sources.
The completed C53 producers use JAX/jaxlib0.9.2, NumPy2.4.4, SciPy1.17.1,
s2fft0.1.dev1047, croissant-sim5.2.1, Astropy7.2.0, healpy1.19.0 and
pygdsm1.7.0. The frozen source/module/binary evidence is authoritative.
Croissant's distribution metadata declares Git58bf56e8af77a5a07fd0669ec6ba38900549e50c;
the executed simulator SHA is
1fb70c4cf99ff7860cd2300585a74d89afd851f0ca78ef67ff9f87dc52df8b8f.
A read-only comparison with existing cached Git objects verifies all18
installed Python files exactly match d01b51db275c6100e649a7e0837d4b93d566f487;
only12/18 match the declared58bf metadata. No Git Python files are missing.
The per-file proof is `063_croissant_python_identity.json`. This establishes
the Python source identity; non-Python payload identity is not inferred from
that Git label. No runtime was installed, changed or rekeyed. Preserve both
identities and the original exact-byte runtime evidence for these products.

Use an independently owned copy of the verified host-compatible CPU runtime,
then install this source component with `--no-deps` in that copy. Never sync
or replace protected host CPU libraries with a default PyPI JAX wheel.
All cache/TMP/Jupyter/Matplotlib/Astropy state must be run-owned. The private
model CLI checks all required version/module pins before loading assets;
its failure sentinel does not replace external CPU/memory/time/product caps.
This is an audited snapshot workflow, not an environment resolver promise.

Fullnight models retain all362 original parents/37697 native rows and2044
children, including wind-excluded periods. The fitter applies the original
wind scope explicitly. All chunks use that identical native row set/global
frame, source d64291c, beam role/hash, calibration and pointing stage hashes,
original034 sky/IERS/horizon/A5 context, and runtime830b19da. A maximum of
three frequencies per setup bounds memory. The six round-robin channel sets
are `[283,603,843]`, `[284,604,844]`, `[285,605,845]`, `[291,611,851]`,
`[292,612,852]`, `[293,613,853]`. Reuse the first completed fullnight chunk;
the318-row cost pilot cannot join the fullnight set.

`python -m fine_night_d5.join --inputs JSON --out NEW_STAGE` verifies pinned
chunk completion/input files, payload hashes, actual historical producer
snapshots and runtime evidence. An explicit runtime read view is allowed
only with the identical original runtime checksum, for archived products.
It requires identical physical context and native/bin identities; channel
coverage must be disjoint and exactly match the declared ordered target.
It independently recomputes each chunk's actual-mask counts and bin means,
retaining empty cells asNaN. Unknown coverage, changed masks/coordinates,
runtime/beam differences or missing snapshots fail. No model is regenerated.
The joined product records its own assembly producer separately from every
original model producer/runtime proof; it never relabels old chunks as new
scientific computations. Existing eigsim/default beam products are unchanged.

No sweep is authorized by this file. Root gates actual costs and resources;
eventual batches may use at most four2CPU jobs only when all-cbye aggregate
CPU≤8 and host RAM/PSI/disk permit. Each job retains its own4GiB and product/
time bound and checkpoints. No automatic retry, cap relaxation or fullnight
fit/adoption follows a source or chunk-assembly test.
