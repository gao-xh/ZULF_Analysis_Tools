# Spin simulation tools

The new backend is based on the sequences in
`ZULF_NMR_Suite/src/core/TwoD_simulation.py` (identical to the copy in
`MUI_backup_continue`). Legacy files remain unchanged. This is a bounded dense
spin-1/2 implementation, not a replacement for all Spinach capabilities.

## Workflow

1. `inspect_simulation_backend`: runs real complex128 CUDA operations, returning
   availability, device and any dependency error.
2. `import_spin_model(folder)`: reads existing `structure.csv` / `symmetry.csv`,
   records file hashes, converts one-based CSV spin indices to zero-based JSON.
   Configure approved read-only directories in `local.json` under `model_roots`.
3. `analyze_spin_symmetry(model)`: validates equivalent-spin groups, reports
   conserved collective-spin sectors, dimensions and commutator residuals.
4. `simulate_spin_dynamics(model, settings)`: background job for `fid`, `2d` or
   `mq`; use `get_job` then `get_result`. Saves numerical arrays and independent
  FID, 1D spectrum and (where applicable) 2D spectrum figures.
  The 2D output also has an independent logarithmic-color figure (display floor
  = maximum magnitude x 1e-6); numerical arrays retain DC and are not clipped.

Example: `examples/simulation.json` is a synthetic A2X model, not an assignment
of an experimental sample. Use `backend: "gpu"` to request CUDA explicitly;
failure is reported rather than silently falling back to CPU. CPU is the default.

## Physics and units

- J couplings: Hz; all time parameters: seconds; fields: microtesla.
- `pulse_field_ut` is `[Bx, By, Bz]`; `field1_ut` and `field2_ut` are static z
  fields during indirect and direct evolution, respectively.
- Hamiltonian: `2*pi*(sum J_ij I_i.I_j + sum gamma_i B.I_i)` in rad/s.
  The positive Zeeman sign follows the legacy code; do not compare signed phases
  to another convention without reconciling it.
- Gamma values are gamma/(2*pi) in Hz/uT. Defaults retain the legacy rounded
  values 42.58 (1H), 10.71 (13C). 15N is -4.316 (approximate), replacing the
  legacy erroneous 60.86. Supply `gamma_hz_per_ut` to override by isotope.
- Each supported nucleus has spin +1/2. The sign of gamma is not negative spin.
  14N, spin 1, is explicitly rejected; an unlabeled nitrogen-containing molecule
  must not be represented by replacing 14N with 15N.
- Initial deviation density matrix and detection: `sum 2*pi*gamma_i I_ix`.
  They are unnormalized, so absolute amplitudes are arbitrary units.
- The finite rectangular pulse includes J coupling and field1 during the pulse.
  It acts on all nuclei, not automatically as an ideal isotope-selective pi pulse.
- `fid`: pulse, then evolve under field2 and detect.
- `2d`: pulse, indirect evolution under field1, pulse, direct evolution under field2.
- `mq`: pulse, mixing delay `tm_s` under field1, pulse, indirect evolution,
  reconversion pulse, direct evolution. Matches legacy `fid2d_MQ`, not its
  inconsistent frequency-domain generator initialization.
- `exp(-t2/t2star_s)` is a phenomenological decay on direct time only. No t1
  relaxation, exchange, quadrupolar interactions or distribution of decay times.
- No apodization, first-point halving, DC subtraction or zero filling is implicit.
  Complex FFTs have signed frequency axes and are normalized by acquired counts.

## Numerical optimization and symmetry

Diagonalize each free-evolution Hamiltonian once per sector. Density matrix
elements acquire phases `exp(-i*(E_a-E_b)*t)` in that basis. The detection signal
is the sum of those phases weighted by `rho_ab * O_ba`. Direct-time evaluation
uses bounded phase chunks and matrix multiplication, replacing a full matrix
propagation at every sample. Indirect-time pulse transfer is still evaluated
once per t1 point. No intensity threshold removes transitions or pathways.

Omitting `symmetry_groups` detects exchange-equivalent spins from isotope and
coupling matrices. Explicit groups are checked, never trusted from labels alone.
Comparison tolerance is absolute 1e-10 Hz. `[]` disables group detection;
`settings.symmetry=false` runs the full Hilbert space for comparison.

For each valid group, diagonalize collective `S_g^2` and partition by total
group spin. Verify that both Hamiltonians, pulse Hamiltonian, preparation and
detection preserve these sectors. Keep all sectors and all multiplicity states;
do not keep only the maximum-spin sector. This is a partial symmetry reduction,
not a full irreducible permutation-group or geometric point-group analysis.
The reported matrix-cube ratio is a rough algebraic indicator, not a speed claim.
See [Spinach's symmetry requirements](https://spindynamics.org/wiki/index.php?title=Basis_set_specification).

## GPU installation and limitations

On the tested Windows machine (RTX 3060 Laptop, 6 GB, driver 566.36):

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-gpu-windows.txt
```

The pinned CUDA 12.6 wheel libraries are local to the virtual environment.
The loader adds their DLL directories in the worker process, without changing
the system driver or persistent environment variables. A CPU-only install needs
none of these optional dependencies. Other platforms should follow the
[CuPy installation guide](https://docs.cupy.dev/en/v13.5.0/install.html).

CPU constructs operators and symmetry bases; GPU performs sector eigensolvers,
pulse transforms and transition sums in complex128. Small sectors can be slower
on GPU because launch/transfer overhead dominates. GPU memory does not remove
the exponential Hilbert-space cost. Initial limits: 8 spins, 4 million output
samples, 256 MB transition-weight array. No sparse or distributed solver yet.

Local six-spin 2-13C-ethanol validation (8 x 128 time samples) matched the
unchanged legacy `fid2d` with maximum error / maximum reference magnitude below
2e-14. Single-run timings including setup: legacy 0.162 s, optimized CPU full
0.053 s, optimized CPU sectors 0.031 s, GPU sectors 1.352 s (first use).
These are a small smoke benchmark, not general speed guarantees. CPU uses
SciPy's MRRR Hermitian eigensolver after the NumPy divide-and-conquer solver
failed to converge on a finite degenerate pulse block in the local LAPACK build.

Tests compare all three sequences with independent scipy.expm propagation,
test an analytic single-spin signal, reject invalid symmetry, check artifacts
and (when CUDA is available) compare CPU/GPU results. Experimental assignments
and relaxation mechanisms still require separate validation.

## Legacy audit

- The old `basis.symmetry_*` fields were not used by the simulation engine.
- Full matrix operators were reconstructed repeatedly by nested objects.
- `fid`/`fid2d` repeatedly propagated density matrices at every direct-time sample.
- The field comment said mT but numeric gamma values correspond to Hz/uT.
- The old 15N gamma and negative spin quantum number are invalid.
- Frequency-domain 2D generators use absolute gaps, have inconsistent pulse
  preparation relative to the time-domain routines, and truncate at max_count.
  The new tool uses time-domain sequence definitions and signed Fourier axes.
