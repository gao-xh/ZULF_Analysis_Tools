# Exploratory isopropylamine J fitting

`build_isopropylamine_model` exports two eight-spin models in the existing
simulation tool format. `fit_isopropylamine_j` searches their parameters against
a saved experimental preprocessing run. MCP fitting returns a background job ID.
The original experiments and legacy simulation source are never modified.

## Scope and parameter mapping

This is a pure, natural-abundance isopropylamine **carbon-bound proton skeleton**
approximation, not a full nitrogen/amine/exchange model. Each isotopomer includes
seven carbon-bound protons and one 13C. Both methyl groups contain three equivalent
protons. NH2, 14N, 15N, multiple-13C species, residual magnetic fields and exchange
are omitted. Their absence must be evaluated against residuals, not assumed proven.

| Parameter (Hz) | Methine-13C isotopomer | Methyl-13C isotopomer |
| --- | --- | --- |
| J_CH_methine | C to directly attached methine H | Not used |
| J_CH_methyl | Not used | C to its three methyl H |
| J_HH_vicinal | Methine H to all six methyl H | Same |
| J_Cmethine_Hmethyl | C to all six methyl H | Not used |
| J_Cmethyl_Hmethine | Not used | C to methine H |
| J_Cmethyl_Hother_methyl | Not used | C to other methyl's three H |

Within-methyl H-H couplings are unobservable in this symmetric preparation and
detection model; they are stored as zero, not estimated. Inter-methyl H-H
coupling is fixed at zero as a simplifying assumption and may cause model error.
Default J values and parameter bounds are search settings, not literature values.
Positive direct C-H and vicinal H-H signs are assumptions; a successful fit does
not eliminate other sign-equivalent solutions.

## Fast simulation and observation model

The two methyl triples are represented by total spin 1/2 (multiplicity 2) and
3/2 (multiplicity 1). The four exact collective blocks have dimensions 16, 32,
32, 64 with multiplicities 4, 2, 2, 1. The weighted dimensions sum to the full
256-state Hilbert space. Cached operator matrices are combined with trial J
values; real Hermitian diagonalization produces transition frequencies and
thermal response weights. Tests compare their time response to the full
eight-spin Hamiltonian from the existing simulation module.

Rotational invariance at zero field permits gamma-weighted Iz preparation and
detection instead of Ix. The fit uses oscillatory transitions only, explicitly
excluding zero-frequency terms. Relative transition strengths assume this
unpulsed thermal response. Unknown pulse excitation can invalidate these relative
strengths; independent global amplitude/phase per isotopomer only partially
addresses that limitation. Frequencies of individual transitions cannot move
independently of the J Hamiltonian.

Each isotopomer has an effective exponential damping rate and a pair of real
cosine/sine coefficients. Their complex spectra are **added before magnitude**.
One complex constant per fitted band accounts for residual background. Linear
coefficients are solved by least squares at each nonlinear J/rate evaluation.
For `objective="complex"` (default) they are a linear least-squares solution.
For `objective="magnitude"` an inner nonlinear solve with analytic derivatives
adjusts those same coefficients to fit the magnitude of the summed complex
signal. This reduces dependence on observed phase, but discards information and
has magnitude-noise bias. It does not fix unknown within-isotopomer excitation
weights. The complex residual is still exported as a diagnostic for that mode.
The natural-abundance 1:2 methine:methyl molecule ratio is absorbed into unknown
response amplitudes: do not interpret fitted gains as isotope concentrations.

Finite sampled exponential sums yield exactly the selected native FFT bins.
The same SG filter as the experimental recipe is applied analytically, including
mirror-boundary corrections, followed by the same recorded-time crop. Mean
removal contributes zero to the strictly positive native bins used here. No
infinite-time Lorentzian approximation or reset of the time origin is used.
Transitions below 1e-7 Hz or numerical weights below 1e-13 of the largest weight
are discarded; degenerate lines within rounded 1e-7-Hz groups are merged by weight.

## Search and output

- Explicit bands are equally weighted by their experimental RMS amplitude;
  a weak second multiplet is therefore not overwhelmed by the first band.
- A seeded Latin-hypercube screen supplies separated starts. Bounded nonlinear
  least squares refines each. This is not an exhaustive global search.
  The explicit initial point is always refined, regardless of screening rank.
- `settings.initial`, `bounds`, and `free_parameters` allow a staged fit;
  fixed parameters are exported separately and are never labeled measured.
- `starts`, `screening_samples`, `max_nfev`, `seed` and `bin_stride` control cost.
  `max_nfev` is the optimizer's budget per start; finite-difference evaluations
  and screening mean the total evaluation count is higher.
- `rate_bounds` and `initial_rate` are in inverse seconds. Damping rates are
  nuisance parameters, not evidence for separate physical relaxation mechanisms.
- Background jobs support cancellation at each objective evaluation. Provisional
  best parameters and completed candidates are recorded while the job runs.

Each run saves candidate parameter sets, both eight-spin J matrices, fit arrays,
independent band overlays, real/imaginary residual plots and local sensitivity
diagnostics. Jacobian column cosines indicate parameter confounding; they are
**not** statistical posterior correlations. Candidate spreads across starts are
not confidence intervals. Boundary hits and optimizer nonconvergence are reported.

No confidence interval is calculated from these correlated processed FFT bins.
No automatic repeat-group validation or validation outside requested bands is
performed. Compare independent repeat averages and alternative preprocessing
recipes before treating any candidate J as an experimental determination. A
visually plausible fit alone does not establish uniqueness or a correct nitrogen
approximation.

See `examples/j_fit.json`. Replace its comparison run ID with a completed local
`compare_preprocessing` run. The first local trial uses SG window 301/order 2,
50 ms recorded-time start, and 110-150 / 230-275 Hz bands. This choice is explicit
and exploratory, not an assertion that it is the optimal preprocessing.
