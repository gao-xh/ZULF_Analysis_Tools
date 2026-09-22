# Natural-abundance 15N hypothesis screening

`inspect_n15_hypotheses` is available through JSON CLI, background jobs and MCP.
It tests explicit coherent Hamiltonians against a frozen carbon J fit. It does
not automatically identify 15N, estimate a nitrogen abundance, or measure an
exchange rate. All generated experimental reports stay under `.analysis`.

## Literature and parameter provenance

- [Pecul and Helgaker, IJMS 4, 143-157 (2003), Tables 1 and 3](https://doi.org/10.3390/i4030143)
  calculates methylamine couplings and compares experiment. Its methylamine
  NH coupling is approximately -65 Hz and geminal N-C-H coupling approximately
  -1 Hz experimentally. These are analog guidance, **not isopropylamine values**.
- [Sodium Isopropyl(trimethylsilyl)amide (NaPTA), experimental section](https://pmc.ncbi.nlm.nih.gov/articles/PMC10042304/)
  reports a 15N-isopropylamine preparation and CDCl3 NMR: methyl H-N splitting
  magnitude 2.7 Hz, carbon-N splittings 3.9 and 2.4 Hz. It supplies no complete
  neat-liquid N-H coupling matrix or signed assignment. The reported boiling
  point and NH integral in that paragraph warrant caution; do not silently
  treat its entire characterization as independently verified. The present
  screen checks both signs of the reported remote H-N magnitude.
- [CIAAW nitrogen](https://ciaaw.org/nitrogen.htm) and
  [natural variations](https://ciaaw.org/natural-variations.htm?el=nitrogen)
  support a representative N15 fraction near 0.003663, not a measured fraction
  for this sample. The C13 fraction 0.0107 is an explicit representative scenario.
- [Barskiy et al., Nature Communications (2019)](https://www.nature.com/articles/s41467-019-10787-9)
  motivates distinguishing coherent coupling from chemical exchange. Omitting
  NH2 here is a decoupled limiting hypothesis, not a finite-rate exchange model.

In the initial screen, J_NH = -55, -65, -75, -85 Hz deliberately spans beyond
the methylamine analog; J_N_methine = -1 or -3 Hz, J_N_methyl = +/-2.7 Hz,
J_HH_vicinal = 6.2 Hz and J_NH_Hmethine = 3 Hz are explicit trial settings.
The last value has no direct isopropylamine measurement in the consulted sources.
Results depending on these choices require sensitivity checks, not reinterpretation
of the settings as literature-derived experimental facts.

## Model and absolute intensity

The slow-exchange coherent hypothesis has ten spin-1/2 nuclei: six equivalent
methyl H, one CH hydrogen, two equivalent NH hydrogen and one N15. All carbon
nuclei are C12. Collective sectors retain the full 1024-state multiplicity;
the largest block is 84-dimensional. NH-to-methyl H-H couplings, fields and
exchange are omitted. The NH2-decoupled limit contains eight active spins.

The magnetization operator uses signed gyromagnetic ratios. Transition weights
are `2 * multiplicity * |M_ij|^2 / Hilbert_dimension`, with no per-isotopomer peak
normalization. This corresponds to unpulsed, gamma-weighted thermal preparation
and detection. For a two-spin AX illustration the oscillatory response scales
with `(gamma_H - gamma_X)^2`; its N15/C13 factor is about 2.17. The full molecule
is computed explicitly rather than multiplied by that two-spin factor.

Probabilities are disjoint: N15 with all three C12 has probability
`pN*(1-pC)^3`; one methyl C13 with N14 and the other two C12 has combined
probability `2*pC*(1-pC)^2*(1-pN)`. Double labels are omitted explicitly.
Absolute theoretical weights and isotope probabilities are multiplied into the
processed spectra. The existing template's internal unit-weight normalization
is undone before that multiplication.

The first prediction assumes the same response calibration as the fitted methyl
carbon component. This is conditional on transfer, pulse, relaxation and detector
response. A second calculation fits just one N15 cosine/sine coefficient pair,
with carbon J, rates, phases, amplitudes and background frozen. Reported response
factor compares this gain with the carbon-calibrated natural-population scenario;
it is **not** an abundance estimate. Spectra add as complex signals before magnitude.

## Interpretation and validation

An improvement on the observations used to screen couplings is not independent
evidence. More candidate spectra and two extra gain parameters can absorb carbon
model mismatch. Save the complete candidate grid, convergence status, required
response factor, out-of-band predictions and exact unbroadened positions.
Compare the latter with experiment before accepting an assignment. Free amplitudes
are not a substitute for evidence about preparation or exchange.

Tests compare collective absolute response with an independent full 128-state
calculation including signed N15 gamma; the isolated NH2-N15 limit produces a
single oscillatory frequency of 1.5*|J_NH|. The ten-spin sector dimensions are
checked against 1024. These checks validate calculation, not the assumed sample
couplings or physical origin of a residual peak.
