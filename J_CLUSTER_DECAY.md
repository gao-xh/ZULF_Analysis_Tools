# Fixed-J cluster decay

`fit_j_cluster_decay` fits repeated FID group means in selected frequency ranges.
Supply a `fit_isopropylamine_j` carbon run, a group-average run, disjoint discovery
and validation indices, and optionally an N15 run containing `couplings_hz`.
The J sources remain conditional models, not accepted assignments.

`cluster_edges` maps `methine`, `methyl`, and optional `N15` to increasing positive
split frequencies. Empty edges mean one group; methyl defaults to `[190]`.
Edges partition every oscillatory transition, including tails outside observed
ranges. A transition on an edge belongs to its upper interval. Up to 32 nonempty
clusters are supported. Each cluster has one fitted real cosine/sine coefficient
pair and one effective decay time, with fixed normalized thermal weights inside
it. Group amplitudes are response parameters, not isotope abundances.

The initial use is a family-level baseline followed by explicit, physically
reviewed finer partitions. The tool does not select a physical component count
or establish whether neighboring transitions can be resolved. Rank, boundaries,
residuals and sensitivity require review before interpreting any decay value.

Default T2* bounds are 0.05–3 seconds; callers should specify appropriate bounds.
`preprocessing` uses the existing full-record SG baseline subtraction and crop
recipe, preserving acquisition-relative time. Discovery and frozen validation
predictions are complex; component magnitudes never add. Each observed band has
equal total squared-error weight, scaled by its discovery spectral norm.

`settings`: starts, max_nfev, max_evaluations, max_seconds, seed, optional
phase_delay_bounds_s. Jobs persist requests/status through the standard worker;
interrupted searches can be rerun, but internal optimizer-state resume is not yet
implemented. All-scan J estimation makes split validation conditional on J.

Artifacts include complete transition tables, explicit partition indices,
source matrices, per-cluster parameters, complex arrays, independent figures,
frozen validation errors, numerical warnings and source hashes. Full peak-cluster
stability assessment and interpretation are separate work, not implied by a
completed tool run.
