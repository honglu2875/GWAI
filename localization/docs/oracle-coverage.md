# Independent Oracle Coverage

The machine-readable inventory is
[`oracle-provenance.tsv`](oracle-provenance.tsv).  Its
`production_exercised` column is the important distinction: storing a trusted
number or testing an oracle implementation is not the same as comparing that
number with a production backend.  `default_ci` separately records whether the
comparison runs without `--ignored`.

Every row also separates its executable identity from its source evidence.
`cargo_target` uses the same target-qualified syntax as the acceptance registry
(`lib`, `test:<name>`, `bin:<name>`, and so on), while `cargo_test` is the exact
name printed by that harness's `--list`.  `source_locator` is a package-relative
file and, for Rust sources, a concrete `path.rs::function` marker.  Some older
oracles are members of the public built-in suite rather than standalone Rust
tests.  Those rows therefore name the real aggregate executable
`testsuite::tests::builtin_suite_passes` in `cargo_test` and name the relevant
helper only in `source_locator`; the inventory does not pretend that the helper
can be selected independently with Cargo.  The audit additionally proves that
each such helper occurs in `run_builtin_tests` and that the aggregate Rust test
invokes that runner, so a leftover helper definition alone cannot satisfy the
row.

The strongest default checks are currently:

- ordinary projective space against direct fixed-locus localization, an
  independent implementation of Zinger's genus-zero formula, and recorded
  Growi 1.0.3 output at positive genus;
- negative split targets against the resolved-conifold multiple-cover formula;
- projective bundles against the validation-only toric fixed-tree backend for
  the `F_1` exceptional section;
- point theory against the closed one-point intersection formula.

The published local-`P^2` grid remains a strong external golden oracle, but its
roughly two-minute debug runtime places it in scheduled acceptance CI rather
than the default push gate.

The bundle fixed-tree backend does not call the projective-bundle I-function,
Birkhoff factorization, Novikov-ray interpolation, `S`/`R` construction, or
stable-graph contraction.  A rank-three fiber-conic comparison is retained as
an ignored acceptance test because it exercises degree-two edge covers but
takes roughly a minute in a debug build.  The fast `F_1` comparison runs in the
default suite.  Both engines intentionally consume the same canonical target
theory (twists, curve coordinates, basis conventions): independence is at the
localization-versus-reconstruction algorithm layer, not a duplicated source of
space data.

Several apparently strong tests share production machinery:

- plane rational-curve seed tests call the same seed implementation used as a
  production fallback;
- zero-twist bundle/product and `F_2` deformation comparisons use different
  target calibrations but the same universal graph contraction and point
  theory;
- alternate ray nodes, coefficient representations, cache modes, unitarity,
  and deeper Laurent windows are valuable internal-consistency tests, not
  independent numerical oracles;
- Virasoro residual tests currently consume the same theory/provider data and
  production correlators that they audit.  They are broad identity checks, not
  external invariant data.

Known coverage gaps are explicit rather than inferred.  Product projective
space has no default production comparison with a direct localization backend
or sourced external table.  Five heavier Growi rows are inventoried but are not
run against production.  Positive-genus/descendant bundle coverage is presently
the ignored `F_2` deformation comparison and therefore still shares the
universal graph engine.  Arbitrary negative twists similarly have excellent
structural coverage, but independently sourced numerical data only on local
`P^2` and the resolved conifold.  The three recorded `O(-1) -> P^2` constants
remain useful default-CI regression fixtures, but their original localization
derivation was not archived; the provenance inventory therefore marks them as
unsourced rather than counting them as an external oracle.

The TSV deliberately uses small closed vocabularies for `target_family`,
`evidence`, and `shared_code`.  The integration test
`oracle_provenance_manifest_is_well_formed_and_gaps_are_explicit` checks those
static semantics without launching a nested Cargo process.  The acceptance
registry audit performs the executable checks: it discovers every Cargo test
harness, verifies each target/test pair and its ignored status, then confirms
that every source file and Rust member marker exists.  Together they reject
unknown labels, stale executable or source identities, duplicate IDs, false
claims of algorithmic independence, loss of the default independent footholds,
or removal of the explicit product-space gap without replacing it by
independent coverage.
