# Legacy Direct Localization Backend

This folder contains the old direct stable-map localization code. It is kept
only for validation and convention checks.

It is not a production backend.  The ordinary-projective-space path covers a
narrow genus-zero primary degree-one slice and graph-enumeration diagnostics.
The independent projective-bundle module covers genus-zero primary fixed trees
through shifted total degree two, including genuine multiple covers.  It
constructs the toric one-skeleton, moving normal-bundle factors, stable and
unstable vertex terms, cover factors, and automorphisms directly.  It calls
none of the projective-bundle I-function, Birkhoff factorization, Novikov-ray
interpolation, descendant `S`/`R`, or Givental graph evaluator.

The bundle regression set includes:

- fiber, negative-section, and mixed rows `1`, `-27`, and `19` for
  `P(O + O(3) + O(3)) -> P^2`;
- a degree-two fiber conic count `1`, which is the multiple-cover check that a
  degree-one localization slice cannot provide;
- the `F_1` exceptional value `-1` and a `P^1 x P^1` mixed value `1`; and
- repetition at disjoint generic rational torus weights, so cancellation to
  an ordinary invariant is checked rather than assumed.

Neither localization module implements arbitrary genus, descendants, or
Hodge factors.  The projective-bundle module also refuses shifted degree above
two and caps the number of labelled fixed-tree candidates before
materialization.  These are independent foothold oracles, not evidence for
the production backend outside their stated slices; unsupported and
resource-limited requests fail closed.

The public computation path for ordinary `P^n` invariants is the Givental
`S/R` graph evaluator.
