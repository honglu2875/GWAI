# Legacy Direct Localization Backend

This folder contains the old direct stable-map localization code. It is kept
only for validation and convention checks.

It is not a production backend.  The ordinary-projective-space path covers a
narrow genus-zero primary degree-one slice and graph-enumeration diagnostics.
The independent projective-bundle module covers genus-zero primary fixed trees
through shifted total degree two, including genuine multiple covers.  Neither
implements arbitrary genus, descendants, or Hodge factors; unsupported
requests fail closed.

The public computation path for ordinary `P^n` invariants is the Givental
`S/R` graph evaluator.
