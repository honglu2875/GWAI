# Legacy Direct Localization Backend

This folder contains the old direct stable-map localization code. It is kept
only for validation and convention checks.

It is not an actively maintained production backend. In particular, it covers a
narrow genus-zero primary degree-one slice and graph-enumeration diagnostics; it
does not implement the full fixed-locus formula for arbitrary genus,
descendants, Hodge factors, and unstable vertices.

The public computation path for ordinary `P^n` invariants is the Givental
`S/R` graph evaluator.
