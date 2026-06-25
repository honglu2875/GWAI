# Symbolic rationalization engine

This module is intentionally separate from the production Givental graph
evaluator.  It provides the quotient-reduction layer used by
`formula --basis rational`: take raw graph expressions, substitute engine
calibration data, contract canonical-root sums, and simplify the result to
graph-wise rational expressions.  The formula command also has a separate
bounded-potential view, enabled by `--d`, that calls the production S/R/T graph
evaluator and formats the resulting q-truncated descendant potential, both as
the full stable-graph sum and as graph-local contributions in each graph
section.

Implemented now:

- `UniPoly`: one-variable polynomials over the existing `RatFun` coefficient
  ring.
- `OneGeneratorQuotient`: normal forms in `K[x]/(P(x))`, denominator inversion
  by Euclidean Bezout, trace sums, and residue-weighted root sums.
- `projective_relation(n)`: the ordinary equivariant `P^n` relation
  `prod_a(x-lambda_a)-q`.
- `projective_residue_polynomial` / `projective_trace_polynomial`: convenience
  contractions for ordinary `P^n` without materializing canonical roots.

The core identities are:

```text
P(u_i) = 0
Delta_i = P'(u_i)
sum_i f(u_i) / P'(u_i) = coefficient of x^(rank-1) in f(x) mod P(x)
sum_i f(u_i) = trace of multiplication by f in K[x]/(P)
```

This is deliberately narrower than a full computer algebra system.  It covers
the first and most important root-sum contractions for `P^n`-type theories.
General semisimple CohFTs should eventually use a matrix/trace reducer rather
than assuming a one-generator quotient.
