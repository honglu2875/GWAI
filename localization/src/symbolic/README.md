# Symbolic rationalization engine

This module is intentionally separate from the production Givental graph
evaluator.  It provides low-level quotient-reduction tools: substitute
engine calibration data, contract canonical-root sums where available, and
simplify expressions in quotient rings.  The previous public formula
`rational` basis has been removed until its mathematical meaning is made
precise; fixed-degree resolvent generating functions now live in the top-level
`resolvent` command, which uses the packed external-leg graph evaluator when
available.

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
