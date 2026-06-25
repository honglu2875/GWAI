# Symbolic rationalization engine

This module is intentionally separate from the production Givental graph
evaluator.  It is the beginning of the future `formula --basis rational` path:
take raw graph expressions, substitute engine calibration data, contract
canonical-root sums, and simplify the result to graph-wise rational `q`-series.

Implemented now:

- `UniPoly`: one-variable polynomials over the existing `RatFun` coefficient
  ring.
- `OneGeneratorQuotient`: normal forms in `K[x]/(P(x))`, denominator inversion
  by Euclidean Bezout, trace sums, and residue-weighted root sums.
- `projective_relation(n)`: the ordinary equivariant `P^n` relation
  `prod_a(x-lambda_a)-q`.

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
