# RESOLVED — Projective-bundle target issues

> Historical note: this file originally described two open bugs in
> `src/givental/bundle.rs`.  Both are now fixed, but the note is kept for the
> deformation dictionary and for future regression triage.

## Summary

The projective-bundle path now handles:

- non-Fano positive-`z` I-functions, including `F_2 = P(O + O(2))`;
- higher-order bundle `R`-matrices, including the zero-twist
  `P(O + O) = P^1 x P^1` genus-one check that originally failed.

The end-to-end acceptance test is:

```bash
cargo test --lib f2_deformation_matches_p1xp1_pointwise -- --ignored
```

It is ignored only because it is slow in debug builds.

## Bug A: non-Fano mirror projection

### Symptom

For `F_2 = P(O + O(2))` over `P^1`, the class `(d1, d2) = (1, 0)` should match
the `P^1 x P^1` invariant in bidegree `(1, 1)`, hence:

```text
<pt, pt, pt>_{0,(1,0)} = 1.
```

The old bundle mirror path either returned a wrong rational number or, after
a guard was added, reported `UnsupportedInvariant`.

### Root Cause

The `(-2)`-section has anticanonical degree zero.  Its I-function coefficient
has positive powers of `z` and a `z^0` part that is not a scalar multiple of
the unit.  Therefore the mirror transform is not just:

1. read a divisor mirror map from the `z^{-1}` part;
2. exponential-gauge it away;
3. invert the divisor variables.

That divisor-only transform is insufficient once `I != 1 + O(1/z)`.

### Fix

The bundle path now ray-restricts the raw bidegree-graded I-function and builds
the fundamental solution from that cone point.  The existing matrix Birkhoff
factorization then supplies the actual Coates-Givental projection:

- the positive factor removes the positive-`z`/polynomial part and encodes the
  projection to the small-J calibration;
- the negative factor gives the descendant S-matrix used by the graph engine.

This keeps Birkhoff one-variable because the rank-two Novikov variables are
already specialized along a rational ray before factorization.

Pinned by:

- `f2_non_fano_positive_z_birkhoff_matches_product_genus_zero`;
- the ignored slow acceptance test `f2_deformation_matches_p1xp1_pointwise`.

## Bug B: higher-order bundle R-matrix

### Symptom

The zero-twist bundle `P(O + O)` over `P^1` is `P^1 x P^1`, but a genus-one,
four-point total-degree-2 computation disagreed with the product engine.
Earlier bundle tests were genus zero with three markings, so they only needed
`R` through first order.

### Root Cause

The bundle's fiber contribution to the classical diagonal R-asymptotic
constants used the Euler-factor sign.  With the convention
`xi|_(i,j) = -c_ij`, the relevant fiber difference for the R-asymptotics is:

```text
(-c_il) - (-c_ij) = c_ij - c_il
```

not `c_il - c_ij`.

This sign error is invisible in the validated low-order cases but affects
higher `R` orders.

### Fix

`BundleRayProvider::graph_kernel` now uses `c_ij - c_il` for the fiber
R-asymptotic differences.  A structural regression compares the zero-twist
bundle calibration against the product calibration with second-factor weights
`-mu_j`, matching the bundle convention for `xi`.

Pinned by:

- `zero_twist_bundle_r_matrix_matches_product_to_higher_order`;
- `zero_twists_have_product_s_matrix`;
- the slow genus-one cases inside `f2_deformation_matches_p1xp1_pointwise`.

## F2 <-> P1 x P1 Dictionary

`F_2` and `F_0 = P^1 x P^1` are deformation equivalent.  Write an `F_2` curve
class as `p f + q B_-`, where `f` is the fiber and `B_-` is the `(-2)`-section.
For bundle degrees `(d1, d2) = (H.beta, xi.beta)`:

```text
d1 = q
d2 = p - 2q
```

Under the deformation:

```text
f   -> f1
B_- -> f2 - f1
```

so:

```text
F_2 class (d1, d2)  <->  P^1 x P^1 bidegree (d2 + d1, d1)
```

The cohomology map is:

```text
H_F2  -> H2
xi_F2 -> H1 - H2
```

Equivalently:

```text
H1 = xi + H
H2 = H
```

The bundle shifted total degree for `F_2` is:

```text
d2 + 3 d1
```

while the product total degree after deformation is:

```text
(d2 + d1) + d1 = d2 + 2 d1
```

so the two engines must be called with their own total-degree conventions.
