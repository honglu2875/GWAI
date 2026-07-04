# TEMPORARY — Open issues in the projective-bundle target

> **Status:** temporary working document / issue writeup, intended to be handed
> to other experts.  Delete once both issues are resolved.
> **Scope:** two independent bugs in `src/givental/bundle.rs` and its
> dependencies.  The rest of the crate (projective space, products, twisted,
> the graph engine) is unaffected and validated.

This document is self-contained.  It assumes familiarity with
Givental–Teleman reconstruction of semisimple CohFTs (an `R`-matrix acting on
a TFT, summed over stable graphs, with a descendant calibration `S`) and with
toric/`I`-function mirror symmetry, but **not** with this codebase.  A one-page
architecture map is in [`docs/architecture.md`](docs/architecture.md); the
design lessons behind the choices are in [`docs/lessons.md`](docs/lessons.md).

The in-code acceptance test for a fix is the `#[ignore]`d
`f2_deformation_matches_p1xp1_pointwise` in `src/givental/bundle.rs` (run with
`cargo test -- --ignored`).  Its dictionary is derived in Appendix B.

---

## 0. TL;DR

The engine computes exact (rational) Gromov–Witten invariants by
Givental–Teleman reconstruction.  Targets are plugged in behind a small
interface; `P^n`, `P^n × P^m`, and projective bundles `P(⊕ O(a_l)) → P^n` are
implemented.  The bundle target has **two open bugs**:

1. **Non-Fano mirror map (Bug A).**  For non-Fano bundles (max twist `A ≥ 2`,
   e.g. the Hirzebruch surface `F_2 = P(O ⊕ O(2))`), the `I`-function has
   positive powers of `z` and a non-unit `z^0` part, so the mirror
   transformation is *not* a divisor change of variables.  The current
   transform is wrong there.  **Currently guarded**: the code detects this and
   returns `UnsupportedInvariant` instead of a wrong number.  A full fix needs
   the Birkhoff/Coates–Givental projection of a positive-`z` `I`-function onto
   the `J`-slice.

2. **`R`-matrix beyond first order (Bug B).**  Every validated bundle test
   only exercises the `R`-matrix to **first order** (`z^1`).  A genus-1,
   four-point invariant of `P^1 × P^1`, computed through the bundle path as
   `P(O ⊕ O)`, disagrees with the (independently validated) product engine.
   So the bundle's `R`-matrix beyond first order is wrong or imprecise.
   **Not root-caused, not guarded**: higher-genus / many-marking bundle output
   should be treated as unverified.

Bug A blocks the mathematically most interesting check (`F_2` is deformation
equivalent to `P^1 × P^1` with a *nontrivial* mirror map, so it is the ideal
cross-validation).  Bug B blocks all higher-genus bundle invariants.  They are
independent.

---

## 1. Engine background (self-contained)

The core (`src/givental/graph.rs`) is a **semisimple-CohFT evaluator**.  It
consumes a *calibration* and a *descendant `S`-matrix* and never inspects
geometry:

- **`SemisimpleCalibration`** (`src/givental/provider.rs`): the flat Poincaré
  metric `η`, the flat↔canonical frame change `Ψ` and `Ψ^{-1}`, the Dubrovin
  connection `Ψ^{-1} q d(Ψ)/dq`, the canonical metric norms `Δ`, and the
  **`R`-matrix** `R(z) = 1 + R_1 z + R_2 z^2 + …` (an upper-triangular
  symplectic loop-group element in the canonical frame).
- **`SeriesSMatrix`** (the descendant `S`): the descendant→ancestor
  dictionary.  The engine consumes the **metric adjoint** `S* = η^{-1} Sᵀ η`.
- The graph sum then reconstructs all-genus invariants.  It is validated to
  genus 4 on `P^n` and via the Behrend product formula on `P^1 × P^1`, so it
  is **not** implicated in either bug.

Coefficients are exact.  Everything runs at *rational* equivariant weights per
target, so all series coefficients are `Rational` and every invariant is a
plain rational number (extracted by `RatFun::as_rational`).

**Truncation.**  For genus `g` with `n` markings the largest vertex `psi`
degree is `dim M̄_{g,n} = 3g − 3 + n`, which bounds the needed `R`-order:
`r_order = graph_dimension + 1 = (3g + n − 3) + 1`.  So `r_order` grows with
genus and markings.  **Genus 0 with 3 markings gives `graph_dimension = 0` and
`r_order = 1`** — this is the regime all bundle tests happen to live in (see
Bug B).

### 1.1 Recipes (how a calibration is built)

A *recipe* (`src/givental/recipe.rs`) manufactures the calibration from more
primitive data.  Two exist:

- **Quantum-ring / QDE recipe.**  From a divisor multiplication operator:
  Newton root series (`newton_root_series`, recipe.rs:177) →
  `divisor_lagrange_frame` (recipe.rs:202) → `calibration_from_canonical_frame`
  (recipe.rs:50, which runs the `R`-flatness recursion in `r_solve.rs`).  `S`
  comes from `descendant_s_from_divisor_qde` (integrate the quantum
  differential equation).  Used by `P^n` and `P^n × P^m`.
- **`I`-function recipe.**  `descendant_s_from_i_function` (recipe.rs:257):
  mirror map from the `H/z` part → exponential gauge to `J` → Birkhoff
  factorization → metric adjoint.  It delegates to
  `descendant_s_from_j_function` (recipe.rs:296) once `J` is known.  Used by
  twisted theories and by the bundle.

The two recipes cross-validate on `P^1` (a rank-zero twist is untwisted `P^1`;
test `i_function_and_qde_recipes_agree_on_projective_space`, recipe.rs).

### 1.2 Rank-2 targets by Novikov ray reconstruction

The series layer (`QSeries`) carries **one** Novikov variable.  Picard-rank-2
targets (`P^n × P^m`, bundles) are handled by *ray specialization*:
`(q_1, q_2) = (t, b·t)` for rational `b` is a Novikov ring homomorphism, so
each ray runs on the unchanged one-variable engine.  A degree-`k` ray
coefficient is `Σ_{d_1 + d_2 = k} b^{d_2} N_{(d_1, d_2)}`; running `k+1`
distinct rays and solving the Vandermonde system over `ℚ` recovers every
bidegree exactly.  See `src/givental/product.rs` (the working, validated
example) and `reconstruct_bundle_invariants` (bundle.rs:988).

---

## 2. The projective-bundle pipeline (`src/givental/bundle.rs`)

**Geometry.**  `X = P(E)`, `E = O(a_1) ⊕ … ⊕ O(a_m)` over `P^n`.  Twists are
normalized so `min a_l = 0` (`P(E) ≅ P(E ⊗ L)`).  `A := max a_l`.
`H*(X) = H*(P^n)[ξ] / ∏_l (ξ + a_l H)`, `dim X = n + m − 1`, Picard rank 2.
Torus fixed points are pairs `(i, j)` (base point `i`, fiber line `j`); fiber
weight `c_{il} = a_l λ_i + μ_l`; `ξ|_{(i,j)} = −c_{ij}`.  Tangent weights at
`(i, j)`: `{λ_k − λ_i : k ≠ i} ∪ {c_{il} − c_{ij} : l ≠ j}`.

**Curve classes.**  `d_1 = H·β`, `d_2 = ξ·β` (which may be negative — the
`(−A)`-section has `d_2 = −A`).  A term of the `I`-function vanishes unless
`d_2 ≥ −A d_1` (a term with every `D_l = d_2 + a_l d_1 < 0` contains the full
ring relation), so the **shifted fiber degree** `d_2' = d_2 + A d_1 ≥ 0` gives
a nonnegative grading covering the effective cone.  The **shifted total
degree** used by the ray reconstruction is `d_1 + d_2' = d_2 + (A+1) d_1`
(this is the `--d` CLI argument; note it is **not** `d_2 + 2 d_1`).

**Cyclic generator.**  `D := ξ + (A+1) H` has pairwise-distinct classical
eigenvalues at generic weights, so the classical ring is cyclic over `D` and
everything runs in the **constant classical `D`-power basis** `1, D, …,
D^{size−1}` (`size = (n+1)m`).  This is the key trick that reduces the
two-generator ring to the existing single-generator machinery.  (Do **not**
use *quantum* powers of `D` as a basis — they are `t`-dependent; see
lessons.md §1.)

**Pipeline, per ray** (`BundleRayProvider`, bundle.rs:832):

1. Build the `I`-function coefficients in **bidegree-graded** form
   (`i_coefficient`, bundle.rs:309; fixed-point restriction
   `i_restriction`, bundle.rs:281), finite per shifted total degree.
2. Mirror-transform in bidegree-graded form to get `J` (`j_container`,
   bundle.rs:341).  **This is where Bug A lives.**
3. Restrict `J` to the ray and Birkhoff-factor to `S`
   (`descendant_s_rational`, bundle.rs:776, via
   `recipe::descendant_s_from_j_function`).
4. Recover quantum multiplication `A_q = A_cl + t d/dt S_1` (bundle.rs:889).
5. Build the canonical frame from the **spectral projectors** of `A_q`
   (`recipe::operator_lagrange_frame`, recipe.rs:465) and the `R`-matrix from
   flatness (`calibration_from_canonical_frame`).  **This is where Bug B
   lives.**
6. Run the graph engine; reconstruct bidegrees from `total+1` rays.

---

## 3. Bug A — non-Fano mirror map (positive-`z` `I`-function)

### 3.1 Symptom

`F_2 = P(O ⊕ O(2))` over `P^1` (twists `[0, 2]`, `A = 2`) returns garbage.
Concretely, the genus-0, three-point invariant in the class `B_+ = (d_1, d_2)
= (1, 0)` should be `1` (it equals the `P^1 × P^1` invariant
`⟨pt, pt, pt⟩_{0, (1,1)} = 1` under deformation), but the raw pipeline returned
`386177852873 / 261003600`.

### 3.2 Root cause

`F_2` is **non-Fano**: its effective `(−2)`-section `B_-` has anticanonical
degree `c_1·B_- = 0`.  (Fano surfaces `F_0`, `F_1` have `c_1·β > 0` for every
effective class.)  For a class with `c_1·β = 0`, the `I`-function term
`I_β(z)` starts at `z^0` and — crucially — carries a **positive power of `z`**
and a `z^0` component that is **not a multiple of the unit**.

This was confirmed directly.  In the classical `D`-power basis (index 0 =
unit `1`, index 1 = `D`, …) the raw `I`-coefficient at the `B_-` grade
`(d_1, d_2') = (1, 0)` (i.e. `d_2 = −2`) is:

```
grade (1,0):
  z^{-2}: [-7372/65, -388/65, 0, 0]
  z^{-1}: [ 323/325,   17/325, 0, 0]
  z^{ 0}: [1919/325,  101/325, 0, 0]   <- non-unit z^0 part (index 1 nonzero)
  z^{+1}: [  76/325,    4/325, 0, 0]   <- positive power of z
```

(For comparison, the Fano fiber grade `(0,1)` is `z^{-2}: [1,0,0,0]`, purely
`1 + O(1/z)`, no positive `z`.)

The presence of `z^{+1}` means `I ≠ 1 + O(1/z)`, so bringing `I` to `J`-form is
a genuine **Birkhoff / Coates–Givental projection onto the Lagrangian cone's
`J`-slice** (an upper-triangular loop-group operation), *not* the
lower-triangular divisor change of variables the current code implements.

The current `j_container` transform (bundle.rs:341) does only: (a) a
multiplicative gauge stripping the `z^{-1}` part, and (b) a divisor
(`H`, `ξ'`) change of variables read from the `z^{-1}` divisor components.  It
has no step for the `z^{≥0}` / positive-`z` part.

**Where it manifests in code.**  The mirror-map read-off
(`decompose_in_degree_two_span`, bundle.rs:702) decomposes the `z^{-1}`
exponent in the span `{1, H, ξ'}`.  For non-Fano targets the **unit
component** `g_unit` of that decomposition is nonzero (for `F_2`,
`g_unit ∈ {11, 33/2, 110/3, …}` at the `B_-` grades — note these are
weight-dependent, `= μ_1 = 11` at leading order).  For Fano targets `g_unit`
is identically zero.

### 3.3 Current guard

bundle.rs:443 — inside the mirror read-off loop:

```rust
if !g_unit.is_zero() {
    return Err(GwError::UnsupportedInvariant(
        "non-Fano projective bundle: ... (e.g. F_2 = P(O + O(2))) is not yet supported"
    ));
}
```

So the code now returns a clean error instead of a wrong number.  Test
`f2_non_fano_is_currently_unsupported` (bundle.rs) pins this.

### 3.4 What a fix requires

Implement the full mirror transform for a positive-`z` `I`-function.  The
standard framework is Coates–Givental: `I(Q, z)` lies on the Lagrangian cone
`L`, and `J(τ(Q), z)` is the unique point on the small-`J` slice above the same
ruling.  Concretely one must handle the `z^{≥0}` polynomial part of `I` — a
Birkhoff factorization producing an *upper*-triangular element that, together
with the mirror map `τ(Q) ∈ H` (which for non-Fano can have components beyond
the divisor classes, including the unit), maps `I` to `J`.  References worth
consulting: Coates–Givental "Quantum Riemann–Roch, Lefschetz and Serre";
Brown "Gromov–Witten invariants of toric fibrations" (the `I`-function of
`P(E)` used here); Iritani on the integral structure / mirror map for toric.
Note the crate already has a single-variable Birkhoff factorization
(`src/twisted/birkhoff_factor.rs`) used for the `J → S` step; the new piece is
the `I → J` projection when `I` has positive-`z` terms.

A correct fix should make the `F_2` deformation cross-check (Appendix B / the
`#[ignore]`d `f2_deformation_matches_p1xp1_pointwise`) pass for genus 0 (and,
once Bug B is fixed, all genus).

### 3.5 Reproduction

```rust
// Currently returns Err(UnsupportedInvariant) (the guard). To SEE the wrong
// number, comment out the `if !g_unit.is_zero()` guard at bundle.rs:443.
use gw_pn::givental::{reconstruct_bundle_invariants, BundleInsertion};
use gw_pn::algebra::Rational;

let point = BundleInsertion::new(0, 1, 1); // H*xi = the F_2 point class
let out = reconstruct_bundle_invariants(
    /* n */ 1, /* twists */ &[0, 2],
    /* weights_base */ &[Rational::from(2), Rational::from(5)],
    /* weights_fiber */ &[Rational::from(11), Rational::from(23)],
    /* genus */ 0, /* shifted total degree */ 3,
    &[point.clone(), point.clone(), point],
);
// Expected once fixed: the (1,0) entry (= B_+ class) equals 1.
// (d1, d2, value) triples; find (a,b) == (1, 0).
```

CLI (returns the unsupported error today):

```bash
cargo run --quiet -- bundle --n 1 --twists 0,2 --g 0 --d 3 \
  --insert 'H*xi' --insert 'H*xi' --insert 'H*xi'
```

---

## 4. Bug B — `R`-matrix beyond first order

### 4.1 Symptom

`P(O ⊕ O)` over `P^1` (twists `[0, 0]`, `A = 0`) **is** `P^1 × P^1` with a
*trivial* mirror map (`I = J`).  So the bundle path and the product engine must
agree exactly.  They agree at shifted total degree 1 but **disagree at total
degree 2, genus 1** (a four-point invariant).  The check took ~68 s and
failed.

### 4.2 Why this was not caught earlier (the coverage gap)

Every passing bundle test is genus 0 with exactly 3 markings, hence
`graph_dimension = 3·0 + 3 − 3 = 0` and `r_order = 1`.  So the `R`-matrix was
**only ever exercised to first order** (`R_1`).  The genus-1 four-point case is
the first to require `r_order = 3·1 + 4 − 3 + 1 = 5`, and it fails.  Meta-point
for reviewers: coverage of *cases* (many classes, insertions, both Hirzebruch
directions) was not coverage of *structure* (the `z`-order of `R`).  See
lessons.md §17.

### 4.3 What is and isn't implicated

- **Graph engine**: not implicated (validated to genus 4 on `P^n`, and via the
  product formula on `P^1 × P^1`).
- **`R`-flatness recursion** (`r_solve.rs:44`,
  `solve_projective_r_coefficients`): shared with `P^n`/product, validated to
  `z^2` by the Behrend product-formula test.  Probably not the bug, but it is
  driven by the frame's connection, which *is* bundle-specific.
- **Classical `R`-asymptotics** (`classical_r_asymptotics_for_point`): shared,
  validated.  The bundle supplies the tangent-weight differences at
  bundle.rs:906–927 — worth double-checking these are the correct tangent
  weights, but they look right.
- **`operator_lagrange_frame`** (recipe.rs:465): **prime suspect.**  This is
  the bundle-specific frame builder.  Unlike `divisor_lagrange_frame`
  (recipe.rs:202), which builds `flat_to_canonical` as the exact Vandermonde
  `canonical_evaluation_matrix(&roots)`, the operator version computes
  `flat_to_canonical = invert_series_matrix_coeff(&transition_to_flat)` (a
  computed matrix inverse) and builds the transition columns as spectral
  projectors `E_p(M)·1`.  Mathematically the two should coincide (the
  Vandermonde is the inverse of the idempotent matrix), but the operator path
  is under-tested and could be wrong/imprecise at higher `q`-order.
- **Quantum multiplication `A_q = A_cl + t d/dt S_1`** (bundle.rs:889): built
  from `S_1` (from Birkhoff, `descendant_s_rational(q_degree, 1)`).  If the
  bundle's Birkhoff `S_1` is imprecise at higher `q`-degree even for `[0,0]`,
  then `A_q` is wrong → frame wrong → `R` wrong.  This is the alternative
  suspect.

`operator_lagrange_frame` is used **only** by the bundle;
`divisor_lagrange_frame` by `P^n`/product.  So a bug there would explain why
products are fine and bundles are not.

### 4.4 Suggested localization (a clean diagnostic)

Distinguish the two suspects by cross-checking on `P^1`, where both frame
builders apply:

1. Build the `P^1` companion `H`-multiplication operator `M` at rational
   weights (relation `(H − w_0)(H − w_1) = q`).
2. Frame A: `divisor_lagrange_frame(roots)`.  Frame B:
   `operator_lagrange_frame(M, seeds, unit, η)`.
3. Feed each through `calibration_from_canonical_frame` at `z_order = 4` and
   compare the `R`-matrices entrywise (exact rationals).
   - If they differ → the bug is in `operator_lagrange_frame` (or
     `invert_series_matrix_coeff` / the spectral-projector construction).
   - If they agree → the bug is upstream, in the bundle's `A_q` / Birkhoff
     `S_1` at higher `q`-degree; next compare the bundle's `A_q` for `[0,0]`
     against the product's `H`-multiplication operator.

(An attempt at exactly this diagnostic was written and then removed because
the hand-built `P^1` metric/operator had a `QSeries` truncation-length bug in
the *test*, not the library — the `from_coeffs` vectors were shorter than
`q_degree + 1`.  Rebuild it carefully with consistent `q_degree` on every
`QSeries`.)

Also worth trying: check whether `P(O ⊕ O)` genus-0 with **4+ markings**
(`r_order ≥ 2`, still genus 0) already disagrees with the product.  If yes, the
bug is purely `r_order`-driven (frame/`R`), independent of genus, which
strongly points at `operator_lagrange_frame`.  If genus-0 higher-marking is
fine but genus-1 is not, look for a genus-specific interaction.

### 4.5 Reproduction

The `#[ignore]`d `f2_deformation_matches_p1xp1_pointwise` in bundle.rs covers
the genus-1 cases (which hit Bug B once Bug A is fixed).  For an isolated,
trivial-mirror reproduction of Bug B alone, add this test back:

```rust
#[test]
fn zero_twist_bundle_matches_product_at_genus_one() {
    use crate::givental::ProductInsertion;
    let point_bundle = BundleInsertion::new(0, 1, 1);
    let point_product = ProductInsertion::new(0, 1, 1);
    for total in 1..=2usize {
        let markings = total + 2;                 // dimension-matched genus-1 point counts
        let bundle_insertions = vec![point_bundle.clone(); markings];
        let product_insertions = vec![point_product.clone(); markings];
        let mut bundle_values = reconstruct_bundle_invariants(
            1, &[0, 0],
            &[Rational::from(2), Rational::from(5)],
            &[Rational::from(11), Rational::from(23)],
            1, total, &bundle_insertions,
        ).unwrap().into_iter().map(|(_, _, v)| v).collect::<Vec<_>>();
        let mut product_values = crate::givental::reconstruct_bidegree_invariants(
            1, 1,
            &[Rational::from(3), Rational::from(7)],
            &[Rational::from(13), Rational::from(29)],
            1, total, &product_insertions,
        ).unwrap();
        bundle_values.sort();
        product_values.sort();
        assert_eq!(bundle_values, product_values, "mismatch at total degree {total}");
        // total 1 passes (both are all-zero -- vacuous); total 2 FAILS.
    }
}
```

Note the `total = 1` case is currently vacuous (the invariants are all zero);
`total = 2` is the meaningful, failing case.  It is slow (~1 min in a debug
build) because it is genus 1 with 4 markings over a 4-color state space, times
3 rays.

---

## 5. Validated envelope (what IS trusted)

Solid and tested (`src/givental/bundle.rs` tests, all passing):

- **Fano bundles, genus 0, `r_order = 1`.**  `F_1 = Bl_pt P^2 = P(O ⊕ O(1))`:
  `∫ H ξ = 1`, `∫ ξ^2 = −1`, the exceptional-curve invariant
  `⟨ξ, ξ, ξ⟩_{e=(1,−1)} = −1` (negative `d_2`, and a *nontrivial* Fano mirror
  map), the fiber count, and the line count `N_h(pt, pt) = 1`.
- `P(O ⊕ O) = P^1 × P^1` reproduces the product engine's genus-0 invariants
  (different pipeline).
- Classical integrals on rank-3 fibers and over a `P^2` base.
- `F_2` (non-Fano) correctly returns `UnsupportedInvariant`.

Not trusted: anything non-Fano (Bug A), anything with `r_order ≥ 2` — i.e.
genus `≥ 1`, or genus 0 with `≥ 4` markings (Bug B).

---

## 6. File / symbol reference

| Symbol | Location | Role |
|---|---|---|
| `ProjectiveBundleRay` | bundle.rs:77 | one ray of a bundle target |
| `ProjectiveBundleRay::i_coefficient` | bundle.rs:309 | bidegree-graded `I`-coefficient |
| `ProjectiveBundleRay::i_restriction` | bundle.rs:281 | fixed-point `z`-Laurent restriction of `I` |
| `ProjectiveBundleRay::j_container` | bundle.rs:341 | **mirror transform `I → J` (Bug A)** |
| `decompose_in_degree_two_span` | bundle.rs:702 | reads mirror map in `{1,H,ξ'}`; `g_unit` = non-Fano signal |
| g_unit guard | bundle.rs:443 | rejects non-Fano (Bug A guard) |
| `ProjectiveBundleRay::descendant_s_rational` | bundle.rs:776 | `J → S` via Birkhoff |
| `BundleRayProvider::graph_kernel` | bundle.rs:865 | `A_q`, frame, `R` — **(Bug B)** |
| `reconstruct_bundle_invariants` | bundle.rs:988 | ray reconstruction entry point |
| `bundle_dimension_matches` | bundle.rs:961 | per-class virtual-dimension filter |
| `operator_lagrange_frame` | recipe.rs:465 | **frame from a multiplication operator (Bug B suspect)** |
| `divisor_lagrange_frame` | recipe.rs:202 | frame from roots (validated; comparison baseline) |
| `calibration_from_canonical_frame` | recipe.rs:50 | frame → `R` via flatness |
| `descendant_s_from_j_function` | recipe.rs:296 | `J → S` (Birkhoff + metric adjoint) |
| `descendant_s_from_i_function` | recipe.rs:257 | `I → S` (single-variable; Fano only) |
| `series_matrix_charpoly` | recipe.rs:422 | Faddeev–LeVerrier charpoly of an operator |
| `newton_root_series` | recipe.rs:177 | root series of the charpoly |
| `solve_projective_r_coefficients` | r_solve.rs:44 | `R`-flatness recursion (shared, validated) |
| `reconstruct_bidegree_invariants` | product.rs | the working rank-2 example (`P^n × P^m`) |

---

## 7. How to build, test, reproduce

From `localization/`:

```bash
cargo build --release
cargo test --lib givental::bundle              # bundle unit tests (all pass; acceptance test ignored)
cargo test --lib givental::bundle -- --ignored # the F_2 acceptance test (fails today)
cargo test --lib givental::recipe              # recipe cross-checks
cargo run --release -- tests                   # 25-case validation oracle
cargo test                                     # full suite (~220 lib tests)
```

Gates every change must pass (CI is at the repo root, one level above the
package): `cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`.

Useful environment flags (see README "Environment Variables"):
`GW_PROFILE=1` (stage timings), `GWAI_DISABLE_RATIONAL_GRAPH`,
`GWAI_DISABLE_FACTORED_GRAPH`.  To dump the raw `I`-coefficients from §3.2,
add a two-line `eprintln!` after bundle.rs:338 printing
`i_container[(d1,d2p)].coefficient(row, z_power)` over the `z` range.

---

## Appendix A — the `I`-function convention actually computed

`i_restriction` (bundle.rs:281) builds the fixed-point restriction of the
`(d_1, d_2)` `I`-coefficient at `(i, j)` as a `z`-Laurent series (`ZLaurent =
BTreeMap<i32, Rational>`):

- Base (`P^n`) factor: for `k = 1..=d_1`, over all base points `i'`, multiply
  by `(λ_i − λ_{i'} + k z)^{-1}` (`zl_mul_inverse_affine`).
- Fiber factors: for each summand `l` with `fiber_degree = d_2 + a_l d_1`:
  - `fiber_degree ≥ 0`: multiply by `(c_{il} − c_{ij} + k z)^{-1}` for
    `k = 1..=fiber_degree` (inverse; negative `z` powers).
  - `fiber_degree < 0`: multiply by `(c_{il} − c_{ij} + k z)` for
    `k = fiber_degree+1..=0` (affine; the `k = −1, −2, …` factors are what
    introduce **positive** `z` powers — the non-Fano source).

`i_coefficient` (bundle.rs:309) assembles the vector in the classical
`D`-power basis by combining fixed-point restrictions with the classical
Lagrange transition (`recipe::classical_lagrange_transition`).  `min_z_power`
(bundle.rs:261) sets the negative-`z` truncation depth; bumping it did **not**
fix Bug B (so Bug B is not a shallow-truncation issue).

---

## Appendix B — the `F_2 ↔ P^1 × P^1` deformation dictionary (the acceptance test)

`F_2` and `F_0 = P^1 × P^1` are deformation equivalent (`F_a` for `a` even), so
**all** GW invariants (every genus, with descendants) match under the induced
identification.  `F_2` has a nontrivial mirror map; `F_0` does not.  This makes
it the ideal acceptance test once Bugs A and B are fixed; it is encoded in the
`#[ignore]`d `f2_deformation_matches_p1xp1_pointwise` (bundle.rs tests).

Derivation.  Write `F_2` curve classes as `p·f + q·B_-` (`f` = fiber, `B_-` =
`(−2)`-section: `f^2 = 0`, `f·B_- = 1`, `B_-^2 = −2`).  Then, with
`(d_1, d_2) = (H·β, ξ·β)`:

- `H` is Poincaré-dual to `f`, and `B_- ≐ ξ` as a divisor, giving
  `H·f = 0, H·B_- = 1, ξ·f = 1, ξ·B_- = −2`.  So `d_1 = q`, `d_2 = p − 2q`.
- Under the deformation `f ↦ f_1`, `B_- ↦ f_2 − f_1` on `F_0` (both have
  self-intersection `−2`), so `p f + q B_- ↦ (p−q) f_1 + q f_2`, i.e. `F_0`
  bidegree `(H_1·β, H_2·β) = (p − q, q) = (d_2 + d_1, d_1)`.

**Curve-class map:** `F_2 (d_1, d_2)  ↔  F_0 bidegree (d_2 + d_1, d_1)`
(first coordinate = `H_1·β`).  Equivalently, the bundle's shifted total degree
`d_2 + (A+1) d_1 = d_2 + 3 d_1` (`A = 2`) and the product's total degree
`(d_2 + d_1) + d_1 = d_2 + 2 d_1` are **different** parametrizations — pass each
engine its own.

**Cohomology map** (from `H·β = d_1 = H_2·β` and `ξ·β = d_2 = (H_1 − H_2)·β`):

```
H_{F_2}  ↔  H_2          (equivalently H_1 = ξ + H,  H_2 = H)
ξ_{F_2}  ↔  H_1 − H_2
```

Consistency: `ξ^2 = −2 H ξ` on `F_2` maps to `(H_1 − H_2)^2 = −2 H_2 (H_1 − H_2)`
on `F_0` (using `H_1^2 = H_2^2 = 0`); and `c_1·β = 4 d_1 + 2 d_2` matches on both
sides.

**Insertion expansion** (for comparing against the monomial-only product
engine).  `τ_k(H^h ξ^x) ↦ τ_k(H_2^h (H_1 − H_2)^x)`; expand
`(H_1 − H_2)^x = Σ_j C(x, j) H_1^j (−H_2)^{x−j}`, times `H_2^h`, and drop any
`H_1^{≥2}` or `H_2^{≥2}` monomial (zero on `P^1 × P^1`).  So each `F_2`
insertion becomes a short signed sum of product monomials `τ_k(H_1^a H_2^b)`,
`a, b ∈ {0, 1}`.  A full multi-insertion invariant is the sum over the
Cartesian product of these expansions (with signs), each term a product-engine
call at bidegree `(d_2 + d_1, d_1)` and total `d_2 + 2 d_1`, read at the `H_2`
index `d_1`.  This is exactly what the in-code helpers
`f2_insertion_to_product_terms` / `product_side_of_f2` / `bundle_side_of_f2`
implement.

**Product-representability caveat.**  The product engine only represents
bidegrees with **both** coordinates `≥ 0`.  `F_0` bidegree `(d_2 + d_1, d_1)`
needs `d_2 + d_1 ≥ 0`, i.e. classes in the cone spanned by the fiber `f` and
the *positive* section `B_+`.  The negative section `B_-` itself maps to
`(−1, 1)` and cannot be checked this way (it is effective on `F_2` but its
`F_0` class is not represented by the product's nonnegative reconstruction).
