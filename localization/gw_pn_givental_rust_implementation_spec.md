# Implementation Specification: High-Genus Gromov-Witten Invariants of `P^n` via Localization and Givental Quantization

**Target package:** `gw-pn`  
**Primary language:** Rust  
**Goal:** compute individual descendant Gromov-Witten invariants of projective space `P^n`, especially high genus and moderate degree, using two mutually validating engines:

1. virtual localization on stable-map fixed loci;
2. Givental-Teleman style semisimple reconstruction / quantization using the Frobenius structure of equivariant quantum cohomology.

This document is written as a build plan for another coding agent. It is intentionally explicit about conventions, truncation, data structures, and test strategy.

---

## 0. Scope and conventions

### 0.1 Mathematical target

The package should compute invariants of the form

```text
< tau_{k_1}(gamma_1) ... tau_{k_m}(gamma_m) >_{g,d}^{P^n}
```

where each insertion `gamma_a` is one of:

```text
H^r                         ordinary cohomology class
phi_i                       equivariant fixed-point class / idempotent-like class
linear combination thereof  exact symbolic coefficients
```

The core computation should be equivariant first. The nonequivariant answer should be obtained only after cancellation.

Use torus weights

```text
lambda_0, ..., lambda_n
```

and hyperplane class `H`. The equivariant cohomology ring is

```text
H_T^*(P^n) = Q(lambda_0,...,lambda_n)[H] / prod_{i=0}^n (H - lambda_i).
```

Small equivariant quantum cohomology is

```text
QH_T^*(P^n) = Q(lambda)[[q]][H] / (prod_i (H - lambda_i) - q).
```

For reconstruction, work at a semisimple equivariant point. For `P^n`, equivariant quantum cohomology is semisimple generically.

### 0.2 Engineering target

The Rust package should expose:

```rust
let target = InvariantRequest {
    n: 2,
    genus: 3,
    degree: 4,
    insertions: vec![tau(0, H.pow(2)), tau(1, H.pow(1)), ...],
    mode: ComputeMode::CompareLocalizationAndGivental,
};
let ans = gw_pn::compute(target)?;
```

The result should be exact:

```text
Rational number, or rational function in lambda_i before nonequivariant limit.
```

The package should also expose lower-level APIs for graph enumeration, Frobenius data, `R`-matrix coefficients, Givental graph expansion, and tautological integral lookup.

---

## 1. High-level architecture

Use a workspace:

```text
gw-pn/
  Cargo.toml
  crates/
    gw-algebra/
    gw-graphs/
    gw-tautological/
    gw-pn-geometry/
    gw-localization/
    gw-frobenius/
    gw-givental/
    gw-validation/
    gw-cli/
```

### 1.1 Crate responsibilities

#### `gw-algebra`

Exact algebra types:

```text
BigInt
BigRational
SparseMonomial
SparsePolynomial
RationalFunction
TruncatedSeries<Variable>
Matrix<T>
```

Must support:

```text
addition, multiplication, division, gcd/canonicalization
multivariate sparse polynomials
rational functions in lambda_i, q, z, psi variables
truncated power series in q and z
Laurent series if needed
matrix multiplication/inversion over rational functions / series
```

Performance priorities:

```text
canonical interning of variable names
monomial order with compact exponent vectors
hash-consing optional for large expressions
lazy simplification with explicit normalize() checkpoints
parallel product/sum over graph contributions
```

#### `gw-graphs`

Generic stable-graph infrastructure:

```text
StableGraph
VertexId
EdgeId
HalfEdgeId
LegId
GraphAutomorphism
CanonicalLabel
```

Should be reused by both localization and Givental graph expansions.

#### `gw-tautological`

Integrals over `Mbar_{g,n}`:

```text
psi_integral(g, psi_powers)
hodge_integral(g, lambda_powers, psi_powers)
kappa_integral(...)
```

At minimum, implement Witten-Kontsevich psi integrals and allow a backend for Hodge integrals. The localization engine needs Hodge integrals. The Givental graph engine can often reduce to psi integrals at vertices because the base theory is the product of KdV tau-functions / point CohFTs.

#### `gw-pn-geometry`

Projective-space data:

```text
EquivariantProjectiveSpace { n, weights }
CohomologyClass
FixedPointBasis
HyperplaneBasis
Pairing
RestrictionMap
```

Responsible for converting between `H^r` and fixed-point restrictions.

#### `gw-localization`

Stable-map localization graph enumeration and contribution evaluation.

#### `gw-frobenius`

Quantum product, canonical coordinates, idempotents, metric, transition matrix `Psi`, canonical norms `Delta_i`, Dubrovin connection, and fundamental solutions.

#### `gw-givental`

`R`-matrix, translation, quantized action, graph expansion, descendant extraction.

#### `gw-validation`

String/dilaton/divisor/TRR checks, comparison between engines, known values, randomized low-degree tests.

#### `gw-cli`

Command-line interface:

```bash
gw-pn compute --n 2 --g 3 --d 4 --insert 'tau0(H^2)' --insert 'tau1(H)'
gw-pn r-matrix --n 2 --q-degree 4 --z-order 8
gw-pn compare --n 1 --g-max 4 --d-max 5
```

---

## 2. Algebra backend

The algebra layer is the most important performance bottleneck. Do not use a general symbolic expression tree for everything. Use structured exact algebra.

### 2.1 Variables

Represent variables by small integer IDs:

```rust
pub struct VarId(u32);

pub enum VarKind {
    EquivariantWeight { i: usize }, // lambda_i
    NovikovQ,
    LoopZ,
    Psi { leg: usize },
    Hbar,
    Auxiliary(String),
}
```

A monomial is an exponent vector:

```rust
pub struct Monomial {
    exponents: SmallVec<[(VarId, i32); 8]>,
}
```

Use signed exponents only for Laurent monomials. For ordinary polynomials, enforce nonnegative exponents.

### 2.2 Sparse polynomials

```rust
pub struct SparsePoly<C> {
    terms: BTreeMap<Monomial, C>,
}
```

where `C = BigRational` initially.

Important methods:

```rust
add_assign
mul_assign_truncated(degree_bound)
derivative(var)
evaluate_substitution(map)
content_gcd
canonicalize
```

### 2.3 Rational functions

```rust
pub struct RatFun {
    num: SparsePoly<BigRational>,
    den: SparsePoly<BigRational>,
}
```

Canonicalization rules:

1. denominator leading coefficient positive;
2. divide numerator/denominator by rational content;
3. optional polynomial gcd only at expensive checkpoints;
4. provide `normalize_light()` and `normalize_full()`.

For graph sums, do not fully simplify after every multiplication. Accumulate terms with bounded light simplification and full simplify at graph-level boundaries.

### 2.4 Truncated series

Use a generic type:

```rust
pub struct Series<T> {
    var: VarId,
    min_order: i32,
    max_order: i32,
    coeffs: BTreeMap<i32, T>,
}
```

Examples:

```text
R(z) = 1 + R_1 z + ... + R_N z^N
I(q,z) = sum_{d=0}^D q^d I_d(z)
```

For multivariate truncation, define an explicit bound object:

```rust
pub struct Truncation {
    q_degree: usize,
    z_order: usize,
    descendant_degree: usize,
    genus: usize,
}
```

Every method taking series should require a `Truncation` parameter rather than silently expanding.

---

## 3. Cohomology and equivariant projective space

### 3.1 Hyperplane basis

Represent classes as vectors of length `n+1` in the basis

```text
1, H, H^2, ..., H^n.
```

Multiplication is reduction modulo

```text
prod_i (H - lambda_i)       classical equivariant
prod_i (H - lambda_i) = q   quantum equivariant
H^{n+1} = q                 nonequivariant quantum
```

### 3.2 Fixed-point restrictions

The restriction of `H` to fixed point `p_i` is

```text
H|_{p_i} = lambda_i.
```

Therefore

```text
(H^r)|_{p_i} = lambda_i^r.
```

A general class `gamma = sum_r c_r H^r` restricts to

```text
gamma_i = sum_r c_r lambda_i^r.
```

### 3.3 Equivariant fixed-point basis

Define the localized idempotent class

```text
phi_i = prod_{j != i} (H - lambda_j) / prod_{j != i} (lambda_i - lambda_j).
```

Then

```text
phi_i|_{p_j} = delta_{ij}.
```

The equivariant pairing in fixed-point basis is

```text
(phi_i, phi_j) = delta_ij / e_i
```

where

```text
e_i = e_T(T_{p_i} P^n) = prod_{j != i} (lambda_i - lambda_j).
```

This is the easiest basis for localization.

---

## 4. Localization engine

The localization engine computes the invariant directly as a sum over decorated stable-map graphs. It should be correct before the Givental engine is trusted.

### 4.1 Stable-map graph data

A localization graph contains:

```rust
pub struct LocGraph {
    vertices: Vec<LocVertex>,
    edges: Vec<LocEdge>,
    legs: Vec<MarkedLeg>,
}

pub struct LocVertex {
    id: VertexId,
    fixed_point: usize, // i_v in {0,...,n}
    genus: usize,
}

pub struct LocEdge {
    from: VertexId,
    to: VertexId,
    degree: usize,
}

pub struct MarkedLeg {
    vertex: VertexId,
    marking: usize,
}
```

Constraints:

```text
sum_e degree(e) = d
sum_v genus(v) + h^1(graph) = g
for each edge e=(v,w), fixed_point(v) != fixed_point(w)
stability: 2g_v - 2 + valence(v) > 0, except unstable vertices must be handled by special rules
```

Here `valence(v)` counts incident edges plus marked legs.

### 4.2 Graph generation algorithm

Recommended approach:

1. Generate all connected abstract multigraphs with bounded number of vertices/edges compatible with `g,d,m`.
2. Assign vertex genera summing to `g - h^1`.
3. Assign fixed points to vertices, rejecting equal fixed points across edges.
4. Assign positive edge degrees summing to `d`.
5. Assign markings to vertices.
6. Canonicalize and divide by automorphism group.

For performance, generate in a canonical form rather than generate-and-deduplicate where possible.

Pseudo-code:

```rust
fn localization_graphs(req: &InvariantRequest) -> impl Iterator<Item = WeightedGraph<LocGraph>> {
    for abstract_graph in connected_multigraphs(bounds(req)) {
        let h1 = abstract_graph.first_betti();
        if h1 > req.genus { continue; }
        for genus_assignment in compositions(req.genus - h1, abstract_graph.num_vertices()) {
            for fixed_assignment in fixed_point_colorings(&abstract_graph, req.n + 1) {
                for degree_assignment in positive_edge_compositions(req.degree, abstract_graph.num_edges()) {
                    for marking_assignment in set_partitions(req.num_insertions, abstract_graph.num_vertices()) {
                        let graph = build(...);
                        if !stable_or_known_unstable(&graph) { continue; }
                        let canon = graph.canonical_label();
                        yield_once(canon, graph, 1 / aut_size(canon));
                    }
                }
            }
        }
    }
}
```

### 4.3 Localization contribution

For a graph `Gamma`, contribution has the form

```text
Cont(Gamma) = 1/|Aut(Gamma)| * prod_v V_v * prod_e E_e * prod_flags F_{v,e} * prod_marks M_a.
```

The exact formula depends on conventions for `phi_i`, Euler classes, and unstable vertices. Therefore implement factors in a convention-tested way:

```rust
trait LocalizationFactorConvention {
    fn vertex(&self, graph: &LocGraph, v: VertexId) -> RatFun;
    fn edge(&self, graph: &LocGraph, e: EdgeId) -> RatFun;
    fn flag(&self, graph: &LocGraph, h: HalfEdgeId) -> RatFun;
    fn marking(&self, graph: &LocGraph, leg: LegId, insertion: &Insertion) -> RatFun;
    fn unstable_replacement(&self, ...) -> RatFun;
}
```

The first implementation should reproduce standard Kontsevich localization for `P^n`.

### 4.4 Flag weights

For an edge of degree `d_e` connecting fixed points `i` and `j`, define the tangent weight at the `i` side:

```text
omega_{i,j,d} = (lambda_i - lambda_j) / d.
```

This is the weight of the tangent direction of the domain branch mapping from `p_i` to `p_j`.

At a stable vertex, denominators usually include factors of the form

```text
1 / (omega_{flag} - psi_flag)
```

which must be expanded as a finite series before integration over `Mbar_{g_v,val(v)}`:

```text
1/(omega - psi) = sum_{a >= 0} psi^a / omega^{a+1}
```

truncate by the dimension of the vertex moduli space.

### 4.5 Vertex integration

For a vertex `v`, the factor is a polynomial/rational expression in psi and Hodge classes integrated over

```text
Mbar_{g_v, val(v)}.
```

The engine should reduce this to calls:

```rust
oracle.integrate(VertexIntegralRequest {
    genus: g_v,
    n_markings: valence,
    psi_powers,
    hodge_lambda_powers,
    coefficient,
})
```

Dimension check:

```text
sum psi powers + weighted Hodge degree = 3g_v - 3 + valence.
```

If not equal, return zero immediately.

### 4.6 Descendant insertions

For an insertion `tau_k(gamma)` attached to vertex `v` at fixed point `i_v`:

```text
restriction = gamma|_{p_{i_v}}
marking factor = restriction * psi_mark^k
```

The `psi_mark^k` is included in the vertex integral.

### 4.7 Nonequivariant limit

Do not set `lambda_i = 0` early. The algorithm should:

1. compute equivariant rational expression;
2. express insertions consistently in the hyperplane basis;
3. simplify the total sum;
4. take limit along a generic specialization if symbolic multivariate limit is expensive.

Practical robust limit method:

```text
lambda_i = c_i * t, with distinct integer c_i, e.g. c_i = i+1
expand as Laurent series in t
extract constant term at t=0
verify independence by trying several c_i choices
```

This is often faster than full multivariate cancellation.

---

## 5. Tautological integral oracle

### 5.1 Required API

```rust
pub trait TautologicalOracle: Send + Sync {
    fn psi_integral(&self, g: usize, powers: &[usize]) -> BigRational;

    fn hodge_integral(
        &self,
        g: usize,
        psi_powers: &[usize],
        lambda_powers: &[(usize, usize)], // (lambda_index, exponent)
    ) -> BigRational;
}
```

### 5.2 Psi integrals

Implement Witten-Kontsevich intersection numbers using string/dilaton/Virasoro recursion. Cache aggressively.

Input:

```text
< tau_{a_1} ... tau_{a_n} >_g
```

Dimension:

```text
sum a_i = 3g - 3 + n.
```

Base cases:

```text
<tau_0^3>_0 = 1
<tau_1>_1 = 1/24
```

String equation:

```text
<tau_0 prod_i tau_{a_i}>_g = sum_i <tau_{a_i-1} prod_{j != i} tau_{a_j}>_g
```

Dilaton equation:

```text
<tau_1 prod_i tau_{a_i}>_g = (2g - 2 + n) <prod_i tau_{a_i}>_g
```

For remaining cases, implement DVV recursion.

### 5.3 Hodge integrals

Options:

1. Use a built-in implementation for lambda/psi integrals up to practical genus.
2. Use precomputed tables serialized as compressed JSON/bincode.
3. Provide an FFI bridge to Sage/admcycles for offline table generation.

The Rust package should define the stable API and caching even if the first version delegates Hodge integrals.

Cache key:

```rust
#[derive(Hash, Eq, PartialEq)]
struct HodgeKey {
    g: u16,
    psi_powers_sorted_with_labels: Vec<u16>,
    lambda_powers: Vec<(u16, u16)>,
}
```

Do not sort marked powers if labels matter in the caller. For symmetric integrals, canonicalize by sorting powers.

---

## 6. Frobenius manifold layer for `P^n`

The Givental engine begins from semisimple Frobenius data.

### 6.1 Frobenius data to compute

At a chosen point of small quantum cohomology, compute:

```text
eta          flat metric in H-basis
star         quantum product
u_i          canonical coordinates / eigenvalues
epsilon_i    idempotent vector fields
e_i          unnormalized idempotents
Delta_i      inverse metric norms or metric norms, depending convention
Psi          transition matrix from normalized canonical basis to flat basis
R(z)         calibration matrix
T(z)         translation / dilaton correction
```

The implementation must store convention metadata:

```rust
pub enum DeltaConvention {
    MetricNorm,        // Delta_i = (epsilon_i, epsilon_i)
    InverseMetricNorm, // Delta_i = 1/(epsilon_i, epsilon_i)
}
```

Many formula errors come from mixing these conventions.

### 6.2 Quantum product matrix

In the hyperplane basis, multiplication by `H` is the companion matrix of

```text
prod_i (H - lambda_i) = q.
```

Let

```text
P(x) = prod_i (x - lambda_i) - q.
```

The canonical eigenvalues of multiplication by `H` are the roots `x_a` of `P(x)`. For exact computation, do not necessarily solve roots. Work formally through symmetric functions or use the fixed-point / canonical branch expansion.

At `q=0`, roots are `lambda_i`. For small `q`, each root has a formal expansion

```text
x_i(q) = lambda_i + O(q).
```

Solve root expansions by Newton iteration in the complete local ring `Q(lambda)[[q]]`:

```text
x_i^{new} = x_i - (P(x_i) / P'(x_i)) mod q^{D+1}
```

Initialize `x_i = lambda_i`.

### 6.3 Canonical coordinates

For conformal Frobenius manifolds, canonical coordinates can be chosen so the Euler vector field is diagonal. For small quantum `P^n`, a practical implementation can treat the canonical coordinate associated to branch `i` as the eigenvalue of quantum multiplication by the Euler field. In many small-quantum calculations, one can work with the canonical roots of multiplication by `H` and derive the needed normalized idempotents and transition matrix directly.

Coding rule:

```text
Do not require closed forms for u_i. Store u_i as truncated q-series.
```

### 6.4 Idempotents

Given a root `x_i` of `P(x)`, the corresponding idempotent in the algebra `Q(lambda)[[q]][H]/P(H)` is

```text
e_i(H) = prod_{j != i} (H - x_j) / prod_{j != i} (x_i - x_j).
```

This satisfies

```text
e_i * e_j = delta_ij e_i,
sum_i e_i = 1.
```

The pairing is residue-like. For `P^n`, the metric can be computed by reducing products in the hyperplane basis and integrating equivariantly.

### 6.5 Norms and normalized idempotents

Compute

```text
g_i = (e_i, e_i).
```

Then choose a square root branch. Common options:

```text
Delta_i = 1/g_i
normalized epsilon_i = sqrt(Delta_i) e_i
```

or

```text
Delta_i = g_i
normalized epsilon_i = e_i / sqrt(Delta_i).
```

Pick one convention globally. Recommended internal convention:

```text
Delta_i = 1 / (e_i, e_i)
normalized basis f_i = sqrt(Delta_i) e_i
(f_i, f_j) = delta_ij
```

This gives an orthonormal canonical basis, simplifying quantization.

### 6.6 Transition matrix `Psi`

Let `phi_alpha` be the flat hyperplane basis and `f_i` the normalized canonical basis. Define

```text
f_i = sum_alpha Psi[alpha, i] phi_alpha.
```

Then `Psi` maps canonical orthonormal coordinates to flat coordinates.

Store both `Psi` and `Psi^{-1}`.

Consistency checks:

```text
Psi^T eta Psi = I
f_i * f_j = delta_ij sqrt(Delta_i) f_i  // depending convention
```

The coding agent must write tests for these identities at every truncation order.

---

## 7. The `R`-matrix: concrete interpretation and computation

This is the part that is often vaguely described. The implementation should not treat `R` as mystical. It is a matrix power series determined by flatness, symplectic/unitarity, and calibration choices.

### 7.1 What `R` is

In normalized canonical coordinates, a formal fundamental solution of the Dubrovin connection has the form

```text
S_tilde(z) = R(z) exp(U/z)
```

where

```text
U = diag(u_0,...,u_n)
R(z) = I + R_1 z + R_2 z^2 + ...
```

and the symplectic/unitarity condition is

```text
R(z) R^T(-z) = I.
```

In flat coordinates,

```text
S(z) = Psi^{-1} R(z) exp(U/z)
```

or equivalently, depending on matrix orientation,

```text
S(z) = Psi R(z) exp(U/z).
```

The package must define one orientation and enforce it with tests. Recommended:

```text
flat_vector = Psi * canonical_vector
S_flat = Psi * R * exp(U/z)
```

Then `Psi^T eta Psi = I`.

### 7.2 Calibration ambiguity

`R` is not unique. Multiplication on the right by

```text
exp(a_1 z + a_3 z^3 + a_5 z^5 + ...)
```

where each `a_{2k-1}` is constant diagonal and skew-compatible, preserves the basic properties. For conformal `P^n`, choose the homogeneous/canonical calibration. In practice, use the calibration coming from the equivariant `J`/`S`-matrix or Lee-Pandharipande materialization.

Implementation rule:

```text
Every RMatrix must carry a CalibrationId.
Never compare R-matrices from different calibration ids without applying the known gauge transformation.
```

### 7.3 Computing `R` from the Dubrovin connection

Let the connection in flat coordinates be

```text
z dS = C(t) S
```

where `C` are quantum multiplication matrices. In canonical coordinates, after normalizing by `Psi`, solve recursively for coefficients of `R`.

A robust general recursion:

1. Compute `Psi`, `U`, and the connection matrices in canonical basis.
2. Write the flatness equation for `S = Psi R exp(U/z)`.
3. Equate coefficients of powers of `z`.
4. Solve off-diagonal entries from differences `u_i - u_j`.
5. Determine diagonal entries from unitarity and homogeneity/calibration.

The coding agent should implement this as a symbolic recursion only after simpler paths are working.

### 7.4 Computing `R` from the `I`/`J`-function

For `P^n`, use the hypergeometric `I`-function:

```text
I(q,z) = exp(H log(q)/z) * sum_{d>=0} q^d / prod_{m=1}^d prod_{j=0}^n (H - lambda_j + m z)
```

Depending on convention, mirror map is trivial for projective space in small variables, but the implementation should still separate:

```text
I_function
mirror_map
J_function
S_matrix
R_matrix_from_S
```

This allows extension to other targets later.

Extraction approach:

```text
S_flat(z) = matrix whose columns are flat sections generated from J derivatives
R(z) = Psi^{-1} S_flat(z) exp(-U/z)
```

Because `exp(-U/z)` introduces negative powers, the multiplication must be handled as a formal asymptotic expansion. Truncate in both `q` and `z`.

### 7.5 Computing `R` from localization / materialization

Alternative and probably better for matching Lee-Pandharipande:

```text
A_i(z)  one-point descendant series in canonical sector i
E_ij(z,w) two-point propagator series
T_i(z)  translation series
```

The rough relationships are:

```text
A_i coefficients encode columns/normalizations of S or R
E_ij(z,w) is the edge propagator generated by R
T_i(z) is the translation/dilaton correction
```

Implement these as first-class objects:

```rust
pub struct Materialization {
    pub a: Vec<Series<RatFun>>,              // A_i
    pub e: Vec<Vec<BivariateSeries<RatFun>>>,// E_ij
    pub t: Vec<Series<RatFun>>,              // T_i
    pub calibration: CalibrationId,
}
```

Then provide conversions:

```rust
impl Materialization {
    fn to_r_matrix(&self) -> RMatrix;
    fn propagator(&self, i: usize, j: usize, k: usize, l: usize) -> RatFun;
    fn translation(&self, i: usize, k: usize) -> RatFun;
}
```

### 7.6 `R`-matrix sanity checks

For every computed `R`:

```text
R_0 = I
R(z) R^T(-z) = I mod z^{N+1}
Psi^T eta Psi = I mod q^{D+1}
S solves Dubrovin equation mod truncation
calibration homogeneity equation holds if using conformal calibration
```

If any check fails, do not use the `R`-matrix for invariants.

---

## 8. Givental quantization: implementable interpretation

There are two equivalent implementation levels:

1. operator quantization on Fock space;
2. stable-graph/Feynman expansion of the quantized action.

For computing individual invariants, implement the graph expansion first. It is finite, easier to truncate, easier to debug, and closer to localization.

### 8.1 Symplectic loop space

Let `V = H_T^*(P^n)` with pairing `( , )`. Define

```text
H_loop = V((z^{-1}))
Omega(f,g) = Res_{z=0} (f(-z), g(z)) dz.
```

Polarization:

```text
H_+ = V[z]
H_- = z^{-1} V[[z^{-1}]]
H_loop = H_+ ⊕ H_-
```

Coordinates:

```text
q_k^alpha  for basis vector phi_alpha z^k in H_+
p_{k,alpha} for dual coordinates in H_-
```

The descendant variables `t_k^alpha` are related by the dilaton shift:

```text
q_1^unit = t_1^unit - 1
q_k^alpha = t_k^alpha otherwise
```

Equivalently,

```text
q(z) = t(z) - z * 1.
```

This shift is a common source of off-by-one errors. Encode it explicitly:

```rust
pub struct DilatonShift {
    unit_basis_index: usize,
    shift_at_descendant_level: usize, // 1
    shift_value: BigRational,          // -1 in q = t - z
}
```

### 8.2 Quantization rules

For quadratic Hamiltonians in Darboux coordinates, use Weyl quantization. In orthonormal canonical coordinates, the simple rules are:

```text
q_i,k q_j,l  ->  q_i,k q_j,l / hbar
q_i,k p_j,l  ->  q_i,k ∂/∂q_j,l
p_i,k p_j,l  ->  hbar ∂^2/(∂q_i,k ∂q_j,l)
```

If not in orthonormal coordinates, insert metric tensors. Therefore the implementation should perform quantization in the normalized canonical basis whenever possible.

### 8.3 Why not expand operators directly?

The operator

```text
hat{R} = exp(hat{r})
```

with

```text
r(z) = log R(z)
```

acts on a product of KdV tau-functions. Direct operator expansion causes massive intermediate expression growth.

Instead use the Feynman graph formula for the action of `R`. This gives exactly the same coefficients but only enumerates stable graphs contributing to the requested genus/number of markings/order.

### 8.4 Base theory: product of point theories

At a semisimple point, the topological field theory decomposes into one copy per canonical idempotent. The ancestor/descendant potential is obtained by acting on

```text
prod_i tau_KdV(q^i_0, q^i_1, ...)
```

Each vertex in the Givental graph expansion is a psi intersection number on `Mbar_{g_v,n_v}` in color `i`.

Vertex rule:

```text
vertex v colored i contributes Delta_i^{?} * <prod_{h incident to v} tau_{a_h}>_{g_v}
```

The exact power of `Delta_i` depends on normalization. With normalized orthonormal idempotents, most metric factors disappear; if using unnormalized idempotents, the common CohFT TFT tensor is

```text
omega_{g,n}(e_i,...,e_i) = Delta_i^{g-1}
```

or inverse depending on whether `Delta_i` is norm or inverse norm. This is why the package must centralize the convention.

Recommended internal rule:

```text
Use normalized canonical basis for R and quantization.
Store any Delta powers in a single method:
TftVertexWeight::weight(color, genus, valence)
```

### 8.5 Leg action

For an insertion vector `v` and descendant power `k`, apply the inverse `R`-matrix to the leg:

```text
R^{-1}(psi) v = sum_{l>=0} (R^{-1})_l v * psi^l.
```

If the input is in flat basis:

```text
v_canonical = Psi^{-1} v_flat
```

Then the leg expansion is in canonical colors.

A leg contribution at a vertex of color `i` is the coefficient of canonical basis vector `i` in `R^{-1}(psi) v_canonical`, multiplied by `psi^k` from the original descendant insertion.

Implementation:

```rust
fn expand_leg(
    insertion: &Insertion,
    frob: &FrobeniusData,
    r_inv: &RMatrix,
    max_psi: usize,
) -> Vec<LegTerm>;

struct LegTerm {
    color: usize,
    psi_power: usize,
    coeff: RatFun,
}
```

### 8.6 Edge propagator

For an edge connecting two half-edges with psi classes `psi'` and `psi''`, the standard Givental edge bivector is

```text
V(psi', psi'') = [ eta^{-1} - R^{-1}(psi') eta^{-1} R^{-1}(psi'')^T ] / (psi' + psi'').
```

In an orthonormal canonical basis, `eta^{-1}` is the identity, so

```text
V^{ij}(x,y) = [delta_ij - sum_a (R^{-1})^i_a(x) (R^{-1})^j_a(y)] / (x+y).
```

Although this has a denominator `x+y`, the numerator is divisible by `x+y` because of the symplectic condition. The implementation should perform formal division by using the identity:

If

```text
N(x,y) = sum_{a,b} N_{a,b} x^a y^b
```

and `N(-y,y)=0`, then solve coefficients `V_{a,b}` from

```text
N(x,y) = (x+y) V(x,y).
```

Do not store actual rational functions in psi variables with denominator `x+y`; expand to polynomial coefficients.

API:

```rust
fn edge_propagator(
    r_inv: &RMatrix,
    max_left: usize,
    max_right: usize,
) -> Vec<EdgeTerm>;

struct EdgeTerm {
    left_color: usize,
    right_color: usize,
    left_psi: usize,
    right_psi: usize,
    coeff: RatFun,
}
```

### 8.7 Translation term

The `R`-action on a CohFT includes a translation, often written

```text
T(z) = z * (1 - R^{-1}(z) 1)
```

or with `R` instead of `R^{-1}` depending on convention and ancestor/descendant orientation.

Implementation rule:

```text
Do not hardcode the sign in multiple places.
Implement TranslationConvention and test it against known genus 0/1 invariants.
```

Translation inserts additional unmarked leaves at vertices. A translation leaf contributes a series term

```text
T_i,k psi^k
```

and is summed over any number of translation leaves, subject to dimension/genus truncation.

For an individual invariant, enumerate translation leaves only up to the maximum needed by dimension:

```text
max_translation_leaves <= 3g - 3 + m + max_extra_from_R
```

In practice, use recursive expansion with pruning by vertex dimension.

### 8.8 Givental stable graph contribution

A Givental graph has:

```text
stable vertices with genus g_v and color i_v
ordinary marked legs corresponding to requested insertions
optional translation legs
edges with propagator terms
```

Contribution:

```text
1/|Aut(Gamma)|
* prod_vertices TFTWeight(color, g_v, valence)
* prod_vertices psi_integral(g_v, incident_psi_powers)
* prod_edges propagator_coeff
* prod_ordinary_legs leg_coeff
* prod_translation_legs T_coeff
```

Then pushforward/integration over boundary strata is already encoded by multiplying vertex psi integrals. For scalar coefficient extraction, no explicit tautological class needs to be returned.

### 8.9 Genus and degree tracking

Givental reconstruction naturally gives generating functions in quantum parameters. For `P^n`, degree is tracked through `q`-series coefficients in Frobenius data and `R`, `Psi`, `Delta`, `T`.

Every coefficient object must retain `q` order. To compute degree `d`, truncate all Frobenius/Givental series to `q^d` and extract coefficient `q^d` at the end.

### 8.10 Givental graph enumeration

For target `(g,m)`, generate stable graphs of genus `g` with `m` ordinary markings plus variable translation markings.

A practical approach:

1. Generate stable graphs with `m + t` legs for `t=0..t_max`.
2. Designate the first `m` as ordinary and remaining `t` as translation legs.
3. Sum over colors of vertices.
4. Expand each ordinary leg into possible `LegTerm`s.
5. Expand each translation leg into possible `TranslationTerm`s.
6. Expand each edge into possible `EdgeTerm`s.
7. Check each vertex dimension exactly before calling psi integral.

Pseudo-code:

```rust
fn givental_invariant(req: &InvariantRequest, trunc: Truncation) -> RatFun {
    let frob = FrobeniusData::for_pn(req.n, trunc.q_degree);
    let r = RMatrix::compute(&frob, trunc);
    let r_inv = r.inverse(trunc.z_order);
    let translation = Translation::compute(&r_inv, &frob, trunc);

    let mut total = RatFun::zero();

    for t in 0..=translation_bound(req, trunc) {
        for graph in stable_graphs(req.genus, req.num_insertions + t) {
            let aut = graph.automorphism_factor_with_marking_types(req.num_insertions, t);
            for coloring in colorings(graph.vertices(), req.n + 1) {
                let mut expander = LocalContributionExpander::new(...);
                total += expander.sum_terms_satisfying_dimension() / aut;
            }
        }
    }

    total.extract_q_coeff(req.degree).nonequivariant_limit_if_requested()
}
```

### 8.11 Pruning rules

For a vertex `v`, dimension is

```text
dim_v = 3g_v - 3 + valence(v).
```

The sum of psi powers incident to `v` must equal `dim_v`. During expansion, track partial psi power and prune if it exceeds `dim_v`.

For an edge term with psi powers `(a,b)`, add `a` to left vertex and `b` to right vertex.

For a leg term, add `original_descendant_k + R_extra_l`.

For translation term, add its psi power.

### 8.12 Output coefficient extraction

The Givental engine returns a coefficient in canonical/flat variables depending on request. For individual invariant, differentiate the potential with respect to requested variables. The graph formula above already computes the coefficient corresponding to labelled insertions, so no factorial correction is needed if markings are labelled.

If the implementation instead extracts from a generating function, remember:

```text
F_g = sum_{d,m} q^d / m! <...>
```

Labelled graph computation avoids the `m!` issue.

---

## 9. Operator quantization layer

This is optional for first production but useful for verifying Givental formalism.

### 9.1 Infinitesimal symplectic transformations

An infinitesimal symplectic operator `A(z)` satisfies

```text
A^*(-z) + A(z) = 0.
```

For `R(z) = exp(r(z))`, `r(z)` is infinitesimal symplectic.

### 9.2 Hamiltonian extraction

For `f in H_loop`, define quadratic Hamiltonian

```text
P_A(f) = 1/2 Omega(Af, f).
```

Expand in Darboux coordinates `q,p`.

### 9.3 Quantized differential operator

Represent a finite quadratic differential operator as terms:

```rust
pub enum DiffOpTerm {
    Constant(RatFun),
    Mul { var: QVar, coeff: RatFun },
    Deriv { var: QVar, coeff: RatFun },
    MulMul { a: QVar, b: QVar, coeff: RatFun },
    MulDeriv { mul: QVar, deriv: QVar, coeff: RatFun },
    DerivDeriv { a: QVar, b: QVar, coeff: RatFun },
}
```

Do not apply this to unrestricted power series. Only use it with finite truncation.

### 9.4 Testing against graph expansion

For tiny cases, compare:

```text
operator expansion of exp(hat r) prod tau_KdV
```

with

```text
Givental graph expansion
```

for low genus and low markings.

---

## 10. Validation strategy

### 10.1 Algebra tests

```text
polynomial multiplication and reduction modulo P(H)
rational normalization invariants
series truncation correctness
matrix inverse mod q^{D+1}
```

### 10.2 Cohomology tests

```text
phi_i|_{p_j} = delta_ij
sum_i phi_i = 1
(phi_i, phi_j) = delta_ij / e_i
prod_i (H - lambda_i) = 0 classically
prod_i (H - lambda_i) = q quantumly
```

### 10.3 Localization tests

Start with:

```text
P^1, genus 0, low degree
P^2, genus 0, low degree
string/dilaton/divisor equations
```

Known checks:

```text
< H^n, H^n >_{0,1}^{P^n} = 1   // line through two points in P^n, interpreted appropriately
P^2 rational plane curve numbers for genus 0 primary point insertions
constant maps reduce to integrals over Mbar_g,m times integral over P^n
```

### 10.4 Frobenius tests

```text
idempotents multiply correctly
Psi^T eta Psi = I
R(z)R^T(-z)=I
Dubrovin flatness equation holds to truncation
```

### 10.5 Givental tests

```text
R = I, T = 0 gives product of KdV theories
edge propagator polynomial division works
translation sign matches string/dilaton equations
Givental engine matches localization for grid:
  n in {1,2}
  g <= 3
  d <= 3
  markings <= 5
```

### 10.6 Differential equation checks

Implement automatic checks:

```text
String:
<tau_0(1) prod tau_{k_i}(gamma_i)> = sum_i <tau_{k_i-1}(gamma_i) prod_{j!=i} ...>

Dilaton:
<tau_1(1) prod> = (2g - 2 + m) <prod>

Divisor for H:
<tau_0(H) prod>_{g,d} = d <prod> + descendant correction terms
```

These tests catch most convention errors.

---

## 11. Performance design

### 11.1 Parallelization

Parallelize at the graph level:

```rust
use rayon::prelude::*;

graphs.par_iter()
    .map(|g| contribution(g))
    .reduce(RatFun::zero, |a,b| a+b)
```

Do not parallelize inside polynomial arithmetic until graph-level parallelism is exhausted.

### 11.2 Memoization

Memoize:

```text
stable graphs by (g,n_legs)
localization graphs by (target n,g,d,m)
automorphism sizes by canonical graph label
psi/hodge integrals
R-matrix coefficients by (n,q_degree,z_order,calibration)
Psi/Delta/idempotents by (n,q_degree)
edge propagators by R hash and max psi orders
leg expansions by insertion and max psi order
```

Use `dashmap` or `parking_lot::RwLock<HashMap<...>>`.

### 11.3 Expression swell control

Rules:

```text
1. Work in fixed-point/canonical basis as long as possible.
2. Delay conversion to H-basis.
3. Use q-truncation everywhere.
4. Use dimension pruning before algebra multiplication.
5. Avoid full gcd normalization inside inner loops.
6. Combine like graph terms before full simplification.
7. Extract target q-degree before nonequivariant limit.
```

### 11.4 Serialization

Cache expensive objects to disk:

```text
~/.cache/gw-pn/
  psi_integrals.bincode
  hodge_integrals.bincode
  r_matrix/pn_n2_q5_z10.bin
  graphs/g3_n8.bin
```

Use versioned cache headers:

```rust
struct CacheHeader {
    package_version: String,
    convention_hash: u64,
    algebra_version: u32,
}
```

If conventions change, invalidate caches.

---

## 12. Implementation roadmap

### Phase 1: exact algebra and projective-space cohomology

Deliverables:

```text
RatFun over lambda_i and q
P^n cohomology ring reduction
fixed-point restriction
pairing
basic tests
```

### Phase 2: psi integral oracle

Deliverables:

```text
Witten-Kontsevich psi integrals
string/dilaton/DVV recursion
cache
```

### Phase 3: genus-zero localization

Deliverables:

```text
localization graph enumeration for g=0
edge/vertex/leg factors
primary + descendant insertions
P^1 and P^2 tests
```

### Phase 4: full localization

Deliverables:

```text
genus > 0 localization
Hodge integral backend API
unstable vertex handling
automorphism factors
nonequivariant limit
```

### Phase 5: Frobenius data

Deliverables:

```text
quantum product companion matrix
root expansions x_i(q)
idempotents
Delta_i
Psi
sanity tests
```

### Phase 6: R-matrix

Deliverables:

```text
R from I/J/S matrix or Dubrovin recursion
R inverse
unitarity tests
edge propagator generation
translation generation
```

### Phase 7: Givental graph engine

Deliverables:

```text
stable graph enumeration
colored graph expansion
leg/edge/translation expansions
psi-integral vertex evaluation
coefficient extraction
comparison with localization
```

### Phase 8: CLI, caching, benchmarks

Deliverables:

```text
CLI compute/compare/r-matrix commands
disk cache
benchmark suite
profiling report
```

---

## 13. Rust API sketch

```rust
pub mod algebra;
pub mod geometry;
pub mod localization;
pub mod frobenius;
pub mod givental;
pub mod tautological;
pub mod validation;

pub fn compute(req: InvariantRequest) -> Result<InvariantResult, GwError> {
    match req.mode {
        ComputeMode::Localization => localization::compute(&req),
        ComputeMode::Givental => givental::compute(&req),
        ComputeMode::CompareLocalizationAndGivental => {
            let a = localization::compute(&req)?;
            let b = givental::compute(&req)?;
            validation::assert_equal(&a, &b)?;
            Ok(a.with_comparison(b))
        }
    }
}
```

### Core request types

```rust
pub struct InvariantRequest {
    pub n: usize,
    pub genus: usize,
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub equivariant: bool,
    pub mode: ComputeMode,
    pub truncation: Option<Truncation>,
}

pub struct Insertion {
    pub descendant_power: usize,
    pub class: CohomologyClass,
}

pub enum ComputeMode {
    Localization,
    Givental,
    CompareLocalizationAndGivental,
}
```

### Error types

```rust
pub enum GwError {
    DimensionMismatch,
    MissingHodgeIntegralBackend,
    NonSemisimplePoint,
    TruncationTooLow,
    ConventionMismatch,
    AlgebraFailure(String),
    ValidationFailure(String),
}
```

---

## 14. Important convention traps

The coding agent should treat these as red flags:

1. **`Psi` orientation:** decide whether `flat = Psi * canonical` or `canonical = Psi * flat`; never mix.
2. **`Delta_i` convention:** metric norm vs inverse metric norm changes vertex weights.
3. **Dilaton shift:** `q = t - z*1`; missing this breaks string/dilaton equations.
4. **`R` vs `R^{-1}` on legs:** depends on whether using CohFT action, ancestor potential, or descendant potential.
5. **Translation sign:** test against genus-zero string equation.
6. **Edge propagator denominator:** must cancel; if not, unitarity or orientation is wrong.
7. **Nonequivariant limit:** never specialize weights before summing graphs.
8. **Automorphisms:** labelled ordinary markings and indistinguishable translation leaves have different automorphism treatment.
9. **Unstable vertices:** localization and Givental formulas both need explicit unstable conventions.
10. **Degree extraction:** Givental coefficients are q-series; extract degree after the full graph sum.

---

## 15. Minimal viable correctness target

The first complete MVP should support:

```text
P^1 and P^2
genus <= 2
degree <= 3
descendant powers <= 3
primary and descendant insertions in H-basis
localization and Givental comparison
exact rational output
```

Only after this should the package optimize for high genus.

---

## 16. References and source anchors

Primary references:

1. Y.-P. Lee and R. Pandharipande, *Frobenius manifolds, Gromov-Witten theory, and Virasoro constraints, Part 1: Frobenius manifolds and Givental's formula*. The table of contents explicitly covers Frobenius manifolds, semisimplicity, canonical coordinates, localization, materialization, and Givental's higher-genus formula.
2. Y.-P. Lee and R. Pandharipande, *Part 2: Quantization and Virasoro*. The table of contents explicitly covers quantization, Givental's descendant formula, `R`/`J` calibrations, and descendant potentials.
3. C. Teleman, *The structure of 2D semi-simple field theories*, for the modern semisimple CohFT classification viewpoint.
4. A. Givental, original papers on semisimple Frobenius structures and quantization.

Implementation seed:

- The prior pipeline design in `Givental Quantization Pipeline.txt` proposed two interoperating engines, shared algebra/graph/tautological infrastructure, localization graph sums, Frobenius data, `R`-matrix computation, quantization, Feynman graph expansion, and cross-validation. This document expands that seed into a Rust implementation specification.

---

## 17. Recommended first task for the coding agent

Implement the following first, in order:

```text
1. gw-algebra RatFun with variables lambda_0,...,lambda_n,q,z.
2. gw-pn-geometry fixed-point basis and restriction maps.
3. psi_integral oracle for pure psi intersections.
4. stable graph generator for Givental graphs, without R, with R=I and T=0.
5. Verify product of KdV tau-functions.
6. Add FrobeniusData for P^1 up to q^2.
7. Add R-matrix placeholder with unitarity checks.
8. Add edge propagator and leg expansion.
9. Compare the simplest P^1 invariants with localization.
```

This ordering gives fast feedback and prevents spending weeks on a symbolic `R`-matrix before the graph expansion and conventions are correct.
