# Formula explanations

This folder is the human-facing companion to the fast Givental graph evaluator.
It is meant for mathematical inspection: which finite pieces of the
Givental-Teleman reconstruction are used, how descendant insertions are moved
to ancestors, and how the stable graph sum is assembled before substituting a
specific `P^n` or twisted calibration.

The production evaluator in `src/givental.rs` should remain the place for fast
contraction.  Code here can be more verbose and explicit because its job is to
help humans trace definitions and intermediate formulas.

Main files:

- `basis.rs`: glossary for the coefficient components `S`, `PsiInv`, `RInv`,
  translation `T`, `Delta`, `EtaInv`, and point-theory psi integrals.
- `expansion.rs`: optional engine-specific dictionaries that explain how the
  universal basis elements are read for ordinary `P^n` or negative split
  twists.  The raw basis uses these dictionaries to display calibration data
  such as canonical roots, hypergeometric/Birkhoff `S`, twisted pairings, and
  QRR `R`-recursions.
- `skeleton.rs`: fixed `(g,m)` formula skeletons, including finite truncation
  orders, stable graph metadata, and expanded graph terms using the legacy
  coefficient basis or packed raw/rational kernels.  Marking factors in
  coefficient mode are expanded into `S/PsiInv/RInv`, and edge factors are
  expanded into `RInv/EtaInv`.  The same skeleton can be rendered
  as plain text, as a TeX fragment, or as a standalone TeX document using
  standard symbols
  \(S_s\), \(R_r^{-1}\), \(\Psi^{-1}\), \((T_p)_i\), \(\Delta_i\), and
  point-theory intersection brackets.  TeX graph sections include TikZ
  drawings generated directly from the stable-graph vertices, edges, loops,
  and labelled markings.  The renderer wraps long displays itself: compact
  graph brackets use `multlined` (from `mathtools`) and the fully expanded
  basis sums use a page-breakable `align*`, so nothing runs past the margin or
  off the bottom of a page.  It avoids giant `\left...\right` delimiter pairs.

The coefficient basis deliberately keeps the lowest-level coefficient data
symbolic.  The raw and rational bases keep descendant variables packed as
resolvent insertions and substitute the leg and edge kernel formulas into each
graph bracket.  The raw basis emphasizes the selected ordinary or twisted
calibration data.  The rational basis is the quotient-reduction path: every
graph is displayed as a packed Laurent expression in the `z_l` variables, and
additional rational contractions are layered on where implemented.  Without a
q-degree, its symbolic seed case is the ordinary `P^n` genus-zero
three-primary one-vertex graph: the root sum `sum_{P(u)=0} f(u)/P'(u)` is
reduced as the top coefficient of `f(H)` modulo `prod_a(H-lambda_a)-q`.  When
a q-degree is supplied with `--d`, rational mode also prints bounded
coefficient extractions from the same S/R/T graph kernel.  These contracted
expressions use labelled variables `x_{ell,k,a}` for `tau_k(H^a)` at marking
`ell`.

Sample commands:

```bash
# Legacy coefficient graph formula.  --twist is accepted but ignored in this mode.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis coefficients \
  --twist -3 \
  --format tex-fragment

# Packed rational descendants with insertion variables z_l.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis rational \
  --format tex-fragment

# Standalone TeX document with ordinary P^2 raw root-sum calibration.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis raw \
  --format tex

# Standalone TeX document with local P^2 = O(-3) over P^2 raw calibration.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --twist -3 \
  --basis raw \
  --format tex

# Quotient-reduced ordinary P^2 primary three-point expression.
cargo run --quiet -- formula --n 2 --g 0 --markings 3 \
  --basis rational \
  --d 1 \
  --format text
```
