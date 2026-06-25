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

- `basis.rs`: glossary for the raw components `S`, `PsiInv`, `RInv`,
  translation `T`, `Delta`, `EtaInv`, and point-theory psi integrals.
- `expansion.rs`: optional engine-specific dictionaries that explain how the
  universal basis elements are read for ordinary `P^n` or negative split
  twists.  The rational basis uses these dictionaries to display calibration
  data such as canonical roots, hypergeometric/Birkhoff `S`, twisted pairings,
  and QRR `R`-recursions.
- `skeleton.rs`: fixed `(g,m)` formula skeletons, including finite truncation
  orders, stable graph metadata, and expanded graph terms using raw basis
  coefficients or packed resolvent/rational kernels.  Marking factors in raw
  mode are expanded into `S/PsiInv/RInv`, and edge factors are expanded into
  `RInv/EtaInv`.  The same skeleton can be rendered
  as plain text, as a TeX fragment, or as a standalone TeX document using
  standard symbols
  \(S_s\), \(R_r^{-1}\), \(\Psi^{-1}\), \((T_p)_i\), \(\Delta_i\), and
  point-theory intersection brackets.  TeX graph sections include TikZ
  drawings generated directly from the stable-graph vertices, edges, loops,
  and labelled markings.  The renderer wraps long displays itself: compact
  graph brackets use `multlined` (from `mathtools`) and the fully expanded
  basis sums use a page-breakable `align*`, so nothing runs past the margin or
  off the bottom of a page.  It avoids giant `\left...\right` delimiter pairs.

The raw basis deliberately keeps the coefficient data symbolic.  The resolvent
basis packs descendants and edge propagators into generating kernels, while the
rational basis displays how those kernels are read in the ordinary or twisted
calibration.

Sample commands:

```bash
# Universal raw graph formula.  --twist is accepted but ignored in this mode.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis raw \
  --twist -3 \
  --format tex-fragment

# Packed descendants with insertion variables z_l.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis resolvent \
  --format tex-fragment

# Standalone TeX document with ordinary P^2 rational/root-sum calibration.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis rational \
  --format tex

# Standalone TeX document with local P^2 = O(-3) over P^2 rational calibration.
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --twist -3 \
  --basis rational \
  --format tex
```
