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

- `atoms.rs`: glossary for the primitive components `S`, `PsiInv`, `RInv`,
  translation `T`, `Delta`, `EtaInv`, and point-theory psi integrals.
- `skeleton.rs`: fixed `(g,m)` formula skeletons, including finite truncation
  orders, stable graph metadata, and expanded graph terms using primitive atom
  coefficients.  Marking factors are expanded into `S/PsiInv/RInv`, and edge
  factors are expanded into `RInv/EtaInv`.  The same skeleton can be rendered
  as plain text or as a TeX fragment using standard symbols
  \(S_s\), \(R_r^{-1}\), \(\Psi^{-1}\), \((T_p)_i\), \(\Delta_i\), and
  point-theory intersection brackets.

The first renderer deliberately keeps calibration data symbolic.  Later stages
can add substitution modes that print actual truncated `R_k`, `S_k`, `T_k`,
`Psi`, and `Delta` data for the ordinary or twisted providers.
