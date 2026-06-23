# Zinger Projective-Space Genus-Zero Path

This module is an independent implementation of a small, explicitly tested
subset of Aleksey Zinger's projective-space formulas from
arXiv:1106.1633, "The Genus 0 Gromov-Witten Invariants of Projective Complete
Intersections".

Implemented:

- genus-zero degree-zero projective-space constant maps;
- the projective-space specialization of Zinger's vanishing criterion
  (Theorem 2);
- the full 3-point projective-space generating-function extraction from
  Theorem 4;
- the explicitly stated primary 4-point degree-one line count and degree-two
  vanishing consequences following Theorem 4.

Not implemented yet:

- the general 4-point descendant extraction from Theorem 4;
- the general N-point structure-constant formula of Theorem A;
- complete-intersection twists with nonempty multi-degree `a`;
- equivariant output.

The code deliberately does not call the crate's Givental `S/R` evaluator.  It is
intended as a cross-check path for genus-zero projective-space invariants.
