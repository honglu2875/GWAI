use crate::algebra::{lambda, q, RatFun};
use crate::error::GwError;

use super::quotient::OneGeneratorQuotient;
use super::univariate::UniPoly;

/// Ordinary equivariant projective-space relation
///
/// `P(x)=prod_{a=0}^n (x-lambda_a)-q`.
///
/// The canonical coordinates of `QH_T(P^n)` are the roots of this relation,
/// and `P'(u_i)` is the corresponding canonical metric denominator in the
/// unnormalized idempotent frame.
pub fn projective_relation(n: usize) -> UniPoly {
    let x = UniPoly::variable();
    let mut relation = UniPoly::one();
    for index in 0..=n {
        let factor = x.sub(&UniPoly::constant(lambda(index)));
        relation = relation.mul(&factor);
    }
    relation.sub(&UniPoly::constant(q()))
}

pub fn projective_quotient(n: usize) -> Result<OneGeneratorQuotient, GwError> {
    OneGeneratorQuotient::new_monic(projective_relation(n))
}

/// Contract `sum_i f(u_i) / P'(u_i)` for an arbitrary polynomial `f`.
pub fn projective_residue_polynomial(n: usize, polynomial: &UniPoly) -> Result<RatFun, GwError> {
    let quotient = projective_quotient(n)?;
    quotient.residue_sum(polynomial)
}

/// Convenience wrapper for `sum_i u_i^power / P'(u_i)`.
pub fn projective_residue_monomial(n: usize, power: usize) -> Result<RatFun, GwError> {
    projective_residue_polynomial(n, &UniPoly::variable().pow_usize(power))
}

/// Contract `sum_i f(u_i)` for an arbitrary polynomial `f`.
pub fn projective_trace_polynomial(n: usize, polynomial: &UniPoly) -> Result<RatFun, GwError> {
    let quotient = projective_quotient(n)?;
    quotient.trace(polynomial)
}

/// Convenience wrapper for `sum_i u_i^power`.
pub fn projective_trace_monomial(n: usize, power: usize) -> Result<RatFun, GwError> {
    projective_trace_polynomial(n, &UniPoly::variable().pow_usize(power))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projective_relation_has_expected_degree() {
        let relation = projective_relation(2);
        assert_eq!(relation.degree(), Some(3));
        assert_eq!(relation.leading_coeff(), Some(RatFun::one()));
    }

    #[test]
    fn projective_residue_pairing_picks_top_remainder_coefficient() {
        assert_eq!(projective_residue_monomial(2, 0).unwrap(), RatFun::zero());
        assert_eq!(projective_residue_monomial(2, 1).unwrap(), RatFun::zero());
        assert_eq!(projective_residue_monomial(2, 2).unwrap(), RatFun::one());
    }

    #[test]
    fn projective_residue_sees_lambda_symmetric_coefficient() {
        let expected = &(&lambda(0) + &lambda(1)) + &lambda(2);
        assert_eq!(projective_residue_monomial(2, 3).unwrap(), expected);
    }

    #[test]
    fn projective_residue_accepts_general_polynomials() {
        let x = UniPoly::variable();
        let polynomial = x.pow_usize(2).add(&x.pow_usize(3));
        let expected = &RatFun::one() + &(&(&lambda(0) + &lambda(1)) + &lambda(2));
        assert_eq!(
            projective_residue_polynomial(2, &polynomial).unwrap(),
            expected
        );
    }

    #[test]
    fn projective_trace_recovers_first_power_sum() {
        let expected = &lambda(0) + &lambda(1);
        assert_eq!(projective_trace_monomial(1, 1).unwrap(), expected);
    }
}
