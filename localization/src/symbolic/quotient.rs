use crate::core::algebra::RatFun;
use crate::core::error::GwError;

use super::univariate::UniPoly;

/// A one-generator semisimple algebra `K[x]/(P(x))`.
///
/// For ordinary `P^n`, `P(x)=prod_a(x-lambda_a)-q`.  For twisted
/// one-generator theories the same type can be fed the characteristic relation
/// extracted from quantum multiplication by `H`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneGeneratorQuotient {
    relation: UniPoly,
    rank: usize,
}

impl OneGeneratorQuotient {
    /// Construct a quotient by a monic relation.
    pub fn new_monic(relation: UniPoly) -> Result<Self, GwError> {
        let Some(rank) = relation.degree() else {
            return Err(GwError::AlgebraFailure(
                "one-generator quotient needs a nonzero relation".to_string(),
            ));
        };
        if rank == 0 {
            return Err(GwError::AlgebraFailure(
                "one-generator quotient relation must have positive degree".to_string(),
            ));
        }
        if relation.leading_coeff() != Some(RatFun::one()) {
            return Err(GwError::AlgebraFailure(
                "one-generator quotient relation must be monic".to_string(),
            ));
        }
        Ok(Self { relation, rank })
    }

    pub fn relation(&self) -> &UniPoly {
        &self.relation
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Reduce a polynomial to the basis `1,x,...,x^{rank-1}`.
    pub fn reduce(&self, poly: &UniPoly) -> Result<UniPoly, GwError> {
        Ok(poly.div_rem(&self.relation)?.1)
    }

    /// Multiply two quotient elements and return their normal form.
    pub fn multiply(&self, left: &UniPoly, right: &UniPoly) -> Result<UniPoly, GwError> {
        self.reduce(&left.mul(right))
    }

    /// Invert a denominator in `K[x]/(P)`.
    ///
    /// This is the operation needed to evaluate rational functions at canonical
    /// roots without introducing the roots themselves.  It fails precisely when
    /// the denominator has a common factor with the relation.
    pub fn invert(&self, denominator: &UniPoly) -> Result<UniPoly, GwError> {
        if denominator.is_zero() {
            return Err(GwError::AlgebraFailure(
                "zero denominator is not invertible in quotient".to_string(),
            ));
        }
        let (gcd, bezout_den, _) = extended_gcd(denominator, &self.relation)?;
        let Some(gcd_constant) = gcd.as_constant() else {
            return Err(GwError::AlgebraFailure(
                "denominator is not invertible modulo the relation".to_string(),
            ));
        };
        if gcd_constant.is_zero() {
            return Err(GwError::AlgebraFailure(
                "zero gcd while inverting quotient denominator".to_string(),
            ));
        }
        let inv_gcd = &RatFun::one() / &gcd_constant;
        self.reduce(&bezout_den.scale(&inv_gcd))
    }

    /// Reduce `numerator/denominator` to a quotient normal form.
    pub fn reduce_rational(
        &self,
        numerator: &UniPoly,
        denominator: &UniPoly,
    ) -> Result<UniPoly, GwError> {
        let inverse = self.invert(denominator)?;
        self.multiply(numerator, &inverse)
    }

    /// Trace of multiplication by a quotient element.
    ///
    /// For a square-free relation this is `sum_i f(u_i)` after reducing `f`.
    pub fn trace(&self, poly: &UniPoly) -> Result<RatFun, GwError> {
        let normal = self.reduce(poly)?;
        let mut total = RatFun::zero();
        for basis_degree in 0..self.rank {
            let basis = UniPoly::monomial(basis_degree, RatFun::one());
            let column = self.multiply(&normal, &basis)?;
            total = &total + &column.coeff(basis_degree);
        }
        Ok(total)
    }

    /// Trace of a rational function `numerator/denominator`.
    pub fn trace_rational(
        &self,
        numerator: &UniPoly,
        denominator: &UniPoly,
    ) -> Result<RatFun, GwError> {
        self.trace(&self.reduce_rational(numerator, denominator)?)
    }

    /// Residue-weighted root sum.
    ///
    /// For monic `P` of degree `r`, the residue theorem gives
    ///
    /// `sum_i f(u_i)/P'(u_i) = [x^{r-1}] (f(x) mod P(x))`.
    ///
    /// This is the basic color-sum contraction behind many `P^n` graph
    /// factors.
    pub fn residue_sum(&self, numerator: &UniPoly) -> Result<RatFun, GwError> {
        let normal = self.reduce(numerator)?;
        Ok(normal.coeff(self.rank - 1))
    }

    /// Residue-weighted root sum of `numerator/denominator`.
    pub fn residue_sum_rational(
        &self,
        numerator: &UniPoly,
        denominator: &UniPoly,
    ) -> Result<RatFun, GwError> {
        self.residue_sum(&self.reduce_rational(numerator, denominator)?)
    }
}

fn extended_gcd(left: &UniPoly, right: &UniPoly) -> Result<(UniPoly, UniPoly, UniPoly), GwError> {
    let mut old_r = left.clone();
    let mut r = right.clone();
    let mut old_s = UniPoly::one();
    let mut s = UniPoly::zero();
    let mut old_t = UniPoly::zero();
    let mut t = UniPoly::one();

    while !r.is_zero() {
        let (quotient, remainder) = old_r.div_rem(&r)?;
        old_r = r;
        r = remainder;

        let next_s = old_s.sub(&quotient.mul(&s));
        old_s = s;
        s = next_s;

        let next_t = old_t.sub(&quotient.mul(&t));
        old_t = t;
        t = next_t;
    }

    Ok((old_r, old_s, old_t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algebra::{q, RatFun};

    fn square_root_quotient() -> OneGeneratorQuotient {
        let relation = UniPoly::variable()
            .pow_usize(2)
            .sub(&UniPoly::constant(q()));
        OneGeneratorQuotient::new_monic(relation).unwrap()
    }

    #[test]
    fn quotient_reduces_by_relation() {
        let quotient = square_root_quotient();
        let x = UniPoly::variable();
        assert_eq!(
            quotient.reduce(&x.pow_usize(2)).unwrap(),
            UniPoly::constant(q())
        );
    }

    #[test]
    fn quotient_inverts_denominators_mod_relation() {
        let quotient = square_root_quotient();
        let x = UniPoly::variable();
        let inverse = quotient.invert(&x).unwrap();
        assert_eq!(
            quotient.multiply(&x, &inverse).unwrap(),
            UniPoly::one(),
            "x * x/q should be one modulo x^2-q"
        );
    }

    #[test]
    fn residue_sum_uses_top_coefficient_normal_form() {
        let quotient = square_root_quotient();
        let x = UniPoly::variable();
        assert_eq!(
            quotient.residue_sum(&UniPoly::one()).unwrap(),
            RatFun::zero()
        );
        assert_eq!(quotient.residue_sum(&x).unwrap(), RatFun::one());
    }

    #[test]
    fn rational_residue_sum_inverts_denominator_first() {
        let quotient = square_root_quotient();
        let x = UniPoly::variable();
        let value = quotient.residue_sum_rational(&UniPoly::one(), &x).unwrap();
        assert_eq!(&value * &q(), RatFun::one());
    }

    #[test]
    fn trace_matches_sum_over_roots_for_basic_powers() {
        let quotient = square_root_quotient();
        let x = UniPoly::variable();
        assert_eq!(
            quotient.trace(&UniPoly::one()).unwrap(),
            RatFun::from(2usize)
        );
        assert_eq!(quotient.trace(&x).unwrap(), RatFun::zero());
        assert_eq!(
            quotient.trace(&x.pow_usize(2)).unwrap(),
            &RatFun::from(2usize) * &q()
        );
    }
}
