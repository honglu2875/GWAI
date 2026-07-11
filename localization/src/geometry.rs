//! Equivariant cohomology of projective space in the hyperplane basis.
//!
//! Classes are stored as coefficients of `1,H,...,H^n`.  Multiplication is
//! reduction modulo either the classical relation `prod_i(H-lambda_i)=0` or the
//! small quantum relation `prod_i(H-lambda_i)=q`.  Fixed-point restrictions and
//! Atiyah-Bott pairings are used throughout the Frobenius and calibration code.

use crate::algebra::{lambda, q, RatFun, Rational};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohomologyClass {
    n: usize,
    coeffs: Vec<RatFun>,
}

impl CohomologyClass {
    pub fn new(n: usize, coeffs: Vec<RatFun>) -> Self {
        let mut normalized = coeffs;
        normalized.resize(n + 1, RatFun::zero());
        normalized.truncate(n + 1);
        Self {
            n,
            coeffs: normalized,
        }
    }

    pub fn zero(n: usize) -> Self {
        Self::new(n, vec![RatFun::zero(); n + 1])
    }

    pub fn one(n: usize) -> Self {
        Self::h_power(n, 0)
    }

    pub fn h_power(n: usize, power: usize) -> Self {
        let mut coeffs = vec![RatFun::zero(); n + 1];
        if power <= n {
            coeffs[power] = RatFun::one();
        }
        Self::new(n, coeffs)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn coeffs(&self) -> &[RatFun] {
        &self.coeffs
    }

    pub fn pure_power(&self) -> Option<usize> {
        let mut found = None;
        for (idx, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if !coeff.is_one() || found.is_some() {
                return None;
            }
            found = Some(idx);
        }
        found
    }

    pub fn restrict_to_fixed_point(&self, fixed_point: usize) -> RatFun {
        // Localization restriction: H evaluates to lambda_i at the i-th
        // torus-fixed point.
        assert!(fixed_point <= self.n, "fixed point index out of range");
        let l = lambda(fixed_point);
        let mut total = RatFun::zero();
        for (power, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            let term = coeff * &l.pow_usize(power);
            total = &total + &term;
        }
        total
    }

    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.n, rhs.n, "cohomology classes have different targets");
        let coeffs = self
            .coeffs
            .iter()
            .zip(rhs.coeffs.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.n, coeffs)
    }

    pub fn scale(&self, scalar: &RatFun) -> Self {
        let coeffs = self.coeffs.iter().map(|c| c * scalar).collect();
        Self::new(self.n, coeffs)
    }

    pub fn multiply_classical_equivariant(&self, rhs: &Self) -> Self {
        self.multiply_with_relation(rhs, false)
    }

    pub fn multiply_quantum_equivariant(&self, rhs: &Self) -> Self {
        self.multiply_with_relation(rhs, true)
    }

    fn multiply_with_relation(&self, rhs: &Self, quantum: bool) -> Self {
        // Multiply polynomials in H, then reduce the high powers by the chosen
        // projective-space relation.
        assert_eq!(self.n, rhs.n, "cohomology classes have different targets");
        let mut product = vec![RatFun::zero(); 2 * self.n + 1];
        for (left_power, left_coeff) in self.coeffs.iter().enumerate() {
            if left_coeff.is_zero() {
                continue;
            }
            for (right_power, right_coeff) in rhs.coeffs.iter().enumerate() {
                if right_coeff.is_zero() {
                    continue;
                }
                let term = left_coeff * right_coeff;
                product[left_power + right_power] = &product[left_power + right_power] + &term;
            }
        }
        Self::new(self.n, reduce_h_polynomial(self.n, product, quantum))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivariantProjectiveSpace {
    pub n: usize,
}

impl EquivariantProjectiveSpace {
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    pub fn weights(&self) -> impl Iterator<Item = RatFun> + '_ {
        (0..=self.n).map(lambda)
    }

    pub fn fixed_point_euler(&self, fixed_point: usize) -> RatFun {
        // e_T(T_{p_i} P^n) = prod_{j != i} (lambda_i - lambda_j).
        assert!(fixed_point <= self.n, "fixed point index out of range");
        let li = lambda(fixed_point);
        let mut product = RatFun::one();
        for j in 0..=self.n {
            if j == fixed_point {
                continue;
            }
            let factor = &li - &lambda(j);
            product = &product * &factor;
        }
        product
    }

    pub fn fixed_point_idempotent(&self, fixed_point: usize) -> CohomologyClass {
        // Lagrange idempotent supported at p_i:
        // prod_{j != i}(H-lambda_j) / prod_{j != i}(lambda_i-lambda_j).
        assert!(fixed_point <= self.n, "fixed point index out of range");
        let mut coeffs = vec![RatFun::one()];
        for j in 0..=self.n {
            if j == fixed_point {
                continue;
            }
            let minus_lambda = -lambda(j);
            let mut next = vec![RatFun::zero(); coeffs.len() + 1];
            for (degree, coeff) in coeffs.iter().enumerate() {
                next[degree] = &next[degree] + &(coeff * &minus_lambda);
                next[degree + 1] = &next[degree + 1] + coeff;
            }
            coeffs = next;
        }
        let euler = self.fixed_point_euler(fixed_point);
        let coeffs = coeffs.iter().map(|c| c / &euler).collect();
        CohomologyClass::new(self.n, coeffs)
    }

    pub fn pairing(&self, a: &CohomologyClass, b: &CohomologyClass) -> RatFun {
        // Atiyah-Bott integration formula:
        // integral a b = sum_i a|_{p_i} b|_{p_i}/e_T(T_{p_i}P^n).
        assert_eq!(a.n(), self.n);
        assert_eq!(b.n(), self.n);
        let mut total = RatFun::zero();
        for i in 0..=self.n {
            let numerator = &a.restrict_to_fixed_point(i) * &b.restrict_to_fixed_point(i);
            let term = &numerator / &self.fixed_point_euler(i);
            total = &total + &term;
        }
        total
    }

    pub fn classical_integral_of_pure_powers(&self, powers: &[usize]) -> Rational {
        let sum: usize = powers.iter().sum();
        if sum == self.n {
            Rational::one()
        } else {
            Rational::zero()
        }
    }

    pub fn genus_zero_three_point_primary(&self, powers: &[usize; 3], degree: usize) -> Rational {
        let sum: usize = powers.iter().sum();
        if sum == self.n + (self.n + 1) * degree {
            Rational::one()
        } else {
            Rational::zero()
        }
    }

    pub fn quantum_relation_rhs(&self) -> Vec<RatFun> {
        h_power_relation_rhs(self.n, true)
    }

    pub fn classical_relation_rhs(&self) -> Vec<RatFun> {
        h_power_relation_rhs(self.n, false)
    }
}

pub fn elementary_symmetric_weights(n: usize) -> Vec<RatFun> {
    let mut elementary = vec![RatFun::zero(); n + 2];
    elementary[0] = RatFun::one();
    for i in 0..=n {
        let li = lambda(i);
        for k in (1..=i + 1).rev() {
            let term = &elementary[k - 1] * &li;
            elementary[k] = &elementary[k] + &term;
        }
    }
    elementary
}

pub fn h_power_relation_rhs(n: usize, quantum: bool) -> Vec<RatFun> {
    // Rewrites H^{n+1} as a linear combination of lower powers from
    // prod_i(H-lambda_i) = q (or 0 in the classical case).
    let elementary = elementary_symmetric_weights(n);
    let mut rhs = vec![RatFun::zero(); n + 1];
    for k in 1..=n + 1 {
        let power = n + 1 - k;
        let signed = if k % 2 == 1 {
            elementary[k].clone()
        } else {
            -elementary[k].clone()
        };
        rhs[power] = &rhs[power] + &signed;
    }
    if quantum {
        rhs[0] = &rhs[0] + &q();
    }
    rhs
}

pub fn reduce_h_polynomial(n: usize, mut coeffs: Vec<RatFun>, quantum: bool) -> Vec<RatFun> {
    let relation = h_power_relation_rhs(n, quantum);
    while coeffs.len() > n + 1 {
        let degree = coeffs.len() - 1;
        let leading = coeffs.pop().unwrap();
        if leading.is_zero() {
            continue;
        }
        let shift = degree - (n + 1);
        for (power, relation_coeff) in relation.iter().enumerate() {
            if relation_coeff.is_zero() {
                continue;
            }
            let term = &leading * relation_coeff;
            coeffs[shift + power] = &coeffs[shift + power] + &term;
        }
    }
    coeffs.resize(n + 1, RatFun::zero());
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_idempotents_restrict_to_delta() {
        let p2 = EquivariantProjectiveSpace::new(2);
        for i in 0..=2 {
            let phi_i = p2.fixed_point_idempotent(i);
            for j in 0..=2 {
                let expected = if i == j {
                    RatFun::one()
                } else {
                    RatFun::zero()
                };
                assert_eq!(phi_i.restrict_to_fixed_point(j), expected);
            }
        }
    }

    #[test]
    fn fixed_point_pairing_is_diagonal() {
        let p1 = EquivariantProjectiveSpace::new(1);
        let phi0 = p1.fixed_point_idempotent(0);
        let phi1 = p1.fixed_point_idempotent(1);
        assert_eq!(
            p1.pairing(&phi0, &phi0),
            &RatFun::one() / &p1.fixed_point_euler(0)
        );
        assert_eq!(p1.pairing(&phi0, &phi1), RatFun::zero());
    }

    #[test]
    fn p1_excess_degree_pairing_is_an_equivariant_class() {
        let p1 = EquivariantProjectiveSpace::new(1);
        let h = CohomologyClass::h_power(1, 1);
        let expected = &lambda(0) + &lambda(1);
        assert!((&p1.pairing(&h, &h) - &expected).is_zero());
    }

    #[test]
    fn quantum_relation_reduces_h_power() {
        let p2 = EquivariantProjectiveSpace::new(2);
        let h = CohomologyClass::h_power(2, 1);
        let h2 = h.multiply_quantum_equivariant(&h);
        let h3 = h2.multiply_quantum_equivariant(&h);
        assert_eq!(h3.coeffs(), p2.quantum_relation_rhs().as_slice());
    }

    #[test]
    fn nonequivariant_relation_is_hn_plus_one_equals_q_after_zero_weights() {
        let rhs = h_power_relation_rhs(2, true);
        assert_eq!(rhs[0].to_string(), "lambda_0*lambda_1*lambda_2 + q");
        assert_eq!(
            rhs[1].to_string(),
            "-lambda_0*lambda_1 - lambda_0*lambda_2 - lambda_1*lambda_2"
        );
        assert_eq!(rhs[2].to_string(), "lambda_0 + lambda_1 + lambda_2");
    }
}
