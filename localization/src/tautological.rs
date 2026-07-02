//! Tautological integrals over moduli spaces of stable curves.
//!
//! The Givental graph expansion reduces every vertex to point-theory psi
//! intersections on `Mbar_{g,n}`.  This module supplies those integrals through
//! Witten-Kontsevich recursion and leaves an explicit table boundary for Hodge
//! integrals needed by direct localization-style checks.

use crate::algebra::Rational;
use std::collections::HashMap;
use std::sync::Mutex;

pub trait TautologicalOracle: Send + Sync {
    /// Integral of `prod_i psi_i^{powers[i]}` over `Mbar_{g,n}`.
    fn psi_integral(&self, genus: usize, powers: &[usize]) -> Rational;

    /// Optional integral with lambda classes.  Implementors may return `None`
    /// when the table does not contain the requested Hodge monomial.
    fn hodge_integral(
        &self,
        _genus: usize,
        _psi_powers: &[usize],
        _lambda_powers: &[(usize, usize)],
    ) -> Option<Rational> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HodgeKey {
    pub genus: usize,
    pub psi_powers: Vec<usize>,
    pub lambda_powers: Vec<(usize, usize)>,
}

impl HodgeKey {
    /// Key for permutation-symmetric Hodge tables, where marked points are
    /// indistinguishable and psi powers can be sorted.
    pub fn symmetric(genus: usize, psi_powers: &[usize], lambda_powers: &[(usize, usize)]) -> Self {
        let mut psi_powers = psi_powers.to_vec();
        psi_powers.sort_unstable();
        let mut lambda_powers = lambda_powers.to_vec();
        lambda_powers.sort_unstable();
        Self {
            genus,
            psi_powers,
            lambda_powers,
        }
    }

    /// Key for labelled-marking tables, where psi powers are tied to specific
    /// markings and must not be sorted.
    pub fn labelled(genus: usize, psi_powers: &[usize], lambda_powers: &[(usize, usize)]) -> Self {
        let mut lambda_powers = lambda_powers.to_vec();
        lambda_powers.sort_unstable();
        Self {
            genus,
            psi_powers: psi_powers.to_vec(),
            lambda_powers,
        }
    }
}

#[derive(Debug, Default)]
pub struct TableTautologicalOracle {
    psi: WittenKontsevich,
    hodge: Mutex<HashMap<HodgeKey, Rational>>,
}

impl TableTautologicalOracle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_symmetric_hodge_integral(
        &self,
        genus: usize,
        psi_powers: &[usize],
        lambda_powers: &[(usize, usize)],
        value: Rational,
    ) {
        self.hodge
            .lock()
            .unwrap()
            .insert(HodgeKey::symmetric(genus, psi_powers, lambda_powers), value);
    }

    pub fn insert_labelled_hodge_integral(
        &self,
        genus: usize,
        psi_powers: &[usize],
        lambda_powers: &[(usize, usize)],
        value: Rational,
    ) {
        self.hodge
            .lock()
            .unwrap()
            .insert(HodgeKey::labelled(genus, psi_powers, lambda_powers), value);
    }
}

impl TautologicalOracle for TableTautologicalOracle {
    fn psi_integral(&self, genus: usize, powers: &[usize]) -> Rational {
        self.psi.psi_integral(genus, powers)
    }

    fn hodge_integral(
        &self,
        genus: usize,
        psi_powers: &[usize],
        lambda_powers: &[(usize, usize)],
    ) -> Option<Rational> {
        if lambda_powers.is_empty() {
            return Some(self.psi_integral(genus, psi_powers));
        }
        let labelled = HodgeKey::labelled(genus, psi_powers, lambda_powers);
        if let Some(value) = self.hodge.lock().unwrap().get(&labelled).cloned() {
            return Some(value);
        }
        let symmetric = HodgeKey::symmetric(genus, psi_powers, lambda_powers);
        self.hodge.lock().unwrap().get(&symmetric).cloned()
    }
}

#[derive(Debug, Default)]
pub struct WittenKontsevich {
    cache: Mutex<HashMap<(usize, Vec<usize>), Rational>>,
}

impl WittenKontsevich {
    pub fn new() -> Self {
        Self::default()
    }

    fn psi_sorted(&self, genus: usize, powers: Vec<usize>) -> Rational {
        let mut powers = powers;
        powers.sort_unstable();
        let key = (genus, powers.clone());
        if let Some(value) = self.cache.lock().unwrap().get(&key).cloned() {
            return value;
        }

        let value = self.psi_uncached(genus, &powers);
        self.cache.lock().unwrap().insert(key, value.clone());
        value
    }

    fn psi_uncached(&self, genus: usize, powers: &[usize]) -> Rational {
        // Dimension check on Mbar_{g,n}: only total psi degree 3g-3+n can
        // contribute.  The remaining cases are reduced by string, dilaton, and
        // finally DVV/Virasoro recursion.
        let n = powers.len();
        let dimension = 3isize * genus as isize - 3 + n as isize;
        if dimension < 0 {
            return Rational::zero();
        }
        let degree: usize = powers.iter().sum();
        if degree as isize != dimension {
            return Rational::zero();
        }

        if genus == 0 && powers == [0, 0, 0] {
            return Rational::one();
        }
        if genus == 1 && powers == [1] {
            return Rational::new(1, 24);
        }

        if let Some(pos) = powers.iter().position(|&p| p == 0) {
            return self.apply_string_equation(genus, powers, pos);
        }
        if let Some(pos) = powers.iter().position(|&p| p == 1) {
            return self.apply_dilaton_equation(genus, powers, pos);
        }

        self.apply_dvv(genus, powers)
    }

    fn apply_string_equation(&self, genus: usize, powers: &[usize], zero_pos: usize) -> Rational {
        // <tau_0 prod tau_d> = sum_j <tau_{d_j-1} prod_{i != j} tau_{d_i}>.
        let mut rest = powers.to_vec();
        rest.remove(zero_pos);
        let mut total = Rational::zero();
        for idx in 0..rest.len() {
            if rest[idx] == 0 {
                continue;
            }
            let mut next = rest.clone();
            next[idx] -= 1;
            total += self.psi_sorted(genus, next);
        }
        total
    }

    fn apply_dilaton_equation(&self, genus: usize, powers: &[usize], one_pos: usize) -> Rational {
        // <tau_1 prod tau_d> = (2g-2+n) <prod tau_d>, after removing tau_1.
        let mut rest = powers.to_vec();
        rest.remove(one_pos);
        let factor = 2isize * genus as isize - 2 + rest.len() as isize;
        if factor == 0 {
            Rational::zero()
        } else {
            Rational::from(factor as i128) * self.psi_sorted(genus, rest)
        }
    }

    fn apply_dvv(&self, genus: usize, powers: &[usize]) -> Rational {
        // DVV recursion for the last insertion tau_{d0}.  The first sum merges
        // tau_{d0} with another marking; the second sum is the boundary term:
        // one irreducible genus reduction plus all separating splittings.
        debug_assert!(powers.iter().all(|&p| p >= 2));
        let mut rest = powers.to_vec();
        let d0 = rest.pop().expect("DVV requires at least one insertion");

        let mut total = Rational::zero();

        for j in 0..rest.len() {
            let dj = rest[j];
            let coeff =
                double_factorial_odd(2 * (d0 + dj) - 1) / double_factorial_odd(2 * dj - 1);
            let mut next = rest.clone();
            next.remove(j);
            next.push(d0 + dj - 1);
            total += coeff * self.psi_sorted(genus, next);
        }

        if d0 >= 2 {
            for a in 0..=(d0 - 2) {
                let b = d0 - 2 - a;
                let coeff = double_factorial_odd(2 * a + 1) * double_factorial_odd(2 * b + 1);
                let mut bracket = Rational::zero();

                if genus > 0 {
                    let mut lower = vec![a, b];
                    lower.extend(rest.iter().copied());
                    bracket += self.psi_sorted(genus - 1, lower);
                }

                for genus_left in 0..=genus {
                    for mask in 0..(1usize << rest.len()) {
                        let mut left = vec![a];
                        let mut right = vec![b];
                        for (idx, power) in rest.iter().copied().enumerate() {
                            if (mask & (1usize << idx)) == 0 {
                                left.push(power);
                            } else {
                                right.push(power);
                            }
                        }
                        bracket += self.psi_sorted(genus_left, left)
                            * self.psi_sorted(genus - genus_left, right);
                    }
                }

                total += Rational::new(1, 2) * coeff * bracket;
            }
        }

        total / double_factorial_odd(2 * d0 + 1)
    }
}

impl TautologicalOracle for WittenKontsevich {
    fn psi_integral(&self, genus: usize, powers: &[usize]) -> Rational {
        self.psi_sorted(genus, powers.to_vec())
    }
}

fn double_factorial_odd(n: usize) -> Rational {
    // Exact big-rational accumulation: (2d+1)!! exceeds i128 already for
    // d >= 28, which one-point integrals reach around genus 10.
    debug_assert!(n % 2 == 1);
    let mut out = Rational::one();
    let mut k = n;
    while k > 1 {
        out = out * Rational::from(k);
        k -= 2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_intersections() {
        let wk = WittenKontsevich::new();
        assert_eq!(wk.psi_integral(0, &[0, 0, 0]), Rational::one());
        assert_eq!(wk.psi_integral(1, &[1]), Rational::new(1, 24));
    }

    #[test]
    fn genus_zero_formula_examples() {
        let wk = WittenKontsevich::new();
        assert_eq!(wk.psi_integral(0, &[1, 0, 0, 0]), Rational::one());
        assert_eq!(wk.psi_integral(0, &[2, 0, 0, 0, 0]), Rational::one());
        assert_eq!(wk.psi_integral(0, &[1, 1, 0, 0, 0]), Rational::from(2));
    }

    #[test]
    fn one_point_high_genus() {
        let wk = WittenKontsevich::new();
        assert_eq!(wk.psi_integral(2, &[4]), Rational::new(1, 1152));
        assert_eq!(wk.psi_integral(3, &[7]), Rational::new(1, 82944));
    }

    #[test]
    fn one_point_genus_ten_matches_closed_form() {
        // <tau_{3g-2}>_g = 1/(24^g g!).  At g = 10 the DVV recursion needs
        // (2*28+1)!! = 57!!, which overflows i128; this locks the exact
        // big-rational path.
        let wk = WittenKontsevich::new();
        let mut denominator = Rational::one();
        for k in 1..=10i128 {
            denominator = denominator * Rational::from(24) * Rational::from(k);
        }
        assert_eq!(
            wk.psi_integral(10, &[28]),
            Rational::one() / denominator
        );
    }

    #[test]
    fn dimension_mismatch_is_zero() {
        let wk = WittenKontsevich::new();
        assert_eq!(wk.psi_integral(1, &[0]), Rational::zero());
    }

    #[test]
    fn table_oracle_delegates_pure_psi_integrals() {
        let oracle = TableTautologicalOracle::new();
        assert_eq!(
            oracle.hodge_integral(1, &[1], &[]),
            Some(Rational::new(1, 24))
        );
    }

    #[test]
    fn table_oracle_returns_inserted_hodge_values() {
        let oracle = TableTautologicalOracle::new();
        oracle.insert_symmetric_hodge_integral(1, &[0, 2], &[(1, 1)], Rational::new(7, 5));
        assert_eq!(
            oracle.hodge_integral(1, &[2, 0], &[(1, 1)]),
            Some(Rational::new(7, 5))
        );
    }
}
