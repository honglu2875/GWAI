//! Closed projective-space formulas used at unstable or otherwise inexpensive
//! scalar leaves of the Givental evaluator.
//!
//! This module is the single implementation owner for these formulas.
//! Validation may consume them as known formulas, but the production target
//! does not depend on the crate-wide validation layer.

use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::tautological::{TautologicalOracle, WittenKontsevich};

use super::api::{InvariantRequest, InvariantResult};
use super::equivariant::{CohomologyClass, EquivariantProjectiveSpace};

/// Evaluate one of the closed projective-space seed families.
pub fn seed_compute(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    req.validate()?;
    // These cases are mathematically closed and cheap. The Givental evaluator
    // uses them only as scalar fallbacks for unsupported unstable ranges;
    // validation callers may also use them explicitly as known formulas.
    if let Some(total_degree) = req.insertion_degree() {
        let virtual_dimension = req.virtual_dimension();
        let forced_zero = if req.equivariant {
            usize::try_from(virtual_dimension)
                .ok()
                .is_some_and(|dimension| total_degree < dimension)
        } else {
            usize::try_from(virtual_dimension).ok() != Some(total_degree)
        };
        if forced_zero {
            return Ok(InvariantResult {
                value: RatFun::zero(),
                engine,
                notes: vec![format!(
                    "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
                )],
            });
        }
    }

    if req.n == 0 {
        return compute_point_theory(req, engine);
    }

    if req.genus == 0 && req.degree == 0 {
        return compute_genus_zero_constant_maps(req, engine);
    }

    if req.genus == 0
        && req.insertions.len() == 3
        && req
            .insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
    {
        if req.equivariant && req.degree > 0 {
            let exact_dimension = req.insertion_degree().is_some_and(|total_degree| {
                usize::try_from(req.virtual_dimension()).ok() == Some(total_degree)
            });
            if !exact_dimension {
                return Err(GwError::UnsupportedInvariant(
                    "positive-degree equivariant three-point closed-form seeds are only implemented in the exact-dimension sector; no seed is available for excess-degree classes"
                        .to_string(),
                ));
            }
        }
        return compute_genus_zero_three_point_primary(req, engine);
    }

    if let Some(value) = p1_stationary_one_descendant_divisor_family(req) {
        return Ok(InvariantResult {
            value: RatFun::from_rational(value),
            engine,
            notes: vec![if req.insertions.is_empty() {
                "Mbar_0,0(P^1,1) is a point, giving the unmarked degree-one invariant".to_string()
            } else {
                "genus-zero P^1 stationary one-descendant family computed from the J-function one-point term and divisor equation"
                    .to_string()
            }],
        });
    }

    if req.n == 2
        && req.genus == 0
        && req.degree > 0
        && req.insertions.iter().all(|insertion| {
            insertion.descendant_power == 0 && insertion.class.pure_power() == Some(2)
        })
        && req.insertions.len() == 3 * req.degree - 1
    {
        return Ok(InvariantResult {
            value: RatFun::from_rational(plane_rational_curve_count(req.degree)),
            engine,
            notes: vec![
                "genus-zero P^2 point invariant computed by Kontsevich recursion".to_string(),
            ],
        });
    }

    Err(GwError::UnsupportedInvariant(format!(
        "implemented seed formulas cover P^0 point theory, genus-zero degree-zero constants, and genus-zero three-point primary small quantum products; requested n={}, g={}, d={}, markings={}",
        req.n,
        req.genus,
        req.degree,
        req.insertions.len()
    )))
}

fn compute_point_theory(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    if req.degree != 0 {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine,
            notes: vec!["P^0 has only degree-zero curve classes".to_string()],
        });
    }
    for insertion in &req.insertions {
        if insertion.class.pure_power() != Some(0) {
            return Ok(InvariantResult {
                value: RatFun::zero(),
                engine,
                notes: vec!["non-unit insertion on P^0 gives zero".to_string()],
            });
        }
    }
    let psi_powers = req
        .insertions
        .iter()
        .map(|insertion| insertion.descendant_power)
        .collect::<Vec<_>>();
    let value = WittenKontsevich::new().psi_integral(req.genus, &psi_powers);
    Ok(InvariantResult {
        value: RatFun::from_rational(value),
        engine,
        notes: vec!["computed by Witten-Kontsevich point theory".to_string()],
    })
}

fn compute_genus_zero_constant_maps(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    if req.insertions.len() < 3 {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine,
            notes: vec![
                "genus-zero degree-zero maps with fewer than three markings have unstable constant domains, so the stable-map moduli space is empty"
                    .to_string(),
            ],
        });
    }
    // Degree-zero genus-zero maps factor as an ordinary P^n intersection times
    // the psi integral on Mbar_{0,n}.
    let classes = req
        .insertions
        .iter()
        .map(|insertion| &insertion.class)
        .collect::<Vec<_>>();
    let class_integral = if req.equivariant {
        let target = EquivariantProjectiveSpace::new(req.n);
        let mut product = CohomologyClass::one(req.n);
        for class in &classes {
            product = product.multiply_classical_equivariant(class);
        }
        target.pairing(&product, &CohomologyClass::one(req.n))
    } else {
        classical_product_integral(req.n, &classes)
    };
    let psi_powers = req
        .insertions
        .iter()
        .map(|insertion| insertion.descendant_power)
        .collect::<Vec<_>>();
    let psi = WittenKontsevich::new().psi_integral(0, &psi_powers);
    let value = &class_integral * &RatFun::from_rational(psi);
    Ok(InvariantResult {
        value,
        engine,
        notes: vec![if req.equivariant {
            "genus-zero degree-zero invariant factorized into the equivariant classical P^n integral and psi integral"
                .to_string()
        } else {
            "genus-zero degree-zero invariant factorized into classical P^n intersection and psi integral"
                .to_string()
        }],
    })
}

fn compute_genus_zero_three_point_primary(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    let classes = req
        .insertions
        .iter()
        .map(|insertion| &insertion.class)
        .collect::<Vec<_>>();
    let value = small_quantum_three_point(req.n, req.degree, &classes);
    Ok(InvariantResult {
        value,
        engine,
        notes: vec![
            "genus-zero three-point primary invariant computed from QH(P^n): H^(n+1)=q".to_string(),
        ],
    })
}

fn p1_stationary_one_descendant_divisor_family(req: &InvariantRequest) -> Option<Rational> {
    // From the P^1 J-function one-point stationary term plus repeated divisor
    // equation:
    // <tau_{2d-2}(H), H,...,H>_{0,d} = d^m/(d!)^2.
    if req.n != 1
        || req.genus != 0
        || req.degree == 0
        || !req
            .insertions
            .iter()
            .all(|insertion| insertion.class.pure_power() == Some(1))
    {
        return None;
    }

    // Mbar_0,0(P^1,1) is a point. This is the base of the same divisor
    // family; the dimension check above has already disposed of d > 1.
    if req.insertions.is_empty() {
        return (req.degree == 1).then(Rational::one);
    }

    let positive_descendants = req
        .insertions
        .iter()
        .filter(|insertion| insertion.descendant_power > 0)
        .count();
    if positive_descendants > 1 {
        return None;
    }

    let descendant_power = req
        .insertions
        .iter()
        .map(|insertion| insertion.descendant_power)
        .max()
        .unwrap_or(0);
    if descendant_power != 2 * req.degree - 2 {
        return None;
    }

    let primary_divisors = req.insertions.len().saturating_sub(1);
    let numerator = Rational::from(req.degree).pow_usize(primary_divisors);
    let denominator = factorial_rational(req.degree).pow_usize(2);
    Some(numerator / denominator)
}

fn factorial_rational(value: usize) -> Rational {
    let mut out = Rational::one();
    for factor in 2..=value {
        out = out * Rational::from(factor);
    }
    out
}

/// Integrate a product of ordinary cohomology classes over `P^n`.
pub fn classical_product_integral(n: usize, classes: &[&CohomologyClass]) -> RatFun {
    let mut total = RatFun::zero();
    for terms in class_term_choices(classes) {
        let power_sum = terms.iter().map(|(power, _)| *power).sum::<usize>();
        if power_sum == n {
            let coeff = multiply_coefficients(&terms);
            total = &total + &coeff;
        }
    }
    total
}

/// Evaluate a genus-zero three-point primary invariant via
/// `QH(P^n) = Q[H,q]/(H^(n+1)-q)`.
pub fn small_quantum_three_point(n: usize, degree: usize, classes: &[&CohomologyClass]) -> RatFun {
    debug_assert_eq!(classes.len(), 3);
    let mut total = RatFun::zero();
    for terms in class_term_choices(classes) {
        let power_sum = terms.iter().map(|(power, _)| *power).sum::<usize>();
        if power_sum == n + (n + 1) * degree {
            let coeff = multiply_coefficients(&terms);
            total = &total + &coeff;
        }
    }
    total
}

/// Kontsevich's recursion for `N_d = <pt^(3d-1)>_(0,d)` on `P^2`.
pub fn plane_rational_curve_count(degree: usize) -> Rational {
    if degree == 0 {
        return Rational::zero();
    }
    let mut counts = vec![Rational::zero(); degree + 1];
    counts[1] = Rational::one();
    for d in 2..=degree {
        let mut total = Rational::zero();
        for d1 in 1..d {
            let d2 = d - d1;
            let bracket = Rational::from(d2) * binomial_rational(3 * d - 4, 3 * d1 - 2)
                - Rational::from(d1) * binomial_rational(3 * d - 4, 3 * d1 - 1);
            let factor = Rational::from(d1 * d1 * d2);
            total += counts[d1].clone() * counts[d2].clone() * factor * bracket;
        }
        counts[d] = total;
    }
    counts[degree].clone()
}

pub(crate) fn binomial_rational(n: usize, k: usize) -> Rational {
    if k > n {
        return Rational::zero();
    }
    let k = k.min(n - k);
    let mut out = Rational::one();
    for i in 0..k {
        out = out * (Rational::from(n - i) / Rational::from(i + 1));
    }
    out
}

fn class_term_choices(classes: &[&CohomologyClass]) -> Vec<Vec<(usize, RatFun)>> {
    fn rec(
        classes: &[&CohomologyClass],
        idx: usize,
        current: &mut Vec<(usize, RatFun)>,
        out: &mut Vec<Vec<(usize, RatFun)>>,
    ) {
        if idx == classes.len() {
            out.push(current.clone());
            return;
        }
        for (power, coeff) in classes[idx].coeffs().iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            current.push((power, coeff.clone()));
            rec(classes, idx + 1, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(classes, 0, &mut Vec::new(), &mut out);
    out
}

fn multiply_coefficients(terms: &[(usize, RatFun)]) -> RatFun {
    terms
        .iter()
        .fold(RatFun::from_rational(Rational::one()), |acc, (_, coeff)| {
            &acc * coeff
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::projective_space::{tau, InvariantRequest};

    #[test]
    fn target_seed_entry_point_handles_unstable_p1_degree_one() {
        let req = InvariantRequest::new(1, 0, 1, vec![tau(0, CohomologyClass::h_power(1, 1))]);
        let result = seed_compute(&req, "projective-seed-test").unwrap();
        assert!(result.value.equivalent(&RatFun::one()));
    }

    #[test]
    fn p1_stationary_divisor_family_matches_initial_degrees() {
        let req = InvariantRequest::new(
            1,
            0,
            3,
            vec![
                tau(4, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        );
        assert_eq!(
            p1_stationary_one_descendant_divisor_family(&req),
            Some(Rational::new(1, 4))
        );

        let req = InvariantRequest::new(
            1,
            0,
            4,
            vec![
                tau(6, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        );
        assert_eq!(
            p1_stationary_one_descendant_divisor_family(&req),
            Some(Rational::new(1, 36))
        );
    }
}
