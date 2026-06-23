//! Small closed-form and oracle-style formulas used for validation and seeds.
//!
//! These are not a replacement for the Givental graph evaluator.  They cover
//! low-complexity cases where the answer follows from point theory, classical
//! intersections, the small quantum product, the divisor equation, or
//! Kontsevich's recursion for plane rational curves.

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::geometry::CohomologyClass;
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::{InvariantRequest, InvariantResult};

pub fn assert_same_value(a: &RatFun, b: &RatFun) -> Result<(), GwError> {
    if a == b {
        Ok(())
    } else {
        Err(GwError::ValidationFailure(format!("{a} != {b}")))
    }
}

pub fn seed_compute(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    // These seed cases are mathematically closed and cheap.  The Givental
    // evaluator uses them only as scalar fallbacks for unsupported unstable
    // ranges; validation tests may call them explicitly.
    if let Some(total_degree) = req.insertion_degree() {
        let virtual_dimension = req.virtual_dimension();
        if virtual_dimension >= 0 && total_degree as isize != virtual_dimension {
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
        return compute_genus_zero_three_point_primary(req, engine);
    }

    if let Some(value) = p1_stationary_one_descendant_divisor_family(req) {
        return Ok(InvariantResult {
            value: RatFun::from_rational(value),
            engine,
            notes: vec![
                "genus-zero P^1 stationary one-descendant family computed from the J-function one-point term and divisor equation"
                    .to_string(),
            ],
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
    // Degree-zero genus-zero maps factor as an ordinary P^n intersection times
    // the psi integral on Mbar_{0,n}.
    let classes = req
        .insertions
        .iter()
        .map(|insertion| &insertion.class)
        .collect::<Vec<_>>();
    let class_integral = classical_product_integral(req.n, &classes);
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
        notes: vec![
            "genus-zero degree-zero invariant factorized into classical P^n intersection and psi integral"
                .to_string(),
        ],
    })
}

pub fn genus_one_degree_zero_one_point_obstruction(
    req: &InvariantRequest,
    engine: &'static str,
) -> Result<InvariantResult, GwError> {
    // For one marked point in degree zero, the obstruction bundle is
    // E^vee \otimes T P^n.  This computes the resulting first Chern/euler
    // contribution against a single H^c insertion.
    let insertion = &req.insertions[0];
    let Some(h_power) = insertion.class.pure_power() else {
        return Err(GwError::UnsupportedInvariant(
            "genus-one degree-zero obstruction oracle currently requires a pure H^c insertion"
                .to_string(),
        ));
    };

    if insertion.descendant_power + h_power != 1 {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine,
            notes: vec![
                "dimension mismatch gives zero for genus-one degree-zero one-point constant maps"
                    .to_string(),
            ],
        });
    }

    let tangent_chern = binomial_rational(req.n + 1, req.n - h_power);
    let hodge_integral = Rational::new(1, 24);
    let signed = if h_power % 2 == 0 {
        tangent_chern
    } else {
        -tangent_chern
    };
    Ok(InvariantResult {
        value: RatFun::from_rational(signed * hodge_integral),
        engine,
        notes: vec![
            "genus-one degree-zero one-point invariant computed from e(E^vee tensor T P^n)"
                .to_string(),
        ],
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
    if req.equivariant
        || req.n != 1
        || req.genus != 0
        || req.degree == 0
        || req.insertions.is_empty()
        || !req
            .insertions
            .iter()
            .all(|insertion| insertion.class.pure_power() == Some(1))
    {
        return None;
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

pub fn plane_rational_curve_count(degree: usize) -> Rational {
    // Kontsevich recursion for N_d = <pt^{3d-1}>_{0,d}^{P^2}.
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

fn binomial_rational(n: usize, k: usize) -> Rational {
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
    use crate::geometry::CohomologyClass;

    #[test]
    fn classical_product_picks_hn_coefficient() {
        let h = CohomologyClass::h_power(2, 1);
        let one = CohomologyClass::one(2);
        assert_eq!(
            classical_product_integral(2, &[&h, &h, &one]),
            RatFun::one()
        );
        assert_eq!(
            classical_product_integral(2, &[&h, &one, &one]),
            RatFun::zero()
        );
    }

    #[test]
    fn quantum_product_tracks_degree() {
        let pt = CohomologyClass::h_power(2, 2);
        let h = CohomologyClass::h_power(2, 1);
        assert_eq!(
            small_quantum_three_point(2, 1, &[&pt, &pt, &h]),
            RatFun::one()
        );
        assert_eq!(
            small_quantum_three_point(2, 0, &[&pt, &pt, &h]),
            RatFun::zero()
        );
    }

    #[test]
    fn kontsevich_plane_curve_counts_start_correctly() {
        assert_eq!(plane_rational_curve_count(1), Rational::one());
        assert_eq!(plane_rational_curve_count(2), Rational::one());
        assert_eq!(plane_rational_curve_count(3), Rational::from(12usize));
        assert_eq!(plane_rational_curve_count(4), Rational::from(620usize));
    }

    #[test]
    fn genus_one_degree_zero_one_point_obstruction_values() {
        let p1_h =
            InvariantRequest::new(1, 1, 0, vec![crate::tau(0, CohomologyClass::h_power(1, 1))]);
        assert_eq!(
            genus_one_degree_zero_one_point_obstruction(&p1_h, "test")
                .unwrap()
                .value,
            RatFun::from_rational(Rational::new(-1, 24))
        );

        let p2_h =
            InvariantRequest::new(2, 1, 0, vec![crate::tau(0, CohomologyClass::h_power(2, 1))]);
        assert_eq!(
            genus_one_degree_zero_one_point_obstruction(&p2_h, "test")
                .unwrap()
                .value,
            RatFun::from_rational(Rational::new(-1, 8))
        );

        let p2_psi_unit =
            InvariantRequest::new(2, 1, 0, vec![crate::tau(1, CohomologyClass::one(2))]);
        assert_eq!(
            genus_one_degree_zero_one_point_obstruction(&p2_psi_unit, "test")
                .unwrap()
                .value,
            RatFun::from_rational(Rational::new(1, 8))
        );
    }

    #[test]
    fn p1_stationary_divisor_family_matches_initial_degrees() {
        let req = InvariantRequest::new(
            1,
            0,
            3,
            vec![
                crate::tau(4, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::h_power(1, 1)),
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
                crate::tau(6, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::h_power(1, 1)),
            ],
        );
        assert_eq!(
            p1_stationary_one_descendant_divisor_family(&req),
            Some(Rational::new(1, 36))
        );
    }
}
