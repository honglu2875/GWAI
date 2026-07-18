//! Independent validation oracles and compatibility access to target-owned
//! projective-space seed formulas.
//!
//! The seed implementations live with the projective-space target. Historical
//! callers through this module continue to resolve to those same functions.

use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::{InvariantRequest, InvariantResult};

pub use crate::spaces::projective_space::seeds::{
    classical_product_integral, plane_rational_curve_count, seed_compute, small_quantum_three_point,
};

pub fn assert_same_value(a: &RatFun, b: &RatFun) -> Result<(), GwError> {
    if a.equivalent(b) {
        Ok(())
    } else {
        Err(GwError::ValidationFailure(format!("{a} != {b}")))
    }
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

    let tangent_chern =
        crate::spaces::projective_space::seeds::binomial_rational(req.n + 1, req.n - h_power);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::projective_space::CohomologyClass;

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
    fn validation_accepts_equivalent_rational_function_representations() {
        let x = RatFun::variable("x");
        let y = RatFun::variable("y");
        let quotient = &(&x.pow_usize(2) - &y.pow_usize(2)) / &(&x - &y);
        let sum = &x + &y;

        assert_ne!(quotient, sum);
        assert_same_value(&quotient, &sum).unwrap();
    }

    #[test]
    fn equivariant_constant_map_seed_keeps_excess_class_degree() {
        let mut req = InvariantRequest::new(
            1,
            0,
            0,
            vec![
                crate::tau(0, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::h_power(1, 1)),
                crate::tau(0, CohomologyClass::one(1)),
            ],
        );
        req.equivariant = true;
        let result = seed_compute(&req, "test-seed").unwrap();
        let expected = &crate::core::algebra::lambda(0) + &crate::core::algebra::lambda(1);
        assert!(result.value.equivalent(&expected));

        let mut deficit = InvariantRequest::new(1, 0, 2, Vec::new());
        deficit.equivariant = true;
        assert!(seed_compute(&deficit, "test-seed").unwrap().value.is_zero());
    }

    #[test]
    fn equivariant_positive_degree_seed_rejects_excess_three_point_degree() {
        let point = CohomologyClass::h_power(2, 2);
        let mut excess = InvariantRequest::new(
            2,
            0,
            1,
            vec![
                crate::tau(0, point.clone()),
                crate::tau(0, point.clone()),
                crate::tau(0, point),
            ],
        );
        excess.equivariant = true;
        let err = seed_compute(&excess, "test-seed").unwrap_err();
        assert!(matches!(err, GwError::UnsupportedInvariant(_)));
        assert!(err.to_string().contains("excess-degree"));

        let mut exact = InvariantRequest::new(
            2,
            0,
            1,
            vec![
                crate::tau(0, CohomologyClass::h_power(2, 2)),
                crate::tau(0, CohomologyClass::h_power(2, 2)),
                crate::tau(0, CohomologyClass::h_power(2, 1)),
            ],
        );
        exact.equivariant = true;
        assert_eq!(
            seed_compute(&exact, "test-seed").unwrap().value,
            RatFun::one()
        );
    }

    #[test]
    fn p1_degree_one_unstable_divisor_family_is_one_in_both_rings() {
        for equivariant in [false, true] {
            for markings in 0..=2 {
                let mut req = InvariantRequest::new(
                    1,
                    0,
                    1,
                    vec![crate::tau(0, CohomologyClass::h_power(1, 1)); markings],
                );
                req.equivariant = equivariant;
                let result = crate::compute(req).unwrap();
                assert!(
                    result.value.equivalent(&RatFun::one()),
                    "equivariant={equivariant}, markings={markings}: {}",
                    result.value
                );
            }
        }
    }

    #[test]
    fn unstable_constant_maps_have_an_explicit_empty_moduli_note() {
        let req = InvariantRequest::new(
            1,
            0,
            0,
            vec![
                crate::tau(0, CohomologyClass::one(1)),
                crate::tau(0, CohomologyClass::one(1)),
            ],
        );
        let result = crate::compute(req).unwrap();
        assert!(result.value.is_zero());
        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("unstable constant domains")));
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
}
