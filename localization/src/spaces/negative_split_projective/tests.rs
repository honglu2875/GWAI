use super::i_function::negative_split_equivariant_i_function_coefficient_coeff;
use super::*;
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{QSeries, SeriesMatrix};
use crate::factored::FactoredRatFun;
use crate::givental::recipe::{
    birkhoff_descendant_s_matrix_from_fundamental,
    birkhoff_descendant_s_matrix_from_fundamental_coeff,
    metric_adjoint_descendant_s_matrix_with_inverse_coeff,
};
use crate::givental::{CalibrationId, SemisimpleCohftProvider, SeriesSMatrix};
use crate::reconstruction::{
    birkhoff_factor_by_q_degree, identity_coeff_matrix, matrix_q_coefficient,
    negative_factor_to_s_coefficients, zero_coeff_matrix,
};
use crate::spaces::projective_space::CohomologyClass;
use crate::tau;
use std::collections::BTreeMap;

#[test]
fn negative_split_degrees_must_be_positive() {
    assert!(NegativeSplitBundleTwist::new(vec![3]).is_ok());
    assert!(NegativeSplitBundleTwist::new(vec![1, 1]).is_ok());
    assert!(NegativeSplitBundleTwist::new(vec![0]).is_err());
    assert!(matches!(
        NegativeSplitBundleTwist::new(vec![usize::MAX, 1]),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert_eq!(
        NegativeSplitBundleTwist::new(vec![3, 1, 2])
            .unwrap()
            .degrees(),
        &[1, 2, 3]
    );
}

#[test]
#[allow(deprecated)]
fn twist_geometry_helpers_delegate_to_the_canonical_theory() {
    use crate::core::theory::{CurveClass, GwTheory};
    use crate::spaces::negative_split_projective::NegativeSplitTotalSpaceTheory;

    let twist = NegativeSplitBundleTwist::new(vec![1, 2]).unwrap();
    let theory = NegativeSplitTotalSpaceTheory::new(2, vec![1, 2]).unwrap();
    assert_eq!(
        twist.try_total_space_dimension(2).unwrap(),
        theory.target_dimension()
    );
    for genus in 0..=2 {
        for degree in 0..=3 {
            assert_eq!(
                twist.try_virtual_dimension(2, genus, degree, 4).unwrap(),
                theory
                    .virtual_dimension(
                        genus,
                        &CurveClass::new(vec![i64::try_from(degree).unwrap()]),
                        4,
                    )
                    .unwrap()
            );
        }
    }
    assert_eq!(
        twist.try_candidate_degrees(2, 0, 5, 1, Some(3)).unwrap(),
        theory
            .candidate_degrees_from_dimension(0, 5, 1, Some(3))
            .unwrap()
    );
}

#[test]
#[allow(deprecated)]
fn empty_twist_geometry_helpers_delegate_to_projective_space() {
    use crate::core::theory::GwTheory;
    use crate::spaces::projective_space::ProjectiveSpaceTheory;

    let twist = NegativeSplitBundleTwist::new(Vec::new()).unwrap();
    let theory = ProjectiveSpaceTheory::try_new(2).unwrap();
    assert_eq!(
        twist.try_total_space_dimension(2).unwrap(),
        theory.target_dimension()
    );
    assert_eq!(
        twist.try_virtual_dimension(2, 0, 1, 1).unwrap(),
        theory
            .virtual_dimension(0, &theory.try_curve(1).unwrap(), 1)
            .unwrap()
    );
}

#[test]
fn twisted_provider_rejects_unrepresentable_canonical_geometry() {
    assert!(matches!(
        TwistedProjectiveSpaceProvider::new(1, vec![usize::MAX], false),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        TwistedProjectiveSpaceProvider::new(usize::MAX, vec![1], false),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn default_twisted_weights_avoid_spurious_nonsemisimple_poles() {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3, 3], false).unwrap();
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 1)),
        tau(0, CohomologyClass::one(2)),
        tau(0, CohomologyClass::one(2)),
    ];
    crate::givental::compute_semisimple_graph_value(&provider, 0, 1, &insertions, None).unwrap();
}

#[test]
fn twisted_request_rejects_class_from_another_target() {
    let err = TwistedInvariantRequest::new(1, vec![3], 0, 1, vec![tau(0, CohomologyClass::one(2))])
        .unwrap_err();
    assert!(matches!(err, GwError::ConventionMismatch(_)));
}

#[test]
#[allow(deprecated)]
fn local_cy_threefold_dimension_is_degree_independent_without_insertions() {
    let local_p2 = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let conifold = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();

    for genus in 0..=4 {
        for degree in 0..=5 {
            assert_eq!(local_p2.virtual_dimension(2, genus, degree, 0), 0);
            assert_eq!(conifold.virtual_dimension(1, genus, degree, 0), 0);
        }
    }
}

#[test]
fn local_cy_provider_returns_all_degree_candidates_when_dimension_matches() {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    assert_eq!(
        provider.candidate_degrees_from_dimension(2, 4, &[]),
        vec![0, 1, 2, 3, 4]
    );
    let h = tau(0, CohomologyClass::h_power(2, 2));
    assert!(provider
        .candidate_degrees_from_dimension(2, 4, &[h])
        .is_empty());
}

#[test]
fn negative_slope_expected_degree_matches_candidate_degree() {
    // For P^1 twisted by O(-3), vdim = 2 - d with these three markings.
    // The insertions have total degree one, so the unique valid degree is one:
    // both the numerator and the virtual-dimension slope are negative.
    let provider = TwistedProjectiveSpaceProvider::new(1, vec![3], false).unwrap();
    let insertions = vec![
        tau(0, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::one(1)),
        tau(0, CohomologyClass::one(1)),
    ];
    assert_eq!(
        provider.expected_degree_from_dimension(0, &insertions),
        Some(1)
    );
    assert_eq!(
        provider.candidate_degrees_from_dimension(0, 3, &insertions),
        vec![1]
    );
}

#[test]
fn twisted_quantum_relation_records_local_p2_symbol() {
    let relation = TwistedQuantumRelation::new(
        2,
        NegativeSplitBundleTwist::new(vec![3]).unwrap(),
        vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(3usize),
        ],
    )
    .unwrap();
    let coefficients = relation.coefficients(1);

    assert_eq!(
        coefficients[0].coeff(0),
        Some(&RatFun::from_rational(Rational::from(-6)))
    );
    assert_eq!(
        coefficients[1].coeff(0),
        Some(&RatFun::from_rational(Rational::from(11)))
    );
    assert_eq!(
        coefficients[2].coeff(0),
        Some(&RatFun::from_rational(Rational::from(-6)))
    );
    assert_eq!(coefficients[3].coeff(0), Some(&RatFun::one()));
    assert_eq!(
        coefficients[3].coeff(1),
        Some(&RatFun::from_rational(Rational::from(27usize)))
    );
}

#[test]
fn local_p2_hypergeometric_i_function_has_expected_first_mirror_term() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let coefficient = negative_split_i_function_coefficient(2, &twist, 1);
    assert_eq!(coefficient.coefficient(1, -1), Rational::from(-6));
    assert_eq!(
        negative_split_mirror_map_coefficients(2, &twist, 2)[1],
        Rational::from(-6)
    );
    assert_eq!(
        negative_split_inverse_mirror_map_coefficients(2, &twist, 2)[2],
        Rational::from(6usize)
    );
}

#[test]
fn projective_i_function_coefficient_records_denominator_series() {
    let coefficient = projective_i_function_coefficient(2, 1);

    assert_eq!(coefficient.coefficient(0, -3), Rational::one());
    assert_eq!(coefficient.coefficient(1, -4), Rational::from(-3));
    assert_eq!(coefficient.coefficient(2, -5), Rational::from(6usize));
}

#[test]
fn equivariant_projective_i_specializes_to_nonequivariant_i_at_zero_weights() {
    let equivariant = projective_equivariant_i_function_coefficient(
        2,
        1,
        &[Rational::zero(), Rational::zero(), Rational::zero()],
        -5,
    )
    .unwrap();

    assert_eq!(equivariant, projective_i_function_coefficient(2, 1));
}

#[test]
fn equivariant_projective_i_records_base_weight_correction() {
    let coefficient = projective_equivariant_i_function_coefficient(
        1,
        1,
        &[Rational::from(2usize), Rational::from(5usize)],
        -3,
    )
    .unwrap();

    assert_eq!(coefficient.coefficient(0, -2), Rational::one());
    assert_eq!(coefficient.coefficient(0, -3), Rational::from(7usize));
    assert_eq!(coefficient.coefficient(1, -3), Rational::from(-2));
}

#[test]
fn equivariant_projective_i_nonnegative_z_floor_is_an_empty_window() {
    let coefficient = projective_equivariant_i_function_coefficient(
        1,
        1,
        &[Rational::from(2usize), Rational::from(5usize)],
        0,
    )
    .unwrap();

    assert_eq!(coefficient, HLaurentSeries::zero(1));

    let degree_zero_above_constant = projective_equivariant_i_function_coefficient(
        1,
        0,
        &[Rational::from(2usize), Rational::from(5usize)],
        1,
    )
    .unwrap();
    assert_eq!(degree_zero_above_constant, HLaurentSeries::zero(1));

    let degree_zero_at_constant = projective_equivariant_i_function_coefficient(
        1,
        0,
        &[Rational::from(2usize), Rational::from(5usize)],
        0,
    )
    .unwrap();
    assert_eq!(degree_zero_at_constant, HLaurentSeries::one(1));
}

#[test]
fn birkhoff_rejects_unrepresentable_bounds_before_factorization() {
    let fundamental = BTreeMap::new();
    let too_large_for_exponents = usize::try_from(i32::MAX).unwrap().checked_add(1).unwrap();

    for z_order in [too_large_for_exponents, usize::MAX] {
        assert!(matches!(
            birkhoff_descendant_s_matrix_from_fundamental(
                1,
                0,
                z_order,
                &fundamental,
                CalibrationId("overflow-guard".to_string()),
            ),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    let model = NegativeSplitHypergeometricModel::new(
        0,
        NegativeSplitBundleTwist::new(Vec::new()).unwrap(),
        0,
    );
    assert!(matches!(
        model.birkhoff_descendant_s_matrix(usize::MAX),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn qrr_factorized_i_function_matches_direct_hypergeometric_i_function() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let qrr = NegativeSplitQrrOperator::new(twist.clone());

    for degree in 0..=4 {
        assert_eq!(
            qrr.apply_to_projective_i_coefficient(2, degree),
            negative_split_i_function_coefficient(2, &twist, degree)
        );
    }
}

#[test]
fn equivariant_negative_split_i_specializes_to_direct_local_i_at_zero_weights() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let equivariant = negative_split_equivariant_i_function_coefficient(
        2,
        &twist,
        1,
        &[Rational::zero(), Rational::zero(), Rational::zero()],
        &[Rational::zero()],
        -5,
    )
    .unwrap();

    assert_eq!(
        equivariant,
        negative_split_i_function_coefficient(2, &twist, 1)
    );
}

#[test]
fn o_minus_five_p1_retained_i_window_matches_deep_laurent_reference() {
    let twist = NegativeSplitBundleTwist::new(vec![5]).unwrap();
    let base_weights = vec![Rational::from(1usize), Rational::from(2usize)];
    let fiber_weights = vec![Rational::from(3usize)];
    let retained_min_z_power = -5;

    let retained = negative_split_equivariant_i_function_coefficient(
        1,
        &twist,
        1,
        &base_weights,
        &fiber_weights,
        retained_min_z_power,
    )
    .unwrap();
    let deep_reference = negative_split_equivariant_i_function_coefficient(
        1,
        &twist,
        1,
        &base_weights,
        &fiber_weights,
        -29,
    )
    .unwrap()
    .truncated_z_below(retained_min_z_power);

    assert_eq!(retained, deep_reference);
}

#[test]
fn multi_summand_qrr_retained_i_window_matches_deep_laurent_reference() {
    let twist = NegativeSplitBundleTwist::new(vec![3, 4]).unwrap();
    let base_weights = vec![Rational::from(2usize), Rational::from(5usize)];
    let fiber_weights = vec![Rational::from(11usize), Rational::from(23usize)];
    let retained_min_z_power = -5;

    // In degree one the two Euler numerators raise z-degree by 2 and 3,
    // respectively.  This specifically exercises their summed source-tail
    // allowance, rather than the one-line maximum used by the simpler tests.
    let retained = negative_split_equivariant_i_function_coefficient(
        1,
        &twist,
        1,
        &base_weights,
        &fiber_weights,
        retained_min_z_power,
    )
    .unwrap();
    let deep_reference = negative_split_equivariant_i_function_coefficient(
        1,
        &twist,
        1,
        &base_weights,
        &fiber_weights,
        -30,
    )
    .unwrap()
    .truncated_z_below(retained_min_z_power);

    assert_eq!(retained, deep_reference);
    assert!(
        (0..=1).any(|h_power| !retained
            .coefficient(h_power, retained_min_z_power)
            .is_zero()),
        "the comparison must exercise the retained Laurent boundary"
    );
}

#[test]
fn negative_split_hypergeometric_windows_reject_arithmetic_overflow() {
    let twist = NegativeSplitBundleTwist::new(vec![usize::MAX]).unwrap();
    let base_weights = vec![Rational::from(1usize), Rational::from(2usize)];
    let fiber_weights = vec![Rational::from(3usize)];

    assert!(matches!(
        negative_split_equivariant_qrr_euler_factor(1, &twist, 2, &base_weights, &fiber_weights),
        Err(GwError::UnsupportedInvariant(_))
    ));
    let exponent_overflow_twist = NegativeSplitBundleTwist::new(vec![usize::try_from(i32::MAX)
        .unwrap()
        .checked_add(2)
        .unwrap()])
    .unwrap();
    assert!(matches!(
        negative_split_equivariant_qrr_euler_factor(
            1,
            &exponent_overflow_twist,
            1,
            &base_weights,
            &fiber_weights,
        ),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        negative_split_equivariant_i_function_coefficient(
            1,
            &NegativeSplitBundleTwist::new(vec![5]).unwrap(),
            1,
            &base_weights,
            &fiber_weights,
            i32::MIN,
        ),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        projective_equivariant_i_function_coefficient(1, 1, &base_weights, i32::MIN),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        projective_equivariant_i_function_coefficient(usize::MAX, 0, &[], -1),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        negative_split_equivariant_qrr_euler_factor(
            usize::MAX,
            &NegativeSplitBundleTwist::new(vec![1]).unwrap(),
            0,
            &[],
            &[Rational::one()],
        ),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        NegativeSplitEquivariantHypergeometricModel::new(
            1,
            NegativeSplitBundleTwist::new(vec![1]).unwrap(),
            usize::MAX,
            base_weights.clone(),
            fiber_weights.clone(),
            -5,
        ),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
            1,
            NegativeSplitBundleTwist::new(vec![1]).unwrap(),
            usize::MAX,
            1,
            base_weights,
            fiber_weights,
        ),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn equivariant_negative_split_qrr_factor_records_fiber_weight() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let factor = negative_split_equivariant_qrr_euler_factor(
        1,
        &twist,
        1,
        &[Rational::zero(), Rational::zero()],
        &[Rational::from(3usize), Rational::from(7usize)],
    )
    .unwrap();

    assert_eq!(factor.coefficient(0, 0), Rational::from(21usize));
    assert_eq!(factor.coefficient(1, 0), Rational::from(-10));
}

#[test]
fn twisted_canonical_roots_solve_principal_relation() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base_weights = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let fiber_weights = vec![Rational::from(7usize)];
    let canonical =
        specialized_twisted_quantum_canonical_data(2, &twist, 3, &base_weights, &fiber_weights)
            .unwrap();

    for root in &canonical.roots {
        let value =
            twisted_relation_series_at_weights(2, &twist, root, &base_weights, &fiber_weights)
                .unwrap();
        assert!(
            value.coeffs().iter().all(RatFun::is_zero),
            "root {root} does not solve relation: {value}"
        );
    }
}

#[test]
fn twisted_metric_norm_q_zero_is_euler_over_tangent_euler() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base_weights = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let fiber_weights = vec![Rational::from(7usize)];
    let canonical =
        specialized_twisted_quantum_canonical_data(2, &twist, 1, &base_weights, &fiber_weights)
            .unwrap();

    for branch in 0..=2 {
        let lambda_i = base_weights[branch].clone();
        let fiber = fiber_weights[0].clone() - Rational::from(3usize) * lambda_i.clone();
        let mut tangent = Rational::one();
        for (other, weight) in base_weights.iter().enumerate() {
            if other != branch {
                tangent = tangent * (lambda_i.clone() - weight.clone());
            }
        }
        let expected = RatFun::from_rational(fiber / tangent);
        assert_eq!(canonical.metric_norms[branch].coeff(0), Some(&expected));
    }
}

#[test]
fn relation_idempotents_do_not_diagonalize_flat_pairing_beyond_q_zero() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base_weights = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let fiber_weights = vec![Rational::from(7usize)];
    let q_degree = 3;
    let canonical = specialized_twisted_quantum_canonical_data(
        2,
        &twist,
        q_degree,
        &base_weights,
        &fiber_weights,
    )
    .unwrap();
    let flat_metric =
        twisted_flat_metric_matrix(2, q_degree, &twist, &base_weights, &fiber_weights).unwrap();
    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
    let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

    let mut found_quantum_off_diagonal = false;
    for row in 0..=2 {
        for col in 0..=2 {
            if row != col
                && canonical_metric
                    .entry(row, col)
                    .coeffs()
                    .iter()
                    .skip(1)
                    .any(|coeff| !coeff.is_zero())
            {
                found_quantum_off_diagonal = true;
            }
        }
    }
    assert!(found_quantum_off_diagonal);
}

#[test]
fn equivariant_birkhoff_s_matrix_builds_from_weighted_i_function() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let model = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
        2,
        twist,
        1,
        1,
        vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ],
        vec![Rational::from(7usize)],
    )
    .unwrap();
    let descendant_s = model.birkhoff_descendant_s_matrix(1).unwrap();

    assert_eq!(descendant_s.size(), 3);
    assert_eq!(descendant_s.q_degree(), 1);
    assert_eq!(descendant_s.z_order(), 1);
    assert_eq!(
        descendant_s.calibration(),
        &CalibrationId("negative-split-equivariant-hypergeometric-birkhoff".to_string())
    );
}

#[test]
fn o_minus_five_p1_default_birkhoff_matches_deep_laurent_reference() {
    let twist = NegativeSplitBundleTwist::new(vec![5]).unwrap();
    let base_weights = vec![Rational::from(1usize), Rational::from(2usize)];
    let fiber_weights = vec![Rational::from(3usize)];
    let bounded = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
        1,
        twist.clone(),
        1,
        1,
        base_weights.clone(),
        fiber_weights.clone(),
    )
    .unwrap()
    .birkhoff_descendant_s_matrix(1)
    .unwrap();
    let deep = NegativeSplitEquivariantHypergeometricModel::new(
        1,
        twist,
        1,
        base_weights,
        fiber_weights,
        -29,
    )
    .unwrap()
    .birkhoff_descendant_s_matrix(1)
    .unwrap();

    assert_eq!(bounded, deep);
}

#[test]
fn o_minus_four_p1_recursive_birkhoff_matches_deep_laurent_reference() {
    let twist = NegativeSplitBundleTwist::new(vec![4]).unwrap();
    let base_weights = vec![Rational::from(1usize), Rational::from(2usize)];
    let fiber_weights = vec![Rational::from(3usize)];
    let bounded = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
        1,
        twist.clone(),
        2,
        1,
        base_weights.clone(),
        fiber_weights.clone(),
    )
    .unwrap()
    .birkhoff_descendant_s_matrix(1)
    .unwrap();
    let deep_model = NegativeSplitEquivariantHypergeometricModel::new(
        1,
        twist,
        2,
        base_weights,
        fiber_weights,
        -31,
    )
    .unwrap();
    let deep_fundamental = deep_model.fundamental_solution_matrix().unwrap();
    let deep = birkhoff_descendant_s_matrix_from_fundamental(
        2,
        2,
        1,
        &deep_fundamental,
        CalibrationId("deep-o4-reference".to_string()),
    )
    .unwrap();

    assert_eq!(bounded.coefficients(), deep.coefficients());
}

#[test]
fn o_minus_eight_p1_dynamic_birkhoff_depth_matches_deep_reference() {
    let twist = NegativeSplitBundleTwist::new(vec![8]).unwrap();
    let base_weights = vec![Rational::one(), Rational::from(2)];
    let fiber_weights = vec![Rational::from(3)];
    let bounded = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
        1,
        twist.clone(),
        2,
        1,
        base_weights.clone(),
        fiber_weights.clone(),
    )
    .unwrap()
    .birkhoff_descendant_s_matrix(1)
    .unwrap();
    let deep_model = NegativeSplitEquivariantHypergeometricModel::new(
        1,
        twist,
        2,
        base_weights,
        fiber_weights,
        -40,
    )
    .unwrap();
    let deep_fundamental = deep_model.fundamental_solution_matrix().unwrap();
    let deep = birkhoff_descendant_s_matrix_from_fundamental(
        2,
        2,
        1,
        &deep_fundamental,
        CalibrationId("deep-o8-reference".to_string()),
    )
    .unwrap();

    assert_eq!(bounded.coefficients(), deep.coefficients());
}

#[test]
fn generic_line_o_minus_eight_uses_dynamic_birkhoff_depth() {
    let twist = NegativeSplitBundleTwist::new(vec![8]).unwrap();
    let base_weights = vec![Rational::one(), Rational::from(2)];
    let fiber_weights = vec![Rational::from(3)];
    let bounded = NegativeSplitLineHypergeometricModel::<Rational>::from_coeff_weights(
        1,
        twist,
        2,
        1,
        base_weights,
        &fiber_weights,
    )
    .unwrap();
    let mut deep = bounded.clone();
    deep.min_z_power = -40;

    assert_eq!(
        bounded.birkhoff_descendant_s_matrix(1).unwrap(),
        deep.birkhoff_descendant_s_matrix(1).unwrap()
    );
}

#[test]
#[ignore = "slow large-twist dynamic Laurent-depth acceptance grid"]
fn p1_twist_grid_uses_dynamic_birkhoff_depth() {
    for bundle_degree in [4usize, 5, 6, 8, 10] {
        let twist = NegativeSplitBundleTwist::new(vec![bundle_degree]).unwrap();
        let base_weights = vec![Rational::one(), Rational::from(2)];
        let fiber_weights = vec![Rational::from(3)];
        let bounded = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
            1,
            twist.clone(),
            2,
            1,
            base_weights.clone(),
            fiber_weights.clone(),
        )
        .unwrap()
        .birkhoff_descendant_s_matrix(1)
        .unwrap();
        let deep_model = NegativeSplitEquivariantHypergeometricModel::new(
            1,
            twist,
            2,
            base_weights,
            fiber_weights,
            -60,
        )
        .unwrap();
        let deep_fundamental = deep_model.fundamental_solution_matrix().unwrap();
        let deep = birkhoff_descendant_s_matrix_from_fundamental(
            2,
            2,
            1,
            &deep_fundamental,
            CalibrationId(format!("deep-o{bundle_degree}-reference")),
        )
        .unwrap();

        assert_eq!(
            bounded.coefficients(),
            deep.coefficients(),
            "O(-{bundle_degree}) on P1"
        );
    }
}

#[test]
fn conifold_birkhoff_quantum_product_is_self_adjoint_and_semisimple() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let base_weights = vec![Rational::from(1usize), Rational::from(3usize)];
    let fiber_weights = vec![Rational::from(5usize), Rational::from(7usize)];
    let canonical =
        specialized_twisted_birkhoff_canonical_data(1, &twist, 1, &base_weights, &fiber_weights)
            .unwrap();
    assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
        1,
        1,
        &twist,
        &base_weights,
        &fiber_weights,
        &canonical,
    );
    assert_eq!(canonical.roots.len(), 2);
}

#[test]
fn local_p2_birkhoff_quantum_product_is_self_adjoint_and_semisimple() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base_weights = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let fiber_weights = vec![Rational::from(7usize)];
    let canonical =
        specialized_twisted_birkhoff_canonical_data(2, &twist, 1, &base_weights, &fiber_weights)
            .unwrap();

    assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
        2,
        1,
        &twist,
        &base_weights,
        &fiber_weights,
        &canonical,
    );
    assert_eq!(canonical.roots.len(), 3);
}

#[test]
fn birkhoff_calibration_skeleton_has_inverse_psi() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let calibration = negative_split_twisted_birkhoff_calibration_skeleton(
        1,
        &twist,
        1,
        1,
        &[Rational::from(1usize), Rational::from(3usize)],
        &[Rational::from(5usize), Rational::from(7usize)],
    )
    .unwrap();

    assert_eq!(
        calibration.psi_inverse.mul(&calibration.psi),
        SeriesMatrix::identity(2, 1)
    );
    calibration.r_matrix.check_identity_calibration().unwrap();
}

#[test]
fn local_p2_birkhoff_r_candidate_is_unitary_at_low_order() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let calibration = negative_split_twisted_birkhoff_calibration_candidate(
        2,
        &twist,
        2,
        2,
        &[
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ],
        &[Rational::from(7usize)],
    )
    .unwrap();

    calibration
        .r_matrix
        .check_unitarity(&calibration.metric)
        .unwrap();
    assert_eq!(
        calibration.r_matrix.calibration(),
        &CalibrationId("negative-split-equivariant-birkhoff-qrr-candidate".to_string())
    );
}

#[test]
fn asymmetric_twisted_birkhoff_calibrations_pass_full_validation_grid() {
    struct Case {
        label: &'static str,
        n: usize,
        degrees: Vec<usize>,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
    }

    let cases = [
        Case {
            label: "O(-1)+O(-2) over P1",
            n: 1,
            degrees: vec![1, 2],
            base_weights: vec![Rational::from(2), Rational::from(7)],
            fiber_weights: vec![Rational::from(19), Rational::from(31)],
        },
        Case {
            label: "O(-1)+O(-2) over P2",
            n: 2,
            degrees: vec![1, 2],
            base_weights: vec![Rational::from(2), Rational::from(7), Rational::from(13)],
            fiber_weights: vec![Rational::from(23), Rational::from(41)],
        },
        Case {
            label: "O(-1)+O(-3) over P2",
            n: 2,
            degrees: vec![1, 3],
            base_weights: vec![Rational::from(1), Rational::from(5), Rational::from(12)],
            fiber_weights: vec![Rational::from(29), Rational::from(47)],
        },
    ];

    for case in cases {
        let twist = NegativeSplitBundleTwist::new(case.degrees).unwrap();
        let calibration =
            negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
                case.n,
                &twist,
                1,
                2,
                &case.base_weights,
                &case.fiber_weights,
                TwistedCalibrationMode::InverseEuler,
                TwistedCalibrationValidation::Full,
            )
            .unwrap_or_else(|error| panic!("{} failed full validation: {error}", case.label));

        assert_eq!(
            calibration.psi_inverse.mul(&calibration.psi),
            SeriesMatrix::identity(case.n + 1, 1),
            "{} has inconsistent canonical-frame transitions",
            case.label
        );
        calibration
            .r_matrix
            .check_unitarity(&calibration.metric)
            .unwrap_or_else(|error| panic!("{} failed explicit unitarity: {error}", case.label));
    }
}

fn assert_factored_qseries_specializes_to_rational(
    factored: &QSeries<FactoredRatFun>,
    rational: &QSeries<Rational>,
    label: &str,
) {
    assert_eq!(factored.max_degree(), rational.max_degree(), "{label}");
    for degree in 0..=rational.max_degree() {
        assert_eq!(
            factored.coeff(degree).unwrap().to_ratfun().as_rational(),
            Some(rational.coeff(degree).unwrap().clone()),
            "{label}, q^{degree}"
        );
    }
}

fn assert_factored_matrix_specializes_to_rational(
    factored: &SeriesMatrix<FactoredRatFun>,
    rational: &SeriesMatrix<Rational>,
    label: &str,
) {
    assert_eq!(
        (factored.rows(), factored.cols()),
        (rational.rows(), rational.cols())
    );
    for row in 0..rational.rows() {
        for col in 0..rational.cols() {
            assert_factored_qseries_specializes_to_rational(
                factored.entry(row, col),
                rational.entry(row, col),
                &format!("{label} entry ({row},{col})"),
            );
        }
    }
}

#[test]
fn asymmetric_calibration_agrees_between_rational_and_factored_coefficient_tiers() {
    let n = 1;
    let twist = NegativeSplitBundleTwist::new(vec![1, 2]).unwrap();
    let base_weights = vec![Rational::from(2), Rational::from(7)];
    let fiber_weights = vec![Rational::from(19), Rational::from(31)];
    let rational =
        negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
            n,
            &twist,
            1,
            1,
            &base_weights,
            &fiber_weights,
            TwistedCalibrationValidation::Full,
        )
        .unwrap();
    let factored_base = base_weights
        .iter()
        .cloned()
        .map(FactoredRatFun::from_rational)
        .collect::<Vec<_>>();
    let factored_fiber = fiber_weights
        .iter()
        .cloned()
        .map(FactoredRatFun::from_rational)
        .collect::<Vec<_>>();
    let factored =
        negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
            n,
            &twist,
            1,
            1,
            &factored_base,
            &factored_fiber,
            TwistedCalibrationValidation::Full,
        )
        .unwrap();

    for (order, (factored_coefficient, rational_coefficient)) in factored
        .r_matrix
        .coefficients()
        .iter()
        .zip(rational.r_matrix.coefficients())
        .enumerate()
    {
        assert_factored_matrix_specializes_to_rational(
            factored_coefficient,
            rational_coefficient,
            &format!("R_{order}"),
        );
    }
    for (label, factored_matrix, rational_matrix) in [
        ("metric", &factored.metric, &rational.metric),
        ("psi", &factored.psi, &rational.psi),
        ("psi inverse", &factored.psi_inverse, &rational.psi_inverse),
        ("connection", &factored.connection, &rational.connection),
    ] {
        assert_factored_matrix_specializes_to_rational(factored_matrix, rational_matrix, label);
    }
    for (label, factored_series, rational_series) in [
        ("delta", &factored.delta, &rational.delta),
        (
            "inverse delta",
            &factored.inverse_delta,
            &rational.inverse_delta,
        ),
        (
            "relative sqrt delta",
            &factored.relative_sqrt_delta,
            &rational.relative_sqrt_delta,
        ),
        (
            "inverse relative sqrt delta",
            &factored.relative_sqrt_delta_inverse,
            &rational.relative_sqrt_delta_inverse,
        ),
    ] {
        assert_eq!(factored_series.len(), rational_series.len(), "{label}");
        for (branch, (factored_entry, rational_entry)) in
            factored_series.iter().zip(rational_series).enumerate()
        {
            assert_factored_qseries_specializes_to_rational(
                factored_entry,
                rational_entry,
                &format!("{label} branch {branch}"),
            );
        }
    }
}

#[test]
fn local_p2_birkhoff_graph_recovers_known_genus_zero_divisor_row() {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 1)),
        tau(0, CohomologyClass::h_power(2, 1)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let expected = [
        (1, RatFun::from(3usize)),
        (2, RatFun::from(-45)),
        (3, RatFun::from(732usize)),
    ];

    for (degree, oracle) in expected {
        let value = crate::givental::compute_semisimple_graph_value(
            &provider,
            0,
            degree,
            &insertions,
            None,
        )
        .unwrap();
        assert_eq!(value, oracle, "local P2 <H,H,H>_0,{degree}");
    }
}

#[test]
fn o_minus_one_p2_birkhoff_graph_matches_recorded_unsourced_row() {
    let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
        2,
        vec![1],
        vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ],
        vec![Rational::from(0usize)],
    )
    .unwrap();
    let cases = [
        (tau(5, CohomologyClass::one(2)), RatFun::zero(), "tau5(1)"),
        (
            tau(4, CohomologyClass::h_power(2, 1)),
            RatFun::from_rational(Rational::new(-1, 480)),
            "tau4(H)",
        ),
        (
            tau(3, CohomologyClass::h_power(2, 2)),
            RatFun::from_rational(Rational::new(-7, 480)),
            "tau3(H^2)",
        ),
    ];

    for (insertion, expected, label) in cases {
        let value =
            crate::givental::compute_semisimple_graph_value(&provider, 2, 2, &[insertion], None)
                .unwrap();
        assert_eq!(value, expected, "O(-1)->P2 g=2 d=2 {label}");
    }
}

fn assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    canonical: &SpecializedTwistedBirkhoffCanonicalData,
) {
    let flat_metric =
        twisted_inverse_euler_flat_metric_matrix(n, q_degree, twist, base_weights, fiber_weights)
            .unwrap();
    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
    let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

    for row in 0..=n {
        for col in 0..=n {
            let expected = if row == col {
                canonical.metric_norms[row].clone()
            } else {
                QSeries::zero(q_degree)
            };
            assert_eq!(canonical_metric.entry(row, col), &expected);
        }
    }
}

#[test]
fn twisted_relation_calibration_skeleton_has_inverse_psi() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let calibration = negative_split_twisted_relation_calibration_skeleton(
        1,
        &twist,
        2,
        2,
        &[Rational::from(1usize), Rational::from(3usize)],
        &[Rational::from(5usize), Rational::from(7usize)],
    )
    .unwrap();

    assert_eq!(
        calibration.psi_inverse.mul(&calibration.psi),
        SeriesMatrix::identity(2, 2)
    );
    calibration.r_matrix.check_identity_calibration().unwrap();
}

#[test]
fn twisted_classical_diagonal_subtracts_inverse_euler_fiber_qrr_term() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let diagonal = twisted_classical_limit_diagonal_coefficients(
        2,
        &twist,
        2,
        &[
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ],
        &[Rational::from(7usize)],
    )
    .unwrap();

    let tangent =
        Rational::one() / Rational::from(1usize) + Rational::one() / Rational::from(3usize);
    let fiber = Rational::one() / Rational::from(4usize);
    let expected = RatFun::from_rational((tangent - fiber) / Rational::from(12usize));
    assert_eq!(diagonal[0][1], expected);
}

#[test]
fn twisted_candidate_r_calibration_fails_unitarity_guard() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let err = negative_split_twisted_relation_calibration_candidate(
        1,
        &twist,
        2,
        2,
        &[Rational::from(1usize), Rational::from(3usize)],
        &[Rational::from(5usize), Rational::from(7usize)],
    )
    .unwrap_err();

    match err {
        GwError::ValidationFailure(message) => {
            assert!(message.contains("R(-z)^T eta R(z)"));
        }
        other => panic!("expected unitarity validation failure, got {other:?}"),
    }
}

#[test]
fn twisted_raw_r_candidate_has_nontrivial_r_coefficients() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let calibration = negative_split_twisted_relation_calibration_raw_candidate(
        2,
        &twist,
        1,
        2,
        &[
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ],
        &[Rational::from(7usize)],
    )
    .unwrap();

    assert!(calibration
        .r_matrix
        .coefficient(1)
        .unwrap()
        .entries()
        .iter()
        .flat_map(|row| row.iter())
        .any(|entry| !entry.is_zero()));
}

#[test]
fn qrr_and_direct_hypergeometric_paths_have_same_mirror_data() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let direct = NegativeSplitHypergeometricModel::new(2, twist.clone(), 3);
    let qrr = NegativeSplitQrrModel::new(2, twist, 3);

    assert_eq!(
        qrr.i_coefficients(),
        direct.i_coefficients(),
        "QRR-factorized coefficients should reproduce the direct local I-function"
    );
    assert_eq!(
        qrr.mirror_map_coefficients(),
        direct.mirror_map_coefficients()
    );
    assert_eq!(
        qrr.inverse_mirror_map_coefficients(),
        direct.inverse_mirror_map_coefficients()
    );
    assert_eq!(
        qrr.mirror_transformed_j_coefficients(),
        direct.mirror_transformed_j_coefficients()
    );
}

#[test]
fn local_p2_mirror_transform_cancels_j_h_over_z_terms() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let model = NegativeSplitHypergeometricModel::new(2, twist, 3);
    let j_coefficients = model.mirror_transformed_j_coefficients();

    for coefficient in j_coefficients.iter().take(4).skip(1) {
        assert_eq!(coefficient.coefficient(1, -1), Rational::zero());
    }
}

#[test]
fn local_p2_fundamental_solution_is_identity_at_q_zero() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let model = NegativeSplitHypergeometricModel::new(2, twist, 2);
    let fundamental = model.fundamental_solution_matrix();

    for (z_power, matrix) in &fundamental {
        let q0 = matrix_q_coefficient(matrix, 0);
        let expected = if *z_power == 0 {
            identity_coeff_matrix(3)
        } else {
            zero_coeff_matrix(3)
        };
        assert_eq!(q0, expected);
    }
}

#[test]
fn twisted_quantum_relation_records_resolved_conifold_symbol() {
    let relation = TwistedQuantumRelation::new(
        1,
        NegativeSplitBundleTwist::new(vec![1, 1]).unwrap(),
        vec![Rational::from(1usize), Rational::from(2usize)],
    )
    .unwrap();
    let coefficients = relation.coefficients(1);

    assert_eq!(
        coefficients[0].coeff(0),
        Some(&RatFun::from_rational(Rational::from(2usize)))
    );
    assert_eq!(
        coefficients[1].coeff(0),
        Some(&RatFun::from_rational(Rational::from(-3)))
    );
    assert_eq!(coefficients[2].coeff(0), Some(&RatFun::one()));
    assert_eq!(
        coefficients[2].coeff(1),
        Some(&RatFun::from_rational(Rational::from(-1)))
    );
}

#[test]
fn twisted_descendant_s_matrix_uses_hypergeometric_birkhoff_model() {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    let descendant_s = provider.descendant_s_matrix(2, 2).unwrap();

    assert_eq!(descendant_s.size(), 3);
    assert_eq!(descendant_s.q_degree(), 2);
    assert_eq!(descendant_s.z_order(), 2);
    assert_eq!(
        descendant_s.calibration(),
        &CalibrationId(
            "negative-split-equivariant-hypergeometric-birkhoff-metric-adjoint".to_string()
        )
    );
    assert_eq!(
        descendant_s.coefficient(0),
        Some(&SeriesMatrix::identity(3, 2))
    );
    assert!(descendant_s
        .coefficient(1)
        .unwrap()
        .entries()
        .iter()
        .flat_map(|row| row.iter())
        .any(|entry| !entry.coeff(1).unwrap().is_zero()));
}

#[test]
fn qrr_birkhoff_s_matches_direct_hypergeometric_birkhoff_s() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let direct = NegativeSplitHypergeometricModel::new(1, twist.clone(), 3);
    let qrr = NegativeSplitQrrModel::new(1, twist, 3);

    assert_eq!(
        qrr.birkhoff_descendant_s_matrix(2).unwrap().coefficients(),
        direct
            .birkhoff_descendant_s_matrix(2)
            .unwrap()
            .coefficients()
    );
}

#[test]
fn qrr_birkhoff_s_has_separate_calibration_label() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let qrr = NegativeSplitQrrModel::new(2, twist, 2);
    let descendant_s = qrr.birkhoff_descendant_s_matrix(2).unwrap();

    assert_eq!(
        descendant_s.calibration(),
        &CalibrationId("negative-split-qrr-hypergeometric-birkhoff".to_string())
    );
}

#[test]
fn negative_split_compute_matches_resolved_conifold_closed_formula() {
    let cases = [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1)];
    for (genus, degree) in cases {
        let req = TwistedInvariantRequest::new(1, vec![1, 1], genus, degree, Vec::new()).unwrap();
        let result = compute_negative_split_twisted(&req).unwrap();
        assert_eq!(
            result.value,
            RatFun::from_rational(
                crate::validation_backends::local_cy::resolved_conifold_gw(genus, degree).unwrap()
            ),
            "resolved conifold g={genus} d={degree}"
        );
        assert_eq!(
            result.engine,
            "twisted-negative-split-givental-birkhoff-early-line"
        );
    }
}

#[test]
fn o_minus_one_p2_genus_zero_two_point_descendant_uses_full_divisor_recursion() {
    let req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        2,
        vec![
            tau(2, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
        ],
    )
    .unwrap();
    let result = compute_negative_split_twisted(&req).unwrap();
    assert_eq!(result.value, RatFun::from_rational(Rational::new(-1, 2)));
    assert!(result.notes[0].contains("full descendant divisor equation"));
}

#[test]
fn o_minus_one_p2_genus_zero_three_primary_uses_frobenius_product() {
    let mut req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ],
    )
    .unwrap();
    req.equivariant = true;

    let result = compute_negative_split_twisted(&req).unwrap();
    assert_eq!(result.value, RatFun::one());
    assert!(result.notes[0].contains("Frobenius quantum product"));
}

#[test]
fn o_minus_two_p2_three_primary_is_permutation_symmetric() {
    let h = tau(0, CohomologyClass::h_power(2, 1));
    let h2 = tau(0, CohomologyClass::h_power(2, 2));
    let expected = RatFun::from_rational(Rational::from(-1));
    for insertions in [
        vec![h2.clone(), h.clone(), h.clone()],
        vec![h.clone(), h2.clone(), h.clone()],
        vec![h.clone(), h.clone(), h2.clone()],
    ] {
        let req = TwistedInvariantRequest::new(2, vec![2], 0, 1, insertions).unwrap();
        let value = compute_negative_split_twisted(&req).unwrap().value;
        assert!(
            value.equivalent(&expected),
            "three-primary value should be -1 for every insertion order, got {value}"
        );
    }
}

#[test]
fn o_minus_two_p2_three_primary_matches_generic_graph_and_divisor_equation() {
    let h = tau(0, CohomologyClass::h_power(2, 1));
    let h2 = tau(0, CohomologyClass::h_power(2, 2));
    let three_insertions = vec![h2.clone(), h.clone(), h.clone()];
    let mut four_insertions = three_insertions.clone();
    four_insertions.push(h.clone());

    let direct_provider = TwistedProjectiveSpaceProvider::new(2, vec![2], false).unwrap();
    let direct = crate::givental::compute_semisimple_graph_value(
        &direct_provider,
        0,
        1,
        &three_insertions,
        None,
    )
    .unwrap();
    // The series entry point deliberately bypasses the provider's scalar
    // `direct_value` shortcut, so this is an independent graph-path value for
    // exactly the same three-point correlator.
    let generic = crate::givental::compute_semisimple_graph_series(
        &direct_provider,
        0,
        1,
        &three_insertions,
        None,
    )
    .unwrap()
    .coeff(1)
    .cloned()
    .unwrap_or_else(RatFun::zero);
    let four_point = crate::givental::compute_semisimple_graph_value(
        &direct_provider,
        0,
        1,
        &four_insertions,
        None,
    )
    .unwrap();
    let expected = RatFun::from_rational(Rational::from(-1));
    assert!(direct.equivalent(&expected), "direct value: {direct}");
    assert!(generic.equivalent(&expected), "generic value: {generic}");
    assert!(
        four_point.equivalent(&expected),
        "four-point value: {four_point}"
    );

    // A degree-two row makes the divisor factor nontrivial: adding H
    // multiplies the invariant by its curve degree, namely two.
    let degree_two_three = vec![h2.clone(), h2.clone(), h.clone()];
    let mut degree_two_four = degree_two_three.clone();
    degree_two_four.push(h);
    let three_value = crate::givental::compute_semisimple_graph_value(
        &direct_provider,
        0,
        2,
        &degree_two_three,
        None,
    )
    .unwrap();
    let four_value = crate::givental::compute_semisimple_graph_value(
        &direct_provider,
        0,
        2,
        &degree_two_four,
        None,
    )
    .unwrap();
    assert!(three_value.equivalent(&RatFun::from_rational(Rational::from(-2))));
    assert!(four_value.equivalent(&(&three_value * &RatFun::from_rational(Rational::from(2)))));
}

#[test]
fn factored_o_minus_two_p2_three_primary_is_permutation_symmetric() {
    let h = tau(0, CohomologyClass::h_power(2, 1));
    let h2 = tau(0, CohomologyClass::h_power(2, 2));
    let expected = RatFun::from_rational(Rational::from(-1));
    for insertions in [
        vec![h2.clone(), h.clone(), h.clone()],
        vec![h.clone(), h2.clone(), h.clone()],
        vec![h.clone(), h.clone(), h2.clone()],
    ] {
        let mut req = TwistedInvariantRequest::new(2, vec![2], 0, 1, insertions).unwrap();
        req.equivariant = true;
        let value = compute_negative_split_twisted_factored(&req)
            .unwrap()
            .to_ratfun();
        assert!(
            value.equivalent(&expected),
            "factored three-primary value should be -1, got {value}"
        );
    }
}

#[test]
fn factored_o_minus_one_p2_three_primary_matches_expanded_product() {
    let mut req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ],
    )
    .unwrap();
    req.equivariant = true;

    let factored = compute_negative_split_twisted_factored(&req).unwrap();
    assert_eq!(factored.to_ratfun(), RatFun::one());
}

#[test]
fn fiber_equivariant_twisted_does_not_prune_dimension_mismatch() {
    let mut zero_req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
        ],
    )
    .unwrap();

    let nonequivariant = compute_negative_split_twisted(&zero_req).unwrap();
    assert_eq!(nonequivariant.value, RatFun::zero());
    assert_eq!(nonequivariant.engine, "twisted-negative-split-dimension");

    zero_req.equivariant = true;
    let expanded_zero = compute_negative_split_twisted(&zero_req).unwrap();
    assert_eq!(expanded_zero.value, RatFun::zero());
    let factored_zero = compute_negative_split_twisted_factored(&zero_req).unwrap();
    assert_eq!(factored_zero.to_ratfun(), RatFun::zero());

    let mut nonzero_req = TwistedInvariantRequest::new(
        2,
        vec![2],
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ],
    )
    .unwrap();
    let expected = RatFun::variable("mu_0");

    let nonequivariant = compute_negative_split_twisted(&nonzero_req).unwrap();
    assert_eq!(nonequivariant.value, RatFun::zero());
    assert_eq!(nonequivariant.engine, "twisted-negative-split-dimension");

    nonzero_req.equivariant = true;
    let expanded = compute_negative_split_twisted(&nonzero_req).unwrap();
    assert_eq!(expanded.value, expected);
    assert!(expanded.notes[0].contains("Frobenius quantum product"));

    let factored = compute_negative_split_twisted_factored(&nonzero_req).unwrap();
    assert_eq!(factored.to_ratfun(), expected);
}

#[test]
fn nonequivariant_negative_virtual_dimension_is_zero() {
    let req = TwistedInvariantRequest::new(2, vec![4], 2, 1, Vec::new()).unwrap();
    let result = compute_negative_split_twisted(&req).unwrap();
    assert_eq!(result.value, RatFun::zero());
    assert_eq!(result.engine, "twisted-negative-split-dimension");
    assert!(result.notes[0].contains("virtual dimension -1"));
}

#[test]
fn factored_negative_virtual_dimension_takes_base_limit() {
    let mut req = TwistedInvariantRequest::new(
        1,
        vec![5],
        0,
        1,
        vec![tau(0, CohomologyClass::h_power(1, 1)); 3],
    )
    .unwrap();
    req.equivariant = true;

    let expected = RatFun::variable("mu_0").pow_usize(4);
    assert_eq!(
        compute_negative_split_twisted(&req).unwrap().value,
        expected
    );
    assert_eq!(
        compute_negative_split_twisted_factored(&req)
            .unwrap()
            .to_ratfun(),
        expected
    );
}

#[test]
fn fiber_equivariant_degree_one_top_terms_match_untwisted_p2() {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let untwisted =
        crate::compute(crate::InvariantRequest::new(2, 0, 1, insertions.clone())).unwrap();
    assert_eq!(untwisted.value, RatFun::one());

    let cases = [
        (vec![2], RatFun::variable("mu_0")),
        {
            let mu = RatFun::variable("mu_0");
            (vec![3], &mu * &mu)
        },
        {
            let mu0 = RatFun::variable("mu_0");
            let mu1 = RatFun::variable("mu_1");
            (vec![2, 2], &mu0 * &mu1)
        },
    ];

    for (twist, expected) in cases {
        let nonequivariant =
            TwistedInvariantRequest::new(2, twist.clone(), 0, 1, insertions.clone()).unwrap();
        let local_constant_term = compute_negative_split_twisted(&nonequivariant).unwrap();
        assert_eq!(
            local_constant_term.value,
            RatFun::zero(),
            "constant term for twist {twist:?}"
        );

        let mut equivariant = nonequivariant;
        equivariant.equivariant = true;
        let value = compute_negative_split_twisted_factored(&equivariant).unwrap();
        let mut zero_fiber_weights = BTreeMap::new();
        for idx in 0..twist.len() {
            zero_fiber_weights.insert(format!("mu_{idx}"), Rational::zero());
        }

        assert_eq!(value.to_ratfun(), expected, "top term for twist {twist:?}");
        assert_eq!(
            value.evaluate_variables(&zero_fiber_weights).unwrap(),
            Rational::zero(),
            "constant term by mu=0 for twist {twist:?}"
        );
    }
}

#[test]
fn fiber_equivariant_degree_one_top_terms_match_higher_projective_spaces() {
    let cases = [
        (
            3,
            vec![4],
            vec![
                tau(0, CohomologyClass::h_power(3, 3)),
                tau(0, CohomologyClass::h_power(3, 3)),
                tau(0, CohomologyClass::h_power(3, 1)),
            ],
            {
                let mu = RatFun::variable("mu_0");
                &(&mu * &mu) * &mu
            },
        ),
        (
            3,
            vec![2, 2],
            vec![
                tau(0, CohomologyClass::h_power(3, 3)),
                tau(0, CohomologyClass::h_power(3, 3)),
                tau(0, CohomologyClass::h_power(3, 1)),
            ],
            {
                let mu0 = RatFun::variable("mu_0");
                let mu1 = RatFun::variable("mu_1");
                &mu0 * &mu1
            },
        ),
    ];

    for (n, twist, insertions, expected) in cases {
        let untwisted =
            crate::compute(crate::InvariantRequest::new(n, 0, 1, insertions.clone())).unwrap();
        assert_eq!(untwisted.value, RatFun::one(), "untwisted P^{n}");

        let nonequivariant =
            TwistedInvariantRequest::new(n, twist.clone(), 0, 1, insertions).unwrap();
        let local_constant_term = compute_negative_split_twisted(&nonequivariant).unwrap();
        assert_eq!(
            local_constant_term.value,
            RatFun::zero(),
            "constant term for P^{n}, twist {twist:?}"
        );

        let mut equivariant = nonequivariant;
        equivariant.equivariant = true;
        let value = compute_negative_split_twisted_factored(&equivariant).unwrap();
        assert_eq!(value.to_ratfun(), expected, "top term for twist {twist:?}");
    }
}

#[test]
fn fiber_equivariant_factored_constant_matches_local_limit() {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(3, 2)),
        tau(0, CohomologyClass::h_power(3, 1)),
        tau(0, CohomologyClass::h_power(3, 1)),
    ];
    let nonequivariant =
        TwistedInvariantRequest::new(3, vec![4], 0, 1, insertions.clone()).unwrap();
    let local = compute_negative_split_twisted(&nonequivariant).unwrap();
    assert_eq!(local.value, RatFun::from_rational(Rational::from(-20)));

    let ordinary =
        crate::compute(crate::InvariantRequest::new(3, 0, 1, insertions.clone())).unwrap();
    assert_eq!(ordinary.value, RatFun::zero());

    let mut divisor_insertions = insertions.clone();
    divisor_insertions.push(tau(0, CohomologyClass::h_power(3, 1)));
    let divisor_request =
        TwistedInvariantRequest::new(3, vec![4], 0, 1, divisor_insertions).unwrap();
    let generic_four_point = compute_negative_split_twisted(&divisor_request).unwrap();
    assert_eq!(
        generic_four_point.value,
        RatFun::from_rational(Rational::from(-20))
    );

    let mut equivariant = TwistedInvariantRequest::new(3, vec![4], 0, 1, insertions).unwrap();
    equivariant.equivariant = true;
    let factored = compute_negative_split_twisted_factored(&equivariant).unwrap();
    for weight in [0usize, 1, 2, 5] {
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(weight));
        assert_eq!(
            factored.evaluate_variables(&values).unwrap(),
            Rational::from(-20),
            "fiber weight mu_0={weight}"
        );
    }
}

#[test]
fn fiber_equivariant_resolvent_uses_factored_packed_path() {
    let req = ResolventRequest {
        target_n: 1,
        genus: 1,
        degree: 1,
        markings: 1,
        virtual_dimension: 1,
    };
    let result =
        compute_negative_split_twisted_resolvent_packed_factored(1, vec![1, 1], &req).unwrap();

    assert_eq!(
        result.engine,
        "twisted-negative-split-fiber-equivariant-factored-packed-resolvent"
    );
    assert_eq!(result.candidate_terms, 2);
    assert_eq!(result.nonzero_terms, 2);
    assert_eq!(result.value.term_count(), 2);
    assert!(result.value.coefficient_text_contains("mu_0"));
    assert!(result.value.coefficient_text_contains("mu_1"));

    for (mu0, mu1) in [(3usize, 5usize), (5, 7)] {
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(mu0));
        values.insert("mu_1".to_string(), Rational::from(mu1));
        let specialized = result.value.evaluate_variables(&values).unwrap();

        let rational_provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            1,
            vec![1, 1],
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(mu0), Rational::from(mu1)],
        )
        .unwrap();
        let rational = crate::givental::compute_packed_resolvent_with_provider(
            &req,
            rational_provider,
            "test-rational-fiber-resolvent",
            "test",
            Ok::<RatFun, GwError>,
        )
        .unwrap();
        assert_eq!(
            specialized, rational.value,
            "specialization mu_0={mu0}, mu_1={mu1}"
        );
    }

    let mut values = BTreeMap::new();
    values.insert("mu_0".to_string(), Rational::from(3usize));
    values.insert("mu_1".to_string(), Rational::from(5usize));
    let left = result.value.evaluate_variables(&values).unwrap();
    values.insert("mu_0".to_string(), Rational::from(5usize));
    values.insert("mu_1".to_string(), Rational::from(3usize));
    let right = result.value.evaluate_variables(&values).unwrap();
    assert_eq!(left, right, "O(-1)+O(-1) symmetry swaps fiber weights");
}

#[test]
fn early_rational_twisted_graph_value_matches_lambda_line_limit() {
    let provider = TwistedProjectiveSpaceProvider::new(1, vec![1, 1], false).unwrap();
    let raw = crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
    let oracle = RatFun::from_rational(
        crate::validation_backends::local_cy::resolved_conifold_gw(2, 1).unwrap(),
    );

    assert_eq!(raw, oracle);
}

#[test]
fn symbolic_raw_twisted_graph_value_has_correct_lambda_line_limit() {
    let provider =
        TwistedProjectiveSpaceProvider::symbolic_lambda_line(1, vec![1, 1], false).unwrap();
    let raw = crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
    let oracle = RatFun::from_rational(
        crate::validation_backends::local_cy::resolved_conifold_gw(2, 1).unwrap(),
    );
    let limit = RatFun::from_rational(
        raw.nonequivariant_limit_line(0, &[Rational::one()])
            .unwrap(),
    );

    assert_eq!(limit, oracle);
}

#[test]
fn fiber_equivariant_i_function_specializes_to_rational_fiber_weights() {
    let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
    let base = vec![Rational::from(1usize), Rational::from(2usize)];
    let rational_fiber = vec![Rational::from(3usize), Rational::from(5usize)];
    let symbolic_base = base
        .iter()
        .cloned()
        .map(RatFun::from_rational)
        .collect::<Vec<_>>();
    let symbolic_fiber = vec![RatFun::variable("mu_0"), RatFun::variable("mu_1")];
    let symbolic = negative_split_equivariant_i_function_coefficient_coeff(
        1,
        &twist,
        1,
        &symbolic_base,
        &symbolic_fiber,
        -4,
    )
    .unwrap();
    let rational =
        negative_split_equivariant_i_function_coefficient(1, &twist, 1, &base, &rational_fiber, -4)
            .unwrap();
    let mut values = BTreeMap::new();
    values.insert("mu_0".to_string(), Rational::from(3usize));
    values.insert("mu_1".to_string(), Rational::from(5usize));

    let rendered = (0..=1)
        .flat_map(|h_power| (-4..=0).map(move |z_power| (h_power, z_power)))
        .map(|(h_power, z_power)| symbolic.coefficient(h_power, z_power).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(rendered.contains("mu_0"));
    assert!(rendered.contains("mu_1"));
    for h_power in 0..=1 {
        for z_power in -4..=0 {
            let specialized = symbolic
                .coefficient(h_power, z_power)
                .evaluate_variables(&values)
                .unwrap();
            assert_eq!(specialized, rational.coefficient(h_power, z_power));
        }
    }
}

#[test]
fn fiber_equivariant_inverse_euler_pairing_specializes_to_rational_weights() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let rational_fiber = vec![Rational::from(7usize)];
    let symbolic_base = base
        .iter()
        .cloned()
        .map(RatFun::from_rational)
        .collect::<Vec<_>>();
    let symbolic_fiber = vec![RatFun::variable("mu_0")];
    let symbolic = twisted_inverse_euler_flat_metric_matrix_ratfun(
        2,
        0,
        &twist,
        &symbolic_base,
        &symbolic_fiber,
    )
    .unwrap();
    let rational =
        twisted_inverse_euler_flat_metric_matrix(2, 0, &twist, &base, &rational_fiber).unwrap();
    let mut values = BTreeMap::new();
    values.insert("mu_0".to_string(), Rational::from(7usize));

    assert!(symbolic
        .entry(0, 0)
        .coeff(0)
        .unwrap()
        .to_string()
        .contains("mu_0"));
    for row in 0..=2 {
        for col in 0..=2 {
            let specialized = symbolic
                .entry(row, col)
                .coeff(0)
                .unwrap()
                .evaluate_variables(&values)
                .unwrap();
            let expected = rational
                .entry(row, col)
                .coeff(0)
                .unwrap()
                .as_rational()
                .unwrap();
            assert_eq!(specialized, expected);
        }
    }
}

#[test]
fn negative_split_compute_matches_local_p2_degree_one() {
    let req = TwistedInvariantRequest::new(2, vec![3], 2, 1, Vec::new()).unwrap();
    let result = compute_negative_split_twisted(&req).unwrap();
    assert_eq!(
        result.value,
        RatFun::from_rational(crate::validation_backends::local_cy::local_p2_gw(2, 1).unwrap(),)
    );
}

#[test]
fn local_p2_unmarked_low_genus_uses_divisor_reconstruction() {
    for (genus, expected) in [(0, Rational::from(3)), (1, Rational::new(1, 4))] {
        let req = TwistedInvariantRequest::new(2, vec![3], genus, 1, Vec::new()).unwrap();
        let result = compute_negative_split_twisted(&req).unwrap();
        assert!(result
            .value
            .equivalent(&RatFun::from_rational(expected.clone())));
        assert!(result
            .notes
            .iter()
            .any(|note| note.contains("divisor equation")));
    }
}

#[test]
fn factored_twisted_compute_requires_equivariant_mode() {
    let req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        2,
        vec![
            tau(2, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
        ],
    )
    .unwrap();
    let err = compute_negative_split_twisted_factored(&req).unwrap_err();

    assert!(err.to_string().contains("--equivariant"));
}

#[test]
fn factored_fundamental_class_cancellation_displays_as_zero() {
    let mut req = TwistedInvariantRequest::new(
        2,
        vec![4],
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::one(2)),
        ],
    )
    .unwrap();
    req.equivariant = true;
    let value = compute_negative_split_twisted_factored(&req).unwrap();
    assert!(value.is_zero());
    assert_eq!(value.to_string(), "0");
}

#[test]
fn factored_twisted_two_point_uses_native_s_matrix() {
    let mut req = TwistedInvariantRequest::new(
        2,
        vec![1],
        0,
        1,
        vec![
            tau(1, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ],
    )
    .unwrap();
    req.equivariant = true;

    let value = compute_negative_split_twisted_factored(&req).unwrap();

    assert_eq!(value.to_ratfun(), RatFun::one());
}

#[test]
fn factored_two_point_raw_s_pairing_matches_metric_adjoint_pairing() {
    let twist = NegativeSplitBundleTwist::new(vec![1]).unwrap();
    let base = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let factored_base = base
        .iter()
        .cloned()
        .map(FactoredRatFun::from_rational)
        .collect::<Vec<_>>();
    let fiber = vec![FactoredRatFun::variable("mu_0")];
    let degree = 1;
    let s_order = 2;

    let raw_s = NegativeSplitLineHypergeometricModel::<FactoredRatFun>::from_coeff_weights(
        2,
        twist.clone(),
        degree,
        s_order,
        factored_base,
        &fiber,
    )
    .unwrap()
    .birkhoff_descendant_s_matrix(s_order)
    .unwrap();
    let (metric, metric_inverse) =
        twisted_inverse_euler_flat_metric_pair_from_rational_base(2, degree, &twist, &base, &fiber)
            .unwrap();
    let adjoint_s = metric_adjoint_descendant_s_matrix_with_inverse_coeff(
        raw_s.clone(),
        &metric,
        &metric_inverse,
    )
    .unwrap();
    let mut descendant = vec![QSeries::<FactoredRatFun>::zero(degree); 3];
    descendant[2] = QSeries::one(degree);
    let mut primary = vec![QSeries::<FactoredRatFun>::zero(degree); 3];
    primary[1] = QSeries::one(degree);

    let direct = genus_zero_two_point_raw_s_matrix_pairing_coeff(
        3,
        degree,
        s_order,
        &raw_s,
        &metric,
        &descendant,
        &primary,
    )
    .unwrap();
    let adjoint = genus_zero_two_point_s_matrix_pairing_coeff(
        3,
        degree,
        s_order,
        &adjoint_s,
        &metric,
        &descendant,
        &primary,
    )
    .unwrap();

    assert_eq!(direct.to_ratfun(), adjoint.to_ratfun());
}

#[test]
fn factored_flat_metric_vandermonde_inverse_is_identity() {
    let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    let base = vec![
        Rational::from(1usize),
        Rational::from(2usize),
        Rational::from(4usize),
    ];
    let fiber = vec![FactoredRatFun::variable("mu_0")];
    let (metric, inverse) =
        twisted_inverse_euler_flat_metric_pair_from_rational_base(2, 0, &twist, &base, &fiber)
            .unwrap();
    let product = metric.mul(&inverse);
    let mut values = BTreeMap::new();
    values.insert("mu_0".to_string(), Rational::from(7usize));

    for row in 0..=2 {
        for col in 0..=2 {
            let actual = product
                .entry(row, col)
                .coeff(0)
                .unwrap()
                .evaluate_variables(&values)
                .unwrap();
            let expected = if row == col {
                Rational::one()
            } else {
                Rational::zero()
            };
            assert_eq!(actual, expected, "entry ({row},{col})");
        }
    }
}

#[test]
fn factored_s_matrix_conversion_round_trips_to_ratfun() {
    let mu = RatFun::variable("mu_0");
    let entry = &mu / &(&mu + &RatFun::from(1usize));
    let matrix = SeriesMatrix::constant(vec![vec![entry.clone()]], 0);
    let expanded =
        SeriesSMatrix::from_coefficients(1, 0, 0, vec![matrix], CalibrationId("test".into()))
            .unwrap();
    let factored = series_s_matrix_to_factored(&expanded).unwrap();

    assert_eq!(
        factored
            .coefficient(0)
            .unwrap()
            .entry(0, 0)
            .coeff(0)
            .unwrap()
            .to_ratfun(),
        expanded
            .coefficient(0)
            .unwrap()
            .entry(0, 0)
            .coeff(0)
            .unwrap()
            .clone()
    );
}

#[test]
fn generic_h_laurent_series_preserves_factored_coefficients() {
    let mu = FactoredRatFun::variable("mu_0");
    let mut series = HCoeffLaurentSeries::<FactoredRatFun>::one(1);
    series.add_term(1, -1, mu.clone());
    let relation = vec![FactoredRatFun::one(), FactoredRatFun::zero()];
    let product = series.multiply_mod_relation(&series, &relation);

    assert_eq!(product.coefficient(0, 0).to_ratfun(), RatFun::one());
    assert_eq!(
        product.coefficient(1, -1).to_ratfun(),
        &RatFun::from(2usize) * &RatFun::variable("mu_0")
    );
}

#[test]
fn generic_birkhoff_split_preserves_factored_coefficients() {
    let mu = FactoredRatFun::variable("mu_0");
    let mut fundamental = BTreeMap::new();
    fundamental.insert(
        0,
        SeriesMatrix::from_entries(vec![vec![QSeries::from_coeffs(vec![
            FactoredRatFun::one(),
            FactoredRatFun::zero(),
        ])]]),
    );
    fundamental.insert(
        -1,
        SeriesMatrix::from_entries(vec![vec![QSeries::from_coeffs(vec![
            FactoredRatFun::zero(),
            mu.clone(),
        ])]]),
    );

    let (_, negative) = birkhoff_factor_by_q_degree(1, 1, &fundamental).unwrap();
    let coefficients = negative_factor_to_s_coefficients(1, 1, 1, &negative);

    assert_eq!(
        coefficients[1].entry(0, 0).coeff(1).unwrap().to_ratfun(),
        RatFun::variable("mu_0")
    );
}

#[test]
fn bounded_q_birkhoff_matches_full_factorization_window() {
    let q_degree = 3;
    let z_order = 1;
    let scalar_matrix = |coeffs: Vec<Rational>| {
        SeriesMatrix::from_entries(vec![vec![QSeries::from_coeffs(coeffs)]])
    };

    let mut fundamental = BTreeMap::new();
    fundamental.insert(
        0,
        scalar_matrix(vec![
            Rational::one(),
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
        ]),
    );
    fundamental.insert(
        2,
        scalar_matrix(vec![
            Rational::zero(),
            Rational::from(3),
            Rational::from(5),
            Rational::zero(),
        ]),
    );
    fundamental.insert(
        -3,
        scalar_matrix(vec![
            Rational::zero(),
            Rational::from(7),
            Rational::zero(),
            Rational::from(11),
        ]),
    );
    fundamental.insert(
        -1,
        scalar_matrix(vec![
            Rational::zero(),
            Rational::zero(),
            Rational::from(13),
            Rational::from(17),
        ]),
    );

    let (_, full_negative) = birkhoff_factor_by_q_degree(1, q_degree, &fundamental).unwrap();
    let full_coefficients = negative_factor_to_s_coefficients(1, q_degree, z_order, &full_negative);
    let bounded = birkhoff_descendant_s_matrix_from_fundamental_coeff(
        1,
        q_degree,
        z_order,
        &fundamental,
        CalibrationId("bounded-q-test".to_string()),
    )
    .unwrap();

    assert_eq!(bounded.coefficients(), full_coefficients.as_slice());
}

#[test]
fn packed_resolvent_matches_invariant_wise_local_p2() {
    let req = crate::spaces::projective_space::resolvent::ResolventRequest {
        target_n: 2,
        genus: 2,
        degree: 1,
        markings: 1,
        virtual_dimension: 1,
    };
    let packed = compute_negative_split_twisted_resolvent_packed(2, vec![3], &req, false).unwrap();
    let invariant_wise =
        crate::spaces::projective_space::resolvent::compute_resolvent_generating_function(
            &req,
            |insertions| {
                let invariant_req =
                    TwistedInvariantRequest::new(2, vec![3], 2, 1, insertions.to_vec())?;
                compute_negative_split_twisted(&invariant_req)
            },
        )
        .unwrap();

    assert_eq!(packed.value, invariant_wise.value);
    assert_eq!(packed.candidate_terms, invariant_wise.candidate_terms);
    assert_eq!(packed.nonzero_terms, invariant_wise.nonzero_terms);
}

#[test]
fn twisted_packed_resolvent_rejects_target_mismatch() {
    let req = ResolventRequest {
        target_n: 1,
        genus: 0,
        degree: 1,
        markings: 3,
        virtual_dimension: 3,
    };
    let expanded =
        compute_negative_split_twisted_resolvent_packed(2, vec![3], &req, false).unwrap_err();
    assert!(matches!(expanded, GwError::ConventionMismatch(_)));

    let factored =
        compute_negative_split_twisted_resolvent_packed_factored(2, vec![3], &req).unwrap_err();
    assert!(matches!(factored, GwError::ConventionMismatch(_)));
}

#[test]
#[allow(deprecated)]
fn non_cy_twist_can_still_select_one_degree() {
    let twist = NegativeSplitBundleTwist::new(vec![1]).unwrap();
    let insertion_degree = Some(3);
    assert_eq!(
        twist.candidate_degrees(2, 0, 5, 1, insertion_degree),
        vec![1]
    );
}
