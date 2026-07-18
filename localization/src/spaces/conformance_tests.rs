//! Cross-target conformance checks for the canonical space theories.

use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, CurveEffectivity, GwTheory, StateSpaceMatrix};
use crate::spaces::negative_split_projective::{
    NegativeSplitProjectiveCompletion, NegativeSplitTotalSpaceTheory,
};
use crate::spaces::product_projective::ProductProjectiveTheory;
use crate::spaces::projective_bundle::ProjectiveBundleTheory;
use crate::spaces::projective_space::ProjectiveSpaceTheory;
use std::collections::BTreeMap;

fn multiply_expansion(
    theory: &dyn GwTheory,
    expansion: &[(BasisId, Rational)],
    right: BasisId,
) -> BTreeMap<BasisId, Rational> {
    let mut out = BTreeMap::new();
    for (left, left_coefficient) in expansion {
        for (basis, coefficient) in theory.classical_product(*left, right).unwrap() {
            *out.entry(basis).or_insert_with(Rational::zero) +=
                left_coefficient.clone() * coefficient;
        }
    }
    out.retain(|_, coefficient| !coefficient.is_zero());
    out
}

fn assert_classical_frobenius_algebra(theory: &dyn GwTheory) {
    let size = theory.state_space().basis.len();
    let unit = theory.state_space().unit;
    let metric = &theory.state_space().pairing.as_ref().unwrap().metric;
    for left in 0..size {
        let left = BasisId(left);
        assert_eq!(
            theory.classical_product(unit, left).unwrap(),
            vec![(left, Rational::one())]
        );
        for right in 0..size {
            let right = BasisId(right);
            assert_eq!(
                theory.classical_product(left, right).unwrap(),
                theory.classical_product(right, left).unwrap(),
                "classical product must be commutative"
            );
            for third in 0..size {
                let third = BasisId(third);
                let left_associated = multiply_expansion(
                    theory,
                    &theory.classical_product(left, right).unwrap(),
                    third,
                );
                let right_associated = multiply_expansion(
                    theory,
                    &theory.classical_product(right, third).unwrap(),
                    left,
                );
                assert_eq!(
                    left_associated, right_associated,
                    "cup product is associative"
                );

                let pairing = |product: Vec<(BasisId, Rational)>, basis: BasisId| {
                    product
                        .into_iter()
                        .fold(Rational::zero(), |total, (output, coefficient)| {
                            total + coefficient * metric.entry(output.0, basis.0).clone()
                        })
                };
                assert_eq!(
                    pairing(theory.classical_product(left, right).unwrap(), third),
                    pairing(theory.classical_product(right, third).unwrap(), left),
                    "Poincare pairing must be Frobenius-invariant"
                );
            }
        }
    }
}

#[test]
fn projective_space_data_and_anomaly_are_exact() {
    let p2 = ProjectiveSpaceTheory::new(2);
    assert_eq!(p2.target_dimension(), 2);
    assert_eq!(
        p2.state_space()
            .pairing
            .as_ref()
            .unwrap()
            .metric
            .entry(0, 2),
        &Rational::one()
    );
    assert_eq!(
        p2.state_space().c1_action.as_ref().unwrap().entry(1, 0),
        &Rational::from(3)
    );
    assert_eq!(
        p2.characteristic_numbers().unwrap().virasoro_anomaly(2),
        Rational::new(-5, 16)
    );
}

#[test]
fn high_dimensional_projective_pairing_has_its_declared_analytic_inverse() {
    let projective = ProjectiveSpaceTheory::new(100);
    let pairing = projective.state_space().pairing.as_ref().unwrap();

    assert_eq!(pairing.inverse, pairing.metric);
    assert_eq!(
        pairing.metric.multiply(&pairing.inverse).unwrap(),
        StateSpaceMatrix::identity(101)
    );
}

#[test]
fn extreme_target_dimensions_are_rejected_fallibly() {
    assert!(matches!(
        ProjectiveSpaceTheory::try_new(usize::MAX),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        ProductProjectiveTheory::new(usize::MAX, 1),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        NegativeSplitTotalSpaceTheory::new(usize::MAX, vec![1]),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        ProjectiveBundleTheory::new(usize::MAX, vec![0, 1]),
        Err(GwError::UnsupportedInvariant(_))
    ));

    let p1 = ProjectiveSpaceTheory::new(1);
    let product = ProductProjectiveTheory::new(1, 1).unwrap();
    let bundle = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
    assert!(matches!(
        p1.try_curve(usize::MAX),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        product.try_curve(usize::MAX, 0),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        bundle.try_curve(usize::MAX, 0),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn permutation_equivalent_bundle_presentations_share_one_fingerprint() {
    let left = ProjectiveBundleTheory::new(1, vec![0, 3, 1]).unwrap();
    let right = ProjectiveBundleTheory::new(1, vec![1, 0, 3]).unwrap();
    assert_eq!(left, right);
    assert_eq!(left.theory_fingerprint(), right.theory_fingerprint());
    assert_eq!(left.theory_id(), "P(O + O(1) + O(3)) over P^1");
    assert_eq!(
        left.canonicalize_summand_payloads(vec![3, 0, 1], vec!["three", "zero", "one"],)
            .unwrap(),
        vec!["zero", "one", "three"]
    );
    assert!(matches!(
        left.canonicalize_summand_payloads(vec![0, 1, 3], vec!["too", "short"]),
        Err(GwError::ConventionMismatch(_))
    ));
    assert!(matches!(
        left.canonicalize_summand_payloads(vec![0, 1, 4], vec!["zero", "one", "four"],),
        Err(GwError::ConventionMismatch(_))
    ));
}

#[test]
fn bundle_xi_multiplication_reduces_in_the_canonical_theory() {
    let theory = ProjectiveBundleTheory::new(2, vec![0, 3, 3]).unwrap();
    let xi_squared = theory.basis_id(0, 2).unwrap();

    assert_eq!(
        theory.multiply_basis_by_xi(xi_squared).unwrap(),
        vec![
            (theory.basis_id(1, 2).unwrap(), Rational::from(-6)),
            (theory.basis_id(2, 1).unwrap(), Rational::from(-9)),
        ],
        "xi^3 = -6 H xi^2 - 9 H^2 xi for P(O + O(3) + O(3))"
    );
    assert!(
        theory
            .multiply_basis_by_xi(theory.basis_id(2, 2).unwrap())
            .unwrap()
            .is_empty(),
        "H^2 xi^3 must vanish after H^3=0"
    );
}

#[test]
fn canonical_theories_own_consistent_classical_frobenius_algebras() {
    assert_classical_frobenius_algebra(&ProjectiveSpaceTheory::new(2));
    assert_classical_frobenius_algebra(&ProductProjectiveTheory::new(1, 2).unwrap());
    assert_classical_frobenius_algebra(&ProjectiveBundleTheory::new(2, vec![0, 3, 3]).unwrap());
}

#[test]
fn canonical_theories_choose_positive_stabilizing_divisors() {
    let projective = ProjectiveSpaceTheory::new(2);
    assert_eq!(
        projective
            .stabilizing_divisor(&projective.curve(3))
            .unwrap(),
        Some((BasisId(1), 3))
    );

    let product = ProductProjectiveTheory::new(1, 2).unwrap();
    assert_eq!(
        product.stabilizing_divisor(&product.curve(0, 4)).unwrap(),
        Some((product.basis_id(0, 1).unwrap(), 4))
    );

    let bundle = ProjectiveBundleTheory::new(2, vec![0, 3, 3]).unwrap();
    assert_eq!(
        bundle.stabilizing_divisor(&bundle.curve(1, -3)).unwrap(),
        Some((bundle.basis_id(1, 0).unwrap(), 1))
    );
    assert_eq!(
        bundle.stabilizing_divisor(&bundle.curve(0, 2)).unwrap(),
        Some((bundle.basis_id(0, 1).unwrap(), 2))
    );
}

#[test]
fn bundle_xi_multiplication_rejects_an_invalid_basis_id() {
    let theory = ProjectiveBundleTheory::new(2, vec![0, 3, 3]).unwrap();
    assert!(matches!(
        theory.multiply_basis_by_xi(BasisId(9)),
        Err(GwError::ConventionMismatch(_))
    ));
}

#[test]
fn permutation_equivalent_local_splits_share_one_fingerprint() {
    let left = NegativeSplitTotalSpaceTheory::new(2, vec![3, 1, 2]).unwrap();
    let right = NegativeSplitTotalSpaceTheory::new(2, vec![2, 3, 1]).unwrap();
    assert_eq!(left, right);
    assert_eq!(left.degrees(), &[1, 2, 3]);
    assert_eq!(left.theory_fingerprint(), right.theory_fingerprint());
    assert_eq!(left.theory_id(), "Tot(O(-[1, 2, 3])) over P^2");
    assert_eq!(
        left.canonicalize_summand_payloads(
            vec![3, 1, 2],
            vec!["degree-three", "degree-one", "degree-two"],
        )
        .unwrap(),
        vec!["degree-one", "degree-two", "degree-three"]
    );
    assert!(matches!(
        left.canonicalize_summand_payloads(vec![3, 1, 2], vec!["too", "short"]),
        Err(GwError::ConventionMismatch(_))
    ));
    assert!(matches!(
        left.canonicalize_summand_payloads(
            vec![4, 1, 2],
            vec!["degree-four", "degree-one", "degree-two"],
        ),
        Err(GwError::ConventionMismatch(_))
    ));
}

#[test]
fn projective_point_has_only_the_zero_curve_class() {
    let point = ProjectiveSpaceTheory::new(0);
    assert_eq!(
        point.bounded_admissible_classes(4).unwrap(),
        vec![point.curve(0)]
    );
    assert_eq!(
        point.effectivity(&point.curve(1)).unwrap(),
        CurveEffectivity::Ineffective
    );
    assert_eq!(
        point.characteristic_numbers().unwrap().virasoro_anomaly(0),
        Rational::new(1, 16)
    );
}

#[test]
fn product_splits_are_geometric_bidegree_splits() {
    let theory = ProductProjectiveTheory::new(1, 2).unwrap();
    let splits = theory
        .admissible_decompositions(&theory.curve(1, 2))
        .unwrap();
    assert_eq!(splits.len(), 6);
    assert_eq!(splits[0].left, theory.curve(0, 0));
    assert_eq!(splits[5].right, theory.curve(0, 0));
}

#[test]
fn ineffective_classes_have_zero_decomposition_count_and_no_splits() {
    let cases: Vec<(Box<dyn GwTheory>, CurveClass)> = vec![
        (
            Box::new(ProjectiveSpaceTheory::new(2)),
            CurveClass::new(vec![-1]),
        ),
        (
            Box::new(ProductProjectiveTheory::new(1, 1).unwrap()),
            CurveClass::new(vec![-1, 0]),
        ),
        (
            Box::new(ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap()),
            CurveClass::new(vec![0, -1]),
        ),
        (
            Box::new(NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap()),
            CurveClass::new(vec![-1]),
        ),
    ];
    for (theory, curve) in cases {
        assert_eq!(theory.admissible_decomposition_count(&curve).unwrap(), 0);
        assert!(theory.admissible_decompositions(&curve).unwrap().is_empty());
    }
}

#[test]
fn product_decomposition_count_overflow_is_fallible() {
    let product = ProductProjectiveTheory::new(1, 1).unwrap();
    let huge = CurveClass::new(vec![i64::MAX, i64::MAX]);
    assert!(matches!(
        product.admissible_decomposition_count(&huge),
        Err(GwError::UnsupportedInvariant(_))
    ));
    assert!(matches!(
        product.admissible_decompositions(&huge),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn bounded_class_counts_match_canonical_theory_enumerations() {
    let theories: Vec<Box<dyn GwTheory>> = vec![
        Box::new(ProjectiveSpaceTheory::new(0)),
        Box::new(ProjectiveSpaceTheory::new(3)),
        Box::new(ProductProjectiveTheory::new(1, 2).unwrap()),
        Box::new(ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap()),
        Box::new(NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap()),
    ];
    for theory in theories {
        for bound in 0..=20 {
            let classes = theory.bounded_admissible_classes(bound).unwrap();
            assert_eq!(
                theory.bounded_admissible_class_count(bound).unwrap(),
                classes.len(),
                "{} at bound {bound}",
                theory.theory_id()
            );
            assert!(classes
                .iter()
                .all(|curve| curve.rank() == theory.curve_class_space().rank()));
            assert_eq!(
                classes
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                classes.len(),
                "{} returned duplicate classes at bound {bound}",
                theory.theory_id()
            );
        }
    }
}

#[test]
fn point_class_count_accepts_an_irrelevant_large_bound() {
    let point = ProjectiveSpaceTheory::new(0);
    assert_eq!(point.bounded_admissible_class_count(usize::MAX).unwrap(), 1);
    assert_eq!(
        point.bounded_admissible_classes(usize::MAX).unwrap(),
        vec![point.curve(0)]
    );
}

#[test]
fn bundle_scan_rejects_shifted_degree_arithmetic_overflow() {
    let twist = (i64::MAX as usize) / 2 + 1;
    let bundle = ProjectiveBundleTheory::new(1, vec![0, twist]).unwrap();
    let error = bundle.bounded_admissible_class_count(2).unwrap_err();
    assert!(matches!(error, GwError::UnsupportedInvariant(_)));
    let error = bundle.bounded_admissible_classes(2).unwrap_err();
    assert!(matches!(error, GwError::UnsupportedInvariant(_)));
}

#[test]
fn trivial_bundle_recovers_product_pairing_and_characteristics() {
    let bundle = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
    let product = ProductProjectiveTheory::new(1, 1).unwrap();
    assert_eq!(bundle.state_space().pairing, product.state_space().pairing);
    assert_eq!(
        bundle.characteristic_numbers().unwrap().top_chern_integral,
        product.characteristic_numbers().unwrap().top_chern_integral
    );
    assert_eq!(
        bundle
            .characteristic_numbers()
            .unwrap()
            .c1_c_dim_minus_one_integral,
        product
            .characteristic_numbers()
            .unwrap()
            .c1_c_dim_minus_one_integral
    );
}

#[test]
fn bundle_splits_use_shifted_effective_coordinates() {
    let theory = ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap();
    let exceptional = theory.curve(1, -2);
    assert_eq!(
        theory.effectivity(&exceptional).unwrap(),
        CurveEffectivity::Unknown
    );
    let classes = theory.bounded_admissible_classes(1).unwrap();
    assert!(classes.contains(&exceptional));
    let splits = theory.admissible_decompositions(&exceptional).unwrap();
    assert_eq!(splits.len(), 2);
    for split in splits {
        assert_eq!(
            split.left.checked_add(&split.right),
            Some(exceptional.clone())
        );
    }
}

#[test]
fn bundle_characteristic_numbers_do_not_depend_on_splitting_twists() {
    for twist in [0, 1, 2, 5] {
        let hirzebruch = ProjectiveBundleTheory::new(1, vec![0, twist]).unwrap();
        let numbers = hirzebruch.characteristic_numbers().unwrap();
        assert_eq!(numbers.top_chern_integral, Rational::from(4));
        assert_eq!(numbers.c1_c_dim_minus_one_integral, Rational::from(8));

        let threefold = ProjectiveBundleTheory::new(1, vec![0, 0, twist]).unwrap();
        let numbers = threefold.characteristic_numbers().unwrap();
        assert_eq!(numbers.top_chern_integral, Rational::from(6));
        assert_eq!(numbers.c1_c_dim_minus_one_integral, Rational::from(24));
    }
}

#[test]
fn local_theory_does_not_fabricate_compact_pairing_or_anomaly() {
    let local = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
    assert!(local.state_space().pairing.is_none());
    assert!(local.state_space().c1_action.is_none());
    assert!(local.characteristic_numbers().is_none());
}

#[test]
fn negative_split_completion_derives_normalized_bundle_geometry() {
    let local = NegativeSplitTotalSpaceTheory::new(3, vec![4, 1, 2]).unwrap();
    let completion = NegativeSplitProjectiveCompletion::new(local).unwrap();

    assert_eq!(completion.local_theory().degrees(), &[1, 2, 4]);
    assert_eq!(completion.max_degree(), 4);
    assert_eq!(completion.compact_theory().twists(), &[0, 2, 3, 4]);
    assert_eq!(completion.local_theory().target_dimension(), 6);
    assert_eq!(completion.compact_theory().target_dimension(), 6);

    let local_curve = CurveClass::new(vec![2]);
    let section_curve = completion.section_curve(2).unwrap();
    assert_eq!(section_curve, completion.compact_theory().curve(2, -8));
    assert_eq!(completion.section_degree(&section_curve), Some(2));
    assert_eq!(
        completion.section_degree(&completion.compact_theory().curve(2, -7)),
        None
    );
    assert_eq!(
        completion.local_theory().c1_pairing(&local_curve).unwrap(),
        -6
    );
    assert_eq!(
        completion
            .compact_theory()
            .c1_pairing(&section_curve)
            .unwrap(),
        -6
    );
    assert_eq!(
        completion
            .local_theory()
            .virtual_dimension(4, &local_curve, 3)
            .unwrap(),
        -12
    );
    assert_eq!(
        completion
            .compact_theory()
            .virtual_dimension(4, &section_curve, 3)
            .unwrap(),
        -12
    );
}

#[test]
fn negative_split_completion_restricts_insertions_with_exact_signs() {
    let local = NegativeSplitTotalSpaceTheory::new(3, vec![1, 2, 4]).unwrap();
    let completion = NegativeSplitProjectiveCompletion::new(local).unwrap();
    let compact = completion.compact_theory();

    let restriction =
        |h, xi| completion.restrict_basis_to_section(compact.basis_id(h, xi).unwrap());
    assert_eq!(restriction(0, 0), Ok(Some((BasisId(0), Rational::one()))));
    assert_eq!(
        restriction(0, 1),
        Ok(Some((BasisId(1), Rational::from(-4))))
    );
    assert_eq!(
        restriction(0, 2),
        Ok(Some((BasisId(2), Rational::from(16))))
    );
    assert_eq!(
        restriction(1, 2),
        Ok(Some((BasisId(3), Rational::from(16))))
    );
    assert_eq!(
        restriction(0, 3),
        Ok(Some((BasisId(3), Rational::from(-64))))
    );
    assert_eq!(restriction(1, 3), Ok(None));
    assert_eq!(
        completion
            .restrict_basis_to_section(BasisId(usize::MAX))
            .unwrap_err(),
        GwError::ConventionMismatch(
            "projective-completion insertion is outside the compact bundle basis".to_string()
        )
    );
}

#[test]
fn negative_split_completion_is_canonical_under_degree_permutations() {
    let left = NegativeSplitProjectiveCompletion::new(
        NegativeSplitTotalSpaceTheory::new(2, vec![1, 3, 2]).unwrap(),
    )
    .unwrap();
    let right = NegativeSplitProjectiveCompletion::new(
        NegativeSplitTotalSpaceTheory::new(2, vec![3, 2, 1]).unwrap(),
    )
    .unwrap();

    assert_eq!(left, right);
    assert_eq!(left.compact_theory().twists(), &[0, 1, 2, 3]);
    assert_eq!(left.section_curve(3).unwrap().coordinates(), &[3, -9]);
    assert_eq!(left.section_degree(&CurveClass::zero(2)), Some(0));
}
