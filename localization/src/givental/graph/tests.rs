use super::super::*;
use super::*;

#[test]
fn stable_range_predicate_is_overflow_free_at_extreme_genus() {
    assert!(!is_stable_cohft_range(0, 2));
    assert!(is_stable_cohft_range(0, 3));
    assert!(!is_stable_cohft_range(1, 0));
    assert!(is_stable_cohft_range(1, 1));
    assert!(is_stable_cohft_range(2, 0));
    assert!(is_stable_cohft_range(usize::MAX, usize::MAX));
    assert!(matches!(
        crate::graphs::stable_graph_dimension(usize::MAX, 0),
        Err(GwError::UnsupportedInvariant(_))
    ));
}

#[test]
fn exact_seed_fallback_survives_the_structured_graph_work_limit() {
    let point = InvariantRequest::new(0, 5, 0, vec![tau(13, CohomologyClass::one(0))]);
    assert!(matches!(
        compute_by_givental_graphs(&point),
        Err(GwError::ResourceLimit { .. })
    ));
    let result = compute(&point).unwrap();
    assert_eq!(result.engine, "givental-seed");
    assert_eq!(
        result.value,
        RatFun::from_rational(WittenKontsevich::shared().psi_integral(5, &[13]))
    );

    let unsupported_seed = InvariantRequest::new(1, 5, 0, vec![tau(9, CohomologyClass::one(1))]);
    assert!(matches!(
        compute(&unsupported_seed),
        Err(GwError::ResourceLimit { .. })
    ));
}
use crate::factored::FactoredRatFun;
use crate::spaces::projective_space::CohomologyClass;
use crate::{tau, ComputeMode, InvariantRequest};

fn usize_factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[test]
fn translation_excess_partitions_group_ordered_compositions() {
    let partitions = translation_excess_partitions(4);
    let mut as_strings = partitions
        .iter()
        .map(|partition| {
            partition
                .iter()
                .map(|(excess, multiplicity)| format!("{excess}^{multiplicity}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>();
    as_strings.sort();
    assert_eq!(as_strings, vec!["1^1 3^1", "1^2 2^1", "1^4", "2^2", "4^1"]);

    let ordered_terms = partitions
        .iter()
        .map(|partition| {
            let translation_count = partition
                .iter()
                .map(|(_, multiplicity)| *multiplicity)
                .sum::<usize>();
            let denominator = partition
                .iter()
                .map(|(_, multiplicity)| usize_factorial(*multiplicity))
                .product::<usize>();
            usize_factorial(translation_count) / denominator
        })
        .sum::<usize>();
    assert_eq!(ordered_terms, 8);
}

#[test]
fn translation_partition_symmetries_recover_ordered_composition_counts() {
    for total in 1..=8 {
        let ordered_terms = translation_excess_partitions(total)
            .iter()
            .map(|partition| {
                let translation_count = partition
                    .iter()
                    .map(|(_, multiplicity)| *multiplicity)
                    .sum::<usize>();
                let denominator = partition
                    .iter()
                    .map(|(_, multiplicity)| usize_factorial(*multiplicity))
                    .product::<usize>();
                usize_factorial(translation_count) / denominator
            })
            .sum::<usize>();
        assert_eq!(ordered_terms, 1usize << (total - 1));
    }
}

#[test]
fn series_identity_r_matrix_is_unitary_for_any_metric() {
    let metric = SeriesMatrix::from_entries(vec![
        vec![
            crate::core::series::QSeries::constant(RatFun::from(2usize), 1),
            crate::core::series::QSeries::q(1),
        ],
        vec![
            crate::core::series::QSeries::q(1),
            crate::core::series::QSeries::constant(RatFun::from(5usize), 1),
        ],
    ]);
    let r = SeriesRMatrix::identity(
        2,
        1,
        3,
        CanonicalFrameConvention::UnnormalizedCanonicalIdempotents,
    );
    r.check_identity_calibration().unwrap();
    r.check_unitarity(&metric).unwrap();
    assert_eq!(r.size(), 2);
    assert_eq!(r.q_degree(), 1);
    assert_eq!(r.z_order(), 3);
    assert_eq!(
        r.convention(),
        CanonicalFrameConvention::UnnormalizedCanonicalIdempotents
    );
}

#[test]
fn projective_j_calibration_matches_p1_classical_limit() {
    let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
    let weights = [
        crate::core::algebra::Rational::from(2),
        crate::core::algebra::Rational::from(5),
    ];
    let r1 = calibration.r_matrix.coefficient(1).unwrap();
    let r2 = calibration.r_matrix.coefficient(2).unwrap();

    assert_eq!(
        r1.entry(0, 0)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::new(1, 36)
    );
    assert_eq!(
        r1.entry(1, 1)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::new(-1, 36)
    );
    assert_eq!(
        r2.entry(0, 0)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::new(1, 2592)
    );
    assert_eq!(
        r1.entry(0, 1)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::zero()
    );
}

#[test]
fn projective_j_calibration_relative_frame_inverts_for_p1() {
    let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
    let product = calibration.psi_inverse.mul(&calibration.psi);
    assert_series_matrix_equal_after_lambda_eval(
        &product,
        &SeriesMatrix::identity(2, 1),
        1,
        &[
            crate::core::algebra::Rational::from(2),
            crate::core::algebra::Rational::from(5),
        ],
    );
}

#[test]
fn projective_j_calibration_is_unitary_for_p1_low_order() {
    let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
    assert_r_matrix_unitary_after_lambda_eval(
        &calibration.r_matrix,
        &calibration.metric,
        1,
        &[
            crate::core::algebra::Rational::from(2),
            crate::core::algebra::Rational::from(5),
        ],
    );
}

#[test]
fn projective_j_calibration_r_coefficients_satisfy_diagonal_flatness() {
    let weights = [
        crate::core::algebra::Rational::from(2),
        crate::core::algebra::Rational::from(5),
        crate::core::algebra::Rational::from(11),
    ];
    let calibration = projective_space_j_calibration_at_lambda_weights(2, 3, 5, &weights).unwrap();

    for order in 0..=calibration.r_matrix.z_order() {
        let coefficient = calibration.r_matrix.coefficient(order).unwrap();
        let source = coefficient
            .q_derivative()
            .add(&calibration.connection.mul(coefficient));
        for branch in 0..calibration.r_matrix.size() {
            let diagonal = source.entry(branch, branch);
            for degree in 0..=diagonal.max_degree() {
                assert_eq!(
                    diagonal.coeff(degree),
                    Some(&RatFun::zero()),
                    "nonzero diagonal flatness source at z^{order}, branch {branch}, q^{degree}"
                );
            }
        }
    }
}

#[test]
fn projective_descendant_s_matrix_matches_p1_low_order() {
    let descendant_s = projective_space_descendant_s_matrix(1, 1, 2).unwrap();
    let weights = [
        crate::core::algebra::Rational::from(2),
        crate::core::algebra::Rational::from(5),
    ];
    assert_eq!(
        descendant_s
            .coefficient(1)
            .unwrap()
            .entry(0, 1)
            .coeff(1)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::one()
    );
    assert_eq!(
        descendant_s
            .coefficient(2)
            .unwrap()
            .entry(1, 1)
            .coeff(1)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::core::algebra::Rational::one()
    );
}

#[test]
fn product_of_point_theories_uses_psi_oracle() {
    let value = product_of_point_theories(2, 1, 0, &[1]).unwrap();
    assert_eq!(value.to_string(), "1/24");
}

#[test]
fn givental_graph_reproduces_degree_zero_classical_product() {
    let req = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(
            2,
            0,
            0,
            vec![
                tau(0, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::one(2)),
            ],
        )
    };
    let result = compute_by_givental_graphs(&req).unwrap();
    assert_eq!(result.value, RatFun::one());
}

#[test]
fn projective_request_rejects_wrong_target_class_before_dimension_pruning() {
    let req = InvariantRequest::new(
        1,
        0,
        0,
        vec![
            tau(0, CohomologyClass::one(2)),
            tau(0, CohomologyClass::one(2)),
            tau(0, CohomologyClass::one(2)),
        ],
    );

    let err = compute_by_givental_graphs(&req).unwrap_err();
    assert!(matches!(err, GwError::ConventionMismatch(_)));
    assert!(err.to_string().contains("insertion 0 belongs to P^2"));
}

#[test]
fn givental_graph_reproduces_genus_one_degree_zero_obstruction_values() {
    let cases = [
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(1, 1, 0, vec![tau(0, CohomologyClass::h_power(1, 1))])
            },
            crate::core::algebra::Rational::new(-1, 24),
        ),
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(2, 1, 0, vec![tau(0, CohomologyClass::h_power(2, 1))])
            },
            crate::core::algebra::Rational::new(-1, 8),
        ),
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(2, 1, 0, vec![tau(1, CohomologyClass::one(2))])
            },
            crate::core::algebra::Rational::new(1, 8),
        ),
    ];

    for (req, expected) in cases {
        let graph = compute_by_givental_graphs(&req).unwrap();
        let obstruction =
            crate::validation::genus_one_degree_zero_one_point_obstruction(&req, "test").unwrap();
        assert_eq!(graph.value, RatFun::from_rational(expected));
        assert_eq!(graph.value, obstruction.value);
        assert_ne!(graph.engine, "givental-seed");
    }
}

#[test]
fn givental_graph_reproduces_p1_degree_one_three_point() {
    let req = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(
            1,
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        )
    };
    let result = compute_by_givental_graphs(&req).unwrap();
    assert_eq!(result.value, RatFun::one());
}

#[test]
fn projective_provider_path_matches_public_graph_path() {
    let insertions = vec![
        tau(2, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::h_power(1, 1)),
    ];
    let req = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(1, 0, 2, insertions.clone())
    };
    let public = compute_by_givental_graphs(&req).unwrap().value;
    let provider = ProjectiveSpaceProvider::lambda_line_nonequivariant(1);
    let generic = compute_semisimple_graph_value(&provider, 0, 2, &insertions, None).unwrap();
    assert_eq!(generic, public);
}

#[test]
fn semisimple_graph_coefficients_match_single_value_path() {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::h_power(1, 1)),
    ];
    let provider = ProjectiveSpaceProvider::lambda_line_nonequivariant(1);
    let coefficients =
        compute_semisimple_graph_coefficients(&provider, 0, 2, &insertions, None).unwrap();

    assert_eq!(coefficients.len(), 3);
    for (degree, coefficient) in coefficients.iter().enumerate() {
        let single =
            compute_semisimple_graph_value(&provider, 0, degree, &insertions, None).unwrap();
        assert_eq!(coefficient, &single, "q^{degree}");
    }
    assert_eq!(coefficients[1], RatFun::one());

    let range =
        compute_semisimple_graph_coefficient_range(&provider, 0, 1, 2, &insertions, None).unwrap();
    assert_eq!(range, coefficients[1..=2].to_vec());
}

#[test]
fn rational_graph_path_matches_dense_evaluator_without_insertions() {
    let provider = ProjectiveSpaceProvider::lambda_line_nonequivariant(1);
    let genus = 2;
    let q_degree = 1;
    let graph_dimension = 3 * genus - 3;
    let kernel = provider
        .graph_kernel(q_degree, graph_dimension + 1, graph_dimension)
        .unwrap();
    let graphs = prepared_stable_graphs(genus, 0, provider.colors()).unwrap();

    let mut dense_profile = GraphEvalProfile::new();
    let dense = evaluate_scalar_graphs_parallel(
        graphs.as_ref(),
        &[],
        &kernel,
        q_degree,
        graph_dimension,
        &mut dense_profile,
    );
    let mut rational_profile = GraphEvalProfile::new();
    let rational = evaluate_rational_graphs_if_possible(
        graphs.as_ref(),
        &[],
        &kernel,
        q_degree,
        graph_dimension,
        &mut rational_profile,
    )
    .expect("nonequivariant P1 graph kernel should be rational");

    assert_eq!(rational, dense);
}

#[test]
fn rational_graph_path_matches_dense_evaluator_with_insertions() {
    let provider = ProjectiveSpaceProvider::lambda_line_nonequivariant(1);
    let genus = 2;
    let markings = 1;
    let q_degree = 1;
    let graph_dimension = 3 * genus + markings - 3;
    let kernel = provider
        .graph_kernel(q_degree, graph_dimension + 1, graph_dimension)
        .unwrap();
    let graphs = prepared_stable_graphs(genus, markings, provider.colors()).unwrap();

    let insertions = vec![tau(4, CohomologyClass::h_power(1, 1))];
    let descendant_s = provider.descendant_s_matrix(q_degree, 4).unwrap();
    let insertion_terms = ancestor_insertion_terms_from_provider(
        &provider,
        &insertions,
        &descendant_s,
        &kernel.calibration.psi_inverse,
        q_degree,
        graph_dimension,
    )
    .unwrap();
    let leg_options = leg_options_by_marking_color(
        &insertion_terms,
        &kernel.inverse_r,
        q_degree,
        graph_dimension,
        provider.colors(),
    );

    let mut dense_profile = GraphEvalProfile::new();
    let dense = evaluate_scalar_graphs_parallel(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut dense_profile,
    );
    let mut rational_profile = GraphEvalProfile::new();
    let rational = evaluate_rational_graphs_if_possible(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut rational_profile,
    )
    .expect("nonequivariant P1 leg options should be rational");

    assert_eq!(rational, dense);
}

#[test]
fn equivariant_compute_limits_to_nonequivariant_value() {
    // The equivariant path now runs over factored coefficients end to end;
    // its expanded result must still specialize to the known number.
    let req = InvariantRequest {
        equivariant: true,
        ..InvariantRequest::new(1, 1, 1, vec![tau(2, CohomologyClass::h_power(1, 1))])
    };
    let result = compute_by_givental_graphs(&req).unwrap();
    assert_eq!(result.engine, "givental-r-graph");
    assert_eq!(
        result
            .value
            .nonequivariant_limit_line(1, &[Rational::from(2), Rational::from(3)])
            .unwrap(),
        Rational::new(1, 24)
    );
}

#[test]
fn equivariant_excess_degree_is_not_pruned() {
    // In degree zero this is the classical Atiyah-Bott integral
    // integral_{P1} H^2 = lambda_0 + lambda_1.
    let insertions = vec![
        tau(0, CohomologyClass::one(1)),
        tau(0, CohomologyClass::h_power(1, 1)),
        tau(0, CohomologyClass::h_power(1, 1)),
    ];
    let nonequivariant =
        compute_by_givental_graphs(&InvariantRequest::new(1, 0, 0, insertions.clone())).unwrap();
    assert_eq!(nonequivariant.value, RatFun::zero());

    let equivariant_req = InvariantRequest {
        equivariant: true,
        ..InvariantRequest::new(1, 0, 0, insertions)
    };
    let equivariant = compute_by_givental_graphs(&equivariant_req).unwrap();
    assert_eq!(equivariant.engine, "givental-r-graph");
    let expected = &crate::core::algebra::lambda(0) + &crate::core::algebra::lambda(1);
    assert!((&equivariant.value - &expected).is_zero());
    assert_eq!(
        equivariant
            .value
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .unwrap(),
        Rational::from(7)
    );
}

#[test]
fn nonequivariant_negative_virtual_dimension_is_zero() {
    let result = compute_by_givental_graphs(&InvariantRequest::new(5, 2, 0, Vec::new())).unwrap();
    assert_eq!(result.value, RatFun::zero());
    assert_eq!(result.engine, "givental-r-graph");
    assert!(result.notes[0].contains("virtual dimension -2"));
}

#[test]
fn point_target_has_no_positive_degree_invariants() {
    // The formal dimension equation for this stable profile suggests d=1,
    // but P^0 is a point and has no positive curve classes.
    let insertions = vec![
        tau(1, CohomologyClass::one(0)),
        tau(0, CohomologyClass::one(0)),
        tau(0, CohomologyClass::one(0)),
    ];
    for equivariant in [false, true] {
        let req = InvariantRequest {
            equivariant,
            ..InvariantRequest::new(0, 0, 1, insertions.clone())
        };
        let result = compute_by_givental_graphs(&req).unwrap();
        assert_eq!(result.value, RatFun::zero());
        assert_eq!(result.engine, "givental-effective-degree");
        assert!(result.notes[0].contains("no effective curve class"));
    }
}

#[test]
fn factored_graph_path_matches_symbolic_evaluator() {
    // Symbolic equivariant kernel: coefficients are genuine rational
    // functions in lambda, so the factored tier engages.  Compare against the
    // plain symbolic evaluator by exact evaluation at generic weights, since
    // the two paths may represent the same value differently.
    let provider = ProjectiveSpaceProvider::symbolic_equivariant(1);
    let genus = 1;
    let markings = 1;
    let q_degree = 1;
    let graph_dimension = 3 * genus + markings - 3;
    let kernel = provider
        .graph_kernel(q_degree, graph_dimension + 1, graph_dimension)
        .unwrap();
    let graphs = prepared_stable_graphs(genus, markings, provider.colors()).unwrap();

    let insertions = vec![tau(2, CohomologyClass::h_power(1, 1))];
    let descendant_s = provider.descendant_s_matrix(q_degree, 2).unwrap();
    let insertion_terms = ancestor_insertion_terms_from_provider(
        &provider,
        &insertions,
        &descendant_s,
        &kernel.calibration.psi_inverse,
        q_degree,
        graph_dimension,
    )
    .unwrap();
    let leg_options = leg_options_by_marking_color(
        &insertion_terms,
        &kernel.inverse_r,
        q_degree,
        graph_dimension,
        provider.colors(),
    );

    let mut dense_profile = GraphEvalProfile::new();
    let dense = evaluate_scalar_graphs_parallel(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut dense_profile,
    );
    let mut factored_profile = GraphEvalProfile::new();
    let factored = evaluate_factored_graphs(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut factored_profile,
    );

    let weights = [
        crate::core::algebra::Rational::from(2),
        crate::core::algebra::Rational::from(5),
    ];
    for degree in 0..=q_degree {
        assert_eq!(
            factored
                .coeff(degree)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            dense
                .coeff(degree)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            "factored/symbolic mismatch at q^{degree}"
        );
    }
}

#[test]
fn graph_kernel_constructor_matches_projective_kernel_builder() {
    let weights = [
        crate::core::algebra::Rational::from(2),
        crate::core::algebra::Rational::from(5),
    ];
    let calibration = projective_space_j_calibration_at_lambda_weights(1, 1, 2, &weights).unwrap();
    let direct = GiventalGraphKernel::from_calibration(calibration.clone(), 2).unwrap();
    let projective = projective_space_graph_kernel(1, 1, 2, 2, false, &weights).unwrap();

    assert_eq!(direct.inverse_r(), projective.inverse_r());
    assert_eq!(direct.translation(), projective.translation());
    assert_eq!(direct.calibration(), projective.calibration());
}

#[test]
fn givental_graph_reproduces_p1_degree_two_stationary_descendant() {
    let req = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(
            1,
            0,
            2,
            vec![
                tau(2, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        )
    };

    let result = compute_by_givental_graphs(&req).unwrap();
    assert_eq!(result.value, RatFun::one());
}

#[test]
fn givental_graph_reproduces_p1_higher_degree_stationary_descendants() {
    let degree_three = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(
            1,
            0,
            3,
            vec![
                tau(4, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        )
    };
    let result = compute_by_givental_graphs(&degree_three).unwrap();
    assert_eq!(
        result.value,
        RatFun::from_rational(crate::core::algebra::Rational::new(1, 4))
    );

    let degree_four = InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(
            1,
            0,
            4,
            vec![
                tau(6, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        )
    };
    let result = compute_by_givental_graphs(&degree_four).unwrap();
    assert_eq!(
        result.value,
        RatFun::from_rational(crate::core::algebra::Rational::new(1, 36))
    );
}

#[test]
fn coloring_orbits_reduce_vertex_automorphism_symmetry() {
    let graph = crate::graphs::StableGraph {
        vertices: vec![
            crate::graphs::StableVertex { genus: 1 },
            crate::graphs::StableVertex { genus: 1 },
        ],
        edges: vec![crate::graphs::StableEdge::new(0, 1)],
        legs: Vec::new(),
    };
    let orbits = vertex_coloring_orbits(&graph, 3).unwrap();
    assert_eq!(orbits.len(), 6);
    assert_eq!(
        orbits.iter().map(|orbit| orbit.multiplicity).sum::<usize>(),
        9
    );
}

#[test]
fn vertex_coloring_work_is_checked_before_materialization() {
    assert!(matches!(
        vertex_colorings(8, 64),
        Err(GwError::ResourceLimit {
            operation,
            limit: MAX_STABLE_GRAPH_COLORING_BYTES,
            ..
        }) if operation == "estimated vertex-coloring storage"
    ));

    let one_vertex_limit = MAX_STABLE_GRAPH_COLORING_BYTES
        / ((std::mem::size_of::<Vec<usize>>() + std::mem::size_of::<usize>())
            * COLORING_STORAGE_AMPLIFICATION);
    assert!(matches!(
        prepared_stable_graphs(0, 3, one_vertex_limit + 1),
        Err(GwError::ResourceLimit {
            operation,
            requested,
            limit: MAX_STABLE_GRAPH_COLORING_BYTES,
        }) if operation == "estimated prepared stable-graph coloring storage"
            && requested > MAX_STABLE_GRAPH_COLORING_BYTES
    ));
}

#[test]
fn prepared_stable_graphs_cache_metadata_matches_raw_graphs() {
    let first = prepared_stable_graphs(2, 0, 3).unwrap();
    let second = prepared_stable_graphs(2, 0, 3).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert!(!first.is_empty());

    for prepared in first.iter() {
        let raw_colorings = vertex_coloring_orbits(&prepared.graph, 3).unwrap();
        assert_eq!(prepared.colorings.len(), raw_colorings.len());
        assert_eq!(
            prepared.vertex_power_caps.len(),
            prepared.graph.vertices.len()
        );
        for (vertex, stable_vertex) in prepared.graph.vertices.iter().enumerate() {
            assert_eq!(
                prepared.vertex_power_caps[vertex],
                3 * stable_vertex.genus + prepared.graph.valence(vertex) - 3
            );
        }

        let automorphism_factor =
            Rational::one() / Rational::from(prepared.graph.automorphism_order());
        for (prepared_coloring, raw_coloring) in prepared.colorings.iter().zip(raw_colorings.iter())
        {
            assert_eq!(prepared_coloring.colors, raw_coloring.colors);
            assert_eq!(
                prepared_coloring.factor,
                automorphism_factor.clone() * Rational::from(raw_coloring.multiplicity)
            );
        }
    }
}

fn assert_r_matrix_unitary_after_lambda_eval(
    r: &SeriesRMatrix,
    metric: &SeriesMatrix,
    target_n: usize,
    weights: &[crate::core::algebra::Rational],
) {
    for z_degree in 0..=r.z_order() {
        let mut total = SeriesMatrix::zero(r.size(), r.size(), r.q_degree());
        for left_order in 0..=z_degree {
            let right_order = z_degree - left_order;
            let term = r
                .coefficient(left_order)
                .unwrap()
                .transpose()
                .mul(metric)
                .mul(r.coefficient(right_order).unwrap());
            total = if left_order % 2 == 0 {
                total.add(&term)
            } else {
                total.sub(&term)
            };
        }
        let expected = if z_degree == 0 {
            metric.clone()
        } else {
            SeriesMatrix::zero(r.size(), r.size(), r.q_degree())
        };
        assert_series_matrix_equal_after_lambda_eval(&total, &expected, target_n, weights);
    }
}

fn assert_series_matrix_equal_after_lambda_eval(
    left: &SeriesMatrix,
    right: &SeriesMatrix,
    target_n: usize,
    weights: &[crate::core::algebra::Rational],
) {
    assert_eq!(left.rows(), right.rows());
    assert_eq!(left.cols(), right.cols());
    for row in 0..left.rows() {
        for col in 0..left.cols() {
            let left_series = left.entry(row, col);
            let right_series = right.entry(row, col);
            assert_eq!(left_series.max_degree(), right_series.max_degree());
            for degree in 0..=left_series.max_degree() {
                let diff = left_series.coeff(degree).unwrap() - right_series.coeff(degree).unwrap();
                let value = diff
                    .evaluate_lambda_weights(target_n, weights)
                    .expect("test specialization should avoid poles");
                assert_eq!(value, crate::core::algebra::Rational::zero());
            }
        }
    }
}

fn factored_identity_kernel(
    q_degree: usize,
    graph_dimension: usize,
) -> (
    Arc<GiventalGraphKernel<FactoredRatFun>>,
    QSeries<FactoredRatFun>,
) {
    let size = 1;
    let matrix = SeriesMatrix::<FactoredRatFun>::identity(size, q_degree);
    let scalar = QSeries::<FactoredRatFun>::one(q_degree);
    let calibration = SemisimpleCalibration::<FactoredRatFun> {
        r_matrix: SeriesRMatrix::<FactoredRatFun>::identity(
            size,
            q_degree,
            0,
            CanonicalFrameConvention::NormalizedCanonicalIdempotents,
        ),
        metric: matrix.clone(),
        psi: matrix.clone(),
        psi_inverse: matrix.clone(),
        connection: matrix,
        delta: vec![scalar.clone()],
        inverse_delta: vec![scalar.clone()],
        relative_sqrt_delta: vec![scalar.clone()],
        relative_sqrt_delta_inverse: vec![scalar.clone()],
    };
    (
        Arc::new(GiventalGraphKernel::from_calibration(calibration, graph_dimension).unwrap()),
        scalar,
    )
}

#[test]
fn scalar_graph_contraction_accepts_factored_coefficients() {
    let q_degree = 0;
    let graph_dimension = 0;
    let size = 1;
    let (kernel, scalar) = factored_identity_kernel(q_degree, graph_dimension);
    let graphs = prepared_stable_graphs(0, 3, size).unwrap();
    let unit_leg = LegFactorOption {
        power: 0,
        coefficient: scalar,
    };
    let leg_options = vec![vec![vec![unit_leg]]; 3];
    let mut profile = GraphEvalProfile::new();

    let total = evaluate_scalar_graphs_parallel(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );

    assert_eq!(total.coeff(0).unwrap(), &<FactoredRatFun as Coeff>::one());
}

#[test]
fn coefficient_generic_evaluator_matches_ratfun_provider_path() {
    let provider = ProjectiveSpaceProvider::new(2, false);
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let direct = compute_semisimple_graph_value(&provider, 0, 1, &insertions, None).unwrap();
    let generic =
        compute_semisimple_graph_value_with_coeff::<RatFun, _>(&provider, 0, 1, &insertions, None)
            .unwrap();

    assert_eq!(generic, direct);
}

#[test]
fn external_leg_contraction_accepts_factored_coefficients() {
    let q_degree = 0;
    let graph_dimension = 0;
    let size = 1;
    let markings = 3;
    let (kernel, scalar) = factored_identity_kernel(q_degree, graph_dimension);
    let graphs = prepared_stable_graphs(0, markings, size).unwrap();
    let mut profile = GraphEvalProfile::new();
    let external_kernel = evaluate_external_graphs_parallel(
        graphs.as_ref(),
        markings,
        size,
        &kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    let unit_leg = LegFactorOption {
        power: 0,
        coefficient: scalar,
    };
    let leg_options = vec![vec![vec![unit_leg]]; markings];

    assert_eq!(
        contract_external_leg_kernel_coeff_generic(&external_kernel, &leg_options, 0),
        <FactoredRatFun as Coeff>::one()
    );
}

#[test]
fn restricted_external_leg_contraction_accepts_factored_coefficients() {
    let q_degree = 0;
    let graph_dimension = 0;
    let size = 1;
    let markings = 3;
    let (kernel, scalar) = factored_identity_kernel(q_degree, graph_dimension);
    let graphs = prepared_stable_graphs(0, markings, size).unwrap();

    let ratfun_unit_leg = LegFactorOption {
        power: 0,
        coefficient: QSeries::one(q_degree),
    };
    let template_leg_options = vec![vec![vec![ratfun_unit_leg]]; markings];
    let template = RestrictedExternalLegKernel::<FactoredRatFun>::from_leg_options(
        markings,
        size,
        graph_dimension,
        q_degree,
        std::iter::once(template_leg_options.as_slice()),
    );
    let mut profile = GraphEvalProfile::new();
    let restricted_kernel = evaluate_restricted_external_graphs_parallel(
        graphs.as_ref(),
        &template,
        &kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    let unit_leg = LegFactorOption {
        power: 0,
        coefficient: scalar,
    };
    let leg_options = vec![vec![vec![unit_leg]]; markings];

    assert_eq!(
        contract_restricted_external_leg_kernel_coeff_generic(&restricted_kernel, &leg_options, 0),
        <FactoredRatFun as Coeff>::one()
    );
}
