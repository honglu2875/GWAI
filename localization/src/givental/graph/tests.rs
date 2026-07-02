use super::super::*;
use super::*;
use crate::factored::FactoredRatFun;
use crate::geometry::CohomologyClass;
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
            crate::series::QSeries::constant(RatFun::from(2usize), 1),
            crate::series::QSeries::q(1),
        ],
        vec![
            crate::series::QSeries::q(1),
            crate::series::QSeries::constant(RatFun::from(5usize), 1),
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
        crate::algebra::Rational::from(2),
        crate::algebra::Rational::from(5),
    ];
    let r1 = calibration.r_matrix.coefficient(1).unwrap();
    let r2 = calibration.r_matrix.coefficient(2).unwrap();

    assert_eq!(
        r1.entry(0, 0)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::algebra::Rational::new(1, 36)
    );
    assert_eq!(
        r1.entry(1, 1)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::algebra::Rational::new(-1, 36)
    );
    assert_eq!(
        r2.entry(0, 0)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::algebra::Rational::new(1, 2592)
    );
    assert_eq!(
        r1.entry(0, 1)
            .coeff(0)
            .unwrap()
            .evaluate_lambda_weights(1, &weights)
            .unwrap(),
        crate::algebra::Rational::zero()
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
            crate::algebra::Rational::from(2),
            crate::algebra::Rational::from(5),
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
            crate::algebra::Rational::from(2),
            crate::algebra::Rational::from(5),
        ],
    );
}

#[test]
fn projective_j_calibration_r_coefficients_satisfy_diagonal_flatness() {
    let weights = [
        crate::algebra::Rational::from(2),
        crate::algebra::Rational::from(5),
        crate::algebra::Rational::from(11),
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
        crate::algebra::Rational::from(2),
        crate::algebra::Rational::from(5),
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
        crate::algebra::Rational::one()
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
        crate::algebra::Rational::one()
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
fn givental_graph_reproduces_genus_one_degree_zero_obstruction_values() {
    let cases = [
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(1, 1, 0, vec![tau(0, CohomologyClass::h_power(1, 1))])
            },
            crate::algebra::Rational::new(-1, 24),
        ),
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(2, 1, 0, vec![tau(0, CohomologyClass::h_power(2, 1))])
            },
            crate::algebra::Rational::new(-1, 8),
        ),
        (
            InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(2, 1, 0, vec![tau(1, CohomologyClass::one(2))])
            },
            crate::algebra::Rational::new(1, 8),
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
fn rational_no_insertion_graph_sidecar_matches_dense_evaluator() {
    let provider = ProjectiveSpaceProvider::lambda_line_nonequivariant(1);
    let genus = 2;
    let q_degree = 1;
    let graph_dimension = 3 * genus - 3;
    let kernel = provider
        .graph_kernel(q_degree, graph_dimension + 1, graph_dimension)
        .unwrap();
    let graphs = prepared_stable_graphs(genus, 0, provider.colors());

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
    let rational = evaluate_rational_no_insertion_graphs_if_possible(
        graphs.as_ref(),
        &kernel,
        q_degree,
        graph_dimension,
        &mut rational_profile,
    )
    .expect("nonequivariant P1 graph kernel should be rational");

    assert_eq!(rational, dense);
}

#[test]
fn graph_kernel_constructor_matches_projective_kernel_builder() {
    let weights = [
        crate::algebra::Rational::from(2),
        crate::algebra::Rational::from(5),
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
        RatFun::from_rational(crate::algebra::Rational::new(1, 4))
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
        RatFun::from_rational(crate::algebra::Rational::new(1, 36))
    );
}

#[test]
fn master_evaluator_matches_scalar_one_point_graph() {
    let series_req = crate::SeriesRequest {
        n: 1,
        genus: 1,
        degree_max: 1,
        max_markings: 1,
        max_descendant_power: 3,
        include_zero: true,
        equivariant: false,
        mode: ComputeMode::Givental,
        truncation: None,
    };
    let insertions = vec![tau(2, CohomologyClass::h_power(1, 1))];
    let mut evaluator = GiventalMasterEvaluator::new(&series_req);
    let master = evaluator.compute_coefficient(1, &insertions).unwrap();
    let scalar = compute_by_givental_graphs(&InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(1, 1, 1, insertions)
    })
    .unwrap()
    .value;
    assert_eq!(master, scalar);
}

#[test]
fn master_evaluator_matches_scalar_two_point_graph() {
    let series_req = crate::SeriesRequest {
        n: 1,
        genus: 1,
        degree_max: 1,
        max_markings: 2,
        max_descendant_power: 1,
        include_zero: true,
        equivariant: false,
        mode: ComputeMode::Givental,
        truncation: None,
    };
    let insertions = vec![
        tau(1, CohomologyClass::h_power(1, 1)),
        tau(1, CohomologyClass::h_power(1, 1)),
    ];
    let mut evaluator = GiventalMasterEvaluator::new(&series_req);
    let master = evaluator.compute_coefficient(1, &insertions).unwrap();
    let scalar = compute_by_givental_graphs(&InvariantRequest {
        mode: ComputeMode::Givental,
        ..InvariantRequest::new(1, 1, 1, insertions)
    })
    .unwrap()
    .value;
    assert_eq!(master, scalar);
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
    let orbits = vertex_coloring_orbits(&graph, 3);
    assert_eq!(orbits.len(), 6);
    assert_eq!(
        orbits.iter().map(|orbit| orbit.multiplicity).sum::<usize>(),
        9
    );
}

#[test]
fn prepared_stable_graphs_cache_metadata_matches_raw_graphs() {
    let first = prepared_stable_graphs(2, 0, 3);
    let second = prepared_stable_graphs(2, 0, 3);
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert!(!first.is_empty());

    for prepared in first.iter() {
        let raw_colorings = vertex_coloring_orbits(&prepared.graph, 3);
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

#[test]
fn sparse_series_planner_uses_restricted_kernel_for_few_one_point_tasks() {
    let req = crate::SeriesRequest {
        n: 2,
        genus: 3,
        degree_max: 2,
        max_markings: 1,
        max_descendant_power: 5,
        include_zero: false,
        equivariant: false,
        mode: ComputeMode::Givental,
        truncation: None,
    };
    let provider = ProjectiveSpaceProvider::new(req.n, req.equivariant);
    let basis = crate::insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];
    for markings in 0..=req.max_markings {
        for insertions in crate::insertion_monomials(&basis, markings) {
            for degree in
                provider.candidate_degrees_from_dimension(req.genus, req.degree_max, &insertions)
            {
                candidates_by_degree[degree].push(insertions.clone());
            }
        }
    }

    let counts = shared_kernel_task_counts(&req, &provider, &candidates_by_degree);
    assert_eq!(counts[1], 5);
    assert!(counts[1] < MASTER_MIN_SHARED_KERNEL_TASKS);
    let mut evaluator = GiventalMasterEvaluator::new(&req);
    evaluator.set_shared_kernel_task_counts(counts);
    assert!(!evaluator.can_use_external_leg_kernel(&[tau(3, CohomologyClass::one(2))]));
    assert!(evaluator.can_use_restricted_external_leg_kernel(&[tau(3, CohomologyClass::one(2))]));
}

#[derive(Debug, Clone)]
struct ShiftedDimensionProvider;

impl SemisimpleCohftProvider for ShiftedDimensionProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        2
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        insertion.descendant_power
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        insertions.iter().try_fold(0usize, |total, insertion| {
            total
                .checked_add(insertion.descendant_power)?
                .checked_add(insertion.class.pure_power()?)
        })
    }

    fn virtual_dimension(&self, _genus: usize, degree: usize, _markings: usize) -> Option<isize> {
        Some(if degree == 1 { 1 } else { 2 })
    }

    fn expected_degree_from_dimension(
        &self,
        _genus: usize,
        _insertions: &[Self::Insertion],
    ) -> Option<usize> {
        Some(1)
    }

    fn descendant_s_matrix(
        &self,
        _q_degree: usize,
        _z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        Err(GwError::UnsupportedInvariant(
            "dimension-only test provider has no S-matrix".to_string(),
        ))
    }

    fn graph_kernel(
        &self,
        _q_degree: usize,
        _r_order: usize,
        _graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        Err(GwError::UnsupportedInvariant(
            "dimension-only test provider has no graph kernel".to_string(),
        ))
    }

    fn insertion_vector(
        &self,
        _insertion: &Self::Insertion,
        _q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        Err(GwError::UnsupportedInvariant(
            "dimension-only test provider has no insertion vectors".to_string(),
        ))
    }
}

#[test]
fn series_planner_uses_provider_dimension_rule() {
    let req = crate::SeriesRequest {
        n: 1,
        genus: 1,
        degree_max: 1,
        max_markings: 1,
        max_descendant_power: 0,
        include_zero: false,
        equivariant: false,
        mode: ComputeMode::Givental,
        truncation: None,
    };
    let provider = ShiftedDimensionProvider;
    let insertions = vec![tau(0, CohomologyClass::h_power(1, 1))];
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];
    for degree in provider.candidate_degrees_from_dimension(req.genus, req.degree_max, &insertions)
    {
        candidates_by_degree[degree].push(insertions.clone());
    }

    let counts = shared_kernel_task_counts(&req, &provider, &candidates_by_degree);
    assert_eq!(counts[1], 1);
}

#[test]
fn restricted_sparse_kernel_matches_scalar_graph_evaluator() {
    let req = crate::SeriesRequest {
        n: 1,
        genus: 1,
        degree_max: 1,
        max_markings: 1,
        max_descendant_power: 2,
        include_zero: true,
        equivariant: false,
        mode: ComputeMode::Givental,
        truncation: None,
    };
    let cases = [
        (1, vec![tau(2, CohomologyClass::h_power(1, 1))]),
        (1, vec![tau(3, CohomologyClass::one(1))]),
    ];
    let mut evaluator = GiventalMasterEvaluator::new(&req);
    evaluator.set_shared_kernel_task_counts(vec![0, MASTER_MIN_RESTRICTED_KERNEL_TASKS]);
    let tasks = cases
        .iter()
        .enumerate()
        .map(|(ordinal, (degree, insertions))| {
            evaluator
                .prepare_sparse_contraction_task(ordinal, *degree, insertions)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let restricted = evaluator
        .contract_restricted_tasks(1, 1, &tasks, true)
        .unwrap();
    for result in restricted {
        let scalar = compute_by_givental_graphs(&InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                req.n,
                req.genus,
                result.coefficient.degree,
                result.coefficient.insertions.clone(),
            )
        })
        .unwrap();
        assert_eq!(result.coefficient.value, scalar.value);
    }
}

#[test]
fn packed_resolvent_matches_invariant_wise_projective_resolver() {
    let req = ResolventRequest {
        target_n: 1,
        genus: 0,
        degree: 0,
        markings: 3,
        virtual_dimension: 1,
    };
    let packed = compute_projective_resolvent_packed(&req, false).unwrap();
    let invariant_wise =
        crate::resolvent::compute_resolvent_generating_function(&req, |insertions| {
            compute_by_givental_graphs(&InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(1, 0, 0, insertions.to_vec())
            })
        })
        .unwrap();

    assert_eq!(packed.value, invariant_wise.value);
    assert_eq!(packed.candidate_terms, invariant_wise.candidate_terms);
    assert_eq!(packed.nonzero_terms, invariant_wise.nonzero_terms);
}

fn assert_r_matrix_unitary_after_lambda_eval(
    r: &SeriesRMatrix,
    metric: &SeriesMatrix,
    target_n: usize,
    weights: &[crate::algebra::Rational],
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
    weights: &[crate::algebra::Rational],
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
                assert_eq!(value, crate::algebra::Rational::zero());
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
    let graphs = prepared_stable_graphs(0, 3, size);
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
    let graphs = prepared_stable_graphs(0, markings, size);
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
    let graphs = prepared_stable_graphs(0, markings, size);

    let ratfun_unit_leg = LegFactorOption {
        power: 0,
        coefficient: QSeries::one(q_degree),
    };
    let template_task = MasterContractionTask {
        ordinal: 0,
        degree: 0,
        insertions: Vec::new(),
        markings,
        leg_options: vec![vec![vec![ratfun_unit_leg]]; markings],
    };
    let template = RestrictedExternalLegKernel::<FactoredRatFun>::from_tasks(
        markings,
        size,
        graph_dimension,
        q_degree,
        &[template_task],
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
