use super::{
    compute_packed_resolvent_with_coeff_provider, compute_packed_resolvent_with_provider,
    shared_kernel_task_counts, GiventalMasterEvaluator, MASTER_MIN_RESTRICTED_KERNEL_TASKS,
    MASTER_MIN_SHARED_KERNEL_TASKS,
};
use crate::core::algebra::RatFun;
use crate::core::error::GwError;
use crate::core::series::QSeries;
use crate::givental::{GiventalGraphKernel, SemisimpleCohftProvider, SeriesSMatrix};
use crate::spaces::projective_space::api::{
    compute_by_givental_graphs, compute_projective_resolvent_packed, insertion_basis,
    insertion_monomials, ComputeMode, Insertion, InvariantRequest, SeriesRequest,
};
use crate::spaces::projective_space::equivariant::CohomologyClass;
use crate::spaces::projective_space::provider::ProjectiveSpaceProvider;
use crate::spaces::projective_space::resolvent::{
    compute_resolvent_generating_function, ResolventRequest,
};
use crate::tau;
use std::sync::Arc;

fn projective_evaluator(req: &SeriesRequest) -> GiventalMasterEvaluator<ProjectiveSpaceProvider> {
    GiventalMasterEvaluator::with_provider(
        req,
        ProjectiveSpaceProvider::new(req.n, req.equivariant),
    )
}

#[test]
fn master_evaluator_matches_scalar_one_point_graph() {
    let series_req = SeriesRequest {
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
    let mut evaluator = projective_evaluator(&series_req);
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
    let series_req = SeriesRequest {
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
    let mut evaluator = projective_evaluator(&series_req);
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
fn sparse_series_planner_uses_restricted_kernel_for_few_one_point_tasks() {
    let req = SeriesRequest {
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
    let basis = insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];
    for markings in 0..=req.max_markings {
        for insertions in insertion_monomials(&basis, markings) {
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
    let mut evaluator = projective_evaluator(&req);
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
    let req = SeriesRequest {
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
    let req = SeriesRequest {
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
    let mut evaluator = projective_evaluator(&req);
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
    let invariant_wise = compute_resolvent_generating_function(&req, |insertions| {
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

#[derive(Debug, Clone)]
struct ExcessFriendlyPackedProvider(ProjectiveSpaceProvider);

impl SemisimpleCohftProvider for ExcessFriendlyPackedProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::colors(&self.0)
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::descendant_power(&self.0, insertion)
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::insertion_degree(&self.0, insertions)?
            .checked_add(1)
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::virtual_dimension(
            &self.0, genus, degree, markings,
        )
    }

    fn degree_is_effective(&self, degree: usize) -> bool {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::degree_is_effective(&self.0, degree)
    }

    fn vanishes_by_dimension(&self, _virtual_dimension: isize, _total_degree: usize) -> bool {
        false
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::descendant_s_matrix(
            &self.0, q_degree, z_order,
        )
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::graph_kernel(
            &self.0,
            q_degree,
            r_order,
            graph_dimension,
        )
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        <ProjectiveSpaceProvider as SemisimpleCohftProvider>::insertion_vector(
            &self.0, insertion, q_degree,
        )
    }
}

#[test]
fn packed_resolvent_keeps_its_exact_slice_under_generic_provider_policy() {
    let req = ResolventRequest::for_projective_space(1, 0, 0, 3);
    let provider = ExcessFriendlyPackedProvider(ProjectiveSpaceProvider::new(1, true));

    let expanded = compute_packed_resolvent_with_provider(
        &req,
        provider.clone(),
        "test-expanded-resolvent",
        "test",
        Ok::<RatFun, GwError>,
    )
    .unwrap();
    let generic = compute_packed_resolvent_with_coeff_provider::<RatFun, _, _>(
        &req,
        provider,
        "test-generic-resolvent",
        "test",
        Ok::<RatFun, GwError>,
    )
    .unwrap();

    assert!(expanded.candidate_terms > 0);
    assert_eq!(generic.candidate_terms, expanded.candidate_terms);
    assert_eq!(expanded.nonzero_terms, 0);
    assert_eq!(generic.nonzero_terms, 0);
    assert!(expanded.value.is_zero());
    assert!(generic.value.is_zero());
}

#[test]
fn point_target_positive_degree_packed_resolvent_is_empty() {
    let req = ResolventRequest {
        target_n: 0,
        genus: 0,
        degree: 1,
        markings: 3,
        virtual_dimension: 1,
    };
    for equivariant in [false, true] {
        let result = compute_projective_resolvent_packed(&req, equivariant).unwrap();
        assert!(result.value.is_zero());
        assert_eq!(result.candidate_terms, 0);
        assert_eq!(result.nonzero_terms, 0);
        assert_eq!(result.engine, "packed-resolvent-ineffective-degree");
    }
}

#[test]
fn packed_resolvent_rejects_inconsistent_provider_dimension() {
    // The request is intentionally inconsistent: its claimed virtual
    // dimension is zero, while P^5 at (g,d,m)=(2,0,0) has dimension -2.
    // Both packed paths must reject it before their no-marking shortcut.
    let req = ResolventRequest {
        target_n: 5,
        genus: 2,
        degree: 0,
        markings: 0,
        virtual_dimension: 0,
    };
    let provider = ProjectiveSpaceProvider::new(5, false);
    let expanded_err = compute_packed_resolvent_with_provider(
        &req,
        provider.clone(),
        "test-expanded-resolvent",
        "test",
        Ok::<RatFun, GwError>,
    )
    .unwrap_err();
    assert!(matches!(expanded_err, GwError::ConventionMismatch(_)));
    assert!(expanded_err
        .to_string()
        .contains("provider virtual dimension -2"));

    let generic_err = compute_packed_resolvent_with_coeff_provider::<RatFun, _, _>(
        &req,
        provider,
        "test-generic-resolvent",
        "test",
        Ok::<RatFun, GwError>,
    )
    .unwrap_err();
    assert!(matches!(generic_err, GwError::ConventionMismatch(_)));
    assert!(generic_err
        .to_string()
        .contains("provider virtual dimension -2"));
}

#[test]
fn packed_resolvent_validates_dimension_before_negative_shortcut() {
    let req = ResolventRequest {
        target_n: 5,
        genus: 2,
        degree: 0,
        markings: 0,
        virtual_dimension: -1,
    };

    let err = compute_projective_resolvent_packed(&req, false).unwrap_err();
    assert!(matches!(err, GwError::ConventionMismatch(_)));
    assert!(err.to_string().contains("provider virtual dimension -2"));
}
