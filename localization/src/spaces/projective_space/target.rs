//! Semisimple calibration data for ordinary projective space.
//!
//! [`ProjectiveTarget`] supplies the projective-space quantum divisor
//! relation and fixed-point eigenvalue seeds to the universal
//! [`GwTarget`] adapter.

use super::{CohomologyClass, ProjectiveSpaceTheory};
use crate::core::algebra::{lambda, Coeff, RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{QSeries, SeriesMatrix};
use crate::core::theory::GwTheory;
use crate::givental::target::GwTarget;

/// Projective space `P^n` as a [`GwTarget`]: the reference implementation.
///
/// `seeds` are the equivariant weights: the symbolic `lambda_i` for the fully
/// equivariant theory or rational constants for the specialized lambda-line
/// theory -- the implementation is identical for both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveTarget {
    theory: ProjectiveSpaceTheory,
    seeds: Vec<RatFun>,
}

impl ProjectiveTarget {
    pub fn equivariant(n: usize) -> Self {
        Self::try_equivariant(n).expect("projective calibration construction failed")
    }

    pub fn try_equivariant(n: usize) -> Result<Self, GwError> {
        let theory = ProjectiveSpaceTheory::try_new(n)?;
        let seeds = (0..theory.state_space().basis.len()).map(lambda).collect();
        Ok(Self { theory, seeds })
    }

    pub fn at_weights(n: usize, weights: &[Rational]) -> Self {
        Self::try_at_weights(n, weights).expect("projective calibration weights are invalid")
    }

    pub fn try_at_weights(n: usize, weights: &[Rational]) -> Result<Self, GwError> {
        let theory = ProjectiveSpaceTheory::try_new(n)?;
        let expected = theory.state_space().basis.len();
        if weights.len() != expected {
            return Err(GwError::ConventionMismatch(format!(
                "P^{n} calibration has {} weights, expected {expected}",
                weights.len()
            )));
        }
        let seeds = weights
            .iter()
            .map(|weight| RatFun::from_rational(weight.clone()))
            .collect();
        Ok(Self { theory, seeds })
    }

    pub fn n(&self) -> usize {
        self.theory.n()
    }

    pub fn theory(&self) -> &ProjectiveSpaceTheory {
        &self.theory
    }

    pub fn seeds(&self) -> &[RatFun] {
        &self.seeds
    }

    /// Coefficients of `prod_i (x - seed_i)` in ascending powers of `x`.
    fn classical_relation_polynomial(&self) -> Vec<RatFun> {
        let mut coefficients = vec![RatFun::one()];
        for seed in &self.seeds {
            let mut next = vec![RatFun::zero(); coefficients.len() + 1];
            for (power, coefficient) in coefficients.iter().enumerate() {
                next[power] = &next[power] - &(coefficient * seed);
                next[power + 1] = &next[power + 1] + coefficient;
            }
            coefficients = next;
        }
        coefficients
    }
}

impl GwTarget for ProjectiveTarget {
    fn canonical_theory(&self) -> &dyn GwTheory {
        &self.theory
    }

    fn cache_key(&self) -> String {
        let seeds = self
            .seeds
            .iter()
            .map(|seed| seed.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("{}[{seeds}]", self.theory.theory_fingerprint())
    }

    fn classical_eigenvalue_seeds(&self) -> Vec<RatFun> {
        self.seeds.clone()
    }

    fn divisor_multiplication(
        &self,
        q_degree: usize,
        quantum: bool,
    ) -> Result<SeriesMatrix, GwError> {
        // Companion matrix for H-multiplication: H^{n+1} reduces via
        // prod_i(H - seed_i) = q (or 0 classically).
        let size = self.theory.state_space().basis.len();
        let relation = self.classical_relation_polynomial();
        let mut matrix = vec![vec![QSeries::zero(q_degree); size]; size];
        for col in 0..size.saturating_sub(1) {
            matrix[col + 1][col] = QSeries::one(q_degree);
        }
        for row in 0..size {
            matrix[row][size - 1] = QSeries::constant(relation[row].neg(), q_degree);
        }
        if quantum {
            matrix[0][size - 1] = matrix[0][size - 1].add(&QSeries::q(q_degree));
        }
        Ok(SeriesMatrix::from_entries(matrix))
    }

    fn insertion_vector(
        &self,
        class: &CohomologyClass,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        let coeffs = class.coeffs();
        let expected = self.theory.state_space().basis.len();
        if coeffs.len() != expected {
            return Err(GwError::ConventionMismatch(format!(
                "P^{} insertion has {} coefficients, expected {}",
                self.n(),
                coeffs.len(),
                expected
            )));
        }
        Ok(coeffs
            .iter()
            .map(|coeff| QSeries::constant(coeff.clone(), q_degree))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::theory::CurveEffectivity;
    use crate::givental::{
        compute_semisimple_graph_value, SemisimpleCohftProvider, TargetProvider,
    };
    use crate::spaces::projective_space::{
        projective_space_descendant_s_matrix, projective_space_j_calibration,
        ProjectiveSpaceProvider,
    };
    use crate::tau;

    fn lambda_eval(value: &RatFun, target_n: usize, weights: &[Rational]) -> Rational {
        value
            .evaluate_lambda_weights(target_n, weights)
            .expect("generic test weights avoid poles")
    }

    #[test]
    fn target_provider_matches_projective_provider_calibration() {
        let weights = [Rational::from(2), Rational::from(5)];
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(1));
        let target_kernel = provider.graph_kernel(1, 3, 2).unwrap();
        let reference = projective_space_j_calibration(1, 1, 3).unwrap();
        let candidate = target_kernel.calibration();

        for order in 0..=3usize {
            let left = candidate.r_matrix.coefficient(order).unwrap();
            let right = reference.r_matrix.coefficient(order).unwrap();
            for row in 0..2 {
                for col in 0..2 {
                    for degree in 0..=1usize {
                        assert_eq!(
                            lambda_eval(left.entry(row, col).coeff(degree).unwrap(), 1, &weights),
                            lambda_eval(right.entry(row, col).coeff(degree).unwrap(), 1, &weights),
                            "R mismatch at z^{order} ({row},{col}) q^{degree}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn generic_projective_target_respects_point_effective_cone() {
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(0));
        assert!(provider.degree_is_effective(0));
        assert!(!provider.degree_is_effective(1));
        let formally_degree_one = [
            tau(1, CohomologyClass::one(0)),
            tau(0, CohomologyClass::one(0)),
            tau(0, CohomologyClass::one(0)),
        ];
        assert_eq!(
            provider.expected_degree_from_dimension(0, &formally_degree_one),
            None
        );
        assert!(provider
            .candidate_degrees_from_dimension(0, 2, &formally_degree_one)
            .is_empty());
    }

    #[test]
    fn target_provider_matches_projective_provider_descendant_s() {
        let weights = [Rational::from(2), Rational::from(5)];
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(1));
        let candidate = provider.descendant_s_matrix(1, 2).unwrap();
        let reference = projective_space_descendant_s_matrix(1, 1, 2).unwrap();
        for order in 0..=2usize {
            let left = candidate.coefficient(order).unwrap();
            let right = reference.coefficient(order).unwrap();
            for row in 0..2 {
                for col in 0..2 {
                    for degree in 0..=1usize {
                        assert_eq!(
                            lambda_eval(left.entry(row, col).coeff(degree).unwrap(), 1, &weights),
                            lambda_eval(right.entry(row, col).coeff(degree).unwrap(), 1, &weights),
                            "S mismatch at z^{order} ({row},{col}) q^{degree}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn target_provider_reproduces_known_invariants_at_weights() {
        // <tau_4(H)>_{2,1} on P^1 = 1/1920, through the generic target path
        // with rational seeds.
        let weights = [Rational::from(2), Rational::from(5)];
        let provider = TargetProvider::new(ProjectiveTarget::at_weights(1, &weights));
        let insertions = vec![tau(4, CohomologyClass::h_power(1, 1))];
        let value = compute_semisimple_graph_value(&provider, 2, 1, &insertions, None).unwrap();
        assert_eq!(value.as_rational(), Some(Rational::new(1, 1920)));

        // Degree-one line count on P^2 through two points and a line.
        let weights = [Rational::from(2), Rational::from(5), Rational::from(11)];
        let provider = TargetProvider::new(ProjectiveTarget::at_weights(2, &weights));
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        let value = compute_semisimple_graph_value(&provider, 0, 1, &insertions, None).unwrap();
        assert_eq!(value.as_rational(), Some(Rational::one()));
    }

    #[test]
    fn target_provider_symbolic_limits_to_known_invariant() {
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(1));
        let insertions = vec![tau(2, CohomologyClass::h_power(1, 1))];
        let value = compute_semisimple_graph_value(&provider, 1, 1, &insertions, None).unwrap();
        assert_eq!(
            value
                .nonequivariant_limit_line(1, &[Rational::from(2), Rational::from(3)])
                .unwrap(),
            Rational::new(1, 24)
        );
    }

    #[test]
    fn target_dimension_bookkeeping_matches_projective_provider() {
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(2));
        let reference = ProjectiveSpaceProvider::new(2, true);
        for genus in 0..3usize {
            for degree in 0..3usize {
                for markings in 0..4usize {
                    assert_eq!(
                        provider.virtual_dimension(genus, degree, markings),
                        reference.virtual_dimension(genus, degree, markings)
                    );
                }
            }
        }
        let insertions = vec![
            tau(1, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        assert_eq!(
            provider.expected_degree_from_dimension(0, &insertions),
            reference.expected_degree_from_dimension(0, &insertions)
        );
    }

    #[test]
    fn target_adapter_uses_its_canonical_theory_for_all_geometry() {
        let target = ProjectiveTarget::equivariant(2);
        assert_eq!(target.n(), 2);
        assert_eq!(
            target.canonical_theory().theory_fingerprint(),
            target.theory().theory_fingerprint()
        );

        let provider = TargetProvider::new(target);
        for degree in 0..=3 {
            let curve = provider.target().theory().curve(degree);
            assert_eq!(
                provider.virtual_dimension(2, degree, 4),
                Some(
                    provider
                        .target()
                        .theory()
                        .virtual_dimension(2, &curve, 4)
                        .unwrap()
                )
            );
            assert_eq!(
                provider.degree_is_effective(degree),
                provider.target().theory().effectivity(&curve).unwrap()
                    != CurveEffectivity::Ineffective
            );
        }
    }

    #[test]
    fn projective_target_rejects_mismatched_calibration_rank() {
        let error = ProjectiveTarget::try_at_weights(2, &[Rational::from(1)]).unwrap_err();
        assert!(matches!(error, GwError::ConventionMismatch(_)));
    }

    #[test]
    fn provider_rank_is_canonical_and_mismatched_calibration_is_rejected() {
        let mut target = ProjectiveTarget::equivariant(2);
        target.seeds.pop();
        let provider = TargetProvider::new(target);
        assert_eq!(provider.colors(), 3);
        assert!(matches!(
            provider.graph_kernel(0, 1, 0),
            Err(GwError::ConventionMismatch(message))
                if message.contains("canonical state space has rank 3")
        ));
    }

    #[test]
    fn target_adapter_rejects_unrepresentable_novikov_degree_without_panicking() {
        let provider = TargetProvider::new(ProjectiveTarget::equivariant(1));
        if usize::BITS > i64::BITS {
            assert_eq!(provider.virtual_dimension(0, usize::MAX, 0), None);
            assert!(!provider.degree_is_effective(usize::MAX));
        }
    }
}
