use super::ast::CorrelatorKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidualStatus {
    VerifiedZero,
    Nonzero,
    Incomplete,
}

/// The exact outcome.  The enum prevents an absent residual from being
/// accidentally reported as a passing constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidualOutcome<C> {
    VerifiedZero { exact_residual: C },
    Nonzero { exact_residual: C },
    Incomplete { exact_partial_sum: Option<C> },
}

impl<C> ResidualOutcome<C> {
    // Outcome status is intentionally private: only `ResidualReport::status`
    // also checks term accounting and missing dependencies before returning a
    // decisive result.
    fn status(&self) -> ResidualStatus {
        match self {
            Self::VerifiedZero { .. } => ResidualStatus::VerifiedZero,
            Self::Nonzero { .. } => ResidualStatus::Nonzero,
            Self::Incomplete { .. } => ResidualStatus::Incomplete,
        }
    }

    pub fn exact_residual(&self) -> Option<&C> {
        match self {
            Self::VerifiedZero { exact_residual } | Self::Nonzero { exact_residual } => {
                Some(exact_residual)
            }
            Self::Incomplete { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum IncompleteReason {
    /// Legacy broad backend support gap.
    Unsupported(String),
    /// Recognized target requiring missing mathematical machinery.
    UnsupportedFeature {
        target: String,
        feature: String,
        witness: String,
    },
    /// Explicit finite-work or retained-state boundary.
    ResourceLimit {
        operation: String,
        requested: usize,
        limit: usize,
    },
    OutsideBounds,
    EvaluationError(String),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MissingCorrelator<D, B> {
    pub correlator: CorrelatorKey<D, B>,
    pub reason: IncompleteReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedTerm<C> {
    pub term_index: usize,
    pub exact_contribution: C,
}

/// Auditable result of substituting exact correlators into one constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualReport<D, B, C> {
    /// Private so callers cannot infer success without the report's missing
    /// dependency and term-accounting checks.  Use [`Self::outcome`] only for
    /// rendering and [`Self::status`] for every pass/fail decision.
    outcome: ResidualOutcome<C>,
    total_terms: usize,
    evaluated_terms: Vec<EvaluatedTerm<C>>,
    missing_correlators: Vec<MissingCorrelator<D, B>>,
    /// Unique dependencies whose value came from the selected evaluator
    /// backend (including its exact cache).
    backend_correlators: Vec<CorrelatorKey<D, B>>,
    /// Unique dependencies proved zero from effectivity, stability, or virtual
    /// dimension without invoking the evaluator backend.
    structural_zero_correlators: Vec<CorrelatorKey<D, B>>,
    /// True when the evaluator retained only a bounded canonical prefix of the
    /// unique dependency closure.  One concrete omitted dependency is retained
    /// separately as an `OutsideBounds` witness.
    dependency_closure_truncated: bool,
    notes: Vec<String>,
}

impl<D, B, C> ResidualReport<D, B, C> {
    pub(crate) fn verified_zero(
        exact_residual: C,
        total_terms: usize,
        evaluated_terms: Vec<EvaluatedTerm<C>>,
    ) -> Self {
        Self::complete_outcome(
            ResidualOutcome::VerifiedZero { exact_residual },
            total_terms,
            evaluated_terms,
        )
    }

    pub(crate) fn nonzero(
        exact_residual: C,
        total_terms: usize,
        evaluated_terms: Vec<EvaluatedTerm<C>>,
    ) -> Self {
        Self::complete_outcome(
            ResidualOutcome::Nonzero { exact_residual },
            total_terms,
            evaluated_terms,
        )
    }

    pub(crate) fn incomplete(
        exact_partial_sum: Option<C>,
        total_terms: usize,
        evaluated_terms: Vec<EvaluatedTerm<C>>,
        missing_correlators: Vec<MissingCorrelator<D, B>>,
    ) -> Self {
        Self {
            outcome: ResidualOutcome::Incomplete { exact_partial_sum },
            total_terms,
            evaluated_terms,
            missing_correlators,
            backend_correlators: Vec::new(),
            structural_zero_correlators: Vec::new(),
            dependency_closure_truncated: false,
            notes: Vec::new(),
        }
    }

    /// Raw exact outcome for rendering.  A caller must use [`Self::status`]
    /// for pass/fail decisions because an otherwise exact outcome can still be
    /// incomplete due to dependency or term-accounting gaps.
    pub fn outcome(&self) -> &ResidualOutcome<C> {
        &self.outcome
    }

    /// Exact residual only when the complete report has a decisive status.
    pub fn exact_residual(&self) -> Option<&C> {
        match self.status() {
            ResidualStatus::VerifiedZero | ResidualStatus::Nonzero => self.outcome.exact_residual(),
            ResidualStatus::Incomplete => None,
        }
    }

    pub fn status(&self) -> ResidualStatus {
        if self.dependency_closure_truncated
            || !self.missing_correlators.is_empty()
            || !self.has_complete_term_accounting()
        {
            ResidualStatus::Incomplete
        } else {
            self.outcome.status()
        }
    }

    pub fn evaluated_term_count(&self) -> usize {
        self.evaluated_terms.len()
    }

    pub fn total_term_count(&self) -> usize {
        self.total_terms
    }

    pub fn evaluated_terms(&self) -> &[EvaluatedTerm<C>] {
        &self.evaluated_terms
    }

    pub fn missing_correlators(&self) -> &[MissingCorrelator<D, B>] {
        &self.missing_correlators
    }

    pub fn backend_correlators(&self) -> &[CorrelatorKey<D, B>] {
        &self.backend_correlators
    }

    pub fn structural_zero_correlators(&self) -> &[CorrelatorKey<D, B>] {
        &self.structural_zero_correlators
    }

    /// Whether at least one unique correlator dependency was omitted by the
    /// evaluator's dependency limit.
    pub fn dependency_closure_truncated(&self) -> bool {
        self.dependency_closure_truncated
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn backend_correlator_count(&self) -> usize {
        self.backend_correlators.len()
    }

    pub fn structural_zero_correlator_count(&self) -> usize {
        self.structural_zero_correlators.len()
    }

    pub fn missing_correlator_count(&self) -> usize {
        self.missing_correlators.len()
    }

    pub fn resolved_correlator_count(&self) -> usize {
        self.backend_correlators
            .len()
            .saturating_add(self.structural_zero_correlators.len())
    }

    /// Number of unique dependency diagnostics retained in this report.
    ///
    /// When [`Self::dependency_closure_truncated`] is true, the actual unique
    /// dependency count is larger.  This count includes the one omitted witness
    /// but not every omitted key.
    pub fn dependency_count(&self) -> usize {
        self.resolved_correlator_count()
            .saturating_add(self.missing_correlators.len())
    }

    pub fn is_complete(&self) -> bool {
        self.status() != ResidualStatus::Incomplete
    }

    fn complete_outcome(
        outcome: ResidualOutcome<C>,
        total_terms: usize,
        evaluated_terms: Vec<EvaluatedTerm<C>>,
    ) -> Self {
        let mut report = Self {
            outcome,
            total_terms,
            evaluated_terms,
            missing_correlators: Vec::new(),
            backend_correlators: Vec::new(),
            structural_zero_correlators: Vec::new(),
            dependency_closure_truncated: false,
            notes: Vec::new(),
        };
        if !report.has_complete_term_accounting() {
            let outcome = std::mem::replace(
                &mut report.outcome,
                ResidualOutcome::Incomplete {
                    exact_partial_sum: None,
                },
            );
            report.outcome = match outcome {
                ResidualOutcome::VerifiedZero { exact_residual }
                | ResidualOutcome::Nonzero { exact_residual } => ResidualOutcome::Incomplete {
                    exact_partial_sum: Some(exact_residual),
                },
                incomplete @ ResidualOutcome::Incomplete { .. } => incomplete,
            };
            report.notes.push(
                "complete residual rejected: every constraint term must be accounted for exactly once"
                    .to_string(),
            );
        }
        report
    }

    pub(crate) fn with_dependency_coverage(
        mut self,
        backend_correlators: Vec<CorrelatorKey<D, B>>,
        structural_zero_correlators: Vec<CorrelatorKey<D, B>>,
    ) -> Self {
        self.backend_correlators = backend_correlators;
        self.structural_zero_correlators = structural_zero_correlators;
        self
    }

    pub(crate) fn with_truncated_dependency_closure(mut self, retained_limit: usize) -> Self {
        self.dependency_closure_truncated = true;
        let outcome = std::mem::replace(
            &mut self.outcome,
            ResidualOutcome::Incomplete {
                exact_partial_sum: None,
            },
        );
        self.outcome = match outcome {
            ResidualOutcome::VerifiedZero { exact_residual }
            | ResidualOutcome::Nonzero { exact_residual } => ResidualOutcome::Incomplete {
                exact_partial_sum: Some(exact_residual),
            },
            incomplete @ ResidualOutcome::Incomplete { .. } => incomplete,
        };
        self.notes.push(format!(
            "dependency closure truncated at {retained_limit} retained canonical keys; one omitted dependency is recorded as an OutsideBounds witness"
        ));
        self
    }

    pub(crate) fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    fn has_complete_term_accounting(&self) -> bool {
        if self.evaluated_terms.len() != self.total_terms {
            return false;
        }
        let mut seen = vec![false; self.total_terms];
        for term in &self.evaluated_terms {
            let Some(slot) = seen.get_mut(term.term_index) else {
                return false;
            };
            if *slot {
                return false;
            }
            *slot = true;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::virasoro::{CorrelatorKey, Descendant};

    #[test]
    fn incomplete_term_accounting_can_never_pass() {
        let report = ResidualReport::<usize, usize, i32>::verified_zero(
            0,
            2,
            vec![EvaluatedTerm {
                term_index: 0,
                exact_contribution: 7,
            }],
        );
        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert!(!report.is_complete());
        assert!(matches!(
            report.outcome(),
            ResidualOutcome::Incomplete { .. }
        ));
        assert!(report.notes()[0].contains("every constraint term"));
    }

    #[test]
    fn missing_correlator_is_explicitly_incomplete() {
        let missing = MissingCorrelator {
            correlator: CorrelatorKey::new(2, 3usize, vec![Descendant::new(1, 0usize)]),
            reason: IncompleteReason::OutsideBounds,
        };
        let report = ResidualReport::<usize, usize, i32>::incomplete(
            Some(5),
            1,
            Vec::new(),
            vec![missing.clone()],
        );
        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(report.missing_correlators(), &[missing]);
        assert_eq!(report.outcome().exact_residual(), None);
        assert_eq!(report.exact_residual(), None);
    }

    #[test]
    fn exact_complete_residual_has_a_decisive_status() {
        let evaluated = vec![EvaluatedTerm {
            term_index: 0,
            exact_contribution: 0,
        }];
        let zero = ResidualReport::<usize, usize, i32>::verified_zero(0, 1, evaluated.clone());
        let nonzero = ResidualReport::<usize, usize, i32>::nonzero(4, 1, evaluated);
        assert_eq!(zero.status(), ResidualStatus::VerifiedZero);
        assert_eq!(zero.outcome().exact_residual(), Some(&0));
        assert_eq!(zero.exact_residual(), Some(&0));
        assert_eq!(nonzero.status(), ResidualStatus::Nonzero);
        assert_eq!(nonzero.outcome().exact_residual(), Some(&4));
        assert_eq!(nonzero.exact_residual(), Some(&4));
    }

    #[test]
    fn dependency_coverage_counts_its_two_resolution_sources() {
        let backend = CorrelatorKey::new(0, 0usize, vec![Descendant::new(0, 0usize)]);
        let structural = CorrelatorKey::new(0, 0usize, Vec::new());
        let report = ResidualReport::<usize, usize, i32>::verified_zero(
            0,
            1,
            vec![EvaluatedTerm {
                term_index: 0,
                exact_contribution: 0,
            }],
        )
        .with_dependency_coverage(vec![backend.clone()], vec![structural.clone()]);

        assert_eq!(report.backend_correlators(), &[backend]);
        assert_eq!(report.structural_zero_correlators(), &[structural]);
        assert_eq!(report.backend_correlator_count(), 1);
        assert_eq!(report.structural_zero_correlator_count(), 1);
        assert_eq!(report.resolved_correlator_count(), 2);
        assert_eq!(report.missing_correlator_count(), 0);
        assert_eq!(report.dependency_count(), 2);
    }

    #[test]
    fn truncated_dependency_metadata_forces_an_incomplete_outcome() {
        let report = ResidualReport::<usize, usize, i32>::verified_zero(0, 0, Vec::new())
            .with_truncated_dependency_closure(0);

        assert!(report.dependency_closure_truncated());
        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert!(!report.is_complete());
        assert!(matches!(
            report.outcome(),
            ResidualOutcome::Incomplete {
                exact_partial_sum: Some(0)
            }
        ));
        assert_eq!(report.exact_residual(), None);
        assert!(report
            .notes()
            .iter()
            .any(|note| note.contains("dependency closure truncated at 0")));
    }
}
