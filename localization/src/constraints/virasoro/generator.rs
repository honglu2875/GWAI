use super::{
    CohomologicalGrading, ConstraintSector, ConstraintTerm, CorrelatorKey, Descendant,
    DilatonShift, FormulaSource, LinearTerm, PotentialConvention, QuadraticTerm,
    StateSpaceConvention, TermOrigin, TheoryLabel, TimeMonomial, TimeNormalization,
    UnstableConvention, VirasoroConstraint, VirasoroConventions, VirasoroOperator,
};
use crate::algebra::Rational;
use crate::error::GwError;
use crate::theory::{
    BasisId, CurveClass, CurveEffectivity, GwTheory, Parity, StateSpace, StateSpaceMatrix,
    VirasoroOperatorKind,
};
use std::collections::BTreeMap;

pub type CanonicalVirasoroConstraint = VirasoroConstraint<CurveClass, BasisId, Rational>;
pub const DEFAULT_GENERATED_TERM_LIMIT: usize = 1_000_000;
/// Explicit work guard for bracket-polynomial and matrix-power generation.
pub const MAX_STANDARD_VIRASORO_OPERATOR_INDEX: usize = 64;
/// Explicit payload guard for labelled correlator keys in one coefficient.
pub const MAX_VIRASORO_MARKINGS: usize = 64;

/// Generate one exact coefficient of Getzler's Virasoro equation.
///
/// The coefficient is obtained from `Z^{-1} L_k Z` by labelled derivatives
/// in the supplied time variables.  Consequently repeated external markings
/// carry their correct binomial multiplicities without an implicit factorial
/// convention.  `operator_index = -1` is the string operator; nonnegative
/// indices use the corrected EHX/Getzler formula.
pub fn generate_constraint<T: GwTheory + ?Sized>(
    theory: &T,
    operator_index: i32,
    genus: usize,
    degree: CurveClass,
    time_coefficient: TimeMonomial<BasisId>,
) -> Result<CanonicalVirasoroConstraint, GwError> {
    generate_constraint_with_term_limit(
        theory,
        operator_index,
        genus,
        degree,
        time_coefficient,
        DEFAULT_GENERATED_TERM_LIMIT,
    )
}

/// Generate with an explicit upper bound on the unaggregated term expansion.
/// The bound is checked before labelled marking partitions or matrix powers
/// are materialized.
pub fn generate_constraint_with_term_limit<T: GwTheory + ?Sized>(
    theory: &T,
    operator_index: i32,
    genus: usize,
    degree: CurveClass,
    time_coefficient: TimeMonomial<BasisId>,
    term_limit: usize,
) -> Result<CanonicalVirasoroConstraint, GwError> {
    if operator_index < -1 {
        return Err(GwError::ConventionMismatch(
            "Virasoro operator index must be at least -1".to_string(),
        ));
    }
    if usize::try_from(operator_index)
        .is_ok_and(|index| index > MAX_STANDARD_VIRASORO_OPERATOR_INDEX)
    {
        return Err(GwError::ResourceLimit {
            operation: "standard Virasoro operator index".to_string(),
            requested: operator_index as usize,
            limit: MAX_STANDARD_VIRASORO_OPERATOR_INDEX,
        });
    }
    match theory.virasoro_operator_kind() {
        VirasoroOperatorKind::StandardCompactGetzler => {}
        VirasoroOperatorKind::QrrConjugatedRequired => {
            return Err(GwError::UnsupportedFeature {
                target: theory.theory_id(),
                feature: "Virasoro operator generation".to_string(),
                witness: "the theory requires a QRR-conjugated operator; the standard compact Getzler operator is not valid".to_string(),
            });
        }
        VirasoroOperatorKind::Unsupported => {
            return Err(GwError::UnsupportedFeature {
                target: theory.theory_id(),
                feature: "Virasoro operator generation".to_string(),
                witness: "the canonical theory does not declare an operator model".to_string(),
            });
        }
    }
    theory.curve_class_space().validate(&degree)?;
    let effectivity = theory.effectivity(&degree)?;
    let marking_count = time_coefficient.total_degree().ok_or_else(|| {
        GwError::AlgebraFailure("time-coefficient marking count overflow".to_string())
    })?;
    if operator_index == -1 {
        let estimate = marking_count.checked_add(2).ok_or_else(|| {
            GwError::UnsupportedInvariant("string coefficient expansion size overflow".to_string())
        })?;
        if estimate > term_limit {
            return Err(GwError::ResourceLimit {
                operation: "string Virasoro coefficient expansion".to_string(),
                requested: estimate,
                limit: term_limit,
            });
        }
    }
    if marking_count > MAX_VIRASORO_MARKINGS {
        return Err(GwError::ResourceLimit {
            operation: "Virasoro markings in one equation".to_string(),
            requested: marking_count,
            limit: MAX_VIRASORO_MARKINGS,
        });
    }
    let degree_split_count = if operator_index <= 0 || effectivity == CurveEffectivity::Ineffective
    {
        0
    } else {
        theory.admissible_decomposition_count(&degree)?
    };
    if operator_index >= 0 {
        enforce_term_budget(
            operator_index as usize,
            genus,
            marking_count,
            theory.state_space().basis.len(),
            degree_split_count,
            term_limit,
        )?;
    }
    // Run allocation and expansion guards before the intentionally thorough
    // provider-data validation.  This keeps a tiny term budget a cheap refusal
    // even for a large extension-provider state space.
    let state_space = validate_compact_virasoro_data(theory)?;
    let external = expand_time_coefficient(&time_coefficient)?;
    for insertion in &external {
        if state_space.element(insertion.class).is_none() {
            return Err(GwError::ConventionMismatch(format!(
                "time coefficient contains unknown basis id {}",
                insertion.class.0
            )));
        }
    }

    let mut terms = TermAccumulator::default();
    if operator_index == -1 {
        generate_string_terms(state_space, genus, &degree, &external, &mut terms);
    } else {
        generate_nonnegative_terms(
            theory,
            state_space,
            operator_index as usize,
            genus,
            &degree,
            &external,
            &mut terms,
            degree_split_count,
            term_limit,
        )?;
    }

    let mut conventions = VirasoroConventions {
        potential: PotentialConvention::LogarithmicPartitionFunctionEquation,
        time_normalization: TimeNormalization::Exponential,
        dilaton_shift: DilatonShift::Explicit(
            "unshifted t-coordinates; the standard unit dilaton shift is expanded explicitly"
                .to_string(),
        ),
        grading: CohomologicalGrading::Complex,
        unstable: UnstableConvention::Excluded,
        state_space: StateSpaceConvention::EvenOnly,
        novikov_variables: theory.curve_class_space().coordinate_names.clone(),
        equivariant_parameters: Vec::new(),
        notes: vec![
            "coefficient extracted by labelled time derivatives of Z^{-1} L_k Z at t=0"
                .to_string(),
            "negative-genus terms are absent; degree-zero unstable correlators are excluded and supplied only by explicit operator corrections"
                .to_string(),
        ],
    };
    if effectivity == CurveEffectivity::Ineffective {
        conventions.notes.push(
            "the requested curve class is canonical-theory-certified ineffective, so every correlator dependency is a forced zero"
                .to_string(),
        );
    } else if effectivity == CurveEffectivity::Unknown {
        conventions.notes.push(
            "the class lies in the canonical theory's admissible support cone, but effectivity is not certified; dependencies will be queried rather than forced to zero"
                .to_string(),
        );
    }
    let mut source =
        FormulaSource::new("Getzler's corrected Eguchi--Hori--Xiong Virasoro operators");
    source.citation = Some("https://arxiv.org/abs/math/9812026".to_string());
    source.locator = Some("equations (Lk), (L-1), and (zkg)".to_string());
    source.derivation = Some(
        "connected coefficient expansion of Z^{-1} L_k Z, including genus and canonical-theory-admissible curve-class splittings"
            .to_string(),
    );

    let label = theory.theory_id();
    Ok(VirasoroConstraint {
        theory: TheoryLabel::new(label, theory.theory_tex()),
        theory_fingerprint: theory.theory_fingerprint(),
        operator: VirasoroOperator::new(operator_index),
        sector: ConstraintSector::new(genus, degree),
        time_coefficient,
        terms: terms.finish(),
        conventions,
        source,
    })
}

fn validate_compact_virasoro_data<T: GwTheory + ?Sized>(
    theory: &T,
) -> Result<&StateSpace, GwError> {
    let state = theory.state_space();
    for (index, basis) in state.basis.iter().enumerate() {
        if basis.id != BasisId(index) {
            return Err(GwError::ConventionMismatch(
                "Virasoro basis ids must be dense and agree with basis order".to_string(),
            ));
        }
    }
    if state.basis.iter().any(|basis| basis.parity == Parity::Odd) {
        return Err(GwError::UnsupportedInvariant(
            "the current Virasoro correlator key supports even state spaces only; Koszul signs are not represented"
                .to_string(),
        ));
    }
    let pairing = state.pairing.as_ref().ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "standard compact Virasoro generation requires the theory's Poincare/twisted pairing and inverse; local twists require the QRR-conjugated operator"
                .to_string(),
        )
    })?;
    let c1 = state.c1_action.as_ref().ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "standard compact Virasoro generation requires cup product by c1(TX)".to_string(),
        )
    })?;
    let size = state.basis.len();
    if pairing.metric.size() != size || pairing.inverse.size() != size || c1.size() != size {
        return Err(GwError::ConventionMismatch(
            "Virasoro geometric matrices do not match the canonical-theory basis".to_string(),
        ));
    }
    if pairing.metric.multiply(&pairing.inverse)? != StateSpaceMatrix::try_identity(size)? {
        return Err(GwError::ConventionMismatch(
            "the supplied inverse Poincare pairing is not an inverse".to_string(),
        ));
    }
    if state.element(state.unit).is_none() {
        return Err(GwError::ConventionMismatch(
            "Virasoro canonical-theory unit is not in its basis".to_string(),
        ));
    }
    if state.element(state.unit).unwrap().hodge_p_degree != 0
        || state.element(state.unit).unwrap().complex_codimension != 0
    {
        return Err(GwError::ConventionMismatch(
            "Virasoro canonical-theory unit must have complex degree zero".to_string(),
        ));
    }
    let dimension = theory.target_dimension();
    for left in 0..size {
        for right in 0..size {
            let metric = pairing.metric.entry(left, right);
            if metric != pairing.metric.entry(right, left) {
                return Err(GwError::ConventionMismatch(
                    "Poincare pairing must be symmetric in the even-state-space convention"
                        .to_string(),
                ));
            }
            if !metric.is_zero()
                && (state.basis[left]
                    .hodge_p_degree
                    .checked_add(state.basis[right].hodge_p_degree)
                    != Some(dimension)
                    || state.basis[left]
                        .complex_codimension
                        .checked_add(state.basis[right].complex_codimension)
                        != Some(dimension))
            {
                return Err(GwError::ConventionMismatch(
                    "nonzero Poincare pairing violates the complex grading".to_string(),
                ));
            }
            if !c1.entry(right, left).is_zero()
                && (state.basis[left].hodge_p_degree.checked_add(1)
                    != Some(state.basis[right].hodge_p_degree)
                    || state.basis[left].complex_codimension.checked_add(1)
                        != Some(state.basis[right].complex_codimension))
            {
                return Err(GwError::ConventionMismatch(
                    "cup product by c1 does not raise complex degree by one".to_string(),
                ));
            }
        }
    }
    // Lower the output index of c1 once, exploiting sparse canonical
    // matrices, instead of recomputing the contraction for every pair.
    let mut lowered_c1 = StateSpaceMatrix::try_zero(size)?;
    for left in 0..size {
        for middle in 0..size {
            let coefficient = c1.entry(middle, left);
            if coefficient.is_zero() {
                continue;
            }
            for right in 0..size {
                let metric = pairing.metric.entry(middle, right);
                if !metric.is_zero() {
                    let value = lowered_c1.entry(left, right).clone()
                        + coefficient.clone() * metric.clone();
                    lowered_c1.set_entry(left, right, value);
                }
            }
        }
    }
    for left in 0..size {
        for right in 0..size {
            if lowered_c1.entry(left, right) != lowered_c1.entry(right, left) {
                return Err(GwError::ConventionMismatch(
                    "cup product by c1 is not self-adjoint for the canonical-theory pairing"
                        .to_string(),
                ));
            }
        }
    }
    let mut grading_supertrace = Rational::zero();
    for basis in &state.basis {
        let mu = Rational::from(basis.hodge_p_degree) - Rational::new(dimension as i128, 2);
        let summand = mu.pow_usize(2) - Rational::new(1, 4);
        grading_supertrace += match basis.parity {
            Parity::Even => summand,
            Parity::Odd => -summand,
        };
    }
    let grading_anomaly = Rational::new(-1, 4) * grading_supertrace;
    let characteristic_anomaly = theory.virasoro_anomaly().ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "standard compact Virasoro generation requires characteristic data for the genus-one anomaly"
                .to_string(),
        )
    })?;
    if characteristic_anomaly != grading_anomaly {
        return Err(GwError::ConventionMismatch(format!(
            "canonical-theory characteristic anomaly {characteristic_anomaly} disagrees with the grading identity {grading_anomaly}"
        )));
    }
    Ok(state)
}

fn expand_time_coefficient(
    time: &TimeMonomial<BasisId>,
) -> Result<Vec<Descendant<BasisId>>, GwError> {
    let capacity = time.total_degree().ok_or_else(|| {
        GwError::AlgebraFailure("time-coefficient marking count overflow".to_string())
    })?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity).map_err(|_| {
        GwError::UnsupportedInvariant(format!(
            "cannot allocate {capacity} labelled time derivatives"
        ))
    })?;
    for (descendant, multiplicity) in time.factors() {
        for _ in 0..multiplicity {
            out.push(descendant.clone());
        }
    }
    Ok(out)
}

fn generate_string_terms(
    state: &StateSpace,
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    terms: &mut TermAccumulator,
) {
    let mut with_unit = external.to_vec();
    with_unit.push(Descendant::new(0, state.unit));
    terms.add_linear(
        -Rational::one(),
        CorrelatorKey::new(genus, degree.clone(), with_unit),
        TermOrigin::LinearOperator,
    );
    for (index, insertion) in external.iter().enumerate() {
        if insertion.psi_power == 0 {
            continue;
        }
        let mut replaced = external.to_vec();
        replaced[index] = Descendant::new(insertion.psi_power - 1, insertion.class);
        terms.add_linear(
            Rational::one(),
            CorrelatorKey::new(genus, degree.clone(), replaced),
            TermOrigin::LinearOperator,
        );
    }
    if genus == 0
        && degree.is_zero()
        && external.len() == 2
        && external.iter().all(|insertion| insertion.psi_power == 0)
    {
        let pairing = state.pairing.as_ref().expect("validated");
        terms.add_constant(
            pairing
                .metric
                .entry(external[0].class.0, external[1].class.0)
                .clone(),
            TermOrigin::UnstableCorrection,
        );
    }
}

fn generate_nonnegative_terms<T: GwTheory + ?Sized>(
    theory: &T,
    state: &StateSpace,
    k: usize,
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    terms: &mut TermAccumulator,
    degree_split_count: usize,
    term_limit: usize,
) -> Result<(), GwError> {
    let pairing = state.pairing.as_ref().expect("validated");
    let c1 = state.c1_action.as_ref().expect("validated");
    let size = state.basis.len();
    let dimension = theory.target_dimension();
    let half = Rational::new(1, 2);
    let shift_x = Rational::new(3 - dimension as i128, 2);
    let degree_splits = if degree_split_count == 0 {
        Vec::new()
    } else {
        let splits = theory.admissible_decompositions(degree)?;
        if splits.len() != degree_split_count {
            return Err(GwError::ValidationFailure(format!(
                "theory {} reported {degree_split_count} admissible decompositions but produced {}",
                theory.theory_id(),
                splits.len()
            )));
        }
        let mut seen = BTreeMap::new();
        for split in &splits {
            theory.curve_class_space().validate(&split.left)?;
            theory.curve_class_space().validate(&split.right)?;
            if split.left.checked_add(&split.right).as_ref() != Some(degree) {
                return Err(GwError::ValidationFailure(format!(
                    "theory {} returned an admissible decomposition that does not sum to {degree}",
                    theory.theory_id()
                )));
            }
            if seen.insert(split.clone(), ()).is_some() {
                return Err(GwError::ValidationFailure(format!(
                    "theory {} returned a duplicate admissible decomposition",
                    theory.theory_id()
                )));
            }
        }
        splits
    };
    let powers = (0..=k + 1)
        .map(|power| c1.pow(power))
        .collect::<Result<Vec<_>, _>>()?;
    let marking_splits = if k > 0 && !degree_splits.is_empty() {
        labelled_marking_splits(external, term_limit)?
    } else {
        Vec::new()
    };

    for i in 0..=k + 1 {
        let power = &powers[i];
        let pure_bracket = getzler_bracket(shift_x.clone(), k, i)?;
        for output in 0..size {
            let coefficient = -pure_bracket.clone() * power.entry(output, state.unit.0).clone();
            if !coefficient.is_zero() {
                let mut insertions = external.to_vec();
                insertions.push(Descendant::new(k + 1 - i, BasisId(output)));
                terms.add_linear(
                    coefficient,
                    CorrelatorKey::new(genus, degree.clone(), insertions),
                    TermOrigin::DilatonShift,
                );
            }
        }

        for (marking, insertion) in external.iter().enumerate() {
            let Some(output_psi) = insertion
                .psi_power
                .checked_add(k)
                .and_then(|value| value.checked_sub(i))
            else {
                continue;
            };
            let basis = state.element(insertion.class).expect("validated basis");
            let mu_plus = Rational::from(basis.hodge_p_degree)
                - Rational::new(dimension as i128, 2)
                + Rational::from(insertion.psi_power)
                + half.clone();
            let bracket = getzler_bracket(mu_plus, k, i)?;
            for output in 0..size {
                let coefficient = bracket.clone() * power.entry(output, insertion.class.0).clone();
                if coefficient.is_zero() {
                    continue;
                }
                let mut replaced = external.to_vec();
                replaced[marking] = Descendant::new(output_psi, BasisId(output));
                terms.add_linear(
                    coefficient,
                    CorrelatorKey::new(genus, degree.clone(), replaced),
                    TermOrigin::LinearOperator,
                );
            }
        }

        if genus > 0 || !degree_splits.is_empty() {
            let lower_m = i as isize - k as isize;
            for m in lower_m..=-1 {
                let left_psi = (-m - 1) as usize;
                let right_psi = (m + k as isize - i as isize) as usize;
                let sign = if m.rem_euclid(2) == 0 {
                    Rational::one()
                } else {
                    -Rational::one()
                };
                // In Getzler's tensor
                //   [mu_c+m+1/2]^k_i eta^{ac} (R^i)^b_c,
                // the grading belongs to the lower input `c` before its
                // index is raised by the inverse pairing.  It cannot be
                // factored out using the degree of the derivative basis `a`.
                let middle_brackets = state
                    .basis
                    .iter()
                    .map(|basis| {
                        let mu_plus = Rational::from(basis.hodge_p_degree)
                            - Rational::new(dimension as i128, 2)
                            + Rational::from(m)
                            + half.clone();
                        getzler_bracket(mu_plus, k, i)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for left_basis in 0..size {
                    for right_basis in 0..size {
                        let raised = weighted_raised_entry(
                            power,
                            &pairing.inverse,
                            &middle_brackets,
                            left_basis,
                            right_basis,
                        );
                        let coefficient = Rational::new(1, 2) * sign.clone() * raised;
                        if coefficient.is_zero() {
                            continue;
                        }
                        if genus > 0 {
                            let mut insertions = external.to_vec();
                            insertions.push(Descendant::new(left_psi, BasisId(left_basis)));
                            insertions.push(Descendant::new(right_psi, BasisId(right_basis)));
                            terms.add_linear(
                                coefficient.clone(),
                                CorrelatorKey::new(genus - 1, degree.clone(), insertions),
                                TermOrigin::GenusReduction,
                            );
                        }
                        for split_genus in 0..=genus {
                            for curve_split in &degree_splits {
                                for (left_markings, right_markings) in &marking_splits {
                                    let mut left_insertions = left_markings.clone();
                                    left_insertions
                                        .push(Descendant::new(left_psi, BasisId(left_basis)));
                                    let mut right_insertions = right_markings.clone();
                                    right_insertions
                                        .push(Descendant::new(right_psi, BasisId(right_basis)));
                                    terms.add_quadratic(
                                        coefficient.clone(),
                                        CorrelatorKey::new(
                                            split_genus,
                                            curve_split.left.clone(),
                                            left_insertions,
                                        ),
                                        CorrelatorKey::new(
                                            genus - split_genus,
                                            curve_split.right.clone(),
                                            right_insertions,
                                        ),
                                        TermOrigin::DegreeSplitting,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if genus == 0
        && degree.is_zero()
        && external.len() == 2
        && external.iter().all(|insertion| insertion.psi_power == 0)
    {
        terms.add_constant(
            lowered_entry(
                &powers[k + 1],
                &pairing.metric,
                external[0].class.0,
                external[1].class.0,
            ),
            TermOrigin::UnstableCorrection,
        );
    }
    if k == 0 && genus == 1 && degree.is_zero() && external.is_empty() {
        let anomaly = theory.virasoro_anomaly().ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "the L_0 genus-one constant requires the canonical theory's top Chern and c1*c_(r-1) characteristic numbers"
                    .to_string(),
            )
        })?;
        terms.add_constant(anomaly, TermOrigin::Other("genus-one anomaly".to_string()));
    }
    Ok(())
}

fn enforce_term_budget(
    k: usize,
    genus: usize,
    markings: usize,
    state_space_size: usize,
    degree_splits: usize,
    term_limit: usize,
) -> Result<(), GwError> {
    let checked_add = |left: usize, right: usize| {
        left.checked_add(right).ok_or_else(|| {
            GwError::UnsupportedInvariant("Virasoro term estimate overflow".to_string())
        })
    };
    let checked_mul = |left: usize, right: usize| {
        left.checked_mul(right).ok_or_else(|| {
            GwError::UnsupportedInvariant("Virasoro term estimate overflow".to_string())
        })
    };
    let operator_width = k.checked_add(2).ok_or_else(|| {
        GwError::UnsupportedInvariant("Virasoro operator index is too large".to_string())
    })?;
    let linear_slots = checked_mul(
        checked_mul(operator_width, state_space_size)?,
        markings.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("Virasoro marking count is too large".to_string())
        })?,
    )?;
    let second_index_pairs = checked_mul(
        k,
        k.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("Virasoro operator index is too large".to_string())
        })?,
    )? / 2;
    let basis_pairs = checked_mul(state_space_size, state_space_size)?;
    let genus_reduction = usize::from(genus > 0);
    let disconnected = if degree_splits == 0 || second_index_pairs == 0 {
        0
    } else {
        let marking_partitions = 1usize
            .checked_shl(u32::try_from(markings).map_err(|_| {
                GwError::UnsupportedInvariant(
                    "too many markings for labelled partitions".to_string(),
                )
            })?)
            .ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "too many markings for labelled partitions".to_string(),
                )
            })?;
        checked_mul(
            genus.checked_add(1).ok_or_else(|| {
                GwError::UnsupportedInvariant("Virasoro genus is too large".to_string())
            })?,
            checked_mul(degree_splits, marking_partitions)?,
        )?
    };
    let genus_and_splitting = checked_add(genus_reduction, disconnected)?;
    let second_slots = checked_mul(
        checked_mul(second_index_pairs, basis_pairs)?,
        genus_and_splitting,
    )?;
    let estimate = checked_add(checked_add(linear_slots, second_slots)?, 2)?;
    if estimate > term_limit {
        return Err(GwError::ResourceLimit {
            operation: "Virasoro coefficient expansion upper bound".to_string(),
            requested: estimate,
            limit: term_limit,
        });
    }
    Ok(())
}

fn weighted_raised_entry(
    power: &StateSpaceMatrix,
    inverse_metric: &StateSpaceMatrix,
    input_weights: &[Rational],
    left: usize,
    right: usize,
) -> Rational {
    let mut value = Rational::zero();
    for middle in 0..power.size() {
        // eta^{ac} [mu_c+m+1/2]^k_i (R^i)^b_c
        value += inverse_metric.entry(left, middle).clone()
            * input_weights[middle].clone()
            * power.entry(right, middle).clone();
    }
    value
}

fn lowered_entry(
    power: &StateSpaceMatrix,
    metric: &StateSpaceMatrix,
    left: usize,
    right: usize,
) -> Rational {
    let mut value = Rational::zero();
    for middle in 0..power.size() {
        // (R^i)_{ab} = (R^i)_a^c eta_cb.
        value += power.entry(middle, left).clone() * metric.entry(middle, right).clone();
    }
    value
}

/// Getzler's `[x]^k_i = e_(k+1-i)(x,x+1,...,x+k)`.
pub fn getzler_bracket(x: Rational, k: usize, i: usize) -> Result<Rational, GwError> {
    let maximum_index = k.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("Getzler bracket index overflow".to_string())
    })?;
    if i > maximum_index {
        return Ok(Rational::zero());
    }
    let wanted = maximum_index - i;
    let width = k.checked_add(2).ok_or_else(|| {
        GwError::UnsupportedInvariant("Getzler bracket index overflow".to_string())
    })?;
    let mut elementary = Vec::new();
    elementary.try_reserve_exact(width).map_err(|_| {
        GwError::UnsupportedInvariant(format!("cannot allocate Getzler bracket of width {width}"))
    })?;
    elementary.resize(width, Rational::zero());
    elementary[0] = Rational::one();
    for offset in 0..=k {
        let value = x.clone() + Rational::from(offset);
        for degree in (1..=offset + 1).rev() {
            elementary[degree] =
                elementary[degree].clone() + elementary[degree - 1].clone() * value.clone();
        }
    }
    Ok(elementary[wanted].clone())
}

fn labelled_marking_splits(
    markings: &[Descendant<BasisId>],
    term_limit: usize,
) -> Result<Vec<(Vec<Descendant<BasisId>>, Vec<Descendant<BasisId>>)>, GwError> {
    fn recurse(
        markings: &[Descendant<BasisId>],
        index: usize,
        left: &mut Vec<Descendant<BasisId>>,
        right: &mut Vec<Descendant<BasisId>>,
        out: &mut Vec<(Vec<Descendant<BasisId>>, Vec<Descendant<BasisId>>)>,
    ) {
        if index == markings.len() {
            out.push((left.clone(), right.clone()));
            return;
        }
        left.push(markings[index].clone());
        recurse(markings, index + 1, left, right, out);
        left.pop();
        right.push(markings[index].clone());
        recurse(markings, index + 1, left, right, out);
        right.pop();
    }

    let split_count = 1usize
        .checked_shl(u32::try_from(markings.len()).map_err(|_| {
            GwError::UnsupportedInvariant("too many markings for labelled partitions".to_string())
        })?)
        .ok_or_else(|| {
            GwError::UnsupportedInvariant("too many markings for labelled partitions".to_string())
        })?;
    if split_count > term_limit {
        return Err(GwError::ResourceLimit {
            operation: "labelled Virasoro marking splits".to_string(),
            requested: split_count,
            limit: term_limit,
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(split_count).map_err(|_| {
        GwError::UnsupportedInvariant(format!(
            "cannot allocate {split_count} labelled marking splits"
        ))
    })?;
    recurse(markings, 0, &mut Vec::new(), &mut Vec::new(), &mut out);
    Ok(out)
}

#[derive(Default)]
struct TermAccumulator {
    constants: BTreeMap<TermOrigin, Rational>,
    linear: BTreeMap<(CorrelatorKey<CurveClass, BasisId>, TermOrigin), Rational>,
    quadratic: BTreeMap<
        (
            CorrelatorKey<CurveClass, BasisId>,
            CorrelatorKey<CurveClass, BasisId>,
            TermOrigin,
        ),
        Rational,
    >,
}

impl TermAccumulator {
    fn add_constant(&mut self, coefficient: Rational, origin: TermOrigin) {
        if coefficient.is_zero() {
            return;
        }
        *self.constants.entry(origin).or_insert_with(Rational::zero) += coefficient;
    }

    fn add_linear(
        &mut self,
        coefficient: Rational,
        correlator: CorrelatorKey<CurveClass, BasisId>,
        origin: TermOrigin,
    ) {
        if coefficient.is_zero() {
            return;
        }
        *self
            .linear
            .entry((correlator, origin))
            .or_insert_with(Rational::zero) += coefficient;
    }

    fn add_quadratic(
        &mut self,
        coefficient: Rational,
        mut left: CorrelatorKey<CurveClass, BasisId>,
        mut right: CorrelatorKey<CurveClass, BasisId>,
        origin: TermOrigin,
    ) {
        if coefficient.is_zero() {
            return;
        }
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        *self
            .quadratic
            .entry((left, right, origin))
            .or_insert_with(Rational::zero) += coefficient;
    }

    fn finish(self) -> Vec<ConstraintTerm<CurveClass, BasisId, Rational>> {
        let constants = self
            .constants
            .into_iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(origin, coefficient)| ConstraintTerm::Constant {
                coefficient,
                origin,
            });
        let linear = self
            .linear
            .into_iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|((correlator, origin), coefficient)| {
                ConstraintTerm::Linear(LinearTerm::new(coefficient, correlator, origin))
            });
        let quadratic = self
            .quadratic
            .into_iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|((left, right, origin), coefficient)| {
                ConstraintTerm::Quadratic(QuadraticTerm::new(coefficient, left, right, origin))
            });
        constants.chain(linear).chain(quadratic).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::{
        BasisElement, CharacteristicNumbers, CurveClassSpace, CurveClassSplit,
        NegativeSplitTotalSpaceTheory, NondegeneratePairing, ProjectiveSpaceTheory,
    };

    struct CountOnlyTheory(ProjectiveSpaceTheory);

    struct OverflowGradingTheory {
        state_space: StateSpace,
        curve_space: CurveClassSpace,
    }

    impl OverflowGradingTheory {
        fn new() -> Self {
            let identity = StateSpaceMatrix::identity(2);
            Self {
                state_space: StateSpace {
                    basis: vec![
                        BasisElement {
                            id: BasisId(0),
                            label: "1".to_string(),
                            hodge_p_degree: 0,
                            complex_codimension: 0,
                            parity: Parity::Even,
                        },
                        BasisElement {
                            id: BasisId(1),
                            label: "overflowing class".to_string(),
                            hodge_p_degree: usize::MAX,
                            complex_codimension: usize::MAX,
                            parity: Parity::Even,
                        },
                    ],
                    unit: BasisId(0),
                    pairing: Some(NondegeneratePairing {
                        metric: identity.clone(),
                        inverse: identity,
                    }),
                    c1_action: Some(StateSpaceMatrix::zero(2)),
                },
                curve_space: CurveClassSpace {
                    coordinate_names: Vec::new(),
                    effective_grading: "zero".to_string(),
                },
            }
        }
    }

    impl GwTheory for OverflowGradingTheory {
        fn theory_id(&self) -> String {
            "overflow-grading test theory".to_string()
        }

        fn target_dimension(&self) -> usize {
            0
        }

        fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
            VirasoroOperatorKind::StandardCompactGetzler
        }

        fn theory_fingerprint(&self) -> String {
            "overflow-grading-test-v1".to_string()
        }

        fn state_space(&self) -> &StateSpace {
            &self.state_space
        }

        fn curve_class_space(&self) -> &CurveClassSpace {
            &self.curve_space
        }

        fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
            self.curve_space.validate(curve)?;
            Ok(0)
        }

        fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
            self.curve_space.validate(curve)?;
            Ok(CurveEffectivity::Effective)
        }

        fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers> {
            None
        }

        fn admissible_decompositions(
            &self,
            total: &CurveClass,
        ) -> Result<Vec<CurveClassSplit>, GwError> {
            self.curve_space.validate(total)?;
            Ok(vec![CurveClassSplit {
                left: CurveClass::zero(0),
                right: CurveClass::zero(0),
            }])
        }

        fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError> {
            self.curve_space.validate(total)?;
            Ok(1)
        }

        fn bounded_admissible_class_count(&self, _max_total: usize) -> Result<usize, GwError> {
            Ok(1)
        }

        fn bounded_admissible_classes(
            &self,
            _max_total: usize,
        ) -> Result<Vec<CurveClass>, GwError> {
            Ok(vec![CurveClass::zero(0)])
        }
    }

    impl GwTheory for CountOnlyTheory {
        fn theory_id(&self) -> String {
            "count-only test theory".to_string()
        }

        fn target_dimension(&self) -> usize {
            self.0.target_dimension()
        }

        fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
            VirasoroOperatorKind::StandardCompactGetzler
        }

        fn theory_fingerprint(&self) -> String {
            "count-only-test-v1".to_string()
        }

        fn state_space(&self) -> &StateSpace {
            self.0.state_space()
        }

        fn curve_class_space(&self) -> &CurveClassSpace {
            self.0.curve_class_space()
        }

        fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
            self.0.c1_pairing(curve)
        }

        fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
            self.0.effectivity(curve)
        }

        fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers> {
            self.0.characteristic_numbers()
        }

        fn admissible_decompositions(
            &self,
            _total: &CurveClass,
        ) -> Result<Vec<CurveClassSplit>, GwError> {
            panic!("term-budget rejection must precede decomposition materialization")
        }

        fn admissible_decomposition_count(&self, _total: &CurveClass) -> Result<usize, GwError> {
            Ok(1_000_000)
        }

        fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError> {
            self.0.bounded_admissible_class_count(max_total)
        }

        fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError> {
            self.0.bounded_admissible_classes(max_total)
        }
    }

    #[test]
    fn getzler_brackets_match_defining_polynomial() {
        let x = Rational::new(1, 2);
        assert_eq!(
            getzler_bracket(x.clone(), 1, 0).unwrap(),
            Rational::new(3, 4)
        );
        assert_eq!(getzler_bracket(x.clone(), 1, 1).unwrap(), Rational::from(2));
        assert_eq!(getzler_bracket(x, 1, 2).unwrap(), Rational::one());
    }

    #[test]
    fn quadratic_grading_precedes_metric_index_raising() {
        let p1 = ProjectiveSpaceTheory::new(1);
        let constraint = generate_constraint(
            &p1,
            2,
            1,
            p1.curve(0),
            TimeMonomial::from_descendants([
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(0)),
            ]),
        )
        .unwrap();
        let expected = CorrelatorKey::new(
            0,
            p1.curve(0),
            vec![
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(1)),
                Descendant::new(0, BasisId(1)),
            ],
        );
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Linear(term)
                if term.origin == TermOrigin::GenusReduction
                    && term.correlator == expected
                    && term.coefficient == Rational::one()
        )));
    }

    #[test]
    fn getzler_bracket_rejects_index_overflow() {
        assert!(matches!(
            getzler_bracket(Rational::zero(), usize::MAX, 0),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    #[test]
    fn generator_caps_operator_polynomial_work() {
        let point = ProjectiveSpaceTheory::new(0);
        let error = generate_constraint(
            &point,
            i32::try_from(MAX_STANDARD_VIRASORO_OPERATOR_INDEX + 1).unwrap(),
            0,
            point.curve(0),
            TimeMonomial::one(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested: 65,
                limit: 64,
                ..
            }
        ));
    }

    #[test]
    fn point_l0_genus_one_constant_has_anomaly_and_tau_one() {
        let point = ProjectiveSpaceTheory::new(0);
        let constraint =
            generate_constraint(&point, 0, 1, point.curve(0), TimeMonomial::one()).unwrap();
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Constant { coefficient, .. }
                if coefficient == &Rational::new(1, 16)
        )));
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Linear(term)
                if term.coefficient == Rational::new(-3, 2)
                    && term.correlator.insertions() == [Descendant::new(1, BasisId(0))]
        )));
    }

    #[test]
    fn string_constraint_has_explicit_unstable_correction() {
        let point = ProjectiveSpaceTheory::new(0);
        let time = TimeMonomial::from_descendants([
            Descendant::new(0, BasisId(0)),
            Descendant::new(0, BasisId(0)),
        ]);
        let constraint = generate_constraint(&point, -1, 0, point.curve(0), time).unwrap();
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Constant { coefficient, .. } if coefficient == &Rational::one()
        )));
    }

    #[test]
    fn local_theory_is_not_misidentified_as_compact() {
        let local = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
        let error =
            generate_constraint(&local, 0, 0, CurveClass::new(vec![0]), TimeMonomial::one())
                .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
        assert!(error.to_string().contains("QRR"));
    }

    #[test]
    fn final_c1_power_uses_primary_derivative_without_unsigned_underflow() {
        let p1 = ProjectiveSpaceTheory::new(1);
        let constraint = generate_constraint(&p1, 0, 0, p1.curve(0), TimeMonomial::one()).unwrap();
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Linear(term)
                if term.correlator.insertions() == [Descendant::new(0, BasisId(1))]
        )));
    }

    #[test]
    fn repeated_markings_are_aggregated_with_labelled_multiplicity() {
        let point = ProjectiveSpaceTheory::new(0);
        let time = TimeMonomial::from_descendants([
            Descendant::new(0, BasisId(0)),
            Descendant::new(0, BasisId(0)),
            Descendant::new(0, BasisId(0)),
            Descendant::new(0, BasisId(0)),
        ]);
        let constraint = generate_constraint(&point, 1, 0, point.curve(0), time).unwrap();
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Linear(term)
                if term.coefficient == Rational::from(3)
                    && term.correlator.insertions().len() == 4
                    && term.correlator.insertions()[3].psi_power == 1
        )));
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Quadratic(term) if term.origin == TermOrigin::DegreeSplitting
        )));
    }

    #[test]
    fn positive_genus_l1_contains_genus_reduction() {
        let point = ProjectiveSpaceTheory::new(0);
        let constraint = generate_constraint(
            &point,
            1,
            1,
            point.curve(0),
            TimeMonomial::from_descendants([Descendant::new(0, BasisId(0))]),
        )
        .unwrap();
        assert!(constraint.terms.iter().any(|term| matches!(
            term,
            ConstraintTerm::Linear(term) if term.origin == TermOrigin::GenusReduction
        )));
    }

    #[test]
    fn string_term_budget_rejects_multiplicity_overflow_before_expansion() {
        let point = ProjectiveSpaceTheory::new(0);
        let time =
            TimeMonomial::try_from_factors([(Descendant::new(0, BasisId(0)), usize::MAX)]).unwrap();
        let error =
            generate_constraint_with_term_limit(&point, -1, 0, point.curve(0), time, usize::MAX)
                .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
        assert!(error.to_string().contains("size overflow"));
    }

    #[test]
    fn large_projective_string_constraint_obeys_term_budget_before_validation() {
        let projective = ProjectiveSpaceTheory::new(100);
        let error = generate_constraint_with_term_limit(
            &projective,
            -1,
            0,
            projective.curve(0),
            TimeMonomial::one(),
            1,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested: 2,
                limit: 1,
                ..
            }
        ));
    }

    #[test]
    fn overflowing_public_grading_is_rejected_without_panicking() {
        let theory = OverflowGradingTheory::new();
        let error = generate_constraint(&theory, -1, 0, CurveClass::zero(0), TimeMonomial::one())
            .unwrap_err();

        assert!(matches!(error, GwError::ConventionMismatch(_)));
        assert!(error.to_string().contains("complex grading"));
    }

    #[test]
    fn marking_cap_precedes_labelled_time_allocation() {
        let point = ProjectiveSpaceTheory::new(0);
        let time =
            TimeMonomial::try_from_factors([(Descendant::new(0, BasisId(0)), usize::MAX - 2)])
                .unwrap();
        let error =
            generate_constraint_with_term_limit(&point, -1, 0, point.curve(0), time, usize::MAX)
                .unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested,
                limit: MAX_VIRASORO_MARKINGS,
                ..
            } if requested == usize::MAX - 2
        ));
    }

    #[test]
    fn nonnegative_operator_checks_marking_cap_before_quadratic_cloning() {
        let point = ProjectiveSpaceTheory::new(0);
        let time = TimeMonomial::try_from_factors([(
            Descendant::new(0, BasisId(0)),
            MAX_VIRASORO_MARKINGS + 1,
        )])
        .unwrap();
        let error =
            generate_constraint_with_term_limit(&point, 0, 0, point.curve(0), time, usize::MAX)
                .unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested,
                limit: MAX_VIRASORO_MARKINGS,
                ..
            } if requested == MAX_VIRASORO_MARKINGS + 1
        ));
    }

    #[test]
    fn term_budget_precedes_curve_split_materialization() {
        let theory = CountOnlyTheory(ProjectiveSpaceTheory::new(0));
        let error = generate_constraint_with_term_limit(
            &theory,
            1,
            0,
            theory.0.curve(0),
            TimeMonomial::one(),
            100,
        )
        .unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { limit: 100, .. }));
    }

    #[test]
    fn l0_does_not_materialize_unused_labelled_partitions() {
        let point = ProjectiveSpaceTheory::new(0);
        let time = TimeMonomial::try_from_factors([(Descendant::new(0, BasisId(0)), 20)]).unwrap();
        let constraint =
            generate_constraint_with_term_limit(&point, 0, 0, point.curve(0), time, 50).unwrap();
        assert!(!constraint.terms.is_empty());
    }
}
