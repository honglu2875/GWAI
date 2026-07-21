//! Equivariant inverse-Euler QRR specialization of the Virasoro `L_0` operator.
//!
//! The implementation deliberately starts with `L_0`: its conjugation can be
//! written in closed form, so no BCH truncation or unrecorded central term is
//! hidden in an invariant test.  Higher operators should be added by exact
//! differential-operator conjugation, not by reusing this special formula.

use crate::constraints::virasoro::{
    generate_constraint_with_term_limit, qrr_bernoulli_number, CohomologicalGrading,
    ConstraintTerm, CorrelatorKey, Descendant, DilatonShift, FormulaSource, LinearTerm,
    PotentialConvention, QrrConjugationFormula, QuadraticTerm, StateSpaceConvention,
    SymbolicVirasoroConstraint, TermOrigin, TheoryLabel, TimeMonomial, TimeNormalization,
    UnstableConvention, VirasoroConventions, DEFAULT_GENERATED_TERM_LIMIT, MAX_QRR_CHERN_INDEX,
    MAX_QRR_POSITIVE_Z_POWER, MAX_VIRASORO_MARKINGS,
};
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use crate::spaces::projective_space::ProjectiveSpaceTheory;
use std::collections::BTreeMap;

use super::super::{provider::default_fiber_parameter_names, NegativeSplitTotalSpaceTheory};

const QRR_ORIGIN: &str = "QRR-conjugated inverse-Euler L_0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InverseEulerL0Correction {
    pub z_power: i32,
    /// Coefficients of `1,H,...,H^n` in the multiplication class `K_r(H)`.
    pub h_coefficients: Vec<RatFun>,
}

/// A finite presentation of the closed-form, equivariant inverse-Euler `L_0`
/// conjugation for a negative split bundle over projective space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InverseEulerQrrL0Operator {
    base_n: usize,
    degrees: Vec<usize>,
    parameter_names: Vec<String>,
    max_positive_z_power: usize,
    twisted_metric: Vec<Vec<RatFun>>,
    inverse_twisted_metric: Vec<Vec<RatFun>>,
    corrections: Vec<InverseEulerL0Correction>,
}

impl InverseEulerQrrL0Operator {
    pub fn new(
        theory: &NegativeSplitTotalSpaceTheory,
        max_positive_z_power: usize,
    ) -> Result<Self, GwError> {
        if theory.base_dimension() > MAX_QRR_CHERN_INDEX {
            return Err(GwError::ResourceLimit {
                operation: "inverse-Euler QRR base Chern-character expansion".to_string(),
                requested: theory.base_dimension(),
                limit: MAX_QRR_CHERN_INDEX,
            });
        }
        if max_positive_z_power > MAX_QRR_POSITIVE_Z_POWER {
            return Err(GwError::ResourceLimit {
                operation: "inverse-Euler QRR positive z-power expansion".to_string(),
                requested: max_positive_z_power,
                limit: MAX_QRR_POSITIVE_Z_POWER,
            });
        }
        let base_n = theory.base_dimension();
        let degrees = theory.degrees().to_vec();
        let parameter_names = default_fiber_parameter_names(degrees.len());
        let parameters = parameter_names
            .iter()
            .cloned()
            .map(RatFun::variable)
            .collect::<Vec<_>>();
        let characteristic_class = inverse_euler_class(base_n, &degrees, &parameters);
        let twisted_metric = pairing_from_class(base_n, &characteristic_class);
        let inverse_twisted_metric = invert_ratfun_matrix(&twisted_metric)?;
        let corrections = l0_corrections(base_n, &degrees, &parameters, max_positive_z_power)?;
        Ok(Self {
            base_n,
            degrees,
            parameter_names,
            max_positive_z_power,
            twisted_metric,
            inverse_twisted_metric,
            corrections,
        })
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    pub fn max_positive_z_power(&self) -> usize {
        self.max_positive_z_power
    }

    pub fn corrections(&self) -> &[InverseEulerL0Correction] {
        &self.corrections
    }

    pub fn qrr_formula(&self) -> Result<QrrConjugationFormula, GwError> {
        let summands = self
            .degrees
            .iter()
            .map(|degree| format!("O(-{degree})"))
            .collect::<Vec<_>>()
            .join(" + ");
        let summands_tex = self
            .degrees
            .iter()
            .map(|degree| format!("\\mathcal{{O}}(-{degree})"))
            .collect::<Vec<_>>()
            .join("\\oplus");
        QrrConjugationFormula::new(
            format!("P^{}", self.base_n),
            format!("\\mathbb{{P}}^{{{}}}", self.base_n),
            summands,
            summands_tex,
            "equivariant inverse Euler",
            self.base_n,
            self.max_positive_z_power,
        )
    }

    pub fn render_text(&self) -> Result<String, GwError> {
        let mut out = self.qrr_formula()?.render_text();
        out.push_str("\nClosed L_0 specialization in native twisted coordinates:\n");
        out.push_str("  B_0 = L_0^(P^n) + sum_r K_r(H) z^r\n");
        out.push_str("  K = H*d_H log(sqrt(c(E))) - (z*d_z+H*d_H) log(Delta)\n");
        out.push_str("  Delta is the unquantized Coates--Givental QRR loop-space multiplier.\n");
        out.push_str("For u_i=mu_i-a_i H (inverse Euler):\n");
        out.push_str(
            "  The mu_i are frozen coefficient-field parameters (there is no mu_i*d/dmu_i term).\n",
        );
        out.push_str("  K_-1 = sum_i [(-a_i H) + mu_i log(mu_i/u_i)]\n");
        out.push_str("  K_0  = sum_i a_i H/(2 u_i)\n");
        out.push_str("  K_(2m-1) = sum_i B_(2m)/(2m) mu_i/u_i^(2m)\n");
        out.push_str("Modulo H^(n+1), the logarithm truncates and K_-1 starts in H^2.\n");
        out.push_str("After pullback to native twisted coordinates, q_tw(z)=t(z)-z; the modes below use q_tw, not q_CG.\n");
        out.push_str("The crate/Getzler normal-ordered modes are:\n");
        out.push_str("  hat_G(K_r z^r) = sum_(j>=0) (K_r q_j)^b d/dq_(j+r)^b\n");
        out.push_str("    + hbar/2 sum_(a+b=r-1) (-1)^(a+1) (K_r)^(alpha beta) d^2/(dq_a^alpha dq_b^beta),  r>0;\n");
        out.push_str("  hat_G(K_-1/z) = sum_(j>=1) (K_-1 q_j)^b d/dq_(j-1)^b + (K_-1 q_0,q_0)_tw/(2 hbar).\n");
        out.push_str("Raised/lowered tensors in these formulas use the twisted pairing.\n");
        out.push_str("For projective space the possible L_0 QRR cocycle is the trace of nilpotent multiplication and vanishes; the ordinary P^n anomaly remains.\n");
        Ok(out)
    }

    pub fn render_tex(&self) -> Result<String, GwError> {
        let mut out = self.qrr_formula()?.render_tex();
        out.push_str("\n\\subsubsection*{Closed inverse-Euler $L_0$ specialization}\n");
        out.push_str("\\[ B_0=L_0^{\\mathbb P^n}+\\sum_r K_r(H)z^r,\\qquad K=H\\partial_H\\log\\sqrt{\\mathbf c(E)}-(z\\partial_z+H\\partial_H)\\log\\Delta. \\]\n");
        out.push_str(
            "Here $\\Delta$ is the unquantized Coates--Givental QRR loop-space multiplier.\n",
        );
        out.push_str("For $u_i=\\mu_i-a_iH$,\n");
        out.push_str("\\[ K_{-1}=\\sum_i\\bigl[-a_iH+\\mu_i\\log(\\mu_i/u_i)\\bigr],\\qquad K_0=\\sum_i\\frac{a_iH}{2u_i}, \\]\n");
        out.push_str("\\[ K_{2m-1}=\\frac{B_{2m}}{2m}\\sum_i\\frac{\\mu_i}{u_i^{2m}}. \\]\n");
        out.push_str("Modulo $H^{n+1}$ the logarithm in $K_{-1}$ truncates and begins in $H^2$.\n");
        out.push_str("After pullback, the following modes use the native twisted coordinate $q_{\\rm tw}(z)=t(z)-z$, not $q_{\\rm CG}$.\n");
        out.push_str("In the crate/Getzler normal ordering, for $r>0$,\n");
        out.push_str("\\[ \\widehat{K_rz^r}^{G}=\\sum_{j\\ge0}(K_rq_j)^\\beta\\partial_{j+r,\\beta}+\\frac{\\hbar}{2}\\sum_{a+b=r-1}(-1)^{a+1}(K_r)^{\\alpha\\beta}\\partial_{a,\\alpha}\\partial_{b,\\beta}. \\]\n");
        out.push_str("All raised and lowered tensors use $(\\ ,\\ )_{\\rm tw}$. The possible extra $L_0$ cocycle is a nilpotent multiplication trace and vanishes for $\\mathbb P^n$.\n");
        Ok(out)
    }

    pub fn generate_constraint(
        &self,
        theory: &NegativeSplitTotalSpaceTheory,
        genus: usize,
        degree: CurveClass,
        time_coefficient: TimeMonomial<BasisId>,
    ) -> Result<SymbolicVirasoroConstraint, GwError> {
        self.generate_constraint_with_term_limit(
            theory,
            genus,
            degree,
            time_coefficient,
            DEFAULT_GENERATED_TERM_LIMIT,
        )
    }

    /// Generate one coefficient while bounding the complete unaggregated
    /// expansion before marking partitions are materialized.
    pub fn generate_constraint_with_term_limit(
        &self,
        theory: &NegativeSplitTotalSpaceTheory,
        genus: usize,
        degree: CurveClass,
        time_coefficient: TimeMonomial<BasisId>,
        term_limit: usize,
    ) -> Result<SymbolicVirasoroConstraint, GwError> {
        if self.base_n != theory.base_dimension() || self.degrees != theory.degrees() {
            return Err(GwError::ConventionMismatch(
                "inverse-Euler QRR operator and target theory differ".to_string(),
            ));
        }
        theory.curve_class_space().validate(&degree)?;
        let external = expand_time_coefficient(&time_coefficient)?;
        if external.len() > MAX_VIRASORO_MARKINGS {
            return Err(GwError::ResourceLimit {
                operation: "QRR Virasoro markings in one equation".to_string(),
                requested: external.len(),
                limit: MAX_VIRASORO_MARKINGS,
            });
        }
        if !cohft_stable(genus, external.len()) {
            return Err(GwError::UnsupportedInvariant(
                "the closed inverse-Euler L_0 coefficient generator currently requires 2g-2+n>0; use a stable high-genus coefficient"
                    .to_string(),
            ));
        }
        let required_positive_z_power =
            certified_l0_positive_z_bound(self.base_n, genus, &degree, &external)?;
        if self.max_positive_z_power < required_positive_z_power {
            return Err(GwError::ResourceLimit {
                operation: "complete QRR L_0 positive-z expansion for this coefficient".to_string(),
                requested: required_positive_z_power,
                limit: self.max_positive_z_power,
            });
        }

        let has_positive_mode = self.corrections.iter().any(|correction| {
            correction.z_power > 0 && correction.z_power as usize <= required_positive_z_power
        });
        let degree_split_count = if has_positive_mode {
            theory.admissible_decomposition_count(&degree)?
        } else {
            0
        };
        let qrr_term_estimate = estimate_qrr_unaggregated_terms(
            self.base_n,
            genus,
            external.len(),
            degree_split_count,
            &self.corrections,
            required_positive_z_power,
        )?;
        if qrr_term_estimate > term_limit {
            return Err(GwError::ResourceLimit {
                operation: "QRR Virasoro coefficient expansion".to_string(),
                requested: qrr_term_estimate,
                limit: term_limit,
            });
        }
        let base = ProjectiveSpaceTheory::try_new(self.base_n)?;
        let ordinary = generate_constraint_with_term_limit(
            &base,
            0,
            genus,
            degree.clone(),
            time_coefficient.clone(),
            term_limit - qrr_term_estimate,
        )?;
        let mut terms = SymbolicTermAccumulator::default();
        for term in ordinary.terms {
            // The R=(n+1)H z^-1 qq term is lowered with the twisted metric in
            // native twisted coordinates, not with the ordinary P^n pairing.
            if matches!(
                term,
                ConstraintTerm::Constant {
                    origin: TermOrigin::UnstableCorrection,
                    ..
                }
            ) {
                continue;
            }
            terms.add_rational_term(term);
        }
        add_base_c1_qq_term(
            self.base_n,
            &self.twisted_metric,
            genus,
            &degree,
            &external,
            &mut terms,
        );

        let splits = if has_positive_mode {
            theory.admissible_decompositions(&degree)?
        } else {
            Vec::new()
        };
        let marking_splits = if splits.is_empty() {
            Vec::new()
        } else {
            labelled_marking_splits(&external)?
        };
        for correction in &self.corrections {
            if correction.z_power > 0 && correction.z_power as usize > required_positive_z_power {
                continue;
            }
            add_vector_terms(correction, genus, &degree, &external, &mut terms);
            match correction.z_power {
                -1 => add_negative_qq_term(
                    correction,
                    self.base_n,
                    &self.twisted_metric,
                    genus,
                    &degree,
                    &external,
                    &mut terms,
                ),
                r if r > 0 => add_positive_dd_terms(
                    correction,
                    self.base_n,
                    &self.inverse_twisted_metric,
                    genus,
                    &degree,
                    &external,
                    &splits,
                    &marking_splits,
                    &mut terms,
                ),
                _ => {}
            }
        }

        let mut source = FormulaSource::new(
            "Coates--Givental QRR conjugation of the base-projective L_0 operator",
        );
        source.citation = Some("https://arxiv.org/pdf/math/0110142".to_string());
        source.locator = Some("Theorem 1, especially equation (7)".to_string());
        source.derivation = Some(
            "exact differential-operator L_0 conjugation after q=sqrt(c(E))(t-z); positive z powers are exhausted by the recorded base stable-map virtual-dimension bound"
                .to_string(),
        );
        source.notes.push(
            "the compact operator belongs to the base P^n; bundle terms occur only through QRR"
                .to_string(),
        );
        source.notes.push(
            "the base L_0 coordinate formula is Getzler's corrected EHX operator: https://arxiv.org/abs/math/9812026"
                .to_string(),
        );
        source
            .notes
            .push("the nonequivariant mu_i->0 limit is not taken termwise".to_string());

        Ok(SymbolicVirasoroConstraint {
            theory: TheoryLabel::new(theory.theory_id(), theory.theory_tex()),
            theory_fingerprint: theory.theory_fingerprint(),
            operator: crate::constraints::virasoro::VirasoroOperator::new(0),
            sector: crate::constraints::virasoro::ConstraintSector::new(genus, degree),
            time_coefficient,
            terms: terms.finish(),
            conventions: VirasoroConventions {
                potential: PotentialConvention::LogarithmicPartitionFunctionEquation,
                time_normalization: TimeNormalization::Exponential,
                dilaton_shift: DilatonShift::Explicit(
                    "q_CG=sqrt(c(E))(t-z) under the Fock-space isometry; after pullback the coefficient AST uses native q_tw=t-z"
                        .to_string(),
                ),
                grading: CohomologicalGrading::Complex,
                unstable: UnstableConvention::Excluded,
                state_space: StateSpaceConvention::EvenOnly,
                novikov_variables: theory.curve_class_space().coordinate_names.clone(),
                equivariant_parameters: self.parameter_names.clone(),
                notes: vec![
                    "inverse Euler is kept equivariant; degree-zero splitting terms are part of the same residual"
                        .to_string(),
                    "the fiber weights mu_i are frozen coefficient-field parameters; this is not an extended equivariant Euler operator"
                        .to_string(),
                    format!(
                        "positive odd z powers through {} are complete for this coefficient",
                        required_positive_z_power
                    ),
                ],
            },
            source,
        })
    }
}

/// Exact positive-`z` cutoff for a stable coefficient of the `L_0` equation.
///
/// A positive QRR Hamiltonian `z^r` inserts two descendants whose powers sum
/// to `r-1`.  The safe cap comes from the base stable-map virtual dimension
/// `V=(1-g)(n-3)+(n+1)d+N`: vector terms have `r<=V-D_M`, and the
/// genus-reduction/splitting terms have `r<=V-D_M+n`.  Only positive odd
/// powers are subsequently materialized.  This remains finite with symbolic
/// fiber weights because, after the base lambda-line limit, inverse Euler is
/// `sum_(j>=0) mu^(-chi(E)-j)c_j` with `codim(c_j)=j`; hence `D_M+j=V`
/// implies `D_M<=V`.  For a boundary term the component virtual dimensions
/// sum to `V+n-1`, while the new psi degrees sum to `r-1`, giving the second
/// bound above.  The argument is specific to inverse Euler and is not a
/// generic equivariant dimension shortcut.
pub fn certified_l0_positive_z_bound(
    base_n: usize,
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
) -> Result<usize, GwError> {
    if !cohft_stable(genus, external.len()) {
        return Err(GwError::UnsupportedInvariant(
            "a stable coefficient is required to certify the QRR descendant bound".to_string(),
        ));
    }
    if degree.rank() != 1 || degree.coordinates()[0] < 0 {
        return Err(GwError::ConventionMismatch(
            "negative-split QRR descendant bounds require one nonnegative degree".to_string(),
        ));
    }
    let degree = i128::from(degree.coordinates()[0]);
    let n = i128::try_from(base_n)
        .map_err(|_| GwError::UnsupportedInvariant("QRR base dimension overflow".to_string()))?;
    let g = i128::try_from(genus)
        .map_err(|_| GwError::UnsupportedInvariant("QRR genus overflow".to_string()))?;
    let markings = i128::try_from(external.len())
        .map_err(|_| GwError::UnsupportedInvariant("QRR marking count overflow".to_string()))?;
    let insertion_degree = external.iter().try_fold(0i128, |total, insertion| {
        let psi = i128::try_from(insertion.psi_power)
            .map_err(|_| GwError::UnsupportedInvariant("QRR psi degree overflow".to_string()))?;
        let basis = i128::try_from(insertion.class.0)
            .map_err(|_| GwError::UnsupportedInvariant("QRR basis degree overflow".to_string()))?;
        total
            .checked_add(psi)
            .and_then(|value| value.checked_add(basis))
            .ok_or_else(|| {
                GwError::UnsupportedInvariant("QRR insertion degree overflow".to_string())
            })
    })?;
    let virtual_dimension = (1i128 - g)
        .checked_mul(n - 3)
        .and_then(|value| value.checked_add((n + 1).checked_mul(degree)?))
        .and_then(|value| value.checked_add(markings))
        .ok_or_else(|| {
            GwError::UnsupportedInvariant("QRR virtual dimension overflow".to_string())
        })?;
    let cap = virtual_dimension - insertion_degree + n;
    if cap <= 0 {
        Ok(0)
    } else {
        let cap = usize::try_from(cap)
            .map_err(|_| GwError::UnsupportedInvariant("QRR z-power bound overflow".to_string()))?;
        Ok(if cap % 2 == 0 { cap - 1 } else { cap })
    }
}

pub fn generate_inverse_euler_qrr_l0_constraint(
    theory: &NegativeSplitTotalSpaceTheory,
    genus: usize,
    degree: CurveClass,
    time_coefficient: TimeMonomial<BasisId>,
) -> Result<SymbolicVirasoroConstraint, GwError> {
    inverse_euler_qrr_l0_operator_for_constraint(theory, genus, &degree, &time_coefficient)?
        .generate_constraint(theory, genus, degree, time_coefficient)
}

/// Construct the exact finite `L_0` operator needed by one coefficient.
///
/// This is the shared entry point for renderers and evaluators: the displayed
/// positive-mode cutoff is therefore the same certified cutoff used to build
/// the constraint AST.
pub fn inverse_euler_qrr_l0_operator_for_constraint(
    theory: &NegativeSplitTotalSpaceTheory,
    genus: usize,
    degree: &CurveClass,
    time_coefficient: &TimeMonomial<BasisId>,
) -> Result<InverseEulerQrrL0Operator, GwError> {
    let external = expand_time_coefficient(time_coefficient)?;
    let bound = certified_l0_positive_z_bound(theory.base_dimension(), genus, degree, &external)?;
    InverseEulerQrrL0Operator::new(theory, bound)
}

fn cohft_stable(genus: usize, markings: usize) -> bool {
    genus
        .checked_mul(2)
        .and_then(|value| value.checked_add(markings))
        .is_some_and(|value| value >= 3)
}

fn estimate_qrr_unaggregated_terms(
    base_n: usize,
    genus: usize,
    markings: usize,
    degree_split_count: usize,
    corrections: &[InverseEulerL0Correction],
    positive_z_bound: usize,
) -> Result<usize, GwError> {
    let checked_add = |left: usize, right: usize| {
        left.checked_add(right)
            .ok_or_else(|| GwError::UnsupportedInvariant("QRR term estimate overflow".to_string()))
    };
    let checked_mul = |left: usize, right: usize| {
        left.checked_mul(right)
            .ok_or_else(|| GwError::UnsupportedInvariant("QRR term estimate overflow".to_string()))
    };
    let state_size = base_n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("QRR state-space size overflow".to_string())
    })?;
    let active_corrections = corrections
        .iter()
        .filter(|correction| {
            correction.z_power <= 0 || correction.z_power as usize <= positive_z_bound
        })
        .count();
    let marking_slots = markings.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("QRR marking-slot count overflow".to_string())
    })?;
    let vector_bound = checked_mul(checked_mul(active_corrections, marking_slots)?, state_size)?;
    let marking_splits = if degree_split_count == 0 {
        0
    } else {
        1usize
            .checked_shl(u32::try_from(markings).map_err(|_| {
                GwError::UnsupportedInvariant("QRR marking-partition count overflow".to_string())
            })?)
            .ok_or_else(|| {
                GwError::UnsupportedInvariant("QRR marking-partition count overflow".to_string())
            })?
    };
    let genus_splits = genus.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("QRR genus-splitting count overflow".to_string())
    })?;
    let disconnected_bound = checked_mul(
        checked_mul(genus_splits, degree_split_count)?,
        marking_splits,
    )?;
    let per_basis_pair = checked_add(usize::from(genus > 0), disconnected_bound)?;
    let basis_pairs = checked_mul(state_size, state_size)?;
    let mut second_order_bound = 0usize;
    for correction in corrections.iter().filter(|correction| {
        correction.z_power > 0 && correction.z_power as usize <= positive_z_bound
    }) {
        let mode_bound = checked_mul(
            checked_mul(correction.z_power as usize, basis_pairs)?,
            per_basis_pair,
        )?;
        second_order_bound = checked_add(second_order_bound, mode_bound)?;
    }
    // One slot covers replacement of the base c1/z unstable contraction by
    // the same contraction lowered with the twisted metric.
    checked_add(checked_add(vector_bound, second_order_bound)?, 1)
}

fn l0_corrections(
    n: usize,
    degrees: &[usize],
    parameters: &[RatFun],
    max_positive_z_power: usize,
) -> Result<Vec<InverseEulerL0Correction>, GwError> {
    let mut negative = vec![RatFun::zero(); n + 1];
    let mut zero = vec![RatFun::zero(); n + 1];
    for (&degree, parameter) in degrees.iter().zip(parameters) {
        let mut degree_power = Rational::one();
        let mut parameter_power = RatFun::one();
        for h_power in 1..=n {
            degree_power = degree_power * Rational::from(degree);
            parameter_power = &parameter_power * parameter;
            add_assign(
                &mut zero[h_power],
                &(&RatFun::from_rational(degree_power.clone())
                    / &(&RatFun::from_rational(Rational::from(2)) * &parameter_power)),
            );
            if h_power >= 2 {
                let denominator = &RatFun::from_rational(Rational::from(h_power))
                    * &parameter.pow_usize(h_power - 1);
                add_assign(
                    &mut negative[h_power],
                    &(&RatFun::from_rational(degree_power.clone()) / &denominator),
                );
            }
        }
    }
    let mut out = vec![
        InverseEulerL0Correction {
            z_power: -1,
            h_coefficients: negative,
        },
        InverseEulerL0Correction {
            z_power: 0,
            h_coefficients: zero,
        },
    ];

    for z_power in (1..=max_positive_z_power).step_by(2) {
        let index = z_power + 1;
        let prefactor = qrr_bernoulli_number(index)? / Rational::from(index);
        let mut polynomial = vec![RatFun::zero(); n + 1];
        for (&degree, parameter) in degrees.iter().zip(parameters) {
            let mut coefficient =
                &RatFun::from_rational(prefactor.clone()) / &parameter.pow_usize(z_power);
            add_assign(&mut polynomial[0], &coefficient);
            for h_power in 0..n {
                coefficient = &coefficient
                    * &RatFun::from_rational(
                        Rational::from(index + h_power) * Rational::from(degree)
                            / Rational::from(h_power + 1),
                    );
                coefficient = &coefficient / parameter;
                add_assign(&mut polynomial[h_power + 1], &coefficient);
            }
        }
        out.push(InverseEulerL0Correction {
            z_power: i32::try_from(z_power).map_err(|_| {
                GwError::UnsupportedInvariant("QRR L_0 z power does not fit i32".to_string())
            })?,
            h_coefficients: polynomial,
        });
    }
    Ok(out)
}

fn inverse_euler_class(n: usize, degrees: &[usize], parameters: &[RatFun]) -> Vec<RatFun> {
    let mut result = vec![RatFun::zero(); n + 1];
    result[0] = RatFun::one();
    for (&degree, parameter) in degrees.iter().zip(parameters) {
        let mut factor = vec![RatFun::zero(); n + 1];
        let mut coefficient = &RatFun::one() / parameter;
        factor[0] = coefficient.clone();
        for entry in factor.iter_mut().skip(1) {
            coefficient =
                &coefficient * &(&RatFun::from_rational(Rational::from(degree)) / parameter);
            *entry = coefficient.clone();
        }
        result = multiply_polynomials(n, &result, &factor);
    }
    result
}

fn multiply_polynomials(n: usize, left: &[RatFun], right: &[RatFun]) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); n + 1];
    for (left_power, left_coefficient) in left.iter().enumerate() {
        for (right_power, right_coefficient) in right.iter().enumerate() {
            if left_power + right_power > n {
                break;
            }
            add_assign(
                &mut out[left_power + right_power],
                &(left_coefficient * right_coefficient),
            );
        }
    }
    out
}

fn pairing_from_class(n: usize, class: &[RatFun]) -> Vec<Vec<RatFun>> {
    let mut metric = vec![vec![RatFun::zero(); n + 1]; n + 1];
    for (left, row) in metric.iter_mut().enumerate() {
        for (right, entry) in row.iter_mut().enumerate() {
            if left + right <= n {
                *entry = class[n - left - right].clone();
            }
        }
    }
    metric
}

fn invert_ratfun_matrix(matrix: &[Vec<RatFun>]) -> Result<Vec<Vec<RatFun>>, GwError> {
    let size = matrix.len();
    let mut left = matrix.to_vec();
    let mut right = vec![vec![RatFun::zero(); size]; size];
    for (index, row) in right.iter_mut().enumerate() {
        row[index] = RatFun::one();
    }
    for column in 0..size {
        let pivot = (column..size)
            .find(|row| !left[*row][column].is_zero())
            .ok_or_else(|| {
                GwError::ConventionMismatch(
                    "twisted inverse-Euler pairing is degenerate".to_string(),
                )
            })?;
        left.swap(column, pivot);
        right.swap(column, pivot);
        let scale = left[column][column].clone();
        for entry in 0..size {
            left[column][entry] = &left[column][entry] / &scale;
            right[column][entry] = &right[column][entry] / &scale;
        }
        for row in 0..size {
            if row == column || left[row][column].is_zero() {
                continue;
            }
            let factor = left[row][column].clone();
            for entry in 0..size {
                left[row][entry] = &left[row][entry] - &(&factor * &left[column][entry]);
                right[row][entry] = &right[row][entry] - &(&factor * &right[column][entry]);
            }
        }
    }
    Ok(right)
}

fn multiplication_matrix(n: usize, polynomial: &[RatFun]) -> Vec<Vec<RatFun>> {
    let mut matrix = vec![vec![RatFun::zero(); n + 1]; n + 1];
    for input in 0..=n {
        for (power, coefficient) in polynomial.iter().enumerate().take(n - input + 1) {
            matrix[input + power][input] = coefficient.clone();
        }
    }
    matrix
}

fn raised_multiplication(
    n: usize,
    polynomial: &[RatFun],
    inverse_metric: &[Vec<RatFun>],
) -> Vec<Vec<RatFun>> {
    let multiplication = multiplication_matrix(n, polynomial);
    let mut raised = vec![vec![RatFun::zero(); n + 1]; n + 1];
    for left in 0..=n {
        for right in 0..=n {
            for middle in 0..=n {
                add_assign(
                    &mut raised[left][right],
                    &(&inverse_metric[left][middle] * &multiplication[right][middle]),
                );
            }
        }
    }
    raised
}

fn lower_multiplication(
    n: usize,
    polynomial: &[RatFun],
    metric: &[Vec<RatFun>],
) -> Vec<Vec<RatFun>> {
    let multiplication = multiplication_matrix(n, polynomial);
    let mut lowered = vec![vec![RatFun::zero(); n + 1]; n + 1];
    for left in 0..=n {
        for right in 0..=n {
            for middle in 0..=n {
                add_assign(
                    &mut lowered[left][right],
                    &(&multiplication[middle][left] * &metric[middle][right]),
                );
            }
        }
    }
    lowered
}

fn add_vector_terms(
    correction: &InverseEulerL0Correction,
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    terms: &mut SymbolicTermAccumulator,
) {
    let matrix = multiplication_matrix(
        correction.h_coefficients.len() - 1,
        &correction.h_coefficients,
    );
    for (marking, insertion) in external.iter().enumerate() {
        let derivative_psi = if correction.z_power == -1 {
            insertion.psi_power.checked_sub(1)
        } else {
            insertion.psi_power.checked_add(correction.z_power as usize)
        };
        let Some(derivative_psi) = derivative_psi else {
            continue;
        };
        for (output, row) in matrix.iter().enumerate() {
            let coefficient = row[insertion.class.0].clone();
            if coefficient.is_zero() {
                continue;
            }
            let mut replaced = external.to_vec();
            replaced[marking] = Descendant::new(derivative_psi, BasisId(output));
            terms.add_linear(
                coefficient,
                CorrelatorKey::new(genus, degree.clone(), replaced),
                TermOrigin::Other(QRR_ORIGIN.to_string()),
            );
        }
    }

    let derivative_psi = if correction.z_power == -1 {
        0
    } else {
        1 + correction.z_power as usize
    };
    for (output, row) in matrix.iter().enumerate() {
        let coefficient = -row[0].clone();
        if coefficient.is_zero() {
            continue;
        }
        let mut insertions = external.to_vec();
        insertions.push(Descendant::new(derivative_psi, BasisId(output)));
        terms.add_linear(
            coefficient,
            CorrelatorKey::new(genus, degree.clone(), insertions),
            TermOrigin::Other(QRR_ORIGIN.to_string()),
        );
    }
}

fn add_negative_qq_term(
    correction: &InverseEulerL0Correction,
    n: usize,
    metric: &[Vec<RatFun>],
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    terms: &mut SymbolicTermAccumulator,
) {
    if genus != 0
        || !degree.is_zero()
        || external.len() != 2
        || external.iter().any(|insertion| insertion.psi_power != 0)
    {
        return;
    }
    let lowered = lower_multiplication(n, &correction.h_coefficients, metric);
    terms.add_constant(
        lowered[external[0].class.0][external[1].class.0].clone(),
        TermOrigin::Other(QRR_ORIGIN.to_string()),
    );
}

fn add_base_c1_qq_term(
    n: usize,
    metric: &[Vec<RatFun>],
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    terms: &mut SymbolicTermAccumulator,
) {
    if genus != 0
        || !degree.is_zero()
        || external.len() != 2
        || external.iter().any(|insertion| insertion.psi_power != 0)
    {
        return;
    }
    let mut c1 = vec![RatFun::zero(); n + 1];
    if n > 0 {
        c1[1] = RatFun::from_rational(Rational::from(n + 1));
    }
    let lowered = lower_multiplication(n, &c1, metric);
    terms.add_constant(
        lowered[external[0].class.0][external[1].class.0].clone(),
        TermOrigin::UnstableCorrection,
    );
}

#[allow(clippy::too_many_arguments)]
fn add_positive_dd_terms(
    correction: &InverseEulerL0Correction,
    n: usize,
    inverse_metric: &[Vec<RatFun>],
    genus: usize,
    degree: &CurveClass,
    external: &[Descendant<BasisId>],
    degree_splits: &[crate::core::theory::CurveClassSplit],
    marking_splits: &[(Vec<Descendant<BasisId>>, Vec<Descendant<BasisId>>)],
    terms: &mut SymbolicTermAccumulator,
) {
    let r = correction.z_power as usize;
    let raised = raised_multiplication(n, &correction.h_coefficients, inverse_metric);
    for left_psi in 0..r {
        let right_psi = r - 1 - left_psi;
        let sign = if left_psi % 2 == 0 {
            -Rational::new(1, 2)
        } else {
            Rational::new(1, 2)
        };
        for left_basis in 0..=n {
            for right_basis in 0..=n {
                let coefficient =
                    &RatFun::from_rational(sign.clone()) * &raised[left_basis][right_basis];
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
                    for curve_split in degree_splits {
                        for (left_markings, right_markings) in marking_splits {
                            let mut left_insertions = left_markings.clone();
                            left_insertions.push(Descendant::new(left_psi, BasisId(left_basis)));
                            let mut right_insertions = right_markings.clone();
                            right_insertions.push(Descendant::new(right_psi, BasisId(right_basis)));
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

fn expand_time_coefficient(
    time: &TimeMonomial<BasisId>,
) -> Result<Vec<Descendant<BasisId>>, GwError> {
    let capacity = time.total_degree().ok_or_else(|| {
        GwError::AlgebraFailure("QRR time-coefficient marking count overflow".to_string())
    })?;
    if capacity > MAX_VIRASORO_MARKINGS {
        return Err(GwError::ResourceLimit {
            operation: "QRR Virasoro markings in one equation".to_string(),
            requested: capacity,
            limit: MAX_VIRASORO_MARKINGS,
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(capacity).map_err(|_| {
        GwError::UnsupportedInvariant("cannot allocate QRR labelled markings".to_string())
    })?;
    for (descendant, multiplicity) in time.factors() {
        out.extend((0..multiplicity).map(|_| descendant.clone()));
    }
    Ok(out)
}

fn labelled_marking_splits(
    markings: &[Descendant<BasisId>],
) -> Result<Vec<(Vec<Descendant<BasisId>>, Vec<Descendant<BasisId>>)>, GwError> {
    let count = 1usize
        .checked_shl(u32::try_from(markings.len()).map_err(|_| {
            GwError::UnsupportedInvariant("too many QRR labelled markings".to_string())
        })?)
        .ok_or_else(|| {
            GwError::UnsupportedInvariant("too many QRR labelled markings".to_string())
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| {
        GwError::UnsupportedInvariant("cannot allocate QRR marking partitions".to_string())
    })?;
    for mask in 0..count {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (index, marking) in markings.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                left.push(marking.clone());
            } else {
                right.push(marking.clone());
            }
        }
        out.push((left, right));
    }
    Ok(out)
}

fn add_assign(target: &mut RatFun, value: &RatFun) {
    *target = &*target + value;
}

#[derive(Default)]
struct SymbolicTermAccumulator {
    constants: BTreeMap<TermOrigin, RatFun>,
    linear: BTreeMap<(CorrelatorKey<CurveClass, BasisId>, TermOrigin), RatFun>,
    quadratic: BTreeMap<
        (
            CorrelatorKey<CurveClass, BasisId>,
            CorrelatorKey<CurveClass, BasisId>,
            TermOrigin,
        ),
        RatFun,
    >,
}

impl SymbolicTermAccumulator {
    fn add_rational_term(&mut self, term: ConstraintTerm<CurveClass, BasisId, Rational>) {
        match term {
            ConstraintTerm::Constant {
                coefficient,
                origin,
            } => self.add_constant(RatFun::from_rational(coefficient), origin),
            ConstraintTerm::Linear(term) => self.add_linear(
                RatFun::from_rational(term.coefficient),
                term.correlator,
                term.origin,
            ),
            ConstraintTerm::Quadratic(term) => self.add_quadratic(
                RatFun::from_rational(term.coefficient),
                term.left,
                term.right,
                term.origin,
            ),
        }
    }

    fn add_constant(&mut self, coefficient: RatFun, origin: TermOrigin) {
        if coefficient.is_zero() {
            return;
        }
        let entry = self.constants.entry(origin).or_insert_with(RatFun::zero);
        add_assign(entry, &coefficient);
    }

    fn add_linear(
        &mut self,
        coefficient: RatFun,
        correlator: CorrelatorKey<CurveClass, BasisId>,
        origin: TermOrigin,
    ) {
        if coefficient.is_zero() {
            return;
        }
        let entry = self
            .linear
            .entry((correlator, origin))
            .or_insert_with(RatFun::zero);
        add_assign(entry, &coefficient);
    }

    fn add_quadratic(
        &mut self,
        coefficient: RatFun,
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
        let entry = self
            .quadratic
            .entry((left, right, origin))
            .or_insert_with(RatFun::zero);
        add_assign(entry, &coefficient);
    }

    fn finish(self) -> Vec<ConstraintTerm<CurveClass, BasisId, RatFun>> {
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
    use crate::constraints::virasoro::{
        evaluate_constraint, CanonicalCorrelatorEvaluator, ConstraintTerm, ResidualStatus,
        VirasoroConstraint,
    };
    use crate::spaces::negative_split_projective::{
        NegativeSplitFixedFiberQrrEvaluator, NegativeSplitQrrEvaluator,
    };

    fn qrr_correlator_value(
        evaluator: &dyn CanonicalCorrelatorEvaluator,
        key: &CorrelatorKey<CurveClass, BasisId>,
    ) -> RatFun {
        if key.degree.is_zero()
            && !crate::core::moduli::pointed_curve_is_stable(key.genus, key.insertions().len())
        {
            RatFun::zero()
        } else {
            evaluator.evaluate_backend(key).unwrap()
        }
    }

    fn residual_change_after_unit_perturbation<C>(
        evaluator: &dyn CanonicalCorrelatorEvaluator,
        constraint: &VirasoroConstraint<CurveClass, BasisId, C>,
        target: &CorrelatorKey<CurveClass, BasisId>,
    ) -> RatFun
    where
        C: Clone + Into<RatFun>,
    {
        let mut delta = RatFun::zero();
        for term in &constraint.terms {
            let change = match term {
                ConstraintTerm::Constant { .. } => RatFun::zero(),
                ConstraintTerm::Linear(term) if &term.correlator == target => {
                    term.coefficient.clone().into()
                }
                ConstraintTerm::Linear(_) => RatFun::zero(),
                ConstraintTerm::Quadratic(term)
                    if &term.left == target || &term.right == target =>
                {
                    let old_left = qrr_correlator_value(evaluator, &term.left);
                    let old_right = qrr_correlator_value(evaluator, &term.right);
                    let new_left = if &term.left == target {
                        &old_left + &RatFun::one()
                    } else {
                        old_left.clone()
                    };
                    let new_right = if &term.right == target {
                        &old_right + &RatFun::one()
                    } else {
                        old_right.clone()
                    };
                    let product_change = &(&new_left * &new_right) - &(&old_left * &old_right);
                    &term.coefficient.clone().into() * &product_change
                }
                ConstraintTerm::Quadratic(_) => RatFun::zero(),
            };
            delta = &delta + &change;
        }
        delta
    }

    #[test]
    fn inverse_euler_l0_closed_coefficients_start_as_expected() {
        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
        let operator = InverseEulerQrrL0Operator::new(&theory, 3).unwrap();
        let mu = RatFun::variable("mu_0");
        let negative = &operator.corrections()[0].h_coefficients;
        assert!(negative[0].is_zero());
        assert!(negative[1].is_zero());
        assert!(negative[2].equivalent(&(&RatFun::from_rational(Rational::new(9, 2)) / &mu)));
        let positive_z = &operator.corrections()[2].h_coefficients;
        assert!(positive_z[0].equivalent(&(&RatFun::from_rational(Rational::new(1, 12)) / &mu)));
    }

    #[test]
    fn genus_two_unmarked_l0_has_nontrivial_qrr_terms_and_source() {
        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
        let constraint = generate_inverse_euler_qrr_l0_constraint(
            &theory,
            2,
            theory.try_curve(1).unwrap(),
            TimeMonomial::one(),
        )
        .unwrap();
        assert!(constraint.terms.iter().any(|term| {
            matches!(term, ConstraintTerm::Linear(term) if term.origin == TermOrigin::GenusReduction)
        }));
        assert!(constraint
            .conventions
            .equivariant_parameters
            .contains(&"mu_0".to_string()));
        assert!(constraint
            .source
            .citation
            .as_deref()
            .unwrap()
            .contains("0110142"));
        assert!(constraint
            .render_symbolic_text_for_theory(&theory)
            .unwrap()
            .contains("QRR conjugation"));
    }

    #[test]
    fn insufficient_positive_z_bound_fails_closed() {
        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
        let operator = InverseEulerQrrL0Operator::new(&theory, 1).unwrap();
        let error = operator
            .generate_constraint(
                &theory,
                2,
                theory.try_curve(1).unwrap(),
                TimeMonomial::one(),
            )
            .unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { .. }), "{error:?}");
    }

    #[test]
    fn certified_cutoff_uses_base_virtual_dimension_and_external_degree() {
        let degree = CurveClass::new(vec![1]);
        assert_eq!(
            certified_l0_positive_z_bound(2, 2, &degree, &[Descendant::new(0, BasisId(1))])
                .unwrap(),
            5
        );
        assert_eq!(
            certified_l0_positive_z_bound(
                2,
                2,
                &degree,
                &[
                    Descendant::new(0, BasisId(2)),
                    Descendant::new(0, BasisId(2)),
                    Descendant::new(0, BasisId(2)),
                    Descendant::new(0, BasisId(2)),
                ],
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn qrr_resource_caps_apply_before_dense_or_mode_construction() {
        let large_base =
            NegativeSplitTotalSpaceTheory::new(MAX_QRR_CHERN_INDEX + 1, vec![1]).unwrap();
        assert!(matches!(
            InverseEulerQrrL0Operator::new(&large_base, 1),
            Err(GwError::ResourceLimit { .. })
        ));

        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![1]).unwrap();
        let error = inverse_euler_qrr_l0_operator_for_constraint(
            &theory,
            2,
            &CurveClass::new(vec![100]),
            &TimeMonomial::one(),
        )
        .unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { .. }), "{error:?}");
    }

    #[test]
    fn selected_holdout_has_live_nonlinear_qrr_dependencies() {
        fn structural_zero(
            evaluator: &dyn CanonicalCorrelatorEvaluator,
            key: &CorrelatorKey<CurveClass, BasisId>,
        ) -> bool {
            (key.degree.is_zero()
                && !crate::core::moduli::pointed_curve_is_stable(key.genus, key.insertions().len()))
                || evaluator.certified_zero(key).unwrap()
        }

        fn live_nonlinear_counts(
            evaluator: &dyn CanonicalCorrelatorEvaluator,
            constraint: &SymbolicVirasoroConstraint,
        ) -> (usize, usize) {
            let genus_reduction = constraint
                .terms
                .iter()
                .filter(|term| {
                    matches!(
                        term,
                        ConstraintTerm::Linear(term)
                            if term.origin == TermOrigin::GenusReduction
                                && !structural_zero(evaluator, &term.correlator)
                    )
                })
                .count();
            let degree_splitting = constraint
                .terms
                .iter()
                .filter(|term| {
                    matches!(
                        term,
                        ConstraintTerm::Quadratic(term)
                            if term.origin == TermOrigin::DegreeSplitting
                                && !structural_zero(evaluator, &term.left)
                                && !structural_zero(evaluator, &term.right)
                    )
                })
                .count();
            (genus_reduction, degree_splitting)
        }

        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![2]).unwrap();
        let evaluator = NegativeSplitQrrEvaluator::new(2, vec![2]).unwrap();
        let generate = |psi_power| {
            generate_inverse_euler_qrr_l0_constraint(
                &theory,
                2,
                theory.try_curve(1).unwrap(),
                TimeMonomial::from_descendants([Descendant::new(psi_power, BasisId(2))]),
            )
            .unwrap()
        };

        // Merely finding nonlinear nodes in the AST is insufficient:
        // tau_3(H^2) has such nodes, but every one is dimensionally zero.
        // Positive-degree pointed-unstable dependencies are deliberately not
        // discarded here: the provider reconstructs them by divisor recursion.
        assert_eq!(live_nonlinear_counts(&evaluator, &generate(3)), (0, 0));
        let unmarked = generate_inverse_euler_qrr_l0_constraint(
            &theory,
            2,
            theory.try_curve(1).unwrap(),
            TimeMonomial::one(),
        )
        .unwrap();
        let live = live_nonlinear_counts(&evaluator, &unmarked);
        assert!(live.0 > 0, "the holdout must exercise genus reduction");
        assert!(live.1 > 0, "the holdout must exercise degree splitting");

        let varies_with_mu = |term: &ConstraintTerm<CurveClass, BasisId, RatFun>| {
            let coefficient = match term {
                ConstraintTerm::Constant { coefficient, .. } => coefficient,
                ConstraintTerm::Linear(term) => &term.coefficient,
                ConstraintTerm::Quadratic(term) => &term.coefficient,
            };
            let at = |value| {
                coefficient
                    .evaluate_variables(&BTreeMap::from([(
                        "mu_0".to_string(),
                        Rational::from(value),
                    )]))
                    .unwrap()
            };
            at(7) != at(11)
        };
        assert!(unmarked.terms.iter().any(|term| {
            matches!(term, ConstraintTerm::Linear(linear)
                if linear.origin == TermOrigin::GenusReduction
                    && !structural_zero(&evaluator, &linear.correlator))
                && varies_with_mu(term)
        }));
        assert!(unmarked.terms.iter().any(|term| {
            matches!(term, ConstraintTerm::Quadratic(quadratic)
                if quadratic.origin == TermOrigin::DegreeSplitting
                    && !structural_zero(&evaluator, &quadratic.left)
                    && !structural_zero(&evaluator, &quadratic.right))
                && varies_with_mu(term)
        }));
    }

    #[test]
    fn cap_zero_does_not_materialize_unused_marking_partitions() {
        let theory = NegativeSplitTotalSpaceTheory::new(1, vec![1]).unwrap();
        let time = TimeMonomial::try_from_factors([(
            Descendant::new(0, BasisId(1)),
            MAX_VIRASORO_MARKINGS,
        )])
        .unwrap();
        let degree = theory.try_curve(0).unwrap();
        let operator =
            inverse_euler_qrr_l0_operator_for_constraint(&theory, 0, &degree, &time).unwrap();
        assert_eq!(operator.max_positive_z_power(), 0);
        operator
            .generate_constraint(&theory, 0, degree, time)
            .unwrap();
    }

    #[test]
    #[ignore = "exactly specialized genus-two inverse-Euler QRR acceptance row; runs live nonlinear degree-zero and positive-degree twisted graph sectors"]
    fn genus_two_o_minus_two_p2_inverse_euler_l0_constraint_vanishes() {
        let theory = NegativeSplitTotalSpaceTheory::new(2, vec![2]).unwrap();
        let constraint = generate_inverse_euler_qrr_l0_constraint(
            &theory,
            2,
            theory.try_curve(1).unwrap(),
            // The unmarked coefficient keeps graph dimension at most four but
            // exercises the z^1, z^3, and z^5 QRR Hamiltonians.  Unlike the
            // superficially small tau_3(H^2) equation, it has genuinely live,
            // fiber-weight-dependent genus-reduction and splitting terms.
            TimeMonomial::one(),
        )
        .unwrap();
        assert!(constraint.conventions.notes.iter().any(
            |note| note == "positive odd z powers through 5 are complete for this coefficient"
        ));
        assert!(constraint.terms.iter().any(|term| {
            matches!(term, ConstraintTerm::Linear(term) if term.origin == TermOrigin::GenusReduction)
        }));
        assert!(constraint.terms.iter().any(|term| {
            matches!(term, ConstraintTerm::Quadratic(term) if term.origin == TermOrigin::DegreeSplitting)
        }));
        eprintln!(
            "QRR holdout: generated {} aggregated terms; evaluating exact dependencies",
            constraint.terms.len()
        );

        // Keep the reviewable operator symbolic, then specialize both its
        // coefficients and the provider to the same regular exact point.
        // This avoids treating a second hand-written numerical equation as
        // an oracle while making the hard graph arithmetic tractable.
        let evaluator =
            NegativeSplitFixedFiberQrrEvaluator::new(2, vec![2], vec![Rational::from(7)]).unwrap();
        let specialized = evaluator.specialize_constraint(&constraint).unwrap();
        assert_eq!(
            specialized.assignments(),
            &BTreeMap::from([("mu_0".to_string(), Rational::from(7))])
        );
        let report = evaluate_constraint(&evaluator, specialized.constraint());
        eprintln!(
            "QRR holdout: status={:?}, backend={}, structural-zero={}, missing={}",
            report.status(),
            report.backend_correlator_count(),
            report.structural_zero_correlator_count(),
            report.missing_correlator_count()
        );
        assert_eq!(report.status(), ResidualStatus::VerifiedZero, "{report:#?}");
        assert!(report
            .backend_correlators()
            .iter()
            .any(|key| key.degree.is_zero()));
        assert!(report
            .backend_correlators()
            .iter()
            .any(|key| key.degree.coordinates() == [1]));
        let coefficient_varies_with_mu = |term: &ConstraintTerm<CurveClass, BasisId, RatFun>| {
            let coefficient = match term {
                ConstraintTerm::Constant { coefficient, .. } => coefficient,
                ConstraintTerm::Linear(term) => &term.coefficient,
                ConstraintTerm::Quadratic(term) => &term.coefficient,
            };
            let evaluate = |value| {
                coefficient
                    .evaluate_variables(&BTreeMap::from([(
                        "mu_0".to_string(),
                        Rational::from(value),
                    )]))
                    .unwrap()
            };
            evaluate(7) != evaluate(11)
        };
        for origin in [TermOrigin::GenusReduction, TermOrigin::DegreeSplitting] {
            assert!(
                report.evaluated_terms().iter().any(|evaluated| {
                    let term = &constraint.terms[evaluated.term_index];
                    !evaluated.exact_contribution.is_zero()
                        && term.origin() == &origin
                        && coefficient_varies_with_mu(term)
                }),
                "the accepted residual must contain a nonzero mu-dependent {origin:?} contribution"
            );
        }

        let occurs_in_mu_dependent_term = |key: &CorrelatorKey<CurveClass, BasisId>| {
            constraint.terms.iter().any(|term| {
                coefficient_varies_with_mu(term)
                    && match term {
                        ConstraintTerm::Constant { .. } => false,
                        ConstraintTerm::Linear(term) => &term.correlator == key,
                        ConstraintTerm::Quadratic(term) => &term.left == key || &term.right == key,
                    }
            })
        };

        let sensitive = report
            .backend_correlators()
            .iter()
            .filter(|key| key.genus >= 2 && key.degree.coordinates() == [1])
            .filter(|key| !qrr_correlator_value(&evaluator, key).is_zero())
            .filter(|key| occurs_in_mu_dependent_term(key))
            .find(|key| {
                !residual_change_after_unit_perturbation(&evaluator, specialized.constraint(), key)
                    .is_zero()
            });
        assert!(
            sensitive.is_some(),
            "the QRR residual must detect corruption of a nonzero positive-degree genus-two dependency"
        );
        eprintln!("QRR holdout: nonzero genus-two corruption changes the exact residual");
    }
}
