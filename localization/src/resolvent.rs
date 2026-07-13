//! Fixed-degree resolvent generating functions.
//!
//! For fixed target, genus, degree, and labelled markings this module computes
//!
//! ```text
//! sum_{a_i,k_i} < prod_i tau_{k_i}(H^{a_i}) >_{g,d}
//!     prod_i t_i^{a_i}/a_i! * z_i^{-k_i-1}.
//! ```
//!
//! The virtual dimension fixes `sum_i (a_i+k_i)`, so the sum is finite.  This
//! module owns the finite index set and Laurent-polynomial output type.  It
//! also provides a callback-based coefficient-wise evaluator, while packed
//! S/R graph evaluators can reuse the same index and polynomial types.

use std::collections::BTreeMap;
use std::fmt;

use crate::algebra::{Coeff, RatFun, Rational};
use crate::error::GwError;
use crate::factored::FactoredRatFun;
use crate::geometry::CohomologyClass;
use crate::{tau, Insertion, InvariantResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventRequest {
    pub target_n: usize,
    pub genus: usize,
    pub degree: usize,
    pub markings: usize,
    /// The exact dimension slice to enumerate. Provider-backed entry points
    /// validate this value before every early return. The generic callback
    /// evaluator cannot infer a theory and therefore treats it as trusted.
    pub virtual_dimension: isize,
}

impl ResolventRequest {
    pub fn for_projective_space(
        target_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Self {
        let virtual_dimension = (1 - genus as isize) * (target_n as isize - 3)
            + (target_n + 1) as isize * degree as isize
            + markings as isize;
        Self {
            target_n,
            genus,
            degree,
            markings,
            virtual_dimension,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventResult<C = RatFun> {
    pub value: ResolventPolynomial<C>,
    pub candidate_terms: usize,
    pub nonzero_terms: usize,
    pub engine: &'static str,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventPolynomial<C = RatFun> {
    terms: BTreeMap<ResolventMonomial, C>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventIndex {
    h_powers: Vec<usize>,
    descendant_powers: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResolventMonomial {
    t_powers: Vec<usize>,
    z_denominator_powers: Vec<usize>,
}

impl ResolventIndex {
    pub fn empty() -> Self {
        Self {
            h_powers: Vec::new(),
            descendant_powers: Vec::new(),
        }
    }

    pub fn new(h_powers: Vec<usize>, descendant_powers: Vec<usize>) -> Self {
        assert_eq!(
            h_powers.len(),
            descendant_powers.len(),
            "resolvent index must have one H-power and one descendant power per marking"
        );
        Self {
            h_powers,
            descendant_powers,
        }
    }

    pub fn h_powers(&self) -> &[usize] {
        &self.h_powers
    }

    pub fn descendant_powers(&self) -> &[usize] {
        &self.descendant_powers
    }

    pub fn to_insertions(&self, target_n: usize) -> Vec<Insertion> {
        self.h_powers
            .iter()
            .zip(self.descendant_powers.iter())
            .map(|(&h_power, &descendant_power)| {
                tau(
                    descendant_power,
                    CohomologyClass::h_power(target_n, h_power),
                )
            })
            .collect()
    }
}

pub trait ResolventCoefficient: Coeff + fmt::Display {
    fn as_resolvent_rational(&self) -> Option<Rational>;
    fn to_resolvent_ratfun(&self) -> RatFun;
}

impl ResolventCoefficient for RatFun {
    fn as_resolvent_rational(&self) -> Option<Rational> {
        self.as_rational()
    }

    fn to_resolvent_ratfun(&self) -> RatFun {
        self.clone()
    }
}

impl ResolventCoefficient for FactoredRatFun {
    fn as_resolvent_rational(&self) -> Option<Rational> {
        self.as_structural_rational()
    }

    fn to_resolvent_ratfun(&self) -> RatFun {
        self.to_ratfun()
    }
}

impl<C: Coeff> ResolventPolynomial<C> {
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    fn add_term(&mut self, monomial: ResolventMonomial, coefficient: C) {
        if coefficient.is_structurally_zero() {
            return;
        }
        let next = self
            .terms
            .remove(&monomial)
            .map(|current| current.add(&coefficient))
            .unwrap_or(coefficient);
        if !next.is_structurally_zero() {
            self.terms.insert(monomial, next);
        }
    }

    pub fn add_coefficient(
        &mut self,
        h_powers: &[usize],
        descendant_powers: &[usize],
        coefficient: C,
    ) {
        assert_eq!(
            h_powers.len(),
            descendant_powers.len(),
            "resolvent coefficient must have one H-power and one descendant power per marking"
        );
        let (monomial, scalar) = resolvent_monomial(h_powers, descendant_powers);
        self.add_term(monomial, coefficient.mul(&C::from_rational(scalar)));
    }

    pub fn add_index_coefficient(&mut self, index: &ResolventIndex, coefficient: C) {
        self.add_coefficient(&index.h_powers, &index.descendant_powers, coefficient);
    }
}

impl<C: ResolventCoefficient> ResolventPolynomial<C> {
    pub fn to_ratfun_polynomial(&self) -> ResolventPolynomial {
        let mut out = ResolventPolynomial::zero();
        for (monomial, coefficient) in &self.terms {
            out.add_term(monomial.clone(), coefficient.to_resolvent_ratfun());
        }
        out
    }

    pub fn coefficient_text_contains(&self, needle: &str) -> bool {
        self.terms
            .values()
            .any(|coefficient| coefficient.to_string().contains(needle))
    }
}

impl ResolventPolynomial<FactoredRatFun> {
    pub fn evaluate_variables(
        &self,
        values: &BTreeMap<String, Rational>,
    ) -> Result<ResolventPolynomial, GwError> {
        let mut out = ResolventPolynomial::zero();
        for (monomial, coefficient) in &self.terms {
            out.add_term(
                monomial.clone(),
                RatFun::from_rational(coefficient.evaluate_variables(values)?),
            );
        }
        Ok(out)
    }
}

impl ResolventPolynomial<RatFun> {
    /// Whether two Laurent polynomials have the same rational-function
    /// coefficient at every labelled resolvent monomial.
    ///
    /// Structural [`PartialEq`] remains available for callers that need to
    /// compare the exact stored numerator/denominator representations.
    pub fn equivalent(&self, rhs: &Self) -> bool {
        self.terms.iter().all(|(monomial, coefficient)| {
            rhs.terms.get(monomial).map_or_else(
                || coefficient.is_zero(),
                |other| coefficient.equivalent(other),
            )
        }) && rhs.terms.iter().all(|(monomial, coefficient)| {
            self.terms.contains_key(monomial) || coefficient.is_zero()
        })
    }
}

impl<C: ResolventCoefficient> fmt::Display for ResolventPolynomial<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (index, (monomial, coefficient)) in self.terms.iter().enumerate() {
            let (negative, text) = format_signed_resolvent_term(coefficient, monomial);
            match (index, negative) {
                (0, true) => write!(f, "-{text}")?,
                (0, false) => write!(f, "{text}")?,
                (_, true) => write!(f, " - {text}")?,
                (_, false) => write!(f, " + {text}")?,
            }
        }
        Ok(())
    }
}

pub fn compute_resolvent_generating_function<F>(
    req: &ResolventRequest,
    mut coefficient: F,
) -> Result<ResolventResult, GwError>
where
    F: FnMut(&[Insertion]) -> Result<InvariantResult, GwError>,
{
    if req.virtual_dimension < 0 {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms: 0,
            nonzero_terms: 0,
            engine: "resolvent-empty-dimension",
            notes: vec![format!(
                "virtual dimension {} is negative, so the resolvent generating function is zero",
                req.virtual_dimension
            )],
        });
    }

    let mut state = Accumulator {
        req,
        coefficient: &mut coefficient,
        value: ResolventPolynomial::zero(),
        candidate_terms: 0,
        nonzero_terms: 0,
        engine: "resolvent",
        notes: Vec::new(),
    };
    enumerate_resolvent_indices(req, |index| {
        add_resolvent_term(index.h_powers(), index.descendant_powers(), &mut state)
    })?;
    Ok(ResolventResult {
        value: state.value,
        candidate_terms: state.candidate_terms,
        nonzero_terms: state.nonzero_terms,
        engine: state.engine,
        notes: state.notes,
    })
}

pub fn enumerate_resolvent_indices<F>(
    req: &ResolventRequest,
    mut visitor: F,
) -> Result<usize, GwError>
where
    F: FnMut(ResolventIndex) -> Result<(), GwError>,
{
    if req.virtual_dimension < 0 {
        return Ok(0);
    }
    let mut h_powers = vec![0; req.markings];
    let mut descendant_powers = vec![0; req.markings];
    let mut count = 0usize;
    let mut visit = |h_powers: &[usize], descendant_powers: &[usize]| {
        count += 1;
        visitor(ResolventIndex::new(
            h_powers.to_vec(),
            descendant_powers.to_vec(),
        ))
    };
    enumerate_h_powers_with_bound(
        0,
        req.virtual_dimension as usize,
        req.target_n,
        &mut h_powers,
        &mut descendant_powers,
        &mut visit,
    )?;
    Ok(count)
}

struct Accumulator<'a, F>
where
    F: FnMut(&[Insertion]) -> Result<InvariantResult, GwError>,
{
    req: &'a ResolventRequest,
    coefficient: &'a mut F,
    value: ResolventPolynomial,
    candidate_terms: usize,
    nonzero_terms: usize,
    engine: &'static str,
    notes: Vec<String>,
}

fn enumerate_h_powers_with_bound<F>(
    marking: usize,
    remaining_degree: usize,
    target_n: usize,
    h_powers: &mut [usize],
    descendant_powers: &mut [usize],
    visitor: &mut F,
) -> Result<(), GwError>
where
    F: FnMut(&[usize], &[usize]) -> Result<(), GwError>,
{
    if marking == h_powers.len() {
        enumerate_descendant_powers(0, remaining_degree, h_powers, descendant_powers, visitor)?;
        return Ok(());
    }

    for h_power in 0..=target_n.min(remaining_degree) {
        h_powers[marking] = h_power;
        enumerate_h_powers_with_bound(
            marking + 1,
            remaining_degree - h_power,
            target_n,
            h_powers,
            descendant_powers,
            visitor,
        )?;
    }
    Ok(())
}

fn enumerate_descendant_powers<F>(
    marking: usize,
    remaining_degree: usize,
    h_powers: &[usize],
    descendant_powers: &mut [usize],
    visitor: &mut F,
) -> Result<(), GwError>
where
    F: FnMut(&[usize], &[usize]) -> Result<(), GwError>,
{
    if marking == descendant_powers.len() {
        if remaining_degree == 0 {
            visitor(h_powers, descendant_powers)?;
        }
        return Ok(());
    }

    for descendant_power in 0..=remaining_degree {
        descendant_powers[marking] = descendant_power;
        enumerate_descendant_powers(
            marking + 1,
            remaining_degree - descendant_power,
            h_powers,
            descendant_powers,
            visitor,
        )?;
    }
    Ok(())
}

fn add_resolvent_term<F>(
    h_powers: &[usize],
    descendant_powers: &[usize],
    state: &mut Accumulator<'_, F>,
) -> Result<(), GwError>
where
    F: FnMut(&[Insertion]) -> Result<InvariantResult, GwError>,
{
    let insertions = h_powers
        .iter()
        .zip(descendant_powers.iter())
        .map(|(&h_power, &descendant_power)| {
            tau(
                descendant_power,
                CohomologyClass::h_power(state.req.target_n, h_power),
            )
        })
        .collect::<Vec<_>>();
    state.candidate_terms += 1;
    let result = match (state.coefficient)(&insertions) {
        Ok(result) => result,
        Err(GwError::UnsupportedInvariant(message)) => {
            push_note_once(
                &mut state.notes,
                format!(
                    "skipped {}: {message}",
                    resolvent_term_label(h_powers, descendant_powers)
                ),
            );
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    state.engine = result.engine;
    for note in result.notes {
        push_note_once(&mut state.notes, note);
    }
    if result.value.is_zero() {
        return Ok(());
    }

    let (monomial, scalar) = resolvent_monomial(h_powers, descendant_powers);
    let coefficient = &result.value * &RatFun::from_rational(scalar);
    state.value.add_term(monomial, coefficient);
    state.nonzero_terms += 1;
    Ok(())
}

fn push_note_once(notes: &mut Vec<String>, note: String) {
    if !notes.iter().any(|existing| existing == &note) {
        notes.push(note);
    }
}

fn resolvent_monomial(
    h_powers: &[usize],
    descendant_powers: &[usize],
) -> (ResolventMonomial, Rational) {
    let mut scalar = Rational::one();
    for &h_power in h_powers {
        if h_power > 1 {
            scalar = scalar / factorial_rational(h_power);
        }
    }
    (
        ResolventMonomial {
            t_powers: h_powers.to_vec(),
            z_denominator_powers: descendant_powers.iter().map(|power| power + 1).collect(),
        },
        scalar,
    )
}

fn format_signed_resolvent_term<C: ResolventCoefficient>(
    coefficient: &C,
    monomial: &ResolventMonomial,
) -> (bool, String) {
    let monomial_text = format_resolvent_monomial(monomial);
    if let Some(rational) = coefficient.as_resolvent_rational() {
        let negative = rational.is_negative();
        let abs = rational.abs();
        if monomial_text == "1" {
            return (negative, abs.to_string());
        }
        let text = if abs == Rational::one() {
            monomial_text
        } else {
            format!("({abs})*{monomial_text}")
        };
        return (negative, text);
    }
    if monomial_text == "1" {
        return (false, coefficient.to_string());
    }
    if coefficient.is_structurally_one() {
        return (false, monomial_text);
    }
    (false, format!("({coefficient})*{monomial_text}"))
}

fn format_resolvent_monomial(monomial: &ResolventMonomial) -> String {
    let numerator = monomial
        .t_powers
        .iter()
        .enumerate()
        .filter_map(|(marking, &power)| variable_power("t", marking, power))
        .collect::<Vec<_>>();
    let denominator = monomial
        .z_denominator_powers
        .iter()
        .enumerate()
        .filter_map(|(marking, &power)| variable_power("z", marking, power))
        .collect::<Vec<_>>();

    match (numerator.is_empty(), denominator.is_empty()) {
        (true, true) => "1".to_string(),
        (false, true) => numerator.join("*"),
        (true, false) => format!("1/({})", denominator.join("*")),
        (false, false) => format!("{}/({})", numerator.join("*"), denominator.join("*")),
    }
}

fn variable_power(prefix: &str, marking: usize, power: usize) -> Option<String> {
    match power {
        0 => None,
        1 => Some(format!("{prefix}{marking}")),
        _ => Some(format!("{prefix}{marking}^{power}")),
    }
}

fn resolvent_term_label(h_powers: &[usize], descendant_powers: &[usize]) -> String {
    h_powers
        .iter()
        .zip(descendant_powers.iter())
        .enumerate()
        .map(|(marking, (&h_power, &descendant_power))| {
            format!(
                "marking {marking}: H^{h_power}/z^{descendant_power_plus_one}",
                descendant_power_plus_one = descendant_power + 1
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn factorial_rational(n: usize) -> Rational {
    let mut out = Rational::one();
    for factor in 2..=n {
        out = out * Rational::from(factor);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(
        expected = "resolvent coefficient must have one H-power and one descendant power per marking"
    )]
    fn polynomial_rejects_mismatched_coefficient_index_lengths() {
        let mut value = ResolventPolynomial::<RatFun>::zero();
        value.add_coefficient(&[1], &[], RatFun::one());
    }

    #[test]
    fn finite_resolvent_sum_matches_manual_one_marking_expression() {
        let req = ResolventRequest {
            target_n: 1,
            genus: 0,
            degree: 0,
            markings: 1,
            virtual_dimension: 1,
        };
        let result = compute_resolvent_generating_function(&req, |_| {
            Ok(InvariantResult {
                value: RatFun::one(),
                engine: "test",
                notes: Vec::new(),
            })
        })
        .unwrap();

        assert_eq!(result.value.to_string(), "1/(z0^2) + t0/(z0)");
        assert_eq!(result.candidate_terms, 2);
        assert_eq!(result.nonzero_terms, 2);
    }

    #[test]
    fn negative_virtual_dimension_is_zero() {
        let req = ResolventRequest {
            target_n: 2,
            genus: 0,
            degree: 0,
            markings: 0,
            virtual_dimension: -1,
        };
        let result =
            compute_resolvent_generating_function(&req, |_| unreachable!("no terms")).unwrap();
        assert!(result.value.is_zero());
        assert_eq!(result.candidate_terms, 0);
    }

    #[test]
    fn enumerated_indices_convert_to_insertions() {
        let req = ResolventRequest {
            target_n: 1,
            genus: 0,
            degree: 0,
            markings: 1,
            virtual_dimension: 1,
        };
        let mut indices = Vec::new();
        let count = enumerate_resolvent_indices(&req, |index| {
            indices.push(index);
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 2);
        assert_eq!(indices[0].h_powers(), &[0]);
        assert_eq!(indices[0].descendant_powers(), &[1]);
        assert_eq!(indices[1].h_powers(), &[1]);
        assert_eq!(indices[1].descendant_powers(), &[0]);
        assert_eq!(
            indices[1].to_insertions(1),
            vec![tau(0, CohomologyClass::h_power(1, 1))]
        );
    }

    #[test]
    fn display_splits_negative_signs() {
        let mut value = ResolventPolynomial::zero();
        value.add_term(
            ResolventMonomial {
                t_powers: vec![0],
                z_denominator_powers: vec![1],
            },
            RatFun::from(Rational::new(-1, 2)),
        );
        value.add_term(
            ResolventMonomial {
                t_powers: vec![1],
                z_denominator_powers: vec![1],
            },
            RatFun::one(),
        );

        assert_eq!(value.to_string(), "-(1/2)*1/(z0) + t0/(z0)");
    }

    #[test]
    fn resolvent_equivalence_compares_coefficients_as_rational_functions() {
        let x = RatFun::variable("x");
        let y = RatFun::variable("y");
        let quotient = &(&x.pow_usize(2) - &y.pow_usize(2)) / &(&x - &y);
        let sum = &x + &y;
        let monomial = ResolventMonomial {
            t_powers: vec![1],
            z_denominator_powers: vec![1],
        };
        let mut left = ResolventPolynomial::zero();
        left.add_term(monomial.clone(), quotient);
        let mut right = ResolventPolynomial::zero();
        right.add_term(monomial, sum);

        assert_ne!(left, right);
        assert!(left.equivalent(&right));

        let extra_monomial = ResolventMonomial {
            t_powers: vec![0],
            z_denominator_powers: vec![2],
        };
        right.terms.insert(extra_monomial.clone(), RatFun::zero());
        assert!(left.equivalent(&right));

        right.terms.insert(extra_monomial, RatFun::one());
        assert!(!left.equivalent(&right));
    }
}
