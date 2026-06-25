//! Fixed-degree resolvent generating functions.
//!
//! For fixed target, genus, degree, and labelled markings this module computes
//!
//! ```text
//! sum_{a_i,k_i} < prod_i tau_{k_i}(H^{a_i}) >_{g,d}
//!     prod_i t_i^{a_i}/a_i! * z_i^{-k_i-1}.
//! ```
//!
//! The virtual dimension fixes `sum_i (a_i+k_i)`, so the sum is finite.  The
//! backend is supplied as a callback, which keeps this layer independent of the
//! ordinary/twisted/future CohFT implementation used to compute individual
//! coefficients.

use std::collections::BTreeMap;
use std::fmt;

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::geometry::CohomologyClass;
use crate::{tau, Insertion, InvariantResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventRequest {
    pub target_n: usize,
    pub genus: usize,
    pub degree: usize,
    pub markings: usize,
    pub virtual_dimension: isize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventResult {
    pub value: ResolventPolynomial,
    pub candidate_terms: usize,
    pub nonzero_terms: usize,
    pub engine: &'static str,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolventPolynomial {
    terms: BTreeMap<ResolventMonomial, RatFun>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResolventMonomial {
    t_powers: Vec<usize>,
    z_denominator_powers: Vec<usize>,
}

impl ResolventPolynomial {
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn add_term(&mut self, monomial: ResolventMonomial, coefficient: RatFun) {
        if coefficient.is_zero() {
            return;
        }
        let next = self
            .terms
            .remove(&monomial)
            .map(|current| &current + &coefficient)
            .unwrap_or(coefficient);
        if !next.is_zero() {
            self.terms.insert(monomial, next);
        }
    }

    pub fn add_coefficient(
        &mut self,
        h_powers: &[usize],
        descendant_powers: &[usize],
        coefficient: RatFun,
    ) {
        let (monomial, scalar) = resolvent_monomial(h_powers, descendant_powers);
        self.add_term(monomial, &coefficient * &scalar);
    }
}

impl fmt::Display for ResolventPolynomial {
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

    let target_degree = req.virtual_dimension as usize;
    let mut h_powers = vec![0; req.markings];
    let mut descendant_powers = vec![0; req.markings];
    let mut state = Accumulator {
        req,
        coefficient: &mut coefficient,
        value: ResolventPolynomial::zero(),
        candidate_terms: 0,
        nonzero_terms: 0,
        engine: "resolvent",
        notes: Vec::new(),
    };
    let mut visit = |h_powers: &[usize], descendant_powers: &[usize]| {
        add_resolvent_term(h_powers, descendant_powers, &mut state)
    };
    enumerate_h_powers_with_bound(
        0,
        target_degree,
        req.target_n,
        &mut h_powers,
        &mut descendant_powers,
        &mut visit,
    )?;
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
    F: FnMut(&[usize], &[usize]) -> Result<(), GwError>,
{
    if req.virtual_dimension < 0 {
        return Ok(0);
    }
    let mut h_powers = vec![0; req.markings];
    let mut descendant_powers = vec![0; req.markings];
    let mut count = 0usize;
    let mut visit = |h_powers: &[usize], descendant_powers: &[usize]| {
        count += 1;
        visitor(h_powers, descendant_powers)
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
    let coefficient = &result.value * &scalar;
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
) -> (ResolventMonomial, RatFun) {
    let mut scalar = RatFun::one();
    for &h_power in h_powers {
        if h_power > 1 {
            scalar = &scalar / &RatFun::from(factorial_rational(h_power));
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

fn format_signed_resolvent_term(
    coefficient: &RatFun,
    monomial: &ResolventMonomial,
) -> (bool, String) {
    let monomial_text = format_resolvent_monomial(monomial);
    if let Some(rational) = coefficient.as_rational() {
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
    if coefficient.is_one() {
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
}
