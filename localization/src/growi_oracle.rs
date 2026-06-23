//! Growi ground-truth values recorded from an isolated Growi 1.0.3 build.
//!
//! These constants are not used as computation shortcuts.  They are regression
//! oracles for the projective-space S/R implementation and for independent
//! cross-check paths.

use crate::algebra::{RatFun, Rational};
use crate::geometry::CohomologyClass;
use crate::{tau, InvariantRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrowiOracleCase {
    pub name: &'static str,
    pub growi_command: &'static str,
    pub request: InvariantRequest,
    pub expected: RatFun,
}

pub fn oracle_cases() -> Vec<GrowiOracleCase> {
    let mut cases = disputed_p2_genus_two_descendants();
    cases.extend(fast_positive_genus_cases());
    cases.extend(p1_genus_two_degree_three_descendants());
    cases.extend([
        case(
            "P2 genus-one elliptic cubics through nine points",
            "growi elliptic cubics in P^2 thru H^2:9",
            InvariantRequest::new(2, 1, 3, vec![tau(0, CohomologyClass::h_power(2, 2)); 9]),
            Rational::one(),
        ),
        case(
            "P2 genus-two quintics through sixteen points",
            "growi G=2,D=5 in P^2 thru H^2:16",
            InvariantRequest::new(2, 2, 5, vec![tau(0, CohomologyClass::h_power(2, 2)); 16]),
            Rational::from(36855usize),
        ),
        case(
            "P2 genus-three quintics through seventeen points",
            "growi G=3,D=5 in P^2 thru H^2:17",
            InvariantRequest::new(2, 3, 5, vec![tau(0, CohomologyClass::h_power(2, 2)); 17]),
            Rational::from(7915usize),
        ),
        case(
            "P3 genus-two degree-three curves meeting twelve codimension-two cycles",
            "growi G=2,D=3 in P^3 thru H^2:12",
            InvariantRequest::new(3, 2, 3, vec![tau(0, CohomologyClass::h_power(3, 2)); 12]),
            Rational::from(5930usize),
        ),
        case(
            "P4 genus-two degree-five descendant",
            "growi G=2,D=5 in P^4 thru H^3*psi^22",
            InvariantRequest::new(4, 2, 5, vec![tau(22, CohomologyClass::h_power(4, 3))]),
            Rational::new(-41369, 110075314176),
        ),
    ]);
    cases
}

pub fn fast_positive_genus_cases() -> Vec<GrowiOracleCase> {
    vec![
        case(
            "P1 genus-one degree-one H psi^2",
            "growi G=1,D=1 in P^1 thru H*psi^2",
            InvariantRequest::new(1, 1, 1, vec![tau(2, CohomologyClass::h_power(1, 1))]),
            Rational::new(1, 24),
        ),
        case(
            "P1 genus-one degree-one psi^3",
            "growi G=1,D=1 in P^1 thru psi^3",
            InvariantRequest::new(1, 1, 1, vec![tau(3, CohomologyClass::one(1))]),
            Rational::zero(),
        ),
    ]
}

pub fn p1_genus_two_degree_three_descendants() -> Vec<GrowiOracleCase> {
    vec![
        case(
            "P1 genus-two degree-three H psi^8",
            "growi G=2,D=3 in P^1 thru H*psi^8",
            InvariantRequest::new(1, 2, 3, vec![tau(8, CohomologyClass::h_power(1, 1))]),
            Rational::new(23, 41472),
        ),
        case(
            "P1 genus-two degree-three psi^9",
            "growi G=2,D=3 in P^1 thru psi^9",
            InvariantRequest::new(1, 2, 3, vec![tau(9, CohomologyClass::one(1))]),
            Rational::new(-977, 622080),
        ),
    ]
}

pub fn disputed_p2_genus_two_descendants() -> Vec<GrowiOracleCase> {
    vec![
        case(
            "P2 genus-two degree-three one-point psi^11",
            "growi G=2,D=3 in P^2 thru psi^11",
            InvariantRequest::new(2, 2, 3, vec![tau(11, CohomologyClass::one(2))]),
            Rational::new(163, 41472),
        ),
        case(
            "P2 genus-two degree-three one-point H psi^10",
            "growi G=2,D=3 in P^2 thru H*psi^10",
            InvariantRequest::new(2, 2, 3, vec![tau(10, CohomologyClass::h_power(2, 1))]),
            Rational::new(-421, 207360),
        ),
        case(
            "P2 genus-two degree-three one-point H^2 psi^9",
            "growi G=2,D=3 in P^2 thru H^2*psi^9",
            InvariantRequest::new(2, 2, 3, vec![tau(9, CohomologyClass::h_power(2, 2))]),
            Rational::new(11, 17280),
        ),
    ]
}

fn case(
    name: &'static str,
    growi_command: &'static str,
    request: InvariantRequest,
    expected: Rational,
) -> GrowiOracleCase {
    GrowiOracleCase {
        name,
        growi_command,
        request,
        expected: RatFun::from_rational(expected),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::givental::compute_by_givental_graphs;

    #[test]
    fn records_disputed_p2_genus_two_descendants() {
        let cases = disputed_p2_genus_two_descendants();
        assert_eq!(cases.len(), 3);
        assert_eq!(
            cases[0].expected,
            RatFun::from_rational(Rational::new(163, 41472))
        );
        assert_eq!(
            cases[1].expected,
            RatFun::from_rational(Rational::new(-421, 207360))
        );
        assert_eq!(
            cases[2].expected,
            RatFun::from_rational(Rational::new(11, 17280))
        );
    }

    #[test]
    fn sr_graph_matches_fast_positive_genus_growi_cases() {
        for case in fast_positive_genus_cases() {
            let actual = compute_by_givental_graphs(&case.request)
                .unwrap_or_else(|err| panic!("{} failed: {err}", case.name));
            assert_eq!(actual.value, case.expected, "{}", case.growi_command);
        }
    }

    #[test]
    fn sr_graph_matches_growi_disputed_descendants() {
        for case in disputed_p2_genus_two_descendants() {
            let actual = compute_by_givental_graphs(&case.request)
                .unwrap_or_else(|err| panic!("{} failed: {err}", case.name));
            assert_eq!(actual.value, case.expected, "{}", case.growi_command);
        }
    }

    #[test]
    fn sr_graph_matches_p1_genus_two_degree_three_growi_cases() {
        for case in p1_genus_two_degree_three_descendants() {
            let actual = compute_by_givental_graphs(&case.request)
                .unwrap_or_else(|err| panic!("{} failed: {err}", case.name));
            assert_eq!(actual.value, case.expected, "{}", case.growi_command);
        }
    }
}
