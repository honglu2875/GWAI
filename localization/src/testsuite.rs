use crate::algebra::{RatFun, Rational};
use crate::frobenius::{canonical_root_series, characteristic_series, FrobeniusData};
use crate::geometry::{CohomologyClass, EquivariantProjectiveSpace};
use crate::givental::{
    compute_by_givental_graphs, projective_space_descendant_s_matrix,
    projective_space_j_calibration, CanonicalFrameConvention, SeriesRMatrix,
};
use crate::growi_oracle::{
    disputed_p2_genus_two_descendants, fast_positive_genus_cases, oracle_cases,
};
use crate::localization::genus_zero_localization_graphs;
use crate::series::{QSeries, SeriesMatrix};
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::{compute, compute_series, tau, ComputeMode, InvariantRequest, SeriesRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinTestCase {
    pub name: &'static str,
    pub passed: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinTestReport {
    pub cases: Vec<BuiltinTestCase>,
}

impl BuiltinTestReport {
    pub fn passed(&self) -> usize {
        self.cases.iter().filter(|case| case.passed).count()
    }

    pub fn failed(&self) -> usize {
        self.cases.len() - self.passed()
    }

    pub fn is_success(&self) -> bool {
        self.failed() == 0
    }
}

pub fn run_builtin_tests() -> BuiltinTestReport {
    let tests: &[(&str, fn() -> Result<(), String>)] = &[
        ("psi genus two one-point", test_psi_genus_two_one_point),
        ("P2 degree-one line coefficient", test_p2_degree_one_line),
        ("P2 conic through five points", test_p2_conic_count),
        ("P2 cubic through eight points", test_p2_cubic_count),
        ("P2 quartic through eleven points", test_p2_quartic_count),
        ("P2 quintic through fourteen points", test_p2_quintic_count),
        (
            "bounded primary potential series",
            test_primary_potential_series,
        ),
        (
            "P2 degree-one localization graph count",
            test_p2_localization_graph_count,
        ),
        (
            "P2 genus-one degree-zero S/R obstruction",
            test_p2_genus_one_degree_zero_sr_obstruction,
        ),
        (
            "P1 equivariant one-edge localization",
            test_p1_equivariant_localization,
        ),
        ("fixed insertion expected degree", test_expected_degree),
        (
            "Frobenius root solves characteristic equation",
            test_frobenius_root,
        ),
        (
            "classical canonical idempotents",
            test_classical_canonical_data,
        ),
        (
            "quantum canonical P1 idempotents",
            test_quantum_canonical_p1_data,
        ),
        (
            "identity R matrix unitarity",
            test_identity_r_matrix_unitarity,
        ),
        ("P1 J-calibrated R matrix", test_p1_j_calibrated_r_matrix),
        ("P1 descendant S matrix", test_p1_descendant_s_matrix),
        (
            "P1 Givental graph expansion",
            test_p1_givental_graph_expansion,
        ),
        (
            "P1 stationary descendant via S/R graph",
            test_p1_stationary_descendant_graph_expansion,
        ),
        (
            "P1 high-degree stationary S/R graph",
            test_p1_high_degree_stationary_graph_expansion,
        ),
        (
            "P1 high-degree stationary shortcut",
            test_p1_high_degree_stationary_shortcut,
        ),
        (
            "Zinger projective path cross-checks",
            test_zinger_projective_cross_checks,
        ),
        ("Growi oracle constants", test_growi_oracle_constants),
        (
            "S/R positive-genus Growi cross-checks",
            test_sr_positive_genus_growi_cross_checks,
        ),
    ];

    BuiltinTestReport {
        cases: tests
            .iter()
            .map(|(name, test)| match test() {
                Ok(()) => BuiltinTestCase {
                    name,
                    passed: true,
                    message: "ok".to_string(),
                },
                Err(message) => BuiltinTestCase {
                    name,
                    passed: false,
                    message,
                },
            })
            .collect(),
    }
}

fn test_psi_genus_two_one_point() -> Result<(), String> {
    expect_eq(
        WittenKontsevich::new().psi_integral(2, &[4]),
        Rational::new(1, 1152),
    )
}

fn test_p2_degree_one_line() -> Result<(), String> {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let req = InvariantRequest {
        mode: ComputeMode::CompareLocalizationAndGivental,
        ..InvariantRequest::new(2, 0, 1, insertions)
    };
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::one(),
    )
}

fn test_p2_conic_count() -> Result<(), String> {
    let req = InvariantRequest {
        mode: ComputeMode::Localization,
        ..InvariantRequest::new(2, 0, 2, vec![tau(0, CohomologyClass::h_power(2, 2)); 5])
    };
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::one(),
    )
}

fn test_p2_cubic_count() -> Result<(), String> {
    let req = InvariantRequest {
        mode: ComputeMode::Localization,
        ..InvariantRequest::new(2, 0, 3, vec![tau(0, CohomologyClass::h_power(2, 2)); 8])
    };
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::from(12usize),
    )
}

fn test_p2_quartic_count() -> Result<(), String> {
    let req = InvariantRequest {
        mode: ComputeMode::Localization,
        ..InvariantRequest::new(2, 0, 4, vec![tau(0, CohomologyClass::h_power(2, 2)); 11])
    };
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::from(620usize),
    )
}

fn test_p2_quintic_count() -> Result<(), String> {
    let req = InvariantRequest {
        mode: ComputeMode::Localization,
        ..InvariantRequest::new(2, 0, 5, vec![tau(0, CohomologyClass::h_power(2, 2)); 14])
    };
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::from(87304usize),
    )
}

fn test_primary_potential_series() -> Result<(), String> {
    let mut req = SeriesRequest::new(2, 0, 1, 3);
    req.mode = ComputeMode::CompareLocalizationAndGivental;
    let series = compute_series(req).map_err(|err| err.to_string())?;
    let found = series.coefficients.iter().any(|coefficient| {
        coefficient.degree == 1
            && coefficient.value == RatFun::one()
            && coefficient.insertion_label() == "tau0(H) tau0(H^2) tau0(H^2)"
    });
    if found {
        Ok(())
    } else {
        Err("missing q^1 tau0(H) tau0(H^2)^2 coefficient".to_string())
    }
}

fn test_p2_localization_graph_count() -> Result<(), String> {
    expect_eq(genus_zero_localization_graphs(2, 1, 0).len(), 3)
}

fn test_p2_genus_one_degree_zero_sr_obstruction() -> Result<(), String> {
    let req = InvariantRequest::new(2, 1, 0, vec![tau(0, CohomologyClass::h_power(2, 1))]);
    let result = compute(req).map_err(|err| err.to_string())?;
    if result.engine == "givental-seed" {
        return Err(
            "genus-one degree-zero check used seed fallback instead of S/R graph engine".into(),
        );
    }
    expect_eq(result.value, RatFun::from_rational(Rational::new(-1, 8)))
}

fn test_p1_equivariant_localization() -> Result<(), String> {
    let req = InvariantRequest {
        n: 1,
        genus: 0,
        degree: 1,
        insertions: vec![tau(0, CohomologyClass::h_power(1, 1)); 2],
        equivariant: true,
        mode: ComputeMode::Localization,
        truncation: None,
    };
    let result = compute(req).map_err(|err| err.to_string())?;
    expect_eq(result.value.clone(), RatFun::one())?;
    let limit = result
        .nonequivariant_limit_line(1, &[Rational::one(), Rational::from(2)])
        .map_err(|err| err.to_string())?;
    expect_eq(limit, Rational::one())
}

fn test_expected_degree() -> Result<(), String> {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let req = InvariantRequest::new(2, 0, 0, insertions);
    expect_eq(req.expected_degree_from_dimension(), Some(1))
}

fn test_frobenius_root() -> Result<(), String> {
    let root = canonical_root_series(1, 0, 3).map_err(|err| err.to_string())?;
    let residual = characteristic_series(1, &root).sub(&QSeries::q(3));
    if residual.coeffs().iter().all(|coeff| coeff.is_zero()) {
        Ok(())
    } else {
        Err(format!("nonzero residual {residual}"))
    }
}

fn test_classical_canonical_data() -> Result<(), String> {
    let frob = FrobeniusData::classical(2);
    let target = EquivariantProjectiveSpace::new(2);
    let data = frob.classical_canonical_data();
    for i in 0..=2 {
        expect_eq(
            data.metric_norms[i].clone(),
            &RatFun::one() / &target.fixed_point_euler(i),
        )?;
    }
    Ok(())
}

fn test_quantum_canonical_p1_data() -> Result<(), String> {
    let frob = FrobeniusData::quantum(1);
    let data = frob
        .quantum_canonical_data(1)
        .map_err(|err| err.to_string())?;
    let sum = data.idempotents.iter().fold(
        crate::frobenius::SeriesCohomologyClass::zero(1, 1),
        |acc, idempotent| acc.add(idempotent),
    );
    expect_eq(sum, crate::frobenius::SeriesCohomologyClass::one(1, 1))
}

fn test_identity_r_matrix_unitarity() -> Result<(), String> {
    let frob = FrobeniusData::quantum(1);
    let data = frob
        .quantum_canonical_data(1)
        .map_err(|err| err.to_string())?;
    let metric = data.metric_norm_matrix();
    let r = SeriesRMatrix::identity(
        metric.rows(),
        metric.max_degree(),
        3,
        CanonicalFrameConvention::UnnormalizedCanonicalIdempotents,
    );
    r.check_identity_calibration()
        .map_err(|err| err.to_string())?;
    r.check_unitarity(&metric).map_err(|err| err.to_string())?;

    let flat_metric = SeriesMatrix::constant(frob.flat_metric_matrix(), 1);
    let diagonalized = data.canonical_metric_from_transition(&flat_metric);
    expect_eq(diagonalized.rows(), metric.rows())
}

fn test_p1_j_calibrated_r_matrix() -> Result<(), String> {
    let calibration = projective_space_j_calibration(1, 1, 2).map_err(|err| err.to_string())?;
    assert_r_matrix_unitary_after_lambda_eval(
        &calibration.r_matrix,
        &calibration.metric,
        1,
        &[Rational::from(2), Rational::from(5)],
    )?;

    let r1 = calibration
        .r_matrix
        .coefficient(1)
        .ok_or_else(|| "missing z^1 R coefficient".to_string())?;
    expect_eq(
        r1.entry(0, 0)
            .coeff(0)
            .ok_or_else(|| "missing q^0 diagonal coefficient".to_string())?
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .map_err(|err| err.to_string())?,
        Rational::new(1, 36),
    )
}

fn test_p1_descendant_s_matrix() -> Result<(), String> {
    let descendant_s =
        projective_space_descendant_s_matrix(1, 1, 2).map_err(|err| err.to_string())?;
    expect_eq(
        descendant_s
            .coefficient(1)
            .ok_or_else(|| "missing z^-1 S coefficient".to_string())?
            .entry(0, 1)
            .coeff(1)
            .ok_or_else(|| "missing q^1 S coefficient".to_string())?
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .map_err(|err| err.to_string())?,
        Rational::one(),
    )?;
    expect_eq(
        descendant_s
            .coefficient(2)
            .ok_or_else(|| "missing z^-2 S coefficient".to_string())?
            .entry(1, 1)
            .coeff(1)
            .ok_or_else(|| "missing q^1 S coefficient".to_string())?
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .map_err(|err| err.to_string())?,
        Rational::one(),
    )
}

fn test_p1_givental_graph_expansion() -> Result<(), String> {
    let req = InvariantRequest::new(
        1,
        0,
        1,
        vec![
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute_by_givental_graphs(&req)
            .map_err(|err| err.to_string())?
            .value,
        RatFun::one(),
    )
}

fn test_p1_stationary_descendant_graph_expansion() -> Result<(), String> {
    let req = InvariantRequest::new(
        1,
        0,
        2,
        vec![
            tau(2, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute_by_givental_graphs(&req)
            .map_err(|err| err.to_string())?
            .value,
        RatFun::one(),
    )
}

fn test_p1_high_degree_stationary_graph_expansion() -> Result<(), String> {
    let req = InvariantRequest::new(
        1,
        0,
        3,
        vec![
            tau(4, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute_by_givental_graphs(&req)
            .map_err(|err| err.to_string())?
            .value,
        RatFun::from_rational(Rational::new(1, 4)),
    )?;

    let req = InvariantRequest::new(
        1,
        0,
        4,
        vec![
            tau(6, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute_by_givental_graphs(&req)
            .map_err(|err| err.to_string())?
            .value,
        RatFun::from_rational(Rational::new(1, 36)),
    )
}

fn test_p1_high_degree_stationary_shortcut() -> Result<(), String> {
    let req = InvariantRequest::new(
        1,
        0,
        3,
        vec![
            tau(4, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::from_rational(Rational::new(1, 4)),
    )?;

    let req = InvariantRequest::new(
        1,
        0,
        4,
        vec![
            tau(6, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ],
    );
    expect_eq(
        compute(req).map_err(|err| err.to_string())?.value,
        RatFun::from_rational(Rational::new(1, 36)),
    )
}

fn test_zinger_projective_cross_checks() -> Result<(), String> {
    let requests = vec![
        InvariantRequest::new(
            1,
            0,
            3,
            vec![
                tau(4, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        ),
        InvariantRequest::new(
            1,
            0,
            4,
            vec![
                tau(6, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
                tau(0, CohomologyClass::h_power(1, 1)),
            ],
        ),
        InvariantRequest::new(
            2,
            0,
            1,
            vec![
                tau(1, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 2)),
            ],
        ),
    ];

    for req in requests {
        let zinger = crate::zinger::compute(&req).map_err(|err| err.to_string())?;
        let givental = compute_by_givental_graphs(&req).map_err(|err| err.to_string())?;
        expect_eq(zinger.value, givental.value)?;
    }
    Ok(())
}

fn test_growi_oracle_constants() -> Result<(), String> {
    let cases = oracle_cases();
    expect_eq(cases.len(), 12)?;
    let disputed = disputed_p2_genus_two_descendants();
    expect_eq(
        disputed[1].expected.clone(),
        RatFun::from_rational(Rational::new(-421, 207360)),
    )?;
    expect_eq(
        disputed[2].expected.clone(),
        RatFun::from_rational(Rational::new(11, 17280)),
    )
}

fn test_sr_positive_genus_growi_cross_checks() -> Result<(), String> {
    for case in fast_positive_genus_cases() {
        let result = compute_by_givental_graphs(&case.request).map_err(|err| err.to_string())?;
        expect_eq(result.value, case.expected)?;
    }
    Ok(())
}

fn assert_r_matrix_unitary_after_lambda_eval(
    r: &SeriesRMatrix,
    metric: &SeriesMatrix,
    target_n: usize,
    weights: &[Rational],
) -> Result<(), String> {
    for z_degree in 0..=r.z_order() {
        let mut total = SeriesMatrix::zero(r.size(), r.size(), r.q_degree());
        for left_order in 0..=z_degree {
            let right_order = z_degree - left_order;
            let term = r
                .coefficient(left_order)
                .ok_or_else(|| format!("missing z^{left_order} coefficient"))?
                .transpose()
                .mul(metric)
                .mul(
                    r.coefficient(right_order)
                        .ok_or_else(|| format!("missing z^{right_order} coefficient"))?,
                );
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
        assert_series_matrix_equal_after_lambda_eval(&total, &expected, target_n, weights)?;
    }
    Ok(())
}

fn assert_series_matrix_equal_after_lambda_eval(
    left: &SeriesMatrix,
    right: &SeriesMatrix,
    target_n: usize,
    weights: &[Rational],
) -> Result<(), String> {
    expect_eq(left.rows(), right.rows())?;
    expect_eq(left.cols(), right.cols())?;
    for row in 0..left.rows() {
        for col in 0..left.cols() {
            let left_series = left.entry(row, col);
            let right_series = right.entry(row, col);
            expect_eq(left_series.max_degree(), right_series.max_degree())?;
            for degree in 0..=left_series.max_degree() {
                let diff = left_series.coeff(degree).unwrap() - right_series.coeff(degree).unwrap();
                let value = diff
                    .evaluate_lambda_weights(target_n, weights)
                    .map_err(|err| err.to_string())?;
                if value != Rational::zero() {
                    return Err(format!(
                        "matrix mismatch at ({row},{col}), q^{degree}: {value}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn expect_eq<T: std::fmt::Debug + PartialEq>(actual: T, expected: T) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("expected {expected:?}, got {actual:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_suite_passes() {
        let report = run_builtin_tests();
        assert!(
            report.is_success(),
            "builtin suite failures: {:?}",
            report
                .cases
                .iter()
                .filter(|case| !case.passed)
                .collect::<Vec<_>>()
        );
    }
}
