use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::{Insertion, InvariantRequest, InvariantResult};
use std::collections::BTreeMap;

type BiKey = (usize, i32);

pub fn compute(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    if req.equivariant {
        return Err(GwError::UnsupportedInvariant(
            "Zinger projective-space path currently returns nonequivariant numbers only"
                .to_string(),
        ));
    }
    if req.genus != 0 {
        return Err(GwError::UnsupportedInvariant(
            "Zinger projective-space path currently implements genus zero only".to_string(),
        ));
    }

    let Some(insertions) = pure_projective_insertions(&req.insertions) else {
        return Err(GwError::UnsupportedInvariant(
            "Zinger projective-space path requires pure tau_b(H^c) insertions".to_string(),
        ));
    };

    if let Some(total_degree) = req.insertion_degree() {
        let virtual_dimension = req.virtual_dimension();
        if virtual_dimension >= 0 && total_degree as isize != virtual_dimension {
            return Ok(result(Rational::zero(), "dimension mismatch gives zero"));
        }
    }

    if req.degree == 0 {
        return Ok(result(
            constant_map_value(req.n, &insertions),
            "Zinger degree-zero projective-space constant-map formula",
        ));
    }

    if zinger_projective_vanishing(req.n, &insertions) {
        return Ok(result(
            Rational::zero(),
            "Zinger projective-space vanishing criterion",
        ));
    }

    match insertions.len() {
        3 => Ok(result(
            three_point_projective_space(req.n, req.degree, &insertions),
            "Zinger projective-space 3-point formula",
        )),
        4 if insertions.iter().all(|insertion| insertion.descendant_power == 0) => {
            four_point_primary_projective_space(req.n, req.degree, &insertions)
                .map(|value| result(value, "Zinger projective-space 4-point primary formula"))
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant(
                        "Zinger 4-point implementation currently covers only the stated primary degree-one and degree-two projective-space consequences"
                            .to_string(),
                    )
                })
        }
        4 => Err(GwError::UnsupportedInvariant(
            "general Zinger 4-point descendant extraction is not implemented yet".to_string(),
        )),
        markings => Err(GwError::UnsupportedInvariant(format!(
            "Zinger projective-space path currently implements N=3, selected N=4, and degree-zero constants; requested N={markings}"
        ))),
    }
}

fn result(value: Rational, note: impl Into<String>) -> InvariantResult {
    InvariantResult {
        value: RatFun::from_rational(value),
        engine: "zinger-projective",
        notes: vec![note.into()],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PureInsertion {
    descendant_power: usize,
    h_power: usize,
}

fn pure_projective_insertions(insertions: &[Insertion]) -> Option<Vec<PureInsertion>> {
    insertions
        .iter()
        .map(|insertion| {
            Some(PureInsertion {
                descendant_power: insertion.descendant_power,
                h_power: insertion.class.pure_power()?,
            })
        })
        .collect()
}

fn constant_map_value(target_dim: usize, insertions: &[PureInsertion]) -> Rational {
    if insertions.len() < 3 {
        return Rational::zero();
    }
    let h_sum = insertions
        .iter()
        .map(|insertion| insertion.h_power)
        .sum::<usize>();
    if h_sum != target_dim {
        return Rational::zero();
    }
    let psi_sum = insertions
        .iter()
        .map(|insertion| insertion.descendant_power)
        .sum::<usize>();
    let moduli_dimension = insertions.len() - 3;
    if psi_sum != moduli_dimension {
        return Rational::zero();
    }
    multinomial_rational(
        moduli_dimension,
        insertions.iter().map(|i| i.descendant_power),
    )
}

fn zinger_projective_vanishing(target_dim: usize, insertions: &[PureInsertion]) -> bool {
    let markings = insertions.len();
    if markings < 3 {
        return false;
    }
    let fano_index = target_dim + 1;
    for mask in 1usize..(1usize << markings) {
        let mut psi_sum = 0usize;
        let mut valid = true;
        for (idx, insertion) in insertions.iter().enumerate() {
            if (mask & (1usize << idx)) == 0 {
                continue;
            }
            if insertion.descendant_power + insertion.h_power >= fano_index {
                valid = false;
                break;
            }
            psi_sum += insertion.descendant_power;
        }
        if valid && psi_sum > markings - 3 {
            return true;
        }
    }
    false
}

fn three_point_projective_space(
    target_dim: usize,
    degree: usize,
    insertions: &[PureInsertion],
) -> Rational {
    debug_assert_eq!(insertions.len(), 3);
    let rank = target_dim + 1;
    let target_p = insertions
        .iter()
        .map(|insertion| target_dim.checked_sub(insertion.h_power))
        .collect::<Option<Vec<_>>>();
    let Some(target_p) = target_p else {
        return Rational::zero();
    };
    let target_y = insertions
        .iter()
        .map(|insertion| insertion.descendant_power + 1)
        .collect::<Vec<_>>();

    let mut total = Rational::zero();
    for d_prime in 0..=1 {
        if d_prime > degree {
            continue;
        }
        let p_sum = (2usize.saturating_sub(d_prime)) * rank;
        if p_sum < 2 {
            continue;
        }
        let p_sum = p_sum - 2;
        for degree_split in weak_compositions(degree - d_prime, 3) {
            for p_split in bounded_compositions(p_sum, 3, rank) {
                let mut term = Rational::one();
                for marking in 0..3 {
                    term = term
                        * zinger_factor_coefficient(
                            rank,
                            degree_split[marking],
                            p_split[marking],
                            target_p[marking],
                            target_y[marking],
                        );
                    if term.is_zero() {
                        break;
                    }
                }
                total += term;
            }
        }
    }
    total
}

fn four_point_primary_projective_space(
    target_dim: usize,
    degree: usize,
    insertions: &[PureInsertion],
) -> Option<Rational> {
    let rank = target_dim + 1;
    let h_sum = insertions
        .iter()
        .map(|insertion| insertion.h_power)
        .sum::<usize>();
    if degree == 1
        && h_sum == 2 * rank
        && insertions
            .iter()
            .all(|insertion| insertion.h_power > 0 && insertion.h_power < rank)
    {
        let value = insertions
            .iter()
            .map(|insertion| {
                let h = insertion.h_power;
                h.min(rank - h)
            })
            .min()
            .unwrap_or(0);
        return Some(Rational::from(value));
    }
    if degree == 2 && h_sum == 3 * rank {
        return Some(Rational::zero());
    }
    None
}

fn zinger_factor_coefficient(
    rank: usize,
    degree_part: usize,
    source_p: usize,
    target_h: usize,
    target_y: usize,
) -> Rational {
    let mut poly = BTreeMap::<BiKey, Rational>::new();
    poly.insert(
        (0, 1 + (rank * degree_part) as i32 - source_p as i32),
        Rational::one(),
    );
    poly = multiply_bivariate(
        &poly,
        &numerator_polynomial(degree_part, source_p),
        target_h,
        target_y,
    );
    if poly.is_empty() {
        return Rational::zero();
    }
    for r in 1..=degree_part {
        poly = multiply_bivariate(
            &poly,
            &inverse_linear_polynomial(rank, r, target_h),
            target_h,
            target_y,
        );
        if poly.is_empty() {
            return Rational::zero();
        }
    }
    poly.get(&(target_h, target_y as i32))
        .cloned()
        .unwrap_or_else(Rational::zero)
}

fn numerator_polynomial(degree_part: usize, source_p: usize) -> BTreeMap<BiKey, Rational> {
    let mut out = BTreeMap::new();
    if source_p == 0 {
        out.insert((0, 0), Rational::one());
        return out;
    }
    if degree_part == 0 {
        out.insert((source_p, source_p as i32), Rational::one());
        return out;
    }
    for h_power in 0..=source_p {
        let coeff = binomial_rational(source_p, h_power)
            * Rational::from(degree_part).pow_usize(source_p - h_power);
        out.insert((h_power, h_power as i32), coeff);
    }
    out
}

fn inverse_linear_polynomial(rank: usize, r: usize, max_h: usize) -> BTreeMap<BiKey, Rational> {
    let mut out = BTreeMap::new();
    let r_rat = Rational::from(r);
    for k in 0..=max_h {
        let sign = if k % 2 == 0 {
            Rational::one()
        } else {
            -Rational::one()
        };
        let coeff = sign * binomial_rational(rank + k - 1, k) / r_rat.pow_usize(rank + k);
        out.insert((k, k as i32), coeff);
    }
    out
}

fn multiply_bivariate(
    left: &BTreeMap<BiKey, Rational>,
    right: &BTreeMap<BiKey, Rational>,
    max_h: usize,
    max_y: usize,
) -> BTreeMap<BiKey, Rational> {
    let mut out = BTreeMap::<BiKey, Rational>::new();
    for ((lh, ly), lc) in left {
        for ((rh, ry), rc) in right {
            let h = lh + rh;
            let y = ly + ry;
            if h > max_h || y > max_y as i32 {
                continue;
            }
            let next =
                out.get(&(h, y)).cloned().unwrap_or_else(Rational::zero) + lc.clone() * rc.clone();
            if next.is_zero() {
                out.remove(&(h, y));
            } else {
                out.insert((h, y), next);
            }
        }
    }
    out
}

fn weak_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    fn rec(total: usize, parts: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() + 1 == parts {
            current.push(total);
            out.push(current.clone());
            current.pop();
            return;
        }
        for value in 0..=total {
            current.push(value);
            rec(total - value, parts, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    if parts == 0 {
        if total == 0 {
            out.push(Vec::new());
        }
    } else {
        rec(total, parts, &mut Vec::new(), &mut out);
    }
    out
}

fn bounded_compositions(total: usize, parts: usize, bound_exclusive: usize) -> Vec<Vec<usize>> {
    weak_compositions(total, parts)
        .into_iter()
        .filter(|composition| composition.iter().all(|&part| part < bound_exclusive))
        .collect()
}

fn multinomial_rational(total: usize, parts: impl Iterator<Item = usize>) -> Rational {
    let mut out = factorial_rational(total);
    for part in parts {
        out = out / factorial_rational(part);
    }
    out
}

fn factorial_rational(value: usize) -> Rational {
    let mut out = Rational::one();
    for factor in 2..=value {
        out = out * Rational::from(factor);
    }
    out
}

fn binomial_rational(n: usize, k: usize) -> Rational {
    if k > n {
        return Rational::zero();
    }
    factorial_rational(n) / (factorial_rational(k) * factorial_rational(n - k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CohomologyClass;
    use crate::{tau, ComputeMode};

    #[test]
    fn zinger_three_point_matches_p1_stationary_values() {
        let req = InvariantRequest {
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
        assert_eq!(
            compute(&req).unwrap().value,
            RatFun::from_rational(Rational::new(1, 36))
        );
    }

    #[test]
    fn zinger_four_point_degree_one_line_count() {
        let req = InvariantRequest::new(
            2,
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 1)),
            ],
        );
        assert_eq!(compute(&req).unwrap().value, RatFun::one());
    }

    #[test]
    fn zinger_vanishing_detects_projective_case() {
        let req = InvariantRequest::new(
            2,
            0,
            1,
            vec![
                tau(1, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 1)),
                tau(0, CohomologyClass::h_power(2, 2)),
            ],
        );
        assert_eq!(compute(&req).unwrap().value, RatFun::zero());
    }
}
