//! Hypergeometric I-function coefficients and genus-zero QRR Euler factors.

use super::twist::NegativeSplitBundleTwist;
use crate::core::algebra::{Coeff, Rational};
use crate::core::error::GwError;
use crate::reconstruction::{base_h_power_relation_coeff, HCoeffLaurentSeries, HLaurentSeries};

pub(super) fn base_h_power_relation(
    n: usize,
    base_weights: &[Rational],
) -> Result<Vec<Rational>, GwError> {
    base_h_power_relation_coeff(n, base_weights)
}
fn h_affine_power_mod_relation_coeff<C: Coeff>(
    max_h_power: usize,
    h_coeff: C,
    constant: C,
    exponent: usize,
    h_power_relation: &[C],
) -> Vec<C> {
    let mut out = vec![C::zero(); max_h_power + 1];
    out[0] = C::one();
    for _ in 0..exponent {
        let mut next = vec![C::zero(); max_h_power + 1];
        for h_power in 0..=max_h_power {
            if out[h_power].is_zero() {
                continue;
            }
            if !constant.is_zero() {
                next[h_power] = next[h_power].add(&out[h_power].mul(&constant));
            }
            if !h_coeff.is_zero() {
                if h_power < max_h_power {
                    next[h_power + 1] = next[h_power + 1].add(&out[h_power].mul(&h_coeff));
                } else {
                    for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                        next[reduced_h] =
                            next[reduced_h].add(&out[h_power].mul(&h_coeff).mul(relation_coeff));
                    }
                }
            }
        }
        out = next;
    }
    out
}

pub fn negative_split_i_function_coefficient(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> HLaurentSeries {
    // Non-equivariant hypergeometric coefficient for local P^n.  The numerator
    // is the concave Euler factor for H^1(C,f^*O(-a)); the denominator is the
    // usual projective-space I-function denominator.
    let mut out = HLaurentSeries::one(n);
    for bundle_degree in twist.degrees() {
        // The factors commute, so enumerate m = 0, -1, ..., -ad + 1.
        // Nested ranges preserve the infallible public API without forming the
        // potentially overflowing product a*d or casting it through isize.
        let mut z_offset = Rational::zero();
        for _ in 0..degree {
            for _ in 0..*bundle_degree {
                out = out.multiply_by_linear(-Rational::from(*bundle_degree), -z_offset.clone());
                z_offset += Rational::one();
            }
        }
    }
    for m in 1..=degree {
        let inverse = inverse_h_plus_mz_power(n, m, n + 1);
        out = out.multiply(&inverse);
    }
    out
}

pub fn negative_split_i_function_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<HLaurentSeries> {
    (0..=q_degree)
        .map(|degree| negative_split_i_function_coefficient(n, twist, degree))
        .collect()
}

pub fn projective_equivariant_i_function_coefficient(
    n: usize,
    degree: usize,
    base_weights: &[Rational],
    min_z_power: i32,
) -> Result<HLaurentSeries, GwError> {
    projective_equivariant_i_function_coefficient_coeff(n, degree, base_weights, min_z_power)
}

pub fn projective_i_function_coefficient(n: usize, degree: usize) -> HLaurentSeries {
    let mut out = HLaurentSeries::one(n);
    for m in 1..=degree {
        let inverse = inverse_h_plus_mz_power(n, m, n + 1);
        out = out.multiply(&inverse);
    }
    out
}

pub fn negative_split_equivariant_qrr_euler_factor(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<HLaurentSeries, GwError> {
    negative_split_equivariant_qrr_euler_factor_coeff(n, twist, degree, base_weights, fiber_weights)
}

pub fn negative_split_qrr_euler_factor(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> HLaurentSeries {
    let mut out = HLaurentSeries::one(n);
    for bundle_degree in twist.degrees() {
        let mut z_offset = Rational::zero();
        for _ in 0..degree {
            for _ in 0..*bundle_degree {
                out = out.multiply_by_linear(-Rational::from(*bundle_degree), -z_offset.clone());
                z_offset += Rational::one();
            }
        }
    }
    out
}

fn negative_split_euler_factor_count(
    bundle_degree: usize,
    curve_degree: usize,
) -> Result<usize, GwError> {
    bundle_degree.checked_mul(curve_degree).ok_or_else(|| {
        GwError::UnsupportedInvariant("negative-split Euler-factor count overflow".to_string())
    })
}

fn negative_split_positive_z_degree(
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> Result<i32, GwError> {
    // For O(-a) in degree d the concave factor contains ad affine factors,
    // one with m=0.  Its largest z power is therefore max(ad-1, 0).  Terms
    // this far below the requested output floor can be shifted back into the
    // retained window when the projective denominator is multiplied by QRR.
    let positive_z_degree = twist.degrees().iter().try_fold(
        0usize,
        |total, bundle_degree| -> Result<usize, GwError> {
            let factor_count = negative_split_euler_factor_count(*bundle_degree, degree)?;
            let summand_z_degree = if factor_count == 0 {
                0
            } else {
                factor_count - 1
            };
            total.checked_add(summand_z_degree).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "negative-split Euler-factor z-degree overflow".to_string(),
                )
            })
        },
    )?;
    i32::try_from(positive_z_degree).map_err(|_| {
        GwError::UnsupportedInvariant(
            "negative-split Euler-factor z-degree exceeds i32 range".to_string(),
        )
    })
}

fn negative_split_projective_source_min_z_power(
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    retained_min_z_power: i32,
) -> Result<i32, GwError> {
    let positive_z_degree = negative_split_positive_z_degree(twist, degree)?;
    retained_min_z_power
        .checked_sub(positive_z_degree)
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split I-function source z-window exceeds i32 range".to_string(),
            )
        })
}

pub fn negative_split_equivariant_i_function_coefficient(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    min_z_power: i32,
) -> Result<HLaurentSeries, GwError> {
    negative_split_equivariant_i_function_coefficient_coeff(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
        min_z_power,
    )
}

pub(super) fn negative_split_equivariant_i_function_coefficient_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[C],
    fiber_weights: &[C],
    min_z_power: i32,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    let projective_min_z_power =
        negative_split_projective_source_min_z_power(twist, degree, min_z_power)?;
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let factor = negative_split_equivariant_qrr_euler_factor_coeff(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
    )?;
    let projective = projective_equivariant_i_function_coefficient_coeff(
        n,
        degree,
        base_weights,
        projective_min_z_power,
    )?;
    Ok(factor
        .multiply_mod_relation(&projective, &h_power_relation)
        .truncated_z_below(min_z_power))
}

fn projective_equivariant_i_function_coefficient_coeff<C: Coeff>(
    n: usize,
    degree: usize,
    base_weights: &[C],
    min_z_power: i32,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    let state_space_size = negative_split_state_space_size(n)?;
    if base_weights.len() != state_space_size {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            state_space_size,
            base_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let mut out = HCoeffLaurentSeries::<C>::one(n);
    for m in 1..=degree {
        for weight in base_weights {
            let inverse = inverse_affine_z_laurent_coeff(
                n,
                C::one(),
                weight.neg(),
                C::from_usize(m),
                min_z_power,
                Some(&h_power_relation),
            )?;
            out = out
                .multiply_mod_relation(&inverse, &h_power_relation)
                .truncated_z_below(min_z_power);
        }
    }
    Ok(out.truncated_z_below(min_z_power))
}

fn negative_split_state_space_size(n: usize) -> Result<usize, GwError> {
    n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "negative-split projective state-space size overflow".to_string(),
        )
    })
}

fn negative_split_equivariant_qrr_euler_factor_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[C],
    fiber_weights: &[C],
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    // `HCoeffLaurentSeries` stores z-exponents in i32.  Validate the
    // cumulative numerator degree before multiplying so the public fallible
    // path reports an error instead of eventually overflowing `z_power + 1`.
    negative_split_positive_z_degree(twist, degree)?;
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let mut out = HCoeffLaurentSeries::<C>::one(n);
    for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
        let factor_count = negative_split_euler_factor_count(*bundle_degree, degree)?;
        for z_offset in 0..factor_count {
            out = out.multiply_by_affine_mod_relation(
                C::from_rational(-Rational::from(*bundle_degree)),
                fiber_weight.clone(),
                C::from_rational(-Rational::from(z_offset)),
                &h_power_relation,
            );
        }
    }
    Ok(out)
}

fn inverse_h_plus_mz_power(max_h_power: usize, m: usize, power: usize) -> HLaurentSeries {
    let mut out = HLaurentSeries::zero(max_h_power);
    let m = Rational::from(m);
    for h_power in 0..=max_h_power {
        let coefficient =
            signed_binomial_negative_power(power, h_power) / m.pow_usize(power + h_power);
        out.add_term(h_power, -((power + h_power) as i32), coefficient);
    }
    out
}

fn inverse_affine_z_laurent_coeff<C: Coeff>(
    max_h_power: usize,
    h_coeff: C,
    constant: C,
    z_coeff: C,
    min_z_power: i32,
    h_power_relation: Option<&[C]>,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    if z_coeff.is_zero() {
        return Err(GwError::AlgebraFailure(
            "cannot expand affine inverse at z=infinity with zero z coefficient".to_string(),
        ));
    }
    if min_z_power >= 0 {
        return Ok(HCoeffLaurentSeries::<C>::zero(max_h_power));
    }

    let mut out = HCoeffLaurentSeries::<C>::zero(max_h_power);
    let max_k = min_z_power
        .checked_neg()
        .and_then(|depth| depth.checked_sub(1))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "Laurent expansion z-window exceeds the supported range".to_string(),
            )
        })?;
    for k in 0..=max_k {
        let sign = if k % 2 == 0 {
            C::one()
        } else {
            C::from_rational(-Rational::one())
        };
        let denominator = z_coeff.pow_usize(k + 1);
        if let Some(relation) = h_power_relation {
            let affine_power = h_affine_power_mod_relation_coeff(
                max_h_power,
                h_coeff.clone(),
                constant.clone(),
                k,
                relation,
            );
            for (h_power, coeff) in affine_power.into_iter().enumerate() {
                out.add_term(
                    h_power,
                    -((k + 1) as i32),
                    sign.mul(&coeff).div(&denominator),
                );
            }
        } else {
            for h_power in 0..=max_h_power.min(k) {
                let coeff = sign
                    .mul(&C::from_rational(binomial_rational(k, h_power)))
                    .mul(&constant.pow_usize(k - h_power))
                    .mul(&h_coeff.pow_usize(h_power))
                    .div(&denominator);
                out.add_term(h_power, -((k + 1) as i32), coeff);
            }
        }
    }
    Ok(out)
}

fn signed_binomial_negative_power(power: usize, h_power: usize) -> Rational {
    let sign = if h_power.is_multiple_of(2) {
        Rational::one()
    } else {
        -Rational::one()
    };
    sign * binomial_rational(power + h_power - 1, h_power)
}

fn binomial_rational(n: usize, k: usize) -> Rational {
    if k > n {
        return Rational::zero();
    }
    let k = k.min(n - k);
    let mut out = Rational::one();
    for idx in 0..k {
        out = out * Rational::from(n - idx) / Rational::from(idx + 1);
    }
    out
}
