//! Cohomology-valued Laurent series in a cyclic power basis.

use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{mul_plain_series, QSeries, SeriesMatrix};
use std::collections::BTreeMap;

/// The rational specialization of the generic coefficient Laurent series.
pub type HLaurentSeries = HCoeffLaurentSeries<Rational>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HCoeffLaurentSeries<C = RatFun> {
    max_h_power: usize,
    coeffs: Vec<BTreeMap<i32, C>>,
}

impl<C: Coeff> HCoeffLaurentSeries<C> {
    pub(crate) fn multiply_by_h(&self) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power + 1, *z_power, coeff.clone());
            }
        }
        out
    }

    pub(crate) fn multiply_by_linear(&self, h_coeff: C, z_coeff: C) -> Self {
        self.multiply_by_affine(h_coeff, C::zero(), z_coeff)
    }

    pub(crate) fn multiply_by_affine(&self, h_coeff: C, constant: C, z_coeff: C) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.mul(&constant));
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.mul(&z_coeff));
                }
                if !h_coeff.is_zero() && h_power < self.max_h_power {
                    out.add_term(h_power + 1, *z_power, coeff.mul(&h_coeff));
                }
            }
        }
        out
    }

    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power - left_h {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        out.add_term(
                            left_h + right_h,
                            left_z + right_z,
                            left_coeff.mul(right_coeff),
                        );
                    }
                }
            }
        }
        out
    }

    pub(crate) fn zero(max_h_power: usize) -> Self {
        Self {
            max_h_power,
            coeffs: vec![BTreeMap::new(); max_h_power + 1],
        }
    }

    pub(crate) fn one(max_h_power: usize) -> Self {
        let mut out = Self::zero(max_h_power);
        out.coeffs[0].insert(0, C::one());
        out
    }

    pub(crate) fn coefficient(&self, h_power: usize, z_power: i32) -> C {
        self.coeffs
            .get(h_power)
            .and_then(|terms| terms.get(&z_power))
            .cloned()
            .unwrap_or_else(C::zero)
    }

    pub(crate) fn max_h_power(&self) -> usize {
        self.max_h_power
    }

    pub(crate) fn terms_at_h_power(&self, h_power: usize) -> Option<&BTreeMap<i32, C>> {
        self.coeffs.get(h_power)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.coeffs.iter().all(BTreeMap::is_empty)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = self.clone();
        for h_power in 0..=rhs.max_h_power {
            for (z_power, coeff) in &rhs.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.clone());
            }
        }
        out
    }

    pub(crate) fn scale(&self, scalar: C) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.max_h_power);
        }
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.mul(&scalar));
            }
        }
        out
    }

    pub(crate) fn shift_z(&self, shift: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, z_power + shift, coeff.clone());
            }
        }
        out
    }

    pub(crate) fn multiply_mod_relation(&self, rhs: &Self, h_power_relation: &[C]) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let basis_powers = h_basis_powers_mod_relation_coeff(self.max_h_power, h_power_relation);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        let scalar = left_coeff.mul(right_coeff);
                        if scalar.is_zero() {
                            continue;
                        }
                        for (reduced_h, reduced_coeff) in
                            basis_powers[left_h + right_h].iter().enumerate()
                        {
                            if reduced_coeff.is_zero() {
                                continue;
                            }
                            out.add_term(reduced_h, left_z + right_z, scalar.mul(reduced_coeff));
                        }
                    }
                }
            }
        }
        out
    }

    pub(crate) fn multiply_by_affine_mod_relation(
        &self,
        h_coeff: C,
        constant: C,
        z_coeff: C,
        h_power_relation: &[C],
    ) -> Self {
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.mul(&constant));
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.mul(&z_coeff));
                }
                if !h_coeff.is_zero() {
                    if h_power < self.max_h_power {
                        out.add_term(h_power + 1, *z_power, coeff.mul(&h_coeff));
                    } else {
                        for (reduced_h, reduced_coeff) in h_power_relation.iter().enumerate() {
                            if !reduced_coeff.is_zero() {
                                out.add_term(
                                    reduced_h,
                                    *z_power,
                                    coeff.mul(&h_coeff).mul(reduced_coeff),
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    }

    pub(crate) fn truncated_z_below(&self, min_z_power: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if *z_power >= min_z_power {
                    out.add_term(h_power, *z_power, coeff.clone());
                }
            }
        }
        out
    }

    pub(crate) fn add_term(&mut self, h_power: usize, z_power: i32, coeff: C) {
        if coeff.is_zero() || h_power > self.max_h_power {
            return;
        }
        let terms = &mut self.coeffs[h_power];
        let next = terms
            .get(&z_power)
            .cloned()
            .unwrap_or_else(C::zero)
            .add(&coeff);
        if next.is_zero() {
            terms.remove(&z_power);
        } else {
            terms.insert(z_power, next);
        }
    }
}

pub(crate) fn h_basis_powers_mod_relation_coeff<C: Coeff>(
    max_h_power: usize,
    h_power_relation: &[C],
) -> Vec<Vec<C>> {
    let mut powers = vec![vec![C::zero(); max_h_power + 1]; 2 * max_h_power + 1];
    powers[0][0] = C::one();
    for power in 1..=2 * max_h_power {
        for h_power in 0..max_h_power {
            powers[power][h_power + 1] =
                powers[power][h_power + 1].add(&powers[power - 1][h_power]);
        }
        let top_coeff = powers[power - 1][max_h_power].clone();
        if !top_coeff.is_zero() {
            for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                powers[power][reduced_h] =
                    powers[power][reduced_h].add(&top_coeff.mul(relation_coeff));
            }
        }
    }
    powers
}

pub(crate) fn base_h_power_relation_coeff<C: Coeff>(
    n: usize,
    base_weights: &[C],
) -> Result<Vec<C>, GwError> {
    let state_space_size = cyclic_state_space_size(n)?;
    if base_weights.len() != state_space_size {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            state_space_size,
            base_weights.len()
        )));
    }
    let mut coefficients = vec![C::one()];
    for weight in base_weights {
        let mut next = vec![C::zero(); coefficients.len() + 1];
        for (power, coeff) in coefficients.iter().enumerate() {
            next[power] = next[power].sub(&coeff.mul(weight));
            next[power + 1] = next[power + 1].add(coeff);
        }
        coefficients = next;
    }
    let leading = coefficients[state_space_size].clone();
    if leading.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    Ok((0..=n)
        .map(|power| coefficients[power].neg().div(&leading))
        .collect())
}

fn cyclic_state_space_size(n: usize) -> Result<usize, GwError> {
    n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("cyclic cohomology state-space size overflow".to_string())
    })
}

pub(crate) fn h_coeff_laurent_columns_to_laurent_matrix<C: Coeff>(
    n: usize,
    q_degree: usize,
    columns: &[Vec<HCoeffLaurentSeries<C>>],
) -> BTreeMap<i32, SeriesMatrix<C>> {
    let size = n + 1;
    let mut by_power: BTreeMap<i32, Vec<Vec<Vec<C>>>> = BTreeMap::new();
    for (col, column_series) in columns.iter().enumerate() {
        for (degree, h_series) in column_series.iter().enumerate().take(q_degree + 1) {
            for h_power in 0..=n {
                for (z_power, coeff) in &h_series.coeffs[h_power] {
                    let entries = by_power
                        .entry(*z_power)
                        .or_insert_with(|| vec![vec![vec![C::zero(); q_degree + 1]; size]; size]);
                    entries[h_power][col][degree] = entries[h_power][col][degree].add(coeff);
                }
            }
        }
    }

    by_power
        .into_iter()
        .map(|(z_power, entries)| {
            let matrix = SeriesMatrix::from_entries(
                entries
                    .into_iter()
                    .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                    .collect(),
            );
            (z_power, matrix)
        })
        .collect()
}

pub(crate) fn h_laurent_columns_to_laurent_matrix(
    n: usize,
    q_degree: usize,
    columns: &[Vec<HLaurentSeries>],
) -> BTreeMap<i32, SeriesMatrix> {
    let size = n + 1;
    let mut by_power: BTreeMap<i32, Vec<Vec<Vec<RatFun>>>> = BTreeMap::new();
    for (col, column_series) in columns.iter().enumerate() {
        for (degree, h_series) in column_series.iter().enumerate().take(q_degree + 1) {
            for h_power in 0..=n {
                for (z_power, coeff) in &h_series.coeffs[h_power] {
                    let entries = by_power.entry(*z_power).or_insert_with(|| {
                        vec![vec![vec![RatFun::zero(); q_degree + 1]; size]; size]
                    });
                    entries[h_power][col][degree] = entries[h_power][col][degree].clone()
                        + RatFun::from_rational(coeff.clone());
                }
            }
        }
    }

    by_power
        .into_iter()
        .map(|(z_power, entries)| {
            let matrix = SeriesMatrix::from_entries(
                entries
                    .into_iter()
                    .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                    .collect(),
            );
            (z_power, matrix)
        })
        .collect()
}

pub(crate) fn exp_minus_h_mirror_over_z_coefficients(
    n: usize,
    mirror: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let mut exponent = vec![HLaurentSeries::zero(n); max_degree + 1];
    for degree in 1..=max_degree {
        let coeff = mirror.get(degree).cloned().unwrap_or_else(Rational::zero);
        if !coeff.is_zero() && n >= 1 {
            exponent[degree].add_term(1, -1, -coeff);
        }
    }

    let mut out = vec![HLaurentSeries::zero(n); max_degree + 1];
    out[0] = HLaurentSeries::one(n);
    for degree in 1..=max_degree {
        let mut sum = HLaurentSeries::zero(n);
        for split in 1..=degree {
            if exponent[split].is_empty() {
                continue;
            }
            let term = exponent[split]
                .multiply(&out[degree - split])
                .scale(Rational::from(split));
            sum = sum.add(&term);
        }
        out[degree] = sum.scale(Rational::one() / Rational::from(degree));
    }
    out
}

pub(crate) fn multiply_h_laurent_q_series(
    left: &[HLaurentSeries],
    right: &[HLaurentSeries],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term = left[left_degree].multiply(&right[right_degree]);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

pub(crate) fn multiply_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    left: &[HCoeffLaurentSeries<C>],
    right: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term =
                left[left_degree].multiply_mod_relation(&right[right_degree], h_power_relation);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

pub(crate) fn full_vector_mirror_gauge_coefficients_coeff<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let mut exponent = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    let mut gauge = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    gauge[0] = HCoeffLaurentSeries::<C>::one(n);

    for degree in 1..=max_degree {
        let mut known_gauge = HCoeffLaurentSeries::<C>::zero(n);
        for split in 1..degree {
            if exponent[split].is_empty() {
                continue;
            }
            let term = exponent[split]
                .multiply_mod_relation(&gauge[degree - split], h_power_relation)
                .scale(C::from_usize(split));
            known_gauge = known_gauge.add(&term);
        }
        known_gauge = known_gauge.scale(C::one().div(&C::from_usize(degree)));
        gauge[degree] = known_gauge;

        let mut gauged_degree = HCoeffLaurentSeries::<C>::zero(n);
        for split in 0..=degree {
            let term = gauge[split]
                .multiply_mod_relation(&i_coefficients[degree - split], h_power_relation);
            gauged_degree = gauged_degree.add(&term);
        }
        let tau = z_power_part_coeff(&gauged_degree, -1);
        exponent[degree] = tau.shift_z(-1).scale(C::from_rational(-Rational::one()));
        gauge[degree] = gauge[degree].add(&exponent[degree]);
    }

    gauge
}

pub(crate) fn z_power_part_coeff<C: Coeff>(
    series: &HCoeffLaurentSeries<C>,
    z_power: i32,
) -> HCoeffLaurentSeries<C> {
    let mut out = HCoeffLaurentSeries::<C>::zero(series.max_h_power());
    for h_power in 0..=series.max_h_power() {
        let coeff = series.coefficient(h_power, z_power);
        if !coeff.is_zero() {
            out.add_term(h_power, 0, coeff);
        }
    }
    out
}

pub(crate) fn compose_h_laurent_q_series(
    series: &[HLaurentSeries],
    input: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    compose_h_laurent_q_series_coeff(series, input, max_degree)
}

pub(crate) fn compose_h_laurent_q_series_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    input: &[C],
    max_degree: usize,
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    let mut power = vec![C::zero(); max_degree + 1];
    power[0] = C::one();
    for source_degree in 0..=max_degree {
        for target_degree in 0..=max_degree {
            if power[target_degree].is_zero() {
                continue;
            }
            let term = series[source_degree].scale(power[target_degree].clone());
            out[target_degree] = out[target_degree].add(&term);
        }
        power = mul_plain_series(&power, input, max_degree);
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series(
    series: &[HLaurentSeries],
) -> Vec<HLaurentSeries> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series.first().map(HLaurentSeries::max_h_power).unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_h());
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(Rational::from(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_affine_mod_relation(
            C::one(),
            C::zero(),
            C::zero(),
            h_power_relation,
        ));
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(C::from_usize(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series_mod_relation(
    series: &[HLaurentSeries],
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    quantum_derivative_h_laurent_q_series_mod_relation_coeff(series, h_power_relation)
}
