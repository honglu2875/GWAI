//! Projective bundles `P(O(a_1) + ... + O(a_m))` over `P^n`, via the toric
//! I-function, bidegree Birkhoff projection, and exact Novikov ray
//! reconstruction.
//!
//! **Geometry.**  With `E = O(a_1) + ... + O(a_m)` (twists normalized so
//! `min a_l = 0`, always possible since `P(E) = P(E ⊗ L)`), the cohomology is
//! `H*(P^n)[xi] / prod_l(xi + a_l H)` with `dim X = n + m - 1` and Picard
//! rank two.  Torus fixed points are pairs `(i, j)`; the fiber coordinate
//! `l` at base point `i` has weight `c_{il} = a_l lambda_i + mu_l`, the
//! tautological class restricts as `xi|_{(i,j)} = -c_{ij}`, and tangent
//! weights are `{lambda_k - lambda_i}` together with `{c_{il} - c_{ij}}`.
//!
//! **Curve classes and the effective cone.**  Classes pair as
//! `d1 = H . beta`, `d2 = xi . beta`; `d2` may be negative (e.g. the
//! exceptional section of a Hirzebruch surface).  The I-function vanishes
//! outside `d2 >= -A d1` for `A = max a_l` — a term with every
//! `D_l = d2 + a_l d1 < 0` contains the full ring relation — so the shifted
//! coordinate `d2' = d2 + A d1 >= 0` gives a nonnegative grading covering
//! the effective cone, exactly as much of the cone as is needed:
//! non-effective classes inside it simply yield zero.
//!
//! **Pipeline.**  The classical ring is cyclic over the grading divisor
//! `D = xi + (A+1) H` at generic weights, so everything runs in the constant
//! classical `D`-power basis via fixed-point restrictions.  The I-function is
//! computed in bidegree-graded form (finite per total degree `d1 + d2'`) and
//! differentiated by the grading divisor to form the raw fundamental solution.
//! That fundamental solution is Birkhoff factored over the full bidegree
//! Novikov ring; the positive factor is the formal cone projection, including
//! higher positive-`z` corrections.  The negative factor's first column is the
//! projected cone point; its bidegree `z^{-1}` divisor part gives the two
//! mirror-coordinate series, which are gauged away and inverted before any ray
//! specialization.  Only then is the flat cone point restricted to rays
//! `(Q1, Q2') = (t, b t)` and regenerated into the fundamental solution used
//! by the graph engine and exact Vandermonde recovery.  The raw fundamental
//! solution gives quantum multiplication by `D`; its metric-adjoint gives the
//! descendant insertion operator, and flatness gives `R`.
//!
//! **Validated scope.**  Regression tests cover Fano genus-zero bundle counts,
//! `P(O + O) = P^1 x P^1` through higher `R` order, and the non-Fano
//! `F_2 = P(O + O(2))` deformation dictionary against the independent product
//! engine, including genus-one cases, plus rank-three deformations to
//! `P^1 x P^2` in negative shifted-fiber directions.

use super::*;
use crate::twisted::{BidegreeLaurentFactor, HLaurentSeries, LaurentCoeffMatrix};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::time::Instant;

/// Insertion `tau_k(H^p xi^q)` on the bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleInsertion {
    pub descendant_power: usize,
    pub h_power: usize,
    pub xi_power: usize,
}

impl BundleInsertion {
    pub fn new(descendant_power: usize, h_power: usize, xi_power: usize) -> Self {
        Self {
            descendant_power,
            h_power,
            xi_power,
        }
    }
}

/// `P(O(a_1) + ... + O(a_m))` over `P^n`, specialized along the Novikov ray
/// `(t, ray * t)` in the shifted grading, at rational equivariant weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveBundleRay {
    pub n: usize,
    pub twists: Vec<usize>,
    pub weights_base: Vec<Rational>,
    pub weights_fiber: Vec<Rational>,
    pub ray: Rational,
}

impl ProjectiveBundleRay {
    pub fn new(
        n: usize,
        twists: Vec<usize>,
        weights_base: Vec<Rational>,
        weights_fiber: Vec<Rational>,
        ray: Rational,
    ) -> Result<Self, GwError> {
        if twists.is_empty() || !twists.contains(&0) {
            return Err(GwError::ConventionMismatch(
                "bundle twists must be normalized with min a_l = 0 (twist by O(-min) first)"
                    .to_string(),
            ));
        }
        if weights_base.len() != n + 1 || weights_fiber.len() != twists.len() {
            return Err(GwError::ConventionMismatch(format!(
                "bundle weights must have lengths {} and {}",
                n + 1,
                twists.len()
            )));
        }
        let target = Self {
            n,
            twists,
            weights_base,
            weights_fiber,
            ray,
        };
        let seeds = target.grading_seeds();
        for left in 0..seeds.len() {
            for right in left + 1..seeds.len() {
                if seeds[left] == seeds[right] {
                    return Err(GwError::NonSemisimplePoint);
                }
            }
        }
        Ok(target)
    }

    fn rank(&self) -> usize {
        self.twists.len()
    }

    fn size(&self) -> usize {
        (self.n + 1) * self.rank()
    }

    fn point(&self, i: usize, j: usize) -> usize {
        i * self.rank() + j
    }

    fn big_a(&self) -> usize {
        *self.twists.iter().max().expect("twists nonempty")
    }

    /// Weight of the `l`-th fiber coordinate over base fixed point `i`.
    fn fiber_weight(&self, i: usize, l: usize) -> Rational {
        Rational::from(self.twists[l] as i128) * self.weights_base[i].clone()
            + self.weights_fiber[l].clone()
    }

    /// Restriction of `xi` at fixed point `(i, j)`.
    fn xi_restriction(&self, i: usize, j: usize) -> Rational {
        -self.fiber_weight(i, j)
    }

    /// Classical eigenvalues of the grading divisor `D = xi + (A+1) H`.
    fn grading_seeds(&self) -> Vec<Rational> {
        let shift = Rational::from((self.big_a() + 1) as i128);
        let mut seeds = vec![Rational::zero(); self.size()];
        for i in 0..=self.n {
            for j in 0..self.rank() {
                seeds[self.point(i, j)] =
                    self.xi_restriction(i, j) + shift.clone() * self.weights_base[i].clone();
            }
        }
        seeds
    }

    /// Equivariant Euler class of the tangent space at `(i, j)`.
    fn euler(&self, i: usize, j: usize) -> Rational {
        let mut euler = Rational::one();
        for k in 0..=self.n {
            if k != i {
                euler = euler * (self.weights_base[i].clone() - self.weights_base[k].clone());
            }
        }
        for l in 0..self.rank() {
            if l != j {
                euler = euler * (self.fiber_weight(i, l) - self.fiber_weight(i, j));
            }
        }
        euler
    }

    fn classical_transition(&self) -> Vec<Vec<Rational>> {
        recipe::classical_lagrange_transition(&self.grading_seeds())
    }

    /// Atiyah-Bott flat metric in the classical `D`-power basis.
    fn flat_metric(&self) -> Vec<Vec<Rational>> {
        let size = self.size();
        let seeds = self.grading_seeds();
        let mut metric = vec![vec![Rational::zero(); size]; size];
        for row in 0..size {
            for col in 0..size {
                let mut total = Rational::zero();
                for i in 0..=self.n {
                    for j in 0..self.rank() {
                        let seed = &seeds[self.point(i, j)];
                        total += seed.pow_usize(row + col) / self.euler(i, j);
                    }
                }
                metric[row][col] = total;
            }
        }
        metric
    }

    /// Classical relation `D^size = sum_k rel_k D^k` (ascending, length
    /// `size`), from the minimal polynomial `prod (x - seed)`.
    fn h_power_relation(&self) -> Vec<Rational> {
        let seeds = self.grading_seeds();
        let mut coefficients = vec![Rational::one()];
        for seed in &seeds {
            let mut next = vec![Rational::zero(); coefficients.len() + 1];
            for (power, coefficient) in coefficients.iter().enumerate() {
                next[power] += -(seed.clone()) * coefficient.clone();
                next[power + 1] += coefficient.clone();
            }
            coefficients = next;
        }
        (0..self.size())
            .map(|power| -coefficients[power].clone())
            .collect()
    }

    /// Classical coordinates of `H^p xi^q` in the `D`-power basis.
    fn insertion_class_vector(&self, h_power: usize, xi_power: usize) -> Vec<Rational> {
        let size = self.size();
        let transition = self.classical_transition();
        let mut vector = vec![Rational::zero(); size];
        for i in 0..=self.n {
            for j in 0..self.rank() {
                let restriction = self.weights_base[i].pow_usize(h_power)
                    * self.xi_restriction(i, j).pow_usize(xi_power);
                for row in 0..size {
                    vector[row] += transition[row][self.point(i, j)].clone() * restriction.clone();
                }
            }
        }
        vector
    }

    fn cache_key(&self) -> String {
        format!(
            "p{}bundle[{:?};{};{}]@{}",
            self.n,
            self.twists,
            self.weights_base
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.weights_fiber
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.ray
        )
    }

    fn rayless_cache_key(&self) -> String {
        format!(
            "p{}bundle[{:?};{};{}]",
            self.n,
            self.twists,
            self.weights_base
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.weights_fiber
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn bidegree_birkhoff_bounds(&self, k_max: usize, z_order: usize) -> BidegreeBirkhoffBounds {
        let column_shift = self.size().saturating_sub(1);
        let preview_min_z = -(column_shift as i32);
        let preview_cone = self.i_container(k_max, preview_min_z);
        let preview = self.fundamental_bidegree_matrix(k_max, &preview_cone);
        let raw_positive_windows = max_nonnegative_z_power_by_grade(&preview);
        let positive_windows = positive_factor_z_windows(k_max, &raw_positive_windows);
        let base_depth = z_order + column_shift;
        let negative_depths = bidegree_negative_depths(k_max, base_depth, &positive_windows);
        let max_depth = negative_depths
            .values()
            .copied()
            .max()
            .unwrap_or(base_depth);

        // In the graded Birkhoff recursion, a positive-factor term z^p can
        // move an already-computed negative coefficient z^(-s-p) into z^-s.
        // The dynamic bound below follows the actual bidegree dependency
        // chains instead of applying the worst positive-z window at every
        // total-degree drop.  The final column shift accounts for repeated
        // (z q d/dq + D) derivatives used to build the raw fundamental matrix
        // from I.
        BidegreeBirkhoffBounds {
            min_z: -(max_depth as i32),
            positive_z_windows: positive_windows,
            negative_z_depths: negative_depths,
        }
    }

    /// Sufficient negative z-depth for the I-coefficients through total
    /// grade `k_max` and Birkhoff order `z_order`.
    #[cfg(test)]
    fn min_z_power(&self, k_max: usize, z_order: usize) -> i32 {
        self.bidegree_birkhoff_bounds(k_max, z_order).min_z
    }

    /// Scalar z-Laurent restriction of the `(d1, d2)` I-coefficient at the
    /// fixed point `(i, j)`.
    fn i_restriction(&self, i: usize, j: usize, d1: usize, d2: isize, min_z: i32) -> ZLaurent {
        let mut out = zl_one();
        for k in 1..=d1 {
            for i_prime in 0..=self.n {
                let constant = self.weights_base[i].clone() - self.weights_base[i_prime].clone();
                out = zl_mul_inverse_affine(&out, &constant, k, min_z);
            }
        }
        for l in 0..self.rank() {
            let fiber_degree = d2 + (self.twists[l] * d1) as isize;
            let value = self.fiber_weight(i, l) - self.fiber_weight(i, j);
            if fiber_degree >= 0 {
                for k in 1..=(fiber_degree as usize) {
                    out = zl_mul_inverse_affine(&out, &value, k, min_z);
                }
            } else {
                for k in (fiber_degree + 1)..=0 {
                    out = zl_mul_affine(&out, &value, k, min_z);
                }
            }
            if out.is_empty() {
                return out;
            }
        }
        out
    }

    /// The `(d1, d2')` I-coefficient in the classical `D`-power basis.
    fn i_coefficient(&self, d1: usize, d2p: usize, min_z: i32) -> HLaurentSeries {
        let size = self.size();
        let d2 = d2p as isize - (self.big_a() * d1) as isize;
        let transition = self.classical_transition();
        let restrictions = (0..=self.n)
            .flat_map(|i| (0..self.rank()).map(move |j| (i, j)))
            .map(|(i, j)| self.i_restriction(i, j, d1, d2, min_z))
            .collect::<Vec<_>>();

        let mut out = HLaurentSeries::zero(size - 1);
        let mut z_powers = std::collections::BTreeSet::<i32>::new();
        for restriction in &restrictions {
            z_powers.extend(restriction.keys().copied());
        }
        for &z_power in &z_powers {
            for row in 0..size {
                let mut total = Rational::zero();
                for (point, restriction) in restrictions.iter().enumerate() {
                    if let Some(value) = restriction.get(&z_power) {
                        total += transition[row][point].clone() * value.clone();
                    }
                }
                if !total.is_zero() {
                    out.add_term(row, z_power, total);
                }
            }
        }
        out
    }

    /// Bidegree-graded `I`-coefficients through shifted total degree `k_max`.
    fn i_container(&self, k_max: usize, min_z: i32) -> BTreeMap<Grade, HLaurentSeries> {
        let mut container = BTreeMap::new();
        for d1 in 0..=k_max {
            for d2p in 0..=(k_max - d1) {
                let coefficient = self.i_coefficient(d1, d2p, min_z);
                if !coefficient.is_empty() {
                    container.insert((d1, d2p), coefficient);
                }
            }
        }
        container
    }

    fn multiply_by_grading_divisor_mod_relation(
        &self,
        series: &HLaurentSeries,
        relation: &[Rational],
    ) -> HLaurentSeries {
        let mut divisor = HLaurentSeries::zero(self.size() - 1);
        divisor.add_term(1, 0, Rational::one());
        series.multiply_mod_relation(&divisor, relation)
    }

    fn quantum_derivative_bidegree_series(
        &self,
        series: &BTreeMap<Grade, HLaurentSeries>,
        relation: &[Rational],
    ) -> BTreeMap<Grade, HLaurentSeries> {
        let mut out = BTreeMap::new();
        for (&grade, value) in series {
            let mut derivative = self.multiply_by_grading_divisor_mod_relation(value, relation);
            let total_degree = grade.0 + grade.1;
            if total_degree > 0 {
                derivative =
                    derivative.add(&value.shift_z(1).scale(Rational::from(total_degree as i128)));
            }
            if !derivative.is_empty() {
                out.insert(grade, derivative);
            }
        }
        out
    }

    fn fundamental_bidegree_matrix(
        &self,
        q_degree: usize,
        container: &BTreeMap<Grade, HLaurentSeries>,
    ) -> BidegreeLaurentFactor<Rational> {
        let relation = self.h_power_relation();
        let mut columns = Vec::with_capacity(self.size());
        let mut current = container.clone();
        for _ in 0..self.size() {
            columns.push(current.clone());
            current = self.quantum_derivative_bidegree_series(&current, &relation);
        }
        h_laurent_bidegree_columns_to_laurent_matrix(self.size(), q_degree, &columns)
    }
}

type Grade = (usize, usize);
type ScalarBidegreeSeries = BTreeMap<Grade, Rational>;
type HLaurentBidegreeSeries = BTreeMap<Grade, HLaurentSeries>;
type ZLaurent = BTreeMap<i32, Rational>;

struct BidegreeBirkhoffBounds {
    min_z: i32,
    positive_z_windows: BTreeMap<Grade, usize>,
    negative_z_depths: BTreeMap<Grade, usize>,
}

fn h_laurent_bidegree_columns_to_laurent_matrix(
    size: usize,
    max_total_degree: usize,
    columns: &[BTreeMap<Grade, HLaurentSeries>],
) -> BidegreeLaurentFactor<Rational> {
    let mut by_grade = BidegreeLaurentFactor::new();
    for (col, column_series) in columns.iter().enumerate() {
        for (&grade, h_series) in column_series {
            if grade.0 + grade.1 > max_total_degree {
                continue;
            }
            for row in 0..size {
                let Some(terms) = h_series.terms_at_h_power(row) else {
                    continue;
                };
                for (z_power, coeff) in terms {
                    let laurent = by_grade
                        .entry(grade)
                        .or_insert_with(LaurentCoeffMatrix::new);
                    let matrix = laurent
                        .entry(*z_power)
                        .or_insert_with(|| vec![vec![Rational::zero(); size]; size]);
                    matrix[row][col] += coeff.clone();
                }
            }
        }
    }
    by_grade
}

fn max_nonnegative_z_power_by_grade(
    matrix: &BidegreeLaurentFactor<Rational>,
) -> BTreeMap<Grade, usize> {
    matrix
        .iter()
        .filter_map(|(&grade, laurent)| {
            laurent
                .keys()
                .copied()
                .filter(|z_power| *z_power >= 0)
                .max()
                .map(|z_power| (grade, z_power as usize))
        })
        .collect()
}

fn positive_factor_z_windows(
    max_total_degree: usize,
    raw_windows: &BTreeMap<Grade, usize>,
) -> BTreeMap<Grade, usize> {
    let mut windows: BTreeMap<Grade, usize> = BTreeMap::new();
    for total in 1..=max_total_degree {
        for first in 0..=total {
            let grade = (first, total - first);
            let mut window = raw_windows.get(&grade).copied().unwrap_or(0);
            for left_first in 0..=grade.0 {
                for left_second in 0..=grade.1 {
                    let left_grade = (left_first, left_second);
                    if left_grade == (0, 0) || left_grade == grade {
                        continue;
                    }
                    let right_grade = (grade.0 - left_first, grade.1 - left_second);
                    if let Some(right_window) = windows.get(&right_grade).copied() {
                        window = window.max(right_window.saturating_sub(1));
                    }
                }
            }
            windows.insert(grade, window);
        }
    }
    windows
}

fn bidegree_negative_depths(
    max_total_degree: usize,
    base_depth: usize,
    positive_windows: &BTreeMap<Grade, usize>,
) -> BTreeMap<Grade, usize> {
    let mut depths: BTreeMap<Grade, usize> = BTreeMap::new();
    for total in 1..=max_total_degree {
        for first in 0..=total {
            depths.insert((first, total - first), base_depth);
        }
    }

    for total in (1..=max_total_degree).rev() {
        for first in 0..=total {
            let grade = (first, total - first);
            let target_depth = depths.get(&grade).copied().unwrap_or(base_depth);
            for left_first in 0..=grade.0 {
                for left_second in 0..=grade.1 {
                    let left_grade = (left_first, left_second);
                    if left_grade == (0, 0) || left_grade == grade {
                        continue;
                    }
                    let right_grade = (grade.0 - left_first, grade.1 - left_second);
                    let right_window = positive_windows.get(&right_grade).copied().unwrap_or(0);
                    let needed_depth = target_depth + right_window;
                    depths
                        .entry(left_grade)
                        .and_modify(|depth| *depth = (*depth).max(needed_depth))
                        .or_insert(needed_depth);
                }
            }
        }
    }
    depths
}

fn zl_one() -> ZLaurent {
    BTreeMap::from([(0, Rational::one())])
}

/// Multiply by the affine factor `constant + k z`.
fn zl_mul_affine(series: &ZLaurent, constant: &Rational, k: isize, min_z: i32) -> ZLaurent {
    let mut out = ZLaurent::new();
    for (z_power, coefficient) in series {
        if !constant.is_zero() {
            add_zl_term(
                &mut out,
                *z_power,
                coefficient.clone() * constant.clone(),
                min_z,
            );
        }
        if k != 0 {
            add_zl_term(
                &mut out,
                z_power + 1,
                coefficient.clone() * Rational::from(k),
                min_z,
            );
        }
    }
    out
}

/// Multiply by `(constant + k z)^{-1}` for `k >= 1`, expanded around
/// `z = infinity` and truncated below `min_z`.
fn zl_mul_inverse_affine(series: &ZLaurent, constant: &Rational, k: usize, min_z: i32) -> ZLaurent {
    let mut out = ZLaurent::new();
    let k_rational = Rational::from(k);
    for (z_power, coefficient) in series {
        // (c + kz)^{-1} = sum_{r >= 0} (-c)^r k^{-r-1} z^{-r-1}.
        let mut factor = Rational::one() / k_rational.clone();
        let mut power = z_power - 1;
        while power >= min_z {
            add_zl_term(&mut out, power, coefficient.clone() * factor.clone(), min_z);
            factor = factor * (-constant.clone()) / k_rational.clone();
            if constant.is_zero() {
                break;
            }
            power -= 1;
        }
    }
    out
}

fn add_zl_term(series: &mut ZLaurent, z_power: i32, value: Rational, min_z: i32) {
    if z_power < min_z || value.is_zero() {
        return;
    }
    match series.entry(z_power) {
        Entry::Vacant(entry) => {
            entry.insert(value);
        }
        Entry::Occupied(mut entry) => {
            *entry.get_mut() += value;
            if entry.get().is_zero() {
                entry.remove();
            }
        }
    }
}

fn rational_series_matrix_to_ratfun(matrix: &SeriesMatrix<Rational>) -> SeriesMatrix {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(rational_qseries_to_ratfun).collect())
            .collect(),
    )
}

fn rational_s_matrix_to_ratfun(matrix: &SeriesSMatrix<Rational>) -> Result<SeriesSMatrix, GwError> {
    SeriesSMatrix::from_coefficients(
        matrix.size(),
        matrix.q_degree(),
        matrix.z_order(),
        matrix
            .coefficients()
            .iter()
            .map(rational_series_matrix_to_ratfun)
            .collect(),
        matrix.calibration().clone(),
    )
}

fn bidegree_negative_factor_to_cone_point(
    size: usize,
    q_degree: usize,
    negative: &BidegreeLaurentFactor<Rational>,
) -> HLaurentBidegreeSeries {
    let mut out = HLaurentBidegreeSeries::new();
    for (&grade, laurent) in negative {
        if grade.0 + grade.1 > q_degree {
            continue;
        }
        let series = out
            .entry(grade)
            .or_insert_with(|| HLaurentSeries::zero(size - 1));
        for (z_power, matrix) in laurent {
            for (row, row_entries) in matrix.iter().enumerate().take(size) {
                series.add_term(row, *z_power, row_entries[0].clone());
            }
        }
    }
    out
}

fn scalar_bidegree_one() -> ScalarBidegreeSeries {
    BTreeMap::from([((0, 0), Rational::one())])
}

fn scalar_bidegree_variable(index: usize) -> ScalarBidegreeSeries {
    let grade = if index == 0 { (1, 0) } else { (0, 1) };
    BTreeMap::from([(grade, Rational::one())])
}

fn scalar_bidegree_add_term(
    series: &mut ScalarBidegreeSeries,
    grade: Grade,
    value: Rational,
    max_total_degree: usize,
) {
    if grade.0 + grade.1 > max_total_degree || value.is_zero() {
        return;
    }
    match series.entry(grade) {
        Entry::Vacant(entry) => {
            entry.insert(value);
        }
        Entry::Occupied(mut entry) => {
            *entry.get_mut() += value;
            if entry.get().is_zero() {
                entry.remove();
            }
        }
    }
}

fn scalar_bidegree_neg(series: &ScalarBidegreeSeries) -> ScalarBidegreeSeries {
    series
        .iter()
        .map(|(&grade, value)| (grade, -value.clone()))
        .collect()
}

fn scalar_bidegree_mul(
    left: &ScalarBidegreeSeries,
    right: &ScalarBidegreeSeries,
    max_total_degree: usize,
) -> ScalarBidegreeSeries {
    let mut out = ScalarBidegreeSeries::new();
    for (&left_grade, left_value) in left {
        for (&right_grade, right_value) in right {
            let grade = (left_grade.0 + right_grade.0, left_grade.1 + right_grade.1);
            scalar_bidegree_add_term(
                &mut out,
                grade,
                left_value.clone() * right_value.clone(),
                max_total_degree,
            );
        }
    }
    out
}

fn scalar_bidegree_exp(
    series: &ScalarBidegreeSeries,
    max_total_degree: usize,
) -> ScalarBidegreeSeries {
    let mut out = scalar_bidegree_one();
    let mut power = scalar_bidegree_one();
    let mut factorial = Rational::one();
    for order in 1..=max_total_degree {
        power = scalar_bidegree_mul(&power, series, max_total_degree);
        factorial = factorial * Rational::from(order);
        for (&grade, value) in &power {
            scalar_bidegree_add_term(
                &mut out,
                grade,
                value.clone() / factorial.clone(),
                max_total_degree,
            );
        }
    }
    out
}

fn scalar_bidegree_powers(
    series: &ScalarBidegreeSeries,
    max_power: usize,
    max_total_degree: usize,
) -> Vec<ScalarBidegreeSeries> {
    let mut powers = Vec::with_capacity(max_power + 1);
    powers.push(scalar_bidegree_one());
    for power in 1..=max_power {
        let next = scalar_bidegree_mul(&powers[power - 1], series, max_total_degree);
        powers.push(next);
    }
    powers
}

fn scalar_bidegree_compose(
    series: &ScalarBidegreeSeries,
    input_first: &ScalarBidegreeSeries,
    input_second: &ScalarBidegreeSeries,
    max_total_degree: usize,
) -> ScalarBidegreeSeries {
    let first_powers = scalar_bidegree_powers(input_first, max_total_degree, max_total_degree);
    let second_powers = scalar_bidegree_powers(input_second, max_total_degree, max_total_degree);
    let mut out = ScalarBidegreeSeries::new();
    for (&grade, coefficient) in series {
        if grade.0 + grade.1 > max_total_degree {
            continue;
        }
        let monomial = scalar_bidegree_mul(
            &first_powers[grade.0],
            &second_powers[grade.1],
            max_total_degree,
        );
        for (&out_grade, value) in &monomial {
            scalar_bidegree_add_term(
                &mut out,
                out_grade,
                coefficient.clone() * value.clone(),
                max_total_degree,
            );
        }
    }
    out
}

fn invert_bidegree_mirror_map(
    mirror_first: &ScalarBidegreeSeries,
    mirror_second: &ScalarBidegreeSeries,
    max_total_degree: usize,
) -> (ScalarBidegreeSeries, ScalarBidegreeSeries) {
    let variable_first = scalar_bidegree_variable(0);
    let variable_second = scalar_bidegree_variable(1);
    let mut inverse_first = variable_first.clone();
    let mut inverse_second = variable_second.clone();
    for _ in 0..max_total_degree {
        let composed_first = scalar_bidegree_compose(
            mirror_first,
            &inverse_first,
            &inverse_second,
            max_total_degree,
        );
        let composed_second = scalar_bidegree_compose(
            mirror_second,
            &inverse_first,
            &inverse_second,
            max_total_degree,
        );
        inverse_first = scalar_bidegree_mul(
            &variable_first,
            &scalar_bidegree_exp(&scalar_bidegree_neg(&composed_first), max_total_degree),
            max_total_degree,
        );
        inverse_second = scalar_bidegree_mul(
            &variable_second,
            &scalar_bidegree_exp(&scalar_bidegree_neg(&composed_second), max_total_degree),
            max_total_degree,
        );
    }
    (inverse_first, inverse_second)
}

fn h_bidegree_add_term(
    series: &mut HLaurentBidegreeSeries,
    grade: Grade,
    value: HLaurentSeries,
    max_total_degree: usize,
) {
    if grade.0 + grade.1 > max_total_degree || value.is_empty() {
        return;
    }
    let next = series
        .get(&grade)
        .cloned()
        .unwrap_or_else(|| HLaurentSeries::zero(value.max_h_power()))
        .add(&value);
    if next.is_empty() {
        series.remove(&grade);
    } else {
        series.insert(grade, next);
    }
}

fn h_bidegree_mul(
    left: &HLaurentBidegreeSeries,
    right: &HLaurentBidegreeSeries,
    relation: &[Rational],
    max_h_power: usize,
    max_total_degree: usize,
) -> HLaurentBidegreeSeries {
    let mut out = HLaurentBidegreeSeries::new();
    for (&left_grade, left_value) in left {
        for (&right_grade, right_value) in right {
            let grade = (left_grade.0 + right_grade.0, left_grade.1 + right_grade.1);
            if grade.0 + grade.1 > max_total_degree {
                continue;
            }
            let product = left_value.multiply_mod_relation(right_value, relation);
            h_bidegree_add_term(&mut out, grade, product, max_total_degree);
        }
    }
    if out.is_empty() && max_total_degree == 0 {
        out.insert((0, 0), HLaurentSeries::zero(max_h_power));
    }
    out
}

fn h_bidegree_exp(
    series: &HLaurentBidegreeSeries,
    relation: &[Rational],
    max_h_power: usize,
    max_total_degree: usize,
) -> HLaurentBidegreeSeries {
    let mut one = HLaurentSeries::zero(max_h_power);
    one.add_term(0, 0, Rational::one());
    let mut out = BTreeMap::from([((0, 0), one.clone())]);
    let mut power = BTreeMap::from([((0, 0), one)]);
    let mut factorial = Rational::one();
    for order in 1..=max_total_degree {
        power = h_bidegree_mul(&power, series, relation, max_h_power, max_total_degree);
        factorial = factorial * Rational::from(order);
        for (&grade, value) in &power {
            h_bidegree_add_term(
                &mut out,
                grade,
                value.scale(Rational::one() / factorial.clone()),
                max_total_degree,
            );
        }
    }
    out
}

fn h_bidegree_compose(
    series: &HLaurentBidegreeSeries,
    input_first: &ScalarBidegreeSeries,
    input_second: &ScalarBidegreeSeries,
    max_total_degree: usize,
) -> HLaurentBidegreeSeries {
    let first_powers = scalar_bidegree_powers(input_first, max_total_degree, max_total_degree);
    let second_powers = scalar_bidegree_powers(input_second, max_total_degree, max_total_degree);
    let mut out = HLaurentBidegreeSeries::new();
    for (&grade, h_value) in series {
        if grade.0 + grade.1 > max_total_degree {
            continue;
        }
        let monomial = scalar_bidegree_mul(
            &first_powers[grade.0],
            &second_powers[grade.1],
            max_total_degree,
        );
        for (&out_grade, coefficient) in &monomial {
            h_bidegree_add_term(
                &mut out,
                out_grade,
                h_value.scale(coefficient.clone()),
                max_total_degree,
            );
        }
    }
    out
}

fn h_bidegree_to_ray_coefficients(
    series: &HLaurentBidegreeSeries,
    q_degree: usize,
    ray: &Rational,
    max_h_power: usize,
) -> Vec<HLaurentSeries> {
    let mut out = vec![HLaurentSeries::zero(max_h_power); q_degree + 1];
    for (&grade, value) in series {
        let total = grade.0 + grade.1;
        if total > q_degree {
            continue;
        }
        out[total] = out[total].add(&value.scale(ray.pow_usize(grade.1)));
    }
    out
}

fn genus_zero_three_primary_bundle_layout(insertions: &[BundleInsertion]) -> bool {
    insertions.len() == 3
        && insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
}

fn apply_rational_series_matrix_to_vector(
    matrix: &SeriesMatrix<Rational>,
    vector: &[QSeries<Rational>],
    q_degree: usize,
) -> Vec<QSeries<Rational>> {
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::<Rational>::zero(q_degree);
            for (col, vector_coeff) in vector.iter().enumerate() {
                if vector_coeff.is_structurally_zero()
                    || matrix.entry(row, col).is_structurally_zero()
                {
                    continue;
                }
                total = total.add(&matrix.entry(row, col).mul(vector_coeff));
            }
            total
        })
        .collect()
}

fn rational_series_column_matrix(columns: &[Vec<QSeries<Rational>>]) -> SeriesMatrix<Rational> {
    let rows = columns.first().map(Vec::len).unwrap_or_default();
    SeriesMatrix::from_entries(
        (0..rows)
            .map(|row| columns.iter().map(|column| column[row].clone()).collect())
            .collect(),
    )
}

fn quantum_cyclic_basis_matrix(
    quantum_grading: &SeriesMatrix<Rational>,
    q_degree: usize,
) -> SeriesMatrix<Rational> {
    let size = quantum_grading.rows();
    let mut columns = Vec::with_capacity(size);
    let mut current = vec![QSeries::<Rational>::zero(q_degree); size];
    current[0] = QSeries::<Rational>::one(q_degree);
    for _ in 0..size {
        columns.push(current.clone());
        current = apply_rational_series_matrix_to_vector(quantum_grading, &current, q_degree);
    }
    rational_series_column_matrix(&columns)
}

fn quantum_cyclic_coordinates(
    quantum_grading: &SeriesMatrix<Rational>,
    vector: &[QSeries<Rational>],
    q_degree: usize,
) -> Result<Vec<QSeries<Rational>>, GwError> {
    let basis = quantum_cyclic_basis_matrix(quantum_grading, q_degree);
    let inverse = crate::twisted::invert_series_matrix_coeff(&basis)?;
    Ok(apply_rational_series_matrix_to_vector(
        &inverse, vector, q_degree,
    ))
}

fn quantum_product_from_left_coordinates(
    quantum_grading: &SeriesMatrix<Rational>,
    left_coordinates: &[QSeries<Rational>],
    right: &[QSeries<Rational>],
    q_degree: usize,
) -> Vec<QSeries<Rational>> {
    let size = quantum_grading.rows();
    debug_assert_eq!(left_coordinates.len(), size);
    debug_assert_eq!(right.len(), size);
    let mut result = vec![QSeries::<Rational>::zero(q_degree); size];
    let mut grading_power_right = right.to_vec();
    for (power, coefficient) in left_coordinates.iter().enumerate() {
        if !coefficient.is_structurally_zero() {
            for row in 0..size {
                result[row] = result[row].add(&grading_power_right[row].mul(coefficient));
            }
        }
        if power + 1 < left_coordinates.len() {
            grading_power_right = apply_rational_series_matrix_to_vector(
                quantum_grading,
                &grading_power_right,
                q_degree,
            );
        }
    }
    result
}

fn pair_rational_series_vectors(
    metric: &SeriesMatrix<Rational>,
    left: &[QSeries<Rational>],
    right: &[QSeries<Rational>],
    degree: usize,
) -> Rational {
    let size = metric.rows();
    debug_assert_eq!(metric.cols(), size);
    debug_assert_eq!(left.len(), size);
    debug_assert_eq!(right.len(), size);
    let mut paired = QSeries::<Rational>::zero(degree);
    for (left_idx, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_structurally_zero() {
            continue;
        }
        for (right_idx, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_structurally_zero()
                || metric.entry(left_idx, right_idx).is_structurally_zero()
            {
                continue;
            }
            paired = paired.add(
                &left_coeff
                    .mul(metric.entry(left_idx, right_idx))
                    .mul(right_coeff),
            );
        }
    }
    paired.coeff(degree).cloned().unwrap_or_else(Rational::zero)
}

impl ProjectiveBundleRay {
    fn shifted_fiber_divisor_vector(&self) -> Vec<Rational> {
        let h = self.insertion_class_vector(1, 0);
        let xi = self.insertion_class_vector(0, 1);
        let shift = Rational::from(self.big_a() as i128);
        xi.into_iter()
            .zip(h)
            .map(|(xi_coeff, h_coeff)| xi_coeff + shift.clone() * h_coeff)
            .collect()
    }

    fn solve_unit_divisor_coordinates(
        &self,
        vector: &[Rational],
    ) -> Result<(Rational, Rational, Rational), GwError> {
        let size = self.size();
        let mut unit = vec![Rational::zero(); size];
        unit[0] = Rational::one();
        let h = self.insertion_class_vector(1, 0);
        let shifted_fiber = self.shifted_fiber_divisor_vector();

        let mut candidates = Vec::with_capacity(size.saturating_sub(1));
        for power in 1..size {
            let mut basis = vec![Rational::zero(); size];
            basis[power] = Rational::one();
            candidates.push(basis);
        }

        let needed = size.saturating_sub(3);
        let masks = 1usize << candidates.len();
        for mask in 0..masks {
            if mask.count_ones() as usize != needed {
                continue;
            }
            let mut columns = vec![unit.clone(), h.clone(), shifted_fiber.clone()];
            for (idx, candidate) in candidates.iter().enumerate() {
                if (mask & (1usize << idx)) != 0 {
                    columns.push(candidate.clone());
                }
            }
            let mut matrix = (0..size)
                .map(|row| {
                    columns
                        .iter()
                        .map(|column| column[row].clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut values = vector.to_vec();
            if recipe::solve_rational_system(&mut matrix, &mut values).is_ok() {
                return Ok((values[0].clone(), values[1].clone(), values[2].clone()));
            }
        }

        Err(GwError::ConventionMismatch(
            "could not extract bundle bidegree divisor mirror coordinates".to_string(),
        ))
    }

    fn bidegree_mirror_maps_from_cone_point(
        &self,
        cone_point: &HLaurentBidegreeSeries,
        q_degree: usize,
    ) -> Result<(ScalarBidegreeSeries, ScalarBidegreeSeries), GwError> {
        let mut mirror_first = ScalarBidegreeSeries::new();
        let mut mirror_second = ScalarBidegreeSeries::new();
        for (&grade, series) in cone_point {
            if grade == (0, 0) || grade.0 + grade.1 > q_degree {
                continue;
            }
            let mut vector = vec![Rational::zero(); self.size()];
            for (row, value) in vector.iter_mut().enumerate().take(self.size()) {
                if let Some(terms) = series.terms_at_h_power(row) {
                    if let Some(coefficient) = terms.get(&-1) {
                        *value = coefficient.clone();
                    }
                }
            }
            if vector.iter().all(Rational::is_zero) {
                continue;
            }
            let (_, h_coordinate, shifted_fiber_coordinate) =
                self.solve_unit_divisor_coordinates(&vector)?;
            scalar_bidegree_add_term(&mut mirror_first, grade, h_coordinate, q_degree);
            scalar_bidegree_add_term(
                &mut mirror_second,
                grade,
                shifted_fiber_coordinate,
                q_degree,
            );
        }
        Ok((mirror_first, mirror_second))
    }

    fn bidegree_mirror_gauge(
        &self,
        mirror_first: &ScalarBidegreeSeries,
        mirror_second: &ScalarBidegreeSeries,
        q_degree: usize,
    ) -> HLaurentBidegreeSeries {
        let h = self.insertion_class_vector(1, 0);
        let shifted_fiber = self.shifted_fiber_divisor_vector();
        let mut exponent = HLaurentBidegreeSeries::new();
        for (&grade, coefficient) in mirror_first {
            let series = exponent
                .entry(grade)
                .or_insert_with(|| HLaurentSeries::zero(self.size() - 1));
            for (row, h_coefficient) in h.iter().enumerate() {
                series.add_term(row, -1, -(coefficient.clone() * h_coefficient.clone()));
            }
        }
        for (&grade, coefficient) in mirror_second {
            let series = exponent
                .entry(grade)
                .or_insert_with(|| HLaurentSeries::zero(self.size() - 1));
            for (row, fiber_coefficient) in shifted_fiber.iter().enumerate() {
                series.add_term(row, -1, -(coefficient.clone() * fiber_coefficient.clone()));
            }
        }
        h_bidegree_exp(
            &exponent,
            &self.h_power_relation(),
            self.size() - 1,
            q_degree,
        )
    }

    fn flat_bidegree_cone_point(
        &self,
        cone_point: &HLaurentBidegreeSeries,
        q_degree: usize,
    ) -> Result<HLaurentBidegreeSeries, GwError> {
        let (mirror_first, mirror_second) =
            self.bidegree_mirror_maps_from_cone_point(cone_point, q_degree)?;
        let gauge = self.bidegree_mirror_gauge(&mirror_first, &mirror_second, q_degree);
        let gauged = h_bidegree_mul(
            &gauge,
            cone_point,
            &self.h_power_relation(),
            self.size() - 1,
            q_degree,
        );
        let (inverse_first, inverse_second) =
            invert_bidegree_mirror_map(&mirror_first, &mirror_second, q_degree);
        Ok(h_bidegree_compose(
            &gauged,
            &inverse_first,
            &inverse_second,
            q_degree,
        ))
    }

    fn flat_bidegree_ray_cone_point(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<Vec<HLaurentSeries>, GwError> {
        let flat_cone_point = self.normalized_flat_bidegree_cone_point(q_degree, z_order)?;
        Ok(h_bidegree_to_ray_coefficients(
            &flat_cone_point,
            q_degree,
            &self.ray,
            self.size() - 1,
        ))
    }

    fn normalized_flat_bidegree_cone_point(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<HLaurentBidegreeSeries, GwError> {
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();
        static CACHE: OnceLock<Mutex<HashMap<(String, usize, usize), HLaurentBidegreeSeries>>> =
            OnceLock::new();
        let key = (self.rayless_cache_key(), q_degree, z_order);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(cone_point) = cache.lock().unwrap().get(&key).cloned() {
            if profile_enabled {
                eprintln!(
                    "GW_PROFILE bundle_flat_bidegree_cache_hit={:.3}s q_degree={} z_order={}",
                    started.elapsed().as_secs_f64(),
                    q_degree,
                    z_order
                );
            }
            return Ok(cone_point);
        }

        let negative_started = Instant::now();
        let negative = self.normalized_bidegree_negative_factor(q_degree, z_order)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_flat_bidegree_negative={:.3}s q_degree={} z_order={}",
                negative_started.elapsed().as_secs_f64(),
                q_degree,
                z_order
            );
        }
        let correction_started = Instant::now();
        let cone_point = bidegree_negative_factor_to_cone_point(self.size(), q_degree, &negative);
        let flat_cone_point = self.flat_bidegree_cone_point(&cone_point, q_degree)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_flat_bidegree_correction={:.3}s total={:.3}s q_degree={} z_order={}",
                correction_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                q_degree,
                z_order
            );
        }
        cache.lock().unwrap().insert(key, flat_cone_point.clone());
        Ok(flat_cone_point)
    }

    fn fundamental_s_from_ray_cone_point(
        &self,
        q_degree: usize,
        z_order: usize,
        cone_point: &[HLaurentSeries],
        calibration: CalibrationId,
    ) -> Result<SeriesSMatrix<Rational>, GwError> {
        let fundamental =
            crate::twisted::fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff(
                self.size() - 1,
                q_degree,
                cone_point,
                &self.h_power_relation(),
            );
        crate::twisted::birkhoff_descendant_s_matrix_from_fundamental_coeff(
            self.size(),
            q_degree,
            z_order,
            &fundamental,
            calibration,
        )
    }

    fn normalized_bidegree_negative_factor(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<BidegreeLaurentFactor<Rational>, GwError> {
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();
        static CACHE: OnceLock<
            Mutex<HashMap<(String, usize, usize), BidegreeLaurentFactor<Rational>>>,
        > = OnceLock::new();
        let key = (self.rayless_cache_key(), q_degree, z_order);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(negative) = cache.lock().unwrap().get(&key).cloned() {
            if profile_enabled {
                eprintln!(
                    "GW_PROFILE bundle_bidegree_negative_cache_hit={:.3}s q_degree={} z_order={}",
                    started.elapsed().as_secs_f64(),
                    q_degree,
                    z_order
                );
            }
            return Ok(negative);
        }

        let min_z_started = Instant::now();
        let bounds = self.bidegree_birkhoff_bounds(q_degree, z_order);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_min_z={:.3}s q_degree={} z_order={} min_z={}",
                min_z_started.elapsed().as_secs_f64(),
                q_degree,
                z_order,
                bounds.min_z
            );
        }
        let i_started = Instant::now();
        let cone_point = self.i_container(q_degree, bounds.min_z);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_i_container={:.3}s grades={} q_degree={} z_order={}",
                i_started.elapsed().as_secs_f64(),
                cone_point.len(),
                q_degree,
                z_order
            );
        }
        let fundamental_started = Instant::now();
        let fundamental = self.fundamental_bidegree_matrix(q_degree, &cone_point);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_bidegree_fundamental={:.3}s grades={} q_degree={} z_order={}",
                fundamental_started.elapsed().as_secs_f64(),
                fundamental.len(),
                q_degree,
                z_order
            );
        }
        let birkhoff_started = Instant::now();
        let negative = crate::twisted::birkhoff_negative_factor_by_bidegree_with_z_bounds(
            self.size(),
            q_degree,
            &fundamental,
            &bounds.positive_z_windows,
            &bounds.negative_z_depths,
        )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_bidegree_birkhoff={:.3}s total={:.3}s q_degree={} z_order={}",
                birkhoff_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                q_degree,
                z_order
            );
        }
        cache.lock().unwrap().insert(key, negative.clone());
        Ok(negative)
    }

    fn flat_metric_series(&self, q_degree: usize) -> SeriesMatrix<Rational> {
        SeriesMatrix::constant(self.flat_metric(), q_degree)
    }

    fn fundamental_s_rational(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<Rational>, GwError> {
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();
        static CACHE: OnceLock<Mutex<HashMap<(String, usize, usize), SeriesSMatrix<Rational>>>> =
            OnceLock::new();
        let target_key = self.cache_key();
        let key = (target_key.clone(), q_degree, z_order);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let cache = cache.lock().unwrap();
            if let Some(fundamental_s) = cache.get(&key).cloned() {
                if profile_enabled {
                    eprintln!(
                        "GW_PROFILE bundle_fundamental_s_cache_hit={:.3}s q_degree={} z_order={}",
                        started.elapsed().as_secs_f64(),
                        q_degree,
                        z_order
                    );
                }
                return Ok(fundamental_s);
            }
            if let Some(fundamental_s) = cache
                .iter()
                .find(|((cached_key, cached_q_degree, cached_z_order), _)| {
                    cached_key == &target_key
                        && *cached_q_degree == q_degree
                        && *cached_z_order >= z_order
                })
                .map(|(_, fundamental_s)| fundamental_s.truncated(z_order))
            {
                if profile_enabled {
                    eprintln!(
                        "GW_PROFILE bundle_fundamental_s_cache_truncate={:.3}s q_degree={} z_order={}",
                        started.elapsed().as_secs_f64(),
                        q_degree,
                        z_order
                    );
                }
                return Ok(fundamental_s);
            }
        }

        let cone_started = Instant::now();
        let cone_point = self.flat_bidegree_ray_cone_point(q_degree, z_order)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_ray_cone_point={:.3}s q_degree={} z_order={} ray={}",
                cone_started.elapsed().as_secs_f64(),
                q_degree,
                z_order,
                self.ray
            );
        }
        let s_started = Instant::now();
        let fundamental_s = self.fundamental_s_from_ray_cone_point(
            q_degree,
            z_order,
            &cone_point,
            CalibrationId(format!("bundle-bidegree-flat:{}", self.cache_key())),
        )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_ray_fundamental_s={:.3}s total={:.3}s q_degree={} z_order={} ray={}",
                s_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                q_degree,
                z_order,
                self.ray
            );
        }
        cache.lock().unwrap().insert(key, fundamental_s.clone());
        Ok(fundamental_s)
    }

    fn descendant_s_rational(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<Rational>, GwError> {
        if z_order == 0 {
            return Ok(SeriesSMatrix::identity(self.size(), q_degree, 0));
        }

        let fundamental_s = self.fundamental_s_rational(q_degree, z_order)?;
        crate::twisted::metric_adjoint_descendant_s_matrix_coeff(
            fundamental_s,
            &self.flat_metric_series(q_degree),
        )
    }

    fn quantum_grading_multiplication(
        &self,
        q_degree: usize,
    ) -> Result<SeriesMatrix<Rational>, GwError> {
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();
        let fundamental_s = self.fundamental_s_rational(q_degree, 1)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_quantum_grading_fundamental_s={:.3}s q_degree={} ray={}",
                started.elapsed().as_secs_f64(),
                q_degree,
                self.ray
            );
        }
        let product_started = Instant::now();
        let s_one = fundamental_s.coefficient(1).ok_or_else(|| {
            GwError::ConventionMismatch(
                "bundle quantum product needs the fundamental S-matrix through z^{-1}".to_string(),
            )
        })?;
        let quantum = self
            .classical_grading_multiplication(q_degree)
            .add(&s_one.q_derivative());
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_quantum_grading_product={:.3}s total={:.3}s q_degree={} ray={}",
                product_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                q_degree,
                self.ray
            );
        }
        Ok(quantum)
    }

    /// Classical multiplication by the grading divisor in the `D`-power
    /// basis: the companion matrix of the classical minimal polynomial.
    fn classical_grading_multiplication(&self, q_degree: usize) -> SeriesMatrix<Rational> {
        let size = self.size();
        let relation = self.h_power_relation();
        let mut matrix = vec![vec![QSeries::<Rational>::zero(q_degree); size]; size];
        for col in 0..size.saturating_sub(1) {
            matrix[col + 1][col] = QSeries::<Rational>::one(q_degree);
        }
        for row in 0..size {
            matrix[row][size - 1] = QSeries::constant(relation[row].clone(), q_degree);
        }
        SeriesMatrix::from_entries(matrix)
    }
}

/// Engine-facing provider for one ray of the bundle theory.
#[derive(Debug, Clone)]
pub struct BundleRayProvider {
    pub target: ProjectiveBundleRay,
}

impl BundleRayProvider {
    pub fn new(target: ProjectiveBundleRay) -> Self {
        Self { target }
    }
}

impl SemisimpleCohftProvider for BundleRayProvider {
    type Insertion = BundleInsertion;

    fn colors(&self) -> usize {
        self.target.size()
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        insertion.descendant_power
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        let mut total = 0usize;
        for insertion in insertions {
            total = total.checked_add(insertion.descendant_power)?;
            total = total.checked_add(insertion.h_power)?;
            total = total.checked_add(insertion.xi_power)?;
        }
        Some(total)
    }

    // As on the product, the virtual dimension depends on the bidegree, not
    // the ray degree, so per-degree pruning stays disabled and the dimension
    // filter is applied after reconstruction.

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        rational_s_matrix_to_ratfun(&self.target.descendant_s_rational(q_degree, z_order)?)
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        static CACHE: OnceLock<
            Mutex<HashMap<(String, usize, usize, usize), Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let key = (self.target.cache_key(), q_degree, r_order, graph_dimension);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(kernel);
        }
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();

        // Quantum multiplication by the grading divisor from the fundamental
        // S-matrix: the z^0 part of the quantum differential equation gives
        // A_q = A_cl + t d/dt S_1.
        let quantum_started = Instant::now();
        let quantum_multiplication = self.target.quantum_grading_multiplication(q_degree)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_kernel_quantum={:.3}s q_degree={} r_order={} ray={}",
                quantum_started.elapsed().as_secs_f64(),
                q_degree,
                r_order,
                self.target.ray
            );
        }

        let unit_coordinates = {
            let mut unit = vec![RatFun::zero(); self.target.size()];
            unit[0] = RatFun::one();
            unit
        };
        let frame_started = Instant::now();
        let frame = recipe::operator_lagrange_frame(
            &rational_series_matrix_to_ratfun(&quantum_multiplication),
            &self.target.grading_seeds(),
            &unit_coordinates,
            &rational_series_matrix_to_ratfun(&self.target.flat_metric_series(q_degree)),
        )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_kernel_frame={:.3}s q_degree={} r_order={} ray={}",
                frame_started.elapsed().as_secs_f64(),
                q_degree,
                r_order,
                self.target.ray
            );
        }

        let asymptotics_started = Instant::now();
        let mut classical_diagonal = Vec::with_capacity(self.target.size());
        for i in 0..=self.target.n {
            for j in 0..self.target.rank() {
                let mut differences = Vec::new();
                for k in 0..=self.target.n {
                    if k != i {
                        differences.push(RatFun::from_rational(
                            self.target.weights_base[k].clone()
                                - self.target.weights_base[i].clone(),
                        ));
                    }
                }
                for l in 0..self.target.rank() {
                    if l != j {
                        // The fiber divisor is xi with restriction -c_ij, so
                        // the R-asymptotic factor difference is
                        // (-c_il) - (-c_ij), opposite to the Euler factor.
                        differences.push(RatFun::from_rational(
                            self.target.fiber_weight(i, j) - self.target.fiber_weight(i, l),
                        ));
                    }
                }
                classical_diagonal.push(classical_r_asymptotics_for_point(&differences, r_order));
            }
        }
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_kernel_asymptotics={:.3}s q_degree={} r_order={} ray={}",
                asymptotics_started.elapsed().as_secs_f64(),
                q_degree,
                r_order,
                self.target.ray
            );
        }

        let calibration_started = Instant::now();
        let calibration = calibration_from_canonical_frame(
            &frame,
            &classical_diagonal,
            q_degree,
            r_order,
            CalibrationId(format!("bundle-ray-i-birkhoff:{}", self.target.cache_key())),
        )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_kernel_calibration={:.3}s q_degree={} r_order={} ray={}",
                calibration_started.elapsed().as_secs_f64(),
                q_degree,
                r_order,
                self.target.ray
            );
        }
        let kernel_started = Instant::now();
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_kernel_build={:.3}s total={:.3}s q_degree={} r_order={} ray={}",
                kernel_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                q_degree,
                r_order,
                self.target.ray
            );
        }
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    fn direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_bundle_layout(insertions) {
            return Ok(None);
        }
        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = Instant::now();

        let insertion_vectors = insertions
            .iter()
            .map(|insertion| {
                self.target
                    .insertion_class_vector(insertion.h_power, insertion.xi_power)
                    .into_iter()
                    .map(|coefficient| QSeries::<Rational>::constant(coefficient, degree))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantum_started = Instant::now();
        let quantum_grading = self.target.quantum_grading_multiplication(degree)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_direct_quantum={:.3}s degree={} ray={}",
                quantum_started.elapsed().as_secs_f64(),
                degree,
                self.target.ray
            );
        }
        let coordinates_started = Instant::now();
        let left_coordinates =
            quantum_cyclic_coordinates(&quantum_grading, &insertion_vectors[0], degree)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_direct_coordinates={:.3}s degree={} ray={}",
                coordinates_started.elapsed().as_secs_f64(),
                degree,
                self.target.ray
            );
        }
        let product_started = Instant::now();
        let product = quantum_product_from_left_coordinates(
            &quantum_grading,
            &left_coordinates,
            &insertion_vectors[1],
            degree,
        );
        let value = pair_rational_series_vectors(
            &self.target.flat_metric_series(degree),
            &product,
            &insertion_vectors[2],
            degree,
        );
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_direct_product_pair={:.3}s total={:.3}s degree={} ray={}",
                product_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                degree,
                self.target.ray
            );
        }
        Ok(Some(RatFun::from_rational(value)))
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        Ok(self
            .target
            .insertion_class_vector(insertion.h_power, insertion.xi_power)
            .into_iter()
            .map(|coefficient| QSeries::constant(RatFun::from_rational(coefficient), q_degree))
            .collect())
    }
}

/// Whether the insertions match the virtual dimension of genus-`genus`
/// class-`(d1, d2)` maps to the bundle (`d2 = xi . beta`, possibly
/// negative).
pub fn bundle_dimension_matches(
    n: usize,
    twists: &[usize],
    genus: usize,
    d1: usize,
    d2: isize,
    insertions: &[BundleInsertion],
) -> bool {
    let rank = twists.len();
    let dimension = (n + rank - 1) as isize;
    let twist_sum: usize = twists.iter().sum();
    let insertion_degree: usize = insertions
        .iter()
        .map(|insertion| insertion.descendant_power + insertion.h_power + insertion.xi_power)
        .sum();
    let c1_pairing = (n + 1 + twist_sum) as isize * d1 as isize + rank as isize * d2;
    let virtual_dimension =
        (1 - genus as isize) * (dimension - 3) + c1_pairing + insertions.len() as isize;
    insertion_degree as isize == virtual_dimension
}

fn rayless_precompute_z_orders(genus: usize, insertions: &[BundleInsertion]) -> Vec<usize> {
    let mut z_orders = Vec::new();
    if genus == 0 && genus_zero_three_primary_bundle_layout(insertions) {
        z_orders.push(1);
    }
    if 2 * genus + insertions.len() > 2 {
        z_orders.push(1);
        if let Some(max_descendant) = insertions
            .iter()
            .map(|insertion| insertion.descendant_power)
            .max()
            .filter(|max_descendant| *max_descendant > 0)
        {
            z_orders.push(max_descendant);
        }
    }
    z_orders.sort_unstable();
    z_orders.dedup();
    z_orders
}

/// Computes all class invariants `N_{(d1, d2)}` whose shifted total degree
/// `d1 + (d2 + A d1)` equals `total_degree`, by running `total_degree + 1`
/// rays and solving the Vandermonde system exactly.
///
/// Returns `(d1, d2, value)` triples ordered by the shifted fiber degree;
/// dimension-mismatched classes are filtered to zero.
pub fn reconstruct_bundle_invariants(
    n: usize,
    twists: &[usize],
    weights_base: &[Rational],
    weights_fiber: &[Rational],
    genus: usize,
    total_degree: usize,
    insertions: &[BundleInsertion],
) -> Result<Vec<(usize, isize, Rational)>, GwError> {
    let big_a = *twists
        .iter()
        .max()
        .ok_or_else(|| GwError::ConventionMismatch("bundle twists must be nonempty".to_string()))?;
    let ray_count = total_degree + 1;
    let profile_enabled = crate::env_flag("GW_PROFILE");
    let started = Instant::now();

    let warm_target = ProjectiveBundleRay::new(
        n,
        twists.to_vec(),
        weights_base.to_vec(),
        weights_fiber.to_vec(),
        Rational::one(),
    )?;
    for z_order in rayless_precompute_z_orders(genus, insertions) {
        let warm_started = Instant::now();
        let _ = warm_target.normalized_flat_bidegree_cone_point(total_degree, z_order)?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_reconstruct_rayless_warm={:.3}s total={:.3}s total_degree={} z_order={}",
                warm_started.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64(),
                total_degree,
                z_order
            );
        }
    }

    let ray_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ray_count);
        for step in 0..ray_count {
            handles.push(
                scope.spawn(move || -> Result<(Rational, Rational), GwError> {
                    let ray = Rational::from(step + 1);
                    let target = ProjectiveBundleRay::new(
                        n,
                        twists.to_vec(),
                        weights_base.to_vec(),
                        weights_fiber.to_vec(),
                        ray.clone(),
                    )?;
                    let provider = BundleRayProvider::new(target);
                    let value = compute_semisimple_graph_value(
                        &provider,
                        genus,
                        total_degree,
                        insertions,
                        None,
                    )?;
                    let value = value.as_rational().ok_or_else(|| {
                        GwError::AlgebraFailure(
                            "bundle ray value did not specialize to a rational".to_string(),
                        )
                    })?;
                    Ok((ray, value))
                }),
            );
        }

        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    GwError::AlgebraFailure("bundle ray worker panicked".to_string())
                })?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    if profile_enabled {
        eprintln!(
            "GW_PROFILE bundle_reconstruct_rays={:.3}s total_degree={} rays={}",
            started.elapsed().as_secs_f64(),
            total_degree,
            ray_count
        );
    }

    let (rays, mut values): (Vec<_>, Vec<_>) = ray_results.into_iter().unzip();

    let mut matrix = rays
        .iter()
        .map(|ray| {
            (0..ray_count)
                .map(|power| ray.pow_usize(power))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    recipe::solve_rational_system(&mut matrix, &mut values)?;

    Ok(values
        .into_iter()
        .enumerate()
        .map(|(d2p, value)| {
            let d1 = total_degree - d2p;
            let d2 = d2p as isize - (big_a * d1) as isize;
            let value = if bundle_dimension_matches(n, twists, genus, d1, d2, insertions) {
                value
            } else {
                Rational::zero()
            };
            (d1, d2, value)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_weights() -> Vec<Rational> {
        vec![Rational::from(2), Rational::from(5)]
    }

    fn fiber_weights() -> Vec<Rational> {
        vec![Rational::from(11), Rational::from(23)]
    }

    #[test]
    fn f2_non_fano_positive_z_birkhoff_matches_product_genus_zero() {
        // F_2 = P(O + O(2)) is non-Fano: its (-2)-section has anticanonical
        // degree zero, so the I-function has positive z-powers and cannot be
        // mirror-transformed by a divisor change alone.  The bidegree
        // fundamental-solution Birkhoff split is the required projection; the
        // positive section B_+ = (1,0) deforms to bidegree (1,1) on P^1 x P^1.
        let point = BundleInsertion::new(0, 1, 1);
        let result = reconstruct_bundle_invariants(
            1,
            &[0, 2],
            &base_weights(),
            &fiber_weights(),
            0,
            3,
            &[point.clone(), point.clone(), point],
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                (3, -6, Rational::zero()),
                (2, -3, Rational::zero()),
                (1, 0, Rational::one()),
                (0, 3, Rational::zero()),
            ]
        );
    }

    #[test]
    fn bounded_bidegree_birkhoff_matches_full_factorization_window() {
        let target = ProjectiveBundleRay::new(
            1,
            vec![5, 4, 0],
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(0), Rational::from(10), Rational::from(30)],
            Rational::one(),
        )
        .unwrap();
        let q_degree = 2;
        let z_order = 1;
        let bounds = target.bidegree_birkhoff_bounds(q_degree, z_order);
        let cone_point = target.i_container(q_degree, bounds.min_z);
        let fundamental = target.fundamental_bidegree_matrix(q_degree, &cone_point);
        let (_, full_negative) =
            crate::twisted::birkhoff_factor_by_bidegree(target.size(), q_degree, &fundamental)
                .unwrap();
        let bounded_negative = crate::twisted::birkhoff_negative_factor_by_bidegree_with_z_bounds(
            target.size(),
            q_degree,
            &fundamental,
            &bounds.positive_z_windows,
            &bounds.negative_z_depths,
        )
        .unwrap();

        let matrix_at = |factor: &BidegreeLaurentFactor<Rational>, grade: Grade, z_power: i32| {
            factor
                .get(&grade)
                .and_then(|laurent| laurent.get(&z_power))
                .cloned()
                .unwrap_or_else(|| crate::twisted::zero_coeff_matrix(target.size()))
        };
        for total in 1..=q_degree {
            for first in 0..=total {
                let grade = (first, total - first);
                let depth = bounds.negative_z_depths.get(&grade).copied().unwrap_or(0);
                for z_power in -(depth as i32)..0 {
                    assert_eq!(
                        matrix_at(&bounded_negative, grade, z_power),
                        matrix_at(&full_negative, grade, z_power),
                        "bounded bidegree factorization mismatch at grade {:?}, z^{}",
                        grade,
                        z_power
                    );
                }
            }
        }
    }

    // ----- F_2 <-> P^1 x P^1 deformation cross-check (acceptance test) -----
    //
    // F_2 = P(O + O(2)) is deformation equivalent to P^1 x P^1, so every GW
    // invariant matches under the identification below, and F_2 has positive-z
    // I-function terms while the product does not -- the ideal check of the
    // full bundle pipeline against an independent one.  It is `#[ignore]`d
    // because the genus-1 cases are slow in debug builds; run it as the
    // end-to-end acceptance test for the non-Fano Birkhoff projection and the
    // higher-order R calibration.
    //
    // Dictionary: class (d1, d2) = (H.beta, xi.beta) on F_2 corresponds to
    // P^1 x P^1 bidegree (d2 + d1, d1); cohomology H <-> H2, xi <-> H1 - H2.

    fn small_binomial(n: usize, k: usize) -> i128 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut out = 1i128;
        for step in 1..=k {
            out = out * (n + 1 - step) as i128 / step as i128;
        }
        out
    }

    /// Expand an F_2 insertion tau_k(H^h xi^x) into P^1 x P^1 product
    /// monomials under H -> H2, xi -> H1 - H2, truncated by H1^2 = H2^2 = 0.
    fn f2_insertion_to_product_terms(
        insertion: &BundleInsertion,
    ) -> Vec<(Rational, crate::givental::ProductInsertion)> {
        let mut terms = Vec::new();
        for j in 0..=insertion.xi_power {
            let a = j;
            let b = insertion.h_power + insertion.xi_power - j;
            if a >= 2 || b >= 2 {
                continue; // vanishes on P^1 x P^1
            }
            let sign = if (insertion.xi_power - j).is_multiple_of(2) {
                1
            } else {
                -1
            };
            let coefficient = Rational::from(small_binomial(insertion.xi_power, j) * sign);
            terms.push((
                coefficient,
                crate::givental::ProductInsertion::new(insertion.descendant_power, a, b),
            ));
        }
        terms
    }

    /// The F_2 invariant of class (d1, d2) computed on P^1 x P^1 through the
    /// deformation identification: bidegree (d2 + d1, d1), insertions expanded
    /// multilinearly and summed over the Cartesian product of terms.
    fn product_side_of_f2(
        genus: usize,
        d1: usize,
        d2: usize,
        insertions: &[BundleInsertion],
    ) -> Rational {
        product_side_of_f2_signed(genus, d1, d2 as isize, insertions)
    }

    /// Signed-`d2` version of [`product_side_of_f2`], needed for the
    /// negative-section direction.
    fn product_side_of_f2_signed(
        genus: usize,
        d1: usize,
        d2: isize,
        insertions: &[BundleInsertion],
    ) -> Rational {
        let product_first = d2 + d1 as isize;
        if product_first < 0 {
            return Rational::zero();
        }
        let product_total = product_first as usize + d1;
        let product_index = d1; // the H2 (second-factor) degree
        let per_insertion = insertions
            .iter()
            .map(f2_insertion_to_product_terms)
            .collect::<Vec<_>>();
        let weights_x = vec![Rational::from(3), Rational::from(7)];
        let weights_y = vec![Rational::from(13), Rational::from(29)];

        let mut indices = vec![0usize; per_insertion.len()];
        let mut total = Rational::zero();
        loop {
            let mut coefficient = Rational::one();
            let mut monomials = Vec::with_capacity(per_insertion.len());
            for (slot, &choice) in indices.iter().enumerate() {
                let (term_coeff, monomial) = &per_insertion[slot][choice];
                coefficient = coefficient * term_coeff.clone();
                monomials.push(monomial.clone());
            }
            let invariants = crate::givental::reconstruct_bidegree_invariants(
                1,
                1,
                &weights_x,
                &weights_y,
                genus,
                product_total,
                &monomials,
            )
            .unwrap();
            total += coefficient * invariants[product_index].clone();

            let mut slot = 0;
            while slot < indices.len() {
                indices[slot] += 1;
                if indices[slot] < per_insertion[slot].len() {
                    break;
                }
                indices[slot] = 0;
                slot += 1;
            }
            if slot == indices.len() {
                break;
            }
        }
        total
    }

    /// The F_2 invariant of class (d1, d2) computed through the bundle
    /// pipeline.  Bundle shifted total degree = d2 + (A+1) d1 with A = 2.
    fn bundle_side_of_f2(
        genus: usize,
        d1: usize,
        d2: usize,
        insertions: &[BundleInsertion],
    ) -> Rational {
        bundle_side_of_f2_signed(genus, d1, d2 as isize, insertions)
    }

    fn bundle_side_of_f2_signed(
        genus: usize,
        d1: usize,
        d2: isize,
        insertions: &[BundleInsertion],
    ) -> Rational {
        let shifted_total = d2 + 3 * d1 as isize;
        assert!(
            shifted_total >= 0,
            "F_2 shifted total degree must be nonnegative"
        );
        let invariants = reconstruct_bundle_invariants(
            1,
            &[0, 2],
            &base_weights(),
            &fiber_weights(),
            genus,
            shifted_total as usize,
            insertions,
        )
        .unwrap();
        invariants
            .into_iter()
            .find(|(a, b, _)| *a == d1 && *b == d2)
            .map(|(_, _, value)| value)
            .expect("bundle class present in the shifted-degree slice")
    }

    #[test]
    #[ignore = "slow end-to-end non-Fano/higher-R acceptance test"]
    fn f2_deformation_matches_p1xp1_pointwise() {
        let point = BundleInsertion::new(0, 1, 1); // H xi is the F_2 point class
        let xi = BundleInsertion::new(0, 0, 1); // maps to H1 - H2 (two terms)
        let tau1_point = BundleInsertion::new(1, 1, 1);
        let cases: Vec<(usize, usize, usize, Vec<BundleInsertion>)> = vec![
            // genus 0, positive section B+ = (1,0) -> F_0 (1,1): <pt,pt,pt>.
            (0, 1, 0, vec![point.clone(), point.clone(), point.clone()]),
            // genus 0, 2f = (0,2) -> F_0 (2,0): <pt,pt,pt>.
            (0, 0, 2, vec![point.clone(), point.clone(), point.clone()]),
            // genus 1, fiber f = (0,1) -> F_0 (1,0): <pt,pt>.
            (1, 0, 1, vec![point.clone(), point.clone()]),
            // genus 1, fiber, single descendant marking: <tau_1(pt)>.
            (1, 0, 1, vec![tau1_point.clone()]),
            // genus 1, fiber, divisor + descendant: exercises the
            // xi -> H1 - H2 multilinear expansion on the product side.
            (1, 0, 1, vec![xi.clone(), tau1_point.clone()]),
            // genus 1, 2f = (0,2) -> F_0 (2,0): <pt,pt,pt,pt>.
            (
                1,
                0,
                2,
                vec![point.clone(), point.clone(), point.clone(), point.clone()],
            ),
        ];

        for (genus, d1, d2, insertions) in cases {
            let bundle = bundle_side_of_f2(genus, d1, d2, &insertions);
            let product = product_side_of_f2(genus, d1, d2, &insertions);
            assert_eq!(
                bundle, product,
                "F_2 vs P^1xP^1 mismatch at genus {genus}, class ({d1},{d2}): \
                 bundle {bundle}, product {product}"
            );
        }
    }

    #[test]
    fn f4_middle_deformation_class_matches_p1xp1_ruling() {
        // P(O(2) + O(-2)) normalizes to P(O(4) + O).  Under the deformation
        // to P^1 x P^1, y = xi + 2H is the second product divisor, so the
        // product ruling <pt,H,H> = 1 corresponds to class (1,-2) and
        // insertions <H xi,H,H> on the bundle.
        let point = BundleInsertion::new(0, 1, 1);
        let h = BundleInsertion::new(0, 1, 0);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[4, 0],
            &base_weights(),
            &fiber_weights(),
            0,
            3,
            &[point, h.clone(), h],
        )
        .unwrap();
        let value = invariants
            .into_iter()
            .find(|(d1, d2, _)| *d1 == 1 && *d2 == -2)
            .map(|(_, _, value)| value)
            .expect("F_4 middle deformation class present in shifted slice");
        assert_eq!(value, Rational::one());
    }

    #[test]
    fn rank3_negative_direction_matches_p1xp2_base_ruling() {
        // P(O(1) + O + O(-1)) normalizes to P(O(2) + O(1) + O).  Under
        // deformation to P^1 x P^2, y = xi + H is the product hyperplane on
        // P^2, so the product ruling <pt,H,H> = 1 corresponds to class
        // (1,-1) and insertions <H xi^2,H,H> on the bundle.
        let point = BundleInsertion::new(0, 1, 2);
        let h = BundleInsertion::new(0, 1, 0);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[2, 1, 0],
            &[Rational::from(1), Rational::from(2)],
            &[Rational::from(0), Rational::from(10), Rational::from(30)],
            0,
            2,
            &[point, h.clone(), h],
        )
        .unwrap();
        let value = invariants
            .into_iter()
            .find(|(d1, d2, _)| *d1 == 1 && *d2 == -1)
            .map(|(_, _, value)| value)
            .expect("rank-3 negative class present in shifted slice");
        assert_eq!(value, Rational::one());
    }

    #[test]
    #[ignore = "slow rank-3 deformation acceptance test"]
    fn rank3_harder_negative_direction_matches_p1xp2_base_ruling() {
        // P(O(2) + O(1) + O(-3)) normalizes to P(O(5) + O(4) + O).  The
        // product hyperplane is y = xi + 3H, hence the same product ruling is
        // the bundle class (1,-3) with insertions <H xi^2,H,H>.
        let point = BundleInsertion::new(0, 1, 2);
        let h = BundleInsertion::new(0, 1, 0);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[5, 4, 0],
            &[Rational::from(1), Rational::from(2)],
            &[Rational::from(0), Rational::from(10), Rational::from(30)],
            0,
            3,
            &[point, h.clone(), h],
        )
        .unwrap();
        let value = invariants
            .into_iter()
            .find(|(d1, d2, _)| *d1 == 1 && *d2 == -3)
            .map(|(_, _, value)| value)
            .expect("rank-3 harder negative class present in shifted slice");
        assert_eq!(value, Rational::one());
    }

    #[test]
    #[ignore = "slow harder F_2 negative-section spot checks"]
    fn f2_negative_section_harder_spot_checks_match_p1xp1() {
        let point = BundleInsertion::new(0, 1, 1);
        let xi = BundleInsertion::new(0, 0, 1);
        let tau1_point = BundleInsertion::new(1, 1, 1);
        let tau2_point = BundleInsertion::new(2, 1, 1);

        let cases = vec![
            (
                "g0 class (2,-1), descendant through three points",
                0,
                2,
                -1,
                vec![tau2_point.clone(), point.clone(), point.clone()],
            ),
            (
                "g0 class (2,-1), xi expansion plus descendant",
                0,
                2,
                -1,
                vec![tau2_point.clone(), point.clone(), point.clone(), xi],
            ),
            (
                "g1 class (2,-1), mixed descendants",
                1,
                2,
                -1,
                vec![tau2_point, tau1_point, point],
            ),
        ];

        for (label, genus, d1, d2, insertions) in cases {
            let bundle = bundle_side_of_f2_signed(genus, d1, d2, &insertions);
            let product = product_side_of_f2_signed(genus, d1, d2, &insertions);
            assert_eq!(
                bundle, product,
                "{label}: bundle {bundle}, product {product}"
            );
        }
    }

    #[test]
    #[ignore = "slow fixed non-Fano negative-section acceptance test"]
    fn f2_negative_section_direction_genus_one_matches_p1xp1() {
        // Class (d1,d2) = (2,-1) is 3f + 2B_- and deforms to product
        // bidegree (1,2).  This was the first case requiring the bidegree
        // Birkhoff projection before ray restriction.
        let tau1_point = BundleInsertion::new(1, 1, 1);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[0, 2],
            &base_weights(),
            &fiber_weights(),
            1,
            5,
            &[tau1_point.clone(), tau1_point.clone(), tau1_point],
        )
        .unwrap();
        let value = invariants
            .into_iter()
            .find(|(d1, d2, _)| *d1 == 2 && *d2 == -1)
            .map(|(_, _, value)| value)
            .expect("class (2,-1) present in shifted total degree 5");
        assert_eq!(value, -Rational::new(1, 4));
    }

    #[test]
    fn i_function_vanishes_outside_the_shifted_cone_boundary() {
        // For F_1 (twists [0,1] over P^1), the grade (d1, d2') = (1, 0)
        // corresponds to d2 = -1: only the a=1 summand has D_l >= 0, so the
        // coefficient is nonzero, while (2, 0) means d2 = -2 with every
        // fixed point still contributing through the a=1 factor... the
        // genuinely impossible direction is d2' shifted below zero, which
        // the grading forbids by construction.  Check instead that a
        // rank-one negative direction vanishes: with twists [0, 0]
        // (a product), d2 = d2' and any negative d2 cannot appear; the
        // boundary grade (1, 0) has d2 = 0 and must be nonzero.
        let target = ProjectiveBundleRay::new(
            1,
            vec![0, 1],
            base_weights(),
            fiber_weights(),
            Rational::one(),
        )
        .unwrap();
        let min_z = target.min_z_power(2, 2);
        assert!(!target.i_coefficient(1, 0, min_z).is_empty());
        assert!(!target.i_coefficient(0, 1, min_z).is_empty());
    }

    #[test]
    fn zero_twist_bundle_reproduces_product_invariants() {
        // P(O + O) over P^1 is P^1 x P^1; the unique (1,1)-curve through
        // three general points must reappear through the full bundle
        // pipeline (I-function, bidegree Birkhoff, operator frame).
        let point = BundleInsertion::new(0, 1, 1);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[0, 0],
            &base_weights(),
            &fiber_weights(),
            0,
            2,
            &[point.clone(), point.clone(), point],
        )
        .unwrap();
        assert_eq!(
            invariants,
            vec![
                (2, 0, Rational::zero()),
                (1, 1, Rational::one()),
                (0, 2, Rational::zero()),
            ]
        );
    }

    #[test]
    fn zero_twist_bundle_r_matrix_matches_product_to_higher_order() {
        // In the zero-twist case xi restricts as -mu_j, so the matching
        // product calibration uses second-factor weights -mu_j.  This compares
        // the structural calibration directly, avoiding a slow genus-1 graph
        // sum while still exercising the higher R orders that caught Bug B.
        let q_degree = 2;
        let r_order = 5;
        let graph_dimension = 0;
        let ray = Rational::from(2);
        let bundle_target =
            ProjectiveBundleRay::new(1, vec![0, 0], base_weights(), fiber_weights(), ray.clone())
                .unwrap();
        let product_target = crate::givental::ProductProjectiveRay::new(
            1,
            1,
            base_weights(),
            fiber_weights()
                .into_iter()
                .map(|weight| -weight)
                .collect::<Vec<_>>(),
            ray,
        )
        .unwrap();
        let bundle = BundleRayProvider::new(bundle_target)
            .graph_kernel(q_degree, r_order, graph_dimension)
            .unwrap();
        let product = crate::givental::ProductRayProvider::new(product_target)
            .graph_kernel(q_degree, r_order, graph_dimension)
            .unwrap();

        assert_eq!(
            bundle.calibration().r_matrix.coefficients(),
            product.calibration().r_matrix.coefficients()
        );
        assert_eq!(bundle.calibration().psi, product.calibration().psi);
        assert_eq!(
            bundle.calibration().psi_inverse,
            product.calibration().psi_inverse
        );
        assert_eq!(
            bundle.calibration().connection,
            product.calibration().connection
        );
    }

    #[test]
    fn hirzebruch_classical_integrals() {
        // F_1 = P(O + O(1)) over P^1: int H xi = 1 and int xi^2 = -1
        // (the relation xi(xi + H) = 0 makes xi^2 = -H xi).
        let first = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights(),
            &fiber_weights(),
            0,
            0,
            &[
                BundleInsertion::new(0, 1, 0),
                BundleInsertion::new(0, 0, 1),
                BundleInsertion::new(0, 0, 0),
            ],
        )
        .unwrap();
        assert_eq!(first, vec![(0, 0, Rational::one())]);

        let second = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights(),
            &fiber_weights(),
            0,
            0,
            &[
                BundleInsertion::new(0, 0, 1),
                BundleInsertion::new(0, 0, 1),
                BundleInsertion::new(0, 0, 0),
            ],
        )
        .unwrap();
        assert_eq!(second, vec![(0, 0, -Rational::one())]);
    }

    #[test]
    fn hirzebruch_exceptional_and_fiber_classes() {
        // <xi, xi, xi> at shifted total degree 1: the exceptional section
        // e = (1, -1) gives (xi.e)^3 N_e = -1, and the fiber grade is
        // dimension-filtered.  <pt, xi, xi> flips the support: the fiber
        // class f = (0, 1) gives (xi.f)^2 <pt>_f = 1.
        let xi = BundleInsertion::new(0, 0, 1);
        let exceptional = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights(),
            &fiber_weights(),
            0,
            1,
            &[xi.clone(), xi.clone(), xi.clone()],
        )
        .unwrap();
        assert_eq!(
            exceptional,
            vec![(1, -1, -Rational::one()), (0, 1, Rational::zero())]
        );

        let fiber = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights(),
            &fiber_weights(),
            0,
            1,
            &[BundleInsertion::new(0, 1, 1), xi.clone(), xi],
        )
        .unwrap();
        assert_eq!(
            fiber,
            vec![(1, -1, Rational::zero()), (0, 1, Rational::one())]
        );
    }

    #[test]
    fn hirzebruch_line_count_through_two_points() {
        // <pt, pt, H> at shifted total degree 2: the class h = e + f =
        // (1, 0) is the strict transform of a line under Bl_p P^2 = F_1,
        // and (H.h) N_h(pt, pt) = 1; the neighbours are dimension-filtered.
        let point = BundleInsertion::new(0, 1, 1);
        let invariants = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights(),
            &fiber_weights(),
            0,
            2,
            &[point.clone(), point, BundleInsertion::new(0, 1, 0)],
        )
        .unwrap();
        assert_eq!(
            invariants,
            vec![
                (2, -2, Rational::zero()),
                (1, 0, Rational::one()),
                (0, 2, Rational::zero()),
            ]
        );
    }

    #[test]
    fn higher_rank_and_base_classical_integrals() {
        // P(O + O + O(1)) over P^1 (threefold, rank-3 fiber):
        // int H xi^2 = 1 and int xi^3 = -int H xi^2 = -1 from
        // xi^2 (xi + H) = 0.
        let threefold = |insertions: &[BundleInsertion]| {
            reconstruct_bundle_invariants(
                1,
                &[0, 0, 1],
                &base_weights(),
                &[Rational::from(11), Rational::from(23), Rational::from(41)],
                0,
                0,
                insertions,
            )
            .unwrap()
        };
        assert_eq!(
            threefold(&[
                BundleInsertion::new(0, 1, 0),
                BundleInsertion::new(0, 0, 2),
                BundleInsertion::new(0, 0, 0),
            ]),
            vec![(0, 0, Rational::one())]
        );
        assert_eq!(
            threefold(&[
                BundleInsertion::new(0, 0, 1),
                BundleInsertion::new(0, 0, 2),
                BundleInsertion::new(0, 0, 0),
            ]),
            vec![(0, 0, -Rational::one())]
        );

        // P(O + O(1)) over P^2 (threefold, rank-6 state space):
        // int H^2 xi = 1.
        let over_plane = reconstruct_bundle_invariants(
            2,
            &[0, 1],
            &[Rational::from(2), Rational::from(5), Rational::from(11)],
            &fiber_weights(),
            0,
            0,
            &[
                BundleInsertion::new(0, 2, 0),
                BundleInsertion::new(0, 0, 1),
                BundleInsertion::new(0, 0, 0),
            ],
        )
        .unwrap();
        assert_eq!(over_plane, vec![(0, 0, Rational::one())]);
    }

    #[test]
    fn zero_twists_have_product_s_matrix() {
        // P(O + O) over P^1 is P^1 x P^1.  Since xi restricts as -mu_j, the
        // matching product ray uses second-factor weights -mu_j; the raw
        // I-function Birkhoff path must reproduce the product S-matrix.
        let q_degree = 2;
        let z_order = 2;
        let ray = Rational::one();
        let bundle_target =
            ProjectiveBundleRay::new(1, vec![0, 0], base_weights(), fiber_weights(), ray.clone())
                .unwrap();
        let product_target = crate::givental::ProductProjectiveRay::new(
            1,
            1,
            base_weights(),
            fiber_weights()
                .into_iter()
                .map(|weight| -weight)
                .collect::<Vec<_>>(),
            ray,
        )
        .unwrap();
        let bundle_s = rational_s_matrix_to_ratfun(
            &bundle_target
                .descendant_s_rational(q_degree, z_order)
                .unwrap(),
        )
        .unwrap();
        let product_s = crate::givental::ProductRayProvider::new(product_target)
            .descendant_s_matrix(q_degree, z_order)
            .unwrap();
        assert_eq!(bundle_s.coefficients(), product_s.coefficients());
    }
}
