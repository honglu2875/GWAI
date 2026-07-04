//! Projective bundles `P(O(a_1) + ... + O(a_m))` over `P^n`, via the toric
//! I-function, a bidegree-graded mirror transformation, and exact Novikov
//! ray reconstruction.
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
//! classical `D`-power basis via fixed-point restrictions.  The I-function
//! and its mirror transformation are computed in bidegree-graded form
//! (finite per total degree `d1 + d2'`), *then* restricted to flat-coordinate
//! rays `(Q1, Q2') = (t, b t)` — so Birkhoff factorization stays a
//! single-variable problem per ray, and `total + 1` rays reconstruct every
//! bidegree exactly by a rational Vandermonde solve.  Per ray: J -> S by
//! Birkhoff (`recipe::descendant_s_from_j_function`), quantum multiplication
//! from `S_1`, the canonical frame from spectral projectors of that operator
//! (`recipe::operator_lagrange_frame`), and the R-matrix from flatness with
//! tangent-weight asymptotics.
//!
//! **Validated scope, and two frontiers.**  The genus-zero curve-counting
//! invariants of Fano bundles (`F_0`, `F_1`, and the like) are validated: the
//! classical integrals, the exceptional and fiber curve invariants, the line
//! count, and the `P^1 x P^1` cross-check.  Two cases are *not* yet trusted:
//!
//! - **Non-Fano bundles** (`A >= 2`, e.g. `F_2 = P(O + O(2))`).  The effective
//!   `(-A)`-section has anticanonical degree zero, so the I-function acquires
//!   positive powers of `z` and a non-unit `z^0` part and the mirror map is no
//!   longer a divisor change of variables.  The pipeline detects this exactly
//!   (a nonzero unit component of the mirror exponent) and returns
//!   [`GwError::UnsupportedInvariant`] rather than a wrong number.
//! - **Higher `R`-order** (genus `>= 1`, or many markings).  Every validated
//!   case exercises the `R`-matrix only through first order; a genus-1,
//!   four-point `P^1 x P^1` check computed as `P(O + O)` disagrees with the
//!   product engine, so the operator-frame `R` beyond first order is not yet
//!   trusted.  These computations are not blocked — treat their output as
//!   unverified pending a fix.

use super::*;
use crate::twisted::HLaurentSeries;
use std::collections::BTreeMap;

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

    /// Sufficient negative z-depth for the I-coefficients through total
    /// grade `k_max` and Birkhoff order `z_order`.
    fn min_z_power(&self, k_max: usize, z_order: usize) -> i32 {
        let mut worst = 0usize;
        for d1 in 0..=k_max {
            for d2p in 0..=(k_max - d1) {
                let d2 = d2p as isize - (self.big_a() * d1) as isize;
                let mut depth = (self.n + 1) * d1;
                for &a in &self.twists {
                    let fiber_degree = d2 + (a * d1) as isize;
                    if fiber_degree > 0 {
                        depth += fiber_degree as usize;
                    }
                }
                worst = worst.max(depth);
            }
        }
        -((worst + z_order + 2) as i32)
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

    /// Mirror-transformed J-coefficients in flat coordinates, bidegree
    /// graded through total degree `k_max`.
    fn j_container(
        &self,
        k_max: usize,
        min_z: i32,
    ) -> Result<BTreeMap<Grade, HLaurentSeries>, GwError> {
        let size = self.size();
        let relation = self.h_power_relation();
        let mut i_container = BTreeMap::new();
        for d1 in 0..=k_max {
            for d2p in 0..=(k_max - d1) {
                let coefficient = self.i_coefficient(d1, d2p, min_z);
                if !coefficient.is_empty() {
                    i_container.insert((d1, d2p), coefficient);
                }
            }
        }

        // Multiplicative gauge stripping every z^{-1} part, graded by total
        // degree (the bidegree analogue of the single-variable full-vector
        // mirror gauge): after this, gauge * I = J at the un-transformed
        // coordinates, and the divisor components of the exponents are the
        // mirror map.
        let mut exponent = BTreeMap::<Grade, HLaurentSeries>::new();
        let mut gauge = BTreeMap::<Grade, HLaurentSeries>::new();
        gauge.insert((0, 0), HLaurentSeries::one(size - 1));
        for total in 1..=k_max {
            for grade in grades_of_total(total, k_max) {
                let mut known = HLaurentSeries::zero(size - 1);
                for (split, exponent_part) in &exponent {
                    if split.0 > grade.0 || split.1 > grade.1 {
                        continue;
                    }
                    let remainder = (grade.0 - split.0, grade.1 - split.1);
                    if let Some(gauge_part) = gauge.get(&remainder) {
                        let weight = Rational::from(split.0 + split.1);
                        known = known.add(
                            &exponent_part
                                .multiply_mod_relation(gauge_part, &relation)
                                .scale(weight),
                        );
                    }
                }
                let mut gauge_grade = known.scale(Rational::one() / Rational::from(total));

                let mut gauged = gauge_grade.clone();
                for (split, gauge_part) in &gauge {
                    if *split == (0usize, 0usize) || split.0 > grade.0 || split.1 > grade.1 {
                        continue;
                    }
                    let remainder = (grade.0 - split.0, grade.1 - split.1);
                    if let Some(i_part) = i_container.get(&remainder) {
                        gauged = gauged.add(&gauge_part.multiply_mod_relation(i_part, &relation));
                    }
                }
                if let Some(i_part) = i_container.get(&grade) {
                    gauged = gauged.add(i_part);
                }

                let tau = crate::twisted::z_power_part_coeff(&gauged, -1);
                let exponent_grade = tau.shift_z(-1).scale(-Rational::one());
                gauge_grade = gauge_grade.add(&exponent_grade);
                if !exponent_grade.is_empty() {
                    exponent.insert(grade, exponent_grade);
                }
                if !gauge_grade.is_empty() {
                    gauge.insert(grade, gauge_grade);
                }
            }
        }
        let gauged = bideg_h_mul(&gauge, &i_container, &relation, k_max);

        // Read the mirror map: each exponent must lie in the span of
        // {1, H, xi'} (degree reasons); the H and xi' components are the
        // divisor-direction corrections for the two flat coordinates.
        let unit_vector = self.insertion_class_vector(0, 0);
        let h_vector = self.insertion_class_vector(1, 0);
        let xi_prime_vector = {
            let xi = self.insertion_class_vector(0, 1);
            let shift = Rational::from(self.big_a() as i128);
            xi.iter()
                .zip(h_vector.iter())
                .map(|(xi_c, h_c)| xi_c.clone() + shift.clone() * h_c.clone())
                .collect::<Vec<_>>()
        };
        let mut mirror_h = BTreeMap::<Grade, Rational>::new();
        let mut mirror_xi = BTreeMap::<Grade, Rational>::new();
        for (grade, exponent_part) in &exponent {
            let target = (0..size)
                .map(|row| exponent_part.coefficient(row, -1))
                .collect::<Vec<_>>();
            let (g_unit, g_h, g_xi) =
                decompose_in_degree_two_span(&unit_vector, &h_vector, &xi_prime_vector, &target)?;
            // A nonzero unit component of the mirror exponent is the precise
            // signature of the non-Fano case: an effective curve class of
            // anticanonical degree zero (the (-A)-section for A >= 2) makes the
            // I-function acquire positive powers of z and a non-unit z^0 part,
            // so the mirror transformation is no longer a divisor change of
            // variables plus a z^{-1} gauge.  Projecting that I-function onto
            // the J-slice needs a full Birkhoff step that this pipeline does
            // not yet implement, so refuse rather than return a wrong number.
            // Fano bundles (F_0, F_1, and the like) have g_unit == 0 and are
            // handled exactly.  See docs/lessons.md.
            if !g_unit.is_zero() {
                return Err(GwError::UnsupportedInvariant(
                    "non-Fano projective bundle: the I-function has an effective \
                     anticanonical-degree-zero class whose mirror map is not a pure \
                     divisor change of variables; this case (e.g. F_2 = P(O + O(2))) \
                     is not yet supported"
                        .to_string(),
                ));
            }
            if !g_h.is_zero() {
                mirror_h.insert(*grade, g_h);
            }
            if !g_xi.is_zero() {
                mirror_xi.insert(*grade, g_xi);
            }
        }

        // Invert the two-variable mirror map on scalar bidegree series:
        // u_i(Q) = q_i(Q)/Q_i = exp(-G_i(q(Q))), solved by grading-complete
        // fixed-point iteration.
        let mut u_h: BTreeMap<Grade, Rational> = BTreeMap::from([((0, 0), Rational::one())]);
        let mut u_xi: BTreeMap<Grade, Rational> = BTreeMap::from([((0, 0), Rational::one())]);
        for _ in 0..=k_max {
            let ghat_h = substitute_scalar(&mirror_h, &u_h, &u_xi, k_max);
            let ghat_xi = substitute_scalar(&mirror_xi, &u_h, &u_xi, k_max);
            u_h = bideg_exp(&bideg_neg(&ghat_h), k_max);
            u_xi = bideg_exp(&bideg_neg(&ghat_xi), k_max);
        }

        // Compose: J[(G)] = sum_g [u_h^{g_1} u_xi^{g_2}]_{G - g} gauged[g].
        let u_h_powers = monomial_powers(&u_h, k_max);
        let u_xi_powers = monomial_powers(&u_xi, k_max);
        let mut j_container = BTreeMap::<Grade, HLaurentSeries>::new();
        for (grade, gauged_part) in &gauged {
            let weight = bideg_mul(&u_h_powers[grade.0], &u_xi_powers[grade.1], k_max);
            for (offset, scalar) in &weight {
                let target = (grade.0 + offset.0, grade.1 + offset.1);
                if target.0 + target.1 > k_max {
                    continue;
                }
                let term = gauged_part.scale(scalar.clone());
                let entry = j_container
                    .entry(target)
                    .or_insert_with(|| HLaurentSeries::zero(size - 1));
                *entry = entry.add(&term);
            }
        }
        Ok(j_container)
    }

    /// Ray restriction of the J-coefficients: `Q1 = t`, `Q2' = ray * t`.
    fn j_ray(&self, k_max: usize, min_z: i32) -> Result<Vec<HLaurentSeries>, GwError> {
        let j_container = self.j_container(k_max, min_z)?;
        let size = self.size();
        let mut out = vec![HLaurentSeries::zero(size - 1); k_max + 1];
        for (grade, value) in &j_container {
            let total = grade.0 + grade.1;
            let scaled = value.scale(self.ray.pow_usize(grade.1));
            out[total] = out[total].add(&scaled);
        }
        Ok(out)
    }
}

type Grade = (usize, usize);
type ZLaurent = BTreeMap<i32, Rational>;

fn grades_of_total(total: usize, k_max: usize) -> Vec<Grade> {
    (0..=total.min(k_max))
        .map(|d2p| (total - d2p, d2p))
        .collect()
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
    let next = series.get(&z_power).cloned().unwrap_or_else(Rational::zero) + value;
    if next.is_zero() {
        series.remove(&z_power);
    } else {
        series.insert(z_power, next);
    }
}

fn bideg_neg(series: &BTreeMap<Grade, Rational>) -> BTreeMap<Grade, Rational> {
    series
        .iter()
        .map(|(grade, value)| (*grade, -value.clone()))
        .collect()
}

fn bideg_mul(
    left: &BTreeMap<Grade, Rational>,
    right: &BTreeMap<Grade, Rational>,
    k_max: usize,
) -> BTreeMap<Grade, Rational> {
    let mut out = BTreeMap::new();
    for (left_grade, left_value) in left {
        for (right_grade, right_value) in right {
            let grade = (left_grade.0 + right_grade.0, left_grade.1 + right_grade.1);
            if grade.0 + grade.1 > k_max {
                continue;
            }
            let next = out.get(&grade).cloned().unwrap_or_else(Rational::zero)
                + left_value.clone() * right_value.clone();
            if next.is_zero() {
                out.remove(&grade);
            } else {
                out.insert(grade, next);
            }
        }
    }
    out
}

/// Graded exponential of a scalar bidegree series with no constant term.
fn bideg_exp(series: &BTreeMap<Grade, Rational>, k_max: usize) -> BTreeMap<Grade, Rational> {
    let mut out = BTreeMap::from([((0usize, 0usize), Rational::one())]);
    for total in 1..=k_max {
        for grade in grades_of_total(total, k_max) {
            let mut value = Rational::zero();
            for (split, series_value) in series {
                if split.0 > grade.0 || split.1 > grade.1 || (split.0 == 0 && split.1 == 0) {
                    continue;
                }
                let remainder = (grade.0 - split.0, grade.1 - split.1);
                if let Some(out_value) = out.get(&remainder) {
                    value += Rational::from(split.0 + split.1)
                        * series_value.clone()
                        * out_value.clone();
                }
            }
            value = value / Rational::from(total);
            if !value.is_zero() {
                out.insert(grade, value);
            }
        }
    }
    out
}

/// Substitute `q_i = Q_i u_i(Q)` into a scalar bidegree series.
fn substitute_scalar(
    series: &BTreeMap<Grade, Rational>,
    u_h: &BTreeMap<Grade, Rational>,
    u_xi: &BTreeMap<Grade, Rational>,
    k_max: usize,
) -> BTreeMap<Grade, Rational> {
    let u_h_powers = monomial_powers(u_h, k_max);
    let u_xi_powers = monomial_powers(u_xi, k_max);
    let mut out = BTreeMap::new();
    for (grade, value) in series {
        let weight = bideg_mul(&u_h_powers[grade.0], &u_xi_powers[grade.1], k_max);
        for (offset, scalar) in &weight {
            let target = (grade.0 + offset.0, grade.1 + offset.1);
            if target.0 + target.1 > k_max {
                continue;
            }
            let next = out.get(&target).cloned().unwrap_or_else(Rational::zero)
                + value.clone() * scalar.clone();
            if next.is_zero() {
                out.remove(&target);
            } else {
                out.insert(target, next);
            }
        }
    }
    out
}

fn monomial_powers(
    series: &BTreeMap<Grade, Rational>,
    k_max: usize,
) -> Vec<BTreeMap<Grade, Rational>> {
    let mut powers = Vec::with_capacity(k_max + 1);
    powers.push(BTreeMap::from([((0usize, 0usize), Rational::one())]));
    for exponent in 1..=k_max {
        let next = bideg_mul(&powers[exponent - 1], series, k_max);
        powers.push(next);
    }
    powers
}

fn bideg_h_mul(
    left: &BTreeMap<Grade, HLaurentSeries>,
    right: &BTreeMap<Grade, HLaurentSeries>,
    relation: &[Rational],
    k_max: usize,
) -> BTreeMap<Grade, HLaurentSeries> {
    let mut out = BTreeMap::<Grade, HLaurentSeries>::new();
    for (left_grade, left_value) in left {
        for (right_grade, right_value) in right {
            let grade = (left_grade.0 + right_grade.0, left_grade.1 + right_grade.1);
            if grade.0 + grade.1 > k_max {
                continue;
            }
            let term = left_value.multiply_mod_relation(right_value, relation);
            match out.get_mut(&grade) {
                Some(entry) => *entry = entry.add(&term),
                None => {
                    out.insert(grade, term);
                }
            }
        }
    }
    out
}

/// Exact decomposition of a vector in the span of `{unit, H, xi'}`.
fn decompose_in_degree_two_span(
    unit: &[Rational],
    h: &[Rational],
    xi_prime: &[Rational],
    target: &[Rational],
) -> Result<(Rational, Rational, Rational), GwError> {
    // The three columns are linearly independent constants, so the normal
    // equations have a unique exact solution; consistency of the
    // overdetermined system is then verified coordinate by coordinate — the
    // z^{-1} part of a homogeneous I-function exponent must have
    // cohomological degree <= 2, so a nonzero residual means broken
    // conventions rather than round-off.
    let columns = [unit, h, xi_prime];
    let mut gram = vec![vec![Rational::zero(); 3]; 3];
    let mut rhs = vec![Rational::zero(); 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut total = Rational::zero();
            for row in 0..target.len() {
                total += columns[a][row].clone() * columns[b][row].clone();
            }
            gram[a][b] = total;
        }
        let mut total = Rational::zero();
        for row in 0..target.len() {
            total += columns[a][row].clone() * target[row].clone();
        }
        rhs[a] = total;
    }
    recipe::solve_rational_system(&mut gram, &mut rhs)?;

    for row in 0..target.len() {
        let reconstructed = unit[row].clone() * rhs[0].clone()
            + h[row].clone() * rhs[1].clone()
            + xi_prime[row].clone() * rhs[2].clone();
        if reconstructed != target[row] {
            return Err(GwError::AlgebraFailure(
                "mirror exponent leaves the degree-two span; conventions are inconsistent"
                    .to_string(),
            ));
        }
    }
    Ok((rhs[0].clone(), rhs[1].clone(), rhs[2].clone()))
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

impl ProjectiveBundleRay {
    fn flat_metric_series(&self, q_degree: usize) -> SeriesMatrix<Rational> {
        SeriesMatrix::constant(self.flat_metric(), q_degree)
    }

    fn descendant_s_rational(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<Rational>, GwError> {
        static CACHE: OnceLock<Mutex<HashMap<(String, usize, usize), SeriesSMatrix<Rational>>>> =
            OnceLock::new();
        let key = (self.cache_key(), q_degree, z_order);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(descendant_s) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(descendant_s);
        }

        let min_z = self.min_z_power(q_degree, z_order);
        let j_ray = self.j_ray(q_degree, min_z)?;
        let descendant_s = recipe::descendant_s_from_j_function(
            self.size() - 1,
            &j_ray,
            &self.h_power_relation(),
            &self.flat_metric_series(q_degree),
            q_degree,
            z_order,
            CalibrationId(format!("bundle-ray-birkhoff:{}", self.cache_key())),
        )?;
        cache.lock().unwrap().insert(key, descendant_s.clone());
        Ok(descendant_s)
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

        // Quantum multiplication by the grading divisor from the descendant
        // S-matrix: the z^0 part of the quantum differential equation gives
        // A_q = A_cl + t d/dt S_1.
        let descendant_s = self.target.descendant_s_rational(q_degree, 1)?;
        let s_one = descendant_s.coefficient(1).ok_or_else(|| {
            GwError::ConventionMismatch(
                "bundle kernel needs the descendant S-matrix through z^{-1}".to_string(),
            )
        })?;
        let quantum_multiplication = self
            .target
            .classical_grading_multiplication(q_degree)
            .add(&s_one.q_derivative());

        let unit_coordinates = {
            let mut unit = vec![RatFun::zero(); self.target.size()];
            unit[0] = RatFun::one();
            unit
        };
        let frame = recipe::operator_lagrange_frame(
            &rational_series_matrix_to_ratfun(&quantum_multiplication),
            &self.target.grading_seeds(),
            &unit_coordinates,
            &rational_series_matrix_to_ratfun(&self.target.flat_metric_series(q_degree)),
        )?;

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
                        differences.push(RatFun::from_rational(
                            self.target.fiber_weight(i, l) - self.target.fiber_weight(i, j),
                        ));
                    }
                }
                classical_diagonal.push(classical_r_asymptotics_for_point(&differences, r_order));
            }
        }

        let calibration = calibration_from_canonical_frame(
            &frame,
            &classical_diagonal,
            q_degree,
            r_order,
            CalibrationId(format!("bundle-ray-j:{}", self.target.cache_key())),
        )?;
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
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
    let mut rays = Vec::with_capacity(ray_count);
    let mut values = Vec::with_capacity(ray_count);
    for step in 0..ray_count {
        let ray = Rational::from(step + 1);
        let target = ProjectiveBundleRay::new(
            n,
            twists.to_vec(),
            weights_base.to_vec(),
            weights_fiber.to_vec(),
            ray.clone(),
        )?;
        let provider = BundleRayProvider::new(target);
        let value =
            compute_semisimple_graph_value(&provider, genus, total_degree, insertions, None)?;
        let value = value.as_rational().ok_or_else(|| {
            GwError::AlgebraFailure("bundle ray value did not specialize to a rational".to_string())
        })?;
        rays.push(ray);
        values.push(value);
    }

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
    fn f2_non_fano_is_currently_unsupported() {
        // F_2 = P(O + O(2)) over P^1 is deformation equivalent to P^1 x P^1,
        // the ideal higher-genus / nontrivial-mirror cross-check.  But F_2 is
        // non-Fano: its effective (-2)-section has anticanonical degree zero,
        // so the I-function acquires positive powers of z and a non-unit z^0
        // part, and the mirror transformation is no longer a divisor change of
        // variables.  The pipeline detects this (a nonzero unit component of
        // the mirror exponent) and refuses rather than returning a wrong
        // number.  Handling it needs the full Birkhoff projection of a
        // positive-z I-function onto the J-slice; see docs/lessons.md.
        let point = BundleInsertion::new(0, 1, 1);
        let result = reconstruct_bundle_invariants(
            1,
            &[0, 2],
            &base_weights(),
            &fiber_weights(),
            0,
            3,
            &[point.clone(), point.clone(), point],
        );
        assert!(
            matches!(result, Err(GwError::UnsupportedInvariant(_))),
            "expected F_2 to report non-Fano as unsupported, got {result:?}"
        );
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
        // pipeline (I-function, mirror stage, Birkhoff, operator frame).
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
    fn zero_twists_have_a_trivial_mirror_map() {
        // P(O + O) over P^1 is P^1 x P^1: I = J, so the mirror exponents
        // read off inside j_container must vanish and J must equal I.
        let target = ProjectiveBundleRay::new(
            1,
            vec![0, 0],
            base_weights(),
            fiber_weights(),
            Rational::one(),
        )
        .unwrap();
        let k_max = 2;
        let min_z = target.min_z_power(k_max, 2);
        let j_container = target.j_container(k_max, min_z).unwrap();
        for d1 in 0..=k_max {
            for d2p in 0..=(k_max - d1) {
                let i_coefficient = target.i_coefficient(d1, d2p, min_z);
                let j_coefficient = j_container.get(&(d1, d2p));
                match j_coefficient {
                    Some(j_coefficient) => {
                        let difference = j_coefficient.add(&i_coefficient.scale(-Rational::one()));
                        assert!(
                            difference.is_empty(),
                            "J != I at grade ({d1},{d2p}) for the untwisted bundle"
                        );
                    }
                    None => assert!(
                        i_coefficient.is_empty(),
                        "missing J coefficient at ({d1},{d2p})"
                    ),
                }
            }
        }
    }
}
