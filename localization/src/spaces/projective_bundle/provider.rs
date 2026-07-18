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
//! projected cone point.  Its positive-degree `z^{-1}` unit coordinate is
//! gauged away, while its two divisor mirror-coordinate series are gauged away
//! and inverted before any ray specialization.  A surviving higher-primary
//! `z^{-1}` coordinate would describe a big-quantum path rather than the small
//! divisor slice, so the backend fails closed until generalized mirror
//! normalization is implemented.  Only a cone point with no such remainder is
//! restricted to rays `(Q1, Q2') = (t, b t)` and regenerated into the
//! fundamental solution used by the graph engine and exact Vandermonde
//! recovery.  The raw fundamental solution gives quantum multiplication by
//! `D`; its metric-adjoint gives the descendant insertion operator, and
//! flatness gives `R`.
//!
//! **Validated scope.**  Regression tests cover Fano genus-zero bundle counts,
//! `P(O + O) = P^1 x P^1` through higher `R` order, and the non-Fano
//! `F_2 = P(O + O(2))` deformation dictionary against the independent product
//! engine, including genus-one cases, and the normalized mixed-sign bundle
//! `P(O + O(3) + O(3)) -> P^2`.  The normalized `F_4` presentation `[0,4]`
//! and tested non-nef rank-three presentations `[0,1,2]` and `[0,4,5]`
//! retain higher-primary `z^{-1}` coordinates and return `UnsupportedFeature`
//! pending generalized mirror normalization; their former isolated numerical
//! coincidences are not small-theory validation.

use crate::core::algebra::{RatFun, Rational};
use crate::core::bounded_cache::{BoundedCache, TARGET_RECONSTRUCTION_CACHE_CAPACITY};
use crate::core::error::GwError;
use crate::core::series::{QSeries, SeriesMatrix};
use crate::core::theory::{CurveClass, GwTheory};
use crate::givental::recipe::{
    birkhoff_descendant_s_matrix_from_fundamental_coeff,
    fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff,
    metric_adjoint_descendant_s_matrix_coeff,
};
use crate::givental::{
    calibration_from_canonical_frame, classical_r_asymptotics_for_point,
    compute_semisimple_graph_value, is_stable_cohft_range, rational_qseries_to_ratfun, recipe,
    CalibrationId, GiventalGraphKernel, SemisimpleCohftProvider, SeriesSMatrix, Truncation,
};
use crate::reconstruction::{
    plan_birkhoff_windows, solve_rational_system, BidegreeLaurentFactor, CyclicCoordinates,
    CyclicQuantumAlgebra, ExactRayInterpolation, HLaurentSeries, LaurentCoeffMatrix,
};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::ProjectiveBundleTheory;

#[cfg(test)]
use crate::givental::compute_semisimple_graph_series;
#[cfg(test)]
use crate::reconstruction::zero_coeff_matrix;
#[cfg(test)]
use crate::reconstruction::MAX_EXACT_RECONSTRUCTION_RAYS;

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
    theory: ProjectiveBundleTheory,
    weights_base: Vec<Rational>,
    weights_fiber: Vec<Rational>,
    ray: Rational,
}

#[derive(Debug, Clone)]
struct ProjectiveBundleClassicalData {
    fiber_weights: Vec<Vec<Rational>>,
    xi_restrictions: Vec<Rational>,
    grading_seeds: Vec<Rational>,
    transition: Vec<Vec<Rational>>,
    flat_metric: Vec<Vec<Rational>>,
    h_power_relation: Vec<Rational>,
}

impl ProjectiveBundleRay {
    pub fn new(
        n: usize,
        twists: Vec<usize>,
        weights_base: Vec<Rational>,
        weights_fiber: Vec<Rational>,
        ray: Rational,
    ) -> Result<Self, GwError> {
        // Validate the target geometry before adapting equivariant weights.
        // In particular, do not let `zip` silently discard summands when a
        // caller supplies too few fiber weights.
        let theory = ProjectiveBundleTheory::new(n, twists.clone())?;
        let weights_fiber = theory.canonicalize_summand_payloads(twists, weights_fiber)?;
        Self::from_theory(theory, weights_base, weights_fiber, ray)
    }

    /// Construct a ray adapter over an already validated canonical theory.
    ///
    /// Fiber weights must follow [`ProjectiveBundleTheory::twists`], whose
    /// order is canonical.  Reconstruction and constraint evaluators use this
    /// entry point so every ray shares the caller's geometric authority.
    pub fn from_theory(
        theory: ProjectiveBundleTheory,
        weights_base: Vec<Rational>,
        weights_fiber: Vec<Rational>,
        ray: Rational,
    ) -> Result<Self, GwError> {
        let expected_base = theory.base_dimension().checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("bundle base dimension is too large".to_string())
        })?;
        expected_base.checked_mul(theory.rank()).ok_or_else(|| {
            GwError::UnsupportedInvariant("bundle state-space size overflow".to_string())
        })?;
        if weights_base.len() != expected_base || weights_fiber.len() != theory.rank() {
            return Err(GwError::ConventionMismatch(format!(
                "bundle weights must have lengths {} and {}",
                expected_base,
                theory.rank()
            )));
        }
        let target = Self {
            theory,
            weights_base,
            weights_fiber,
            ray,
        };
        // Validate the fixed-point grading before constructing the Lagrange
        // frame.  `classical_data()` divides by pairwise seed and Euler-class
        // differences, so discovering a collision through that path would
        // panic in exact rational arithmetic instead of returning the
        // intended non-semisimple error.
        let seeds = target.raw_grading_seeds();
        for left in 0..seeds.len() {
            for right in left + 1..seeds.len() {
                if seeds[left] == seeds[right] {
                    return Err(GwError::NonSemisimplePoint);
                }
            }
        }
        Ok(target)
    }

    /// Ordinary geometric data before Novikov-ray specialization.
    pub fn canonical_theory(&self) -> &ProjectiveBundleTheory {
        &self.theory
    }

    /// Dimension of the projective-space base.
    pub fn base_dimension(&self) -> usize {
        self.theory.base_dimension()
    }

    /// Canonically ordered bundle twists.
    pub fn twists(&self) -> &[usize] {
        self.theory.twists()
    }

    /// Rank of the vector bundle being projectivized.
    pub fn rank(&self) -> usize {
        self.theory.rank()
    }

    /// Dimension of the canonical cohomology state space.
    pub fn size(&self) -> usize {
        self.theory.state_space().basis.len()
    }

    /// Equivariant weights of the base fixed points.
    pub fn base_weights(&self) -> &[Rational] {
        &self.weights_base
    }

    /// Equivariant fiber weights, in the same canonical summand order as
    /// [`Self::twists`].
    pub fn fiber_weights(&self) -> &[Rational] {
        &self.weights_fiber
    }

    /// Novikov-ray specialization in the shifted fiber grading.
    pub fn ray(&self) -> &Rational {
        &self.ray
    }

    fn point(&self, i: usize, j: usize) -> usize {
        i * self.rank() + j
    }

    fn big_a(&self) -> usize {
        *self.twists().iter().max().expect("twists nonempty")
    }

    fn raw_fiber_weight(
        twists: &[usize],
        weights_base: &[Rational],
        weights_fiber: &[Rational],
        i: usize,
        l: usize,
    ) -> Rational {
        Rational::from(twists[l] as i128) * weights_base[i].clone() + weights_fiber[l].clone()
    }

    fn raw_grading_seeds(&self) -> Vec<Rational> {
        let shift = Rational::from((self.big_a() + 1) as i128);
        let mut seeds = Vec::with_capacity(self.size());
        for i in 0..=self.base_dimension() {
            for j in 0..self.rank() {
                seeds.push(
                    -Self::raw_fiber_weight(
                        self.twists(),
                        &self.weights_base,
                        &self.weights_fiber,
                        i,
                        j,
                    ) + shift.clone() * self.weights_base[i].clone(),
                );
            }
        }
        seeds
    }

    fn build_classical_data(&self) -> ProjectiveBundleClassicalData {
        let rank = self.rank();
        let size = self.size();
        let big_a = self.big_a();
        let mut fiber_weights = vec![vec![Rational::zero(); rank]; self.base_dimension() + 1];
        for (i, row) in fiber_weights.iter_mut().enumerate() {
            for (l, value) in row.iter_mut().enumerate() {
                *value = Self::raw_fiber_weight(
                    self.twists(),
                    &self.weights_base,
                    &self.weights_fiber,
                    i,
                    l,
                );
            }
        }

        let mut xi_restrictions = vec![Rational::zero(); size];
        let mut grading_seeds = vec![Rational::zero(); size];
        let shift = Rational::from((big_a + 1) as i128);
        for i in 0..=self.base_dimension() {
            for j in 0..rank {
                let point = self.point(i, j);
                xi_restrictions[point] = -fiber_weights[i][j].clone();
                grading_seeds[point] =
                    xi_restrictions[point].clone() + shift.clone() * self.weights_base[i].clone();
            }
        }

        let mut eulers = vec![Rational::one(); size];
        for i in 0..=self.base_dimension() {
            for j in 0..rank {
                let mut euler = Rational::one();
                for k in 0..=self.base_dimension() {
                    if k != i {
                        euler =
                            euler * (self.weights_base[i].clone() - self.weights_base[k].clone());
                    }
                }
                for l in 0..rank {
                    if l != j {
                        euler = euler * (fiber_weights[i][l].clone() - fiber_weights[i][j].clone());
                    }
                }
                eulers[self.point(i, j)] = euler;
            }
        }

        let transition = recipe::classical_lagrange_transition(&grading_seeds);
        let mut flat_metric = vec![vec![Rational::zero(); size]; size];
        for row in 0..size {
            for col in 0..size {
                let mut total = Rational::zero();
                for point in 0..size {
                    total += grading_seeds[point].pow_usize(row + col) / eulers[point].clone();
                }
                flat_metric[row][col] = total;
            }
        }

        let mut coefficients = vec![Rational::one()];
        for seed in &grading_seeds {
            let mut next = vec![Rational::zero(); coefficients.len() + 1];
            for (power, coefficient) in coefficients.iter().enumerate() {
                next[power] += -(seed.clone()) * coefficient.clone();
                next[power + 1] += coefficient.clone();
            }
            coefficients = next;
        }
        let h_power_relation = (0..size)
            .map(|power| -coefficients[power].clone())
            .collect();

        ProjectiveBundleClassicalData {
            fiber_weights,
            xi_restrictions,
            grading_seeds,
            transition,
            flat_metric,
            h_power_relation,
        }
    }

    fn classical_data(&self) -> Arc<ProjectiveBundleClassicalData> {
        static CACHE: OnceLock<Mutex<BoundedCache<String, Arc<ProjectiveBundleClassicalData>>>> =
            OnceLock::new();
        let key = self.rayless_cache_key();
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some(data) = cache.lock().unwrap().get(&key).cloned() {
            return data;
        }
        let data = Arc::new(self.build_classical_data());
        cache.lock().unwrap().insert(key, data.clone());
        data
    }

    /// Weight of the `l`-th fiber coordinate over base fixed point `i`.
    fn fiber_weight(&self, i: usize, l: usize) -> Rational {
        self.classical_data().fiber_weights[i][l].clone()
    }

    /// Classical eigenvalues of the grading divisor `D = xi + (A+1) H`.
    fn grading_seeds(&self) -> Vec<Rational> {
        self.classical_data().grading_seeds.clone()
    }

    /// Atiyah-Bott flat metric in the classical `D`-power basis.
    fn flat_metric(&self) -> Vec<Vec<Rational>> {
        self.classical_data().flat_metric.clone()
    }

    /// Classical relation `D^size = sum_k rel_k D^k` (ascending, length
    /// `size`), from the minimal polynomial `prod (x - seed)`.
    fn h_power_relation(&self) -> Vec<Rational> {
        self.classical_data().h_power_relation.clone()
    }

    /// Classical coordinates of `H^p xi^q` in the `D`-power basis.
    fn insertion_class_vector(&self, h_power: usize, xi_power: usize) -> Vec<Rational> {
        let size = self.size();
        let data = self.classical_data();
        let mut vector = vec![Rational::zero(); size];
        for i in 0..=self.base_dimension() {
            for j in 0..self.rank() {
                let restriction = self.weights_base[i].pow_usize(h_power)
                    * data.xi_restrictions[self.point(i, j)].pow_usize(xi_power);
                for row in 0..size {
                    vector[row] +=
                        data.transition[row][self.point(i, j)].clone() * restriction.clone();
                }
            }
        }
        vector
    }

    fn cache_key(&self) -> String {
        format!(
            "p{}bundle[{:?};{};{}]@{}",
            self.base_dimension(),
            self.twists(),
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
            self.base_dimension(),
            self.twists(),
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

    fn bidegree_birkhoff_bounds(
        &self,
        k_max: usize,
        z_order: usize,
    ) -> Result<BidegreeBirkhoffBounds, GwError> {
        let column_shift = self.size().checked_sub(1).ok_or_else(|| {
            GwError::AlgebraFailure("bundle state space must be nonempty".to_string())
        })?;
        let column_shift_i32 = i32::try_from(column_shift).map_err(|_| {
            GwError::UnsupportedInvariant(
                "bundle state-space dimension exceeds Laurent exponent range".to_string(),
            )
        })?;
        let preview_min_z = column_shift_i32.checked_neg().ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "bundle preview Laurent exponent exceeds supported range".to_string(),
            )
        })?;
        let preview_cone = self.i_container(k_max, preview_min_z)?;
        let preview = self.fundamental_bidegree_matrix(k_max, &preview_cone);
        let raw_positive_windows = max_nonnegative_z_power_by_grade(&preview);
        let base_depth = z_order.checked_add(column_shift).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "bundle Birkhoff Laurent-depth calculation overflowed".to_string(),
            )
        })?;
        let grades = (1..=k_max)
            .flat_map(|total| (0..=total).map(move |first| (first, total - first)))
            .collect::<Vec<_>>();
        let truncation_plan = plan_birkhoff_windows(
            &grades,
            &raw_positive_windows,
            base_depth,
            |grade| {
                let mut splits = Vec::new();
                for left_first in 0..=grade.0 {
                    for left_second in 0..=grade.1 {
                        let left_grade = (left_first, left_second);
                        if left_grade == (0, 0) || left_grade == *grade {
                            continue;
                        }
                        splits.push((left_grade, (grade.0 - left_first, grade.1 - left_second)));
                    }
                }
                splits
            },
            "bundle Birkhoff Laurent-depth calculation overflowed",
        )?;
        let positive_windows = truncation_plan.positive_windows;
        let negative_depths = truncation_plan.negative_depths;
        let max_depth = negative_depths
            .values()
            .copied()
            .max()
            .unwrap_or(base_depth);
        let max_depth_i32 = i32::try_from(max_depth).map_err(|_| {
            GwError::UnsupportedInvariant(
                "bundle Birkhoff Laurent depth exceeds supported exponent range".to_string(),
            )
        })?;

        // In the graded Birkhoff recursion, a positive-factor term z^p can
        // move an already-computed negative coefficient z^(-s-p) into z^-s.
        // The dynamic bound below follows the actual bidegree dependency
        // chains instead of applying the worst positive-z window at every
        // total-degree drop.  The final column shift accounts for repeated
        // (z q d/dq + D) derivatives used to build the raw fundamental matrix
        // from I.
        Ok(BidegreeBirkhoffBounds {
            min_z: max_depth_i32.checked_neg().ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "bundle Birkhoff Laurent exponent exceeds supported range".to_string(),
                )
            })?,
            positive_z_windows: positive_windows,
            negative_z_depths: negative_depths,
        })
    }

    /// Sufficient negative z-depth for the I-coefficients through total
    /// grade `k_max` and Birkhoff order `z_order`.
    #[cfg(test)]
    fn min_z_power(&self, k_max: usize, z_order: usize) -> Result<i32, GwError> {
        Ok(self.bidegree_birkhoff_bounds(k_max, z_order)?.min_z)
    }

    fn shifted_fiber_degrees(&self, d1: usize, d2p: usize) -> Result<Vec<i64>, GwError> {
        let curve = self.theory.curve_from_shifted(d1, d2p)?;
        let d1 = i64::try_from(d1).map_err(|_| {
            GwError::UnsupportedInvariant(
                "bundle base degree exceeds the signed curve lattice".to_string(),
            )
        })?;
        let d2 = curve.coordinate(1).ok_or_else(|| {
            GwError::AlgebraFailure("bundle curve class has the wrong rank".to_string())
        })?;

        self.twists()
            .iter()
            .map(|&twist| {
                let twist = i64::try_from(twist).map_err(|_| {
                    GwError::UnsupportedInvariant(
                        "bundle twist exceeds the signed curve lattice".to_string(),
                    )
                })?;
                twist
                    .checked_mul(d1)
                    .and_then(|offset| d2.checked_add(offset))
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant(
                            "bundle fiber degree calculation overflowed".to_string(),
                        )
                    })
            })
            .collect()
    }

    /// Scalar z-Laurent restriction of the `(d1, d2)` I-coefficient at the
    /// fixed point `(i, j)`.
    fn i_restriction_with_data(
        &self,
        data: &ProjectiveBundleClassicalData,
        i: usize,
        j: usize,
        d1: usize,
        fiber_degrees: &[i64],
        min_z: i32,
    ) -> Result<ZLaurent, GwError> {
        let mut out = zl_one();

        // Negative-degree fiber factors are polynomials in z.  Apply all of
        // them exactly before truncating any inverse-factor expansion: a term
        // such as (c-z) can raise a coefficient from z^(-s-1) to z^-s.
        // Scalar factors commute, and every remaining inverse factor only
        // lowers z-degree, so truncation at `min_z` is safe after this phase.
        for (l, &fiber_degree) in fiber_degrees.iter().enumerate() {
            if fiber_degree >= 0 {
                continue;
            }
            let value = data.fiber_weights[i][l].clone() - data.fiber_weights[i][j].clone();
            let first_factor = fiber_degree.checked_add(1).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "bundle negative fiber-factor range overflowed".to_string(),
                )
            })?;
            for k in first_factor..=0 {
                out = zl_mul_affine_exact(&out, &value, k)?;
            }
            if out.is_empty() {
                return Ok(out);
            }
        }

        for k in 1..=d1 {
            for i_prime in 0..=self.base_dimension() {
                let constant = self.weights_base[i].clone() - self.weights_base[i_prime].clone();
                out = zl_mul_inverse_affine(&out, &constant, k, min_z);
            }
        }
        for (l, &fiber_degree) in fiber_degrees.iter().enumerate() {
            if fiber_degree < 0 {
                continue;
            }
            let value = data.fiber_weights[i][l].clone() - data.fiber_weights[i][j].clone();
            let fiber_degree = usize::try_from(fiber_degree).map_err(|_| {
                GwError::UnsupportedInvariant(
                    "bundle fiber degree exceeds the platform range".to_string(),
                )
            })?;
            for k in 1..=fiber_degree {
                out = zl_mul_inverse_affine(&out, &value, k, min_z);
            }
            if out.is_empty() {
                return Ok(out);
            }
        }
        out.retain(|z_power, _| *z_power >= min_z);
        Ok(out)
    }

    /// The `(d1, d2')` I-coefficient in the classical `D`-power basis.
    fn i_coefficient(&self, d1: usize, d2p: usize, min_z: i32) -> Result<HLaurentSeries, GwError> {
        let size = self.size();
        let fiber_degrees = self.shifted_fiber_degrees(d1, d2p)?;
        let data = self.classical_data();
        let restrictions = (0..=self.base_dimension())
            .flat_map(|i| (0..self.rank()).map(move |j| (i, j)))
            .map(|(i, j)| self.i_restriction_with_data(&data, i, j, d1, &fiber_degrees, min_z))
            .collect::<Result<Vec<_>, _>>()?;

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
                        total += data.transition[row][point].clone() * value.clone();
                    }
                }
                if !total.is_zero() {
                    out.add_term(row, z_power, total);
                }
            }
        }
        Ok(out)
    }

    /// Bidegree-graded `I`-coefficients through shifted total degree `k_max`.
    fn i_container(
        &self,
        k_max: usize,
        min_z: i32,
    ) -> Result<BTreeMap<Grade, HLaurentSeries>, GwError> {
        let mut container = BTreeMap::new();
        for d1 in 0..=k_max {
            for d2p in 0..=(k_max - d1) {
                let coefficient = self.i_coefficient(d1, d2p, min_z)?;
                if !coefficient.is_empty() {
                    container.insert((d1, d2p), coefficient);
                }
            }
        }
        Ok(container)
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

fn zl_one() -> ZLaurent {
    BTreeMap::from([(0, Rational::one())])
}

/// Multiply exactly by the affine factor `constant + k z`.
fn zl_mul_affine_exact(
    series: &ZLaurent,
    constant: &Rational,
    k: i64,
) -> Result<ZLaurent, GwError> {
    let mut out = ZLaurent::new();
    for (z_power, coefficient) in series {
        if !constant.is_zero() {
            add_zl_term_exact(&mut out, *z_power, coefficient.clone() * constant.clone());
        }
        if k != 0 {
            let raised_power = z_power.checked_add(1).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "bundle Laurent numerator exceeds supported exponent range".to_string(),
                )
            })?;
            add_zl_term_exact(
                &mut out,
                raised_power,
                coefficient.clone() * Rational::from(k),
            );
        }
    }
    Ok(out)
}

/// Multiply by `(constant + k z)^{-1}` for `k >= 1`, expanded around
/// `z = infinity` and truncated below `min_z`.
fn zl_mul_inverse_affine(series: &ZLaurent, constant: &Rational, k: usize, min_z: i32) -> ZLaurent {
    let mut out = ZLaurent::new();
    let k_rational = Rational::from(k);
    for (z_power, coefficient) in series {
        // (c + kz)^{-1} = sum_{r >= 0} (-c)^r k^{-r-1} z^{-r-1}.
        let mut factor = Rational::one() / k_rational.clone();
        let Some(mut power) = z_power.checked_sub(1) else {
            continue;
        };
        while power >= min_z {
            add_zl_term(&mut out, power, coefficient.clone() * factor.clone(), min_z);
            factor = factor * (-constant.clone()) / k_rational.clone();
            if constant.is_zero() {
                break;
            }
            let Some(next_power) = power.checked_sub(1) else {
                break;
            };
            power = next_power;
        }
    }
    out
}

fn add_zl_term_exact(series: &mut ZLaurent, z_power: i32, value: Rational) {
    if value.is_zero() {
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
            if solve_rational_system(&mut matrix, &mut values).is_ok() {
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
    ) -> Result<
        (
            ScalarBidegreeSeries,
            ScalarBidegreeSeries,
            ScalarBidegreeSeries,
        ),
        GwError,
    > {
        let mut mirror_unit = ScalarBidegreeSeries::new();
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
            let (unit_coordinate, h_coordinate, shifted_fiber_coordinate) =
                self.solve_unit_divisor_coordinates(&vector)?;
            scalar_bidegree_add_term(&mut mirror_unit, grade, unit_coordinate, q_degree);
            scalar_bidegree_add_term(&mut mirror_first, grade, h_coordinate, q_degree);
            scalar_bidegree_add_term(
                &mut mirror_second,
                grade,
                shifted_fiber_coordinate,
                q_degree,
            );
        }
        Ok((mirror_unit, mirror_first, mirror_second))
    }

    fn bidegree_mirror_gauge(
        &self,
        mirror_unit: &ScalarBidegreeSeries,
        mirror_first: &ScalarBidegreeSeries,
        mirror_second: &ScalarBidegreeSeries,
        q_degree: usize,
    ) -> HLaurentBidegreeSeries {
        // The J-slice fixes the flat coordinate along the unit as well as the
        // two divisor coordinates.  Leaving a q-dependent unit coefficient
        // in z^-1 evaluates the CohFT at a moving string-direction point; in
        // particular q d/dq then acquires an extra unit derivative and no
        // longer agrees with insertion of the grading divisor.  Non-Fano,
        // mixed-sign bundles can have this unit mirror coordinate even after
        // the upper-loop Birkhoff projection, so gauge it away explicitly.
        let unit = self.insertion_class_vector(0, 0);
        let h = self.insertion_class_vector(1, 0);
        let shifted_fiber = self.shifted_fiber_divisor_vector();
        let mut exponent = HLaurentBidegreeSeries::new();
        for (&grade, coefficient) in mirror_unit {
            let series = exponent
                .entry(grade)
                .or_insert_with(|| HLaurentSeries::zero(self.size() - 1));
            for (row, unit_coefficient) in unit.iter().enumerate() {
                series.add_term(row, -1, -(coefficient.clone() * unit_coefficient.clone()));
            }
        }
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
        let (mirror_unit, mirror_first, mirror_second) =
            self.bidegree_mirror_maps_from_cone_point(cone_point, q_degree)?;
        let gauge =
            self.bidegree_mirror_gauge(&mirror_unit, &mirror_first, &mirror_second, q_degree);
        let gauged = h_bidegree_mul(
            &gauge,
            cone_point,
            &self.h_power_relation(),
            self.size() - 1,
            q_degree,
        );
        let (inverse_first, inverse_second) =
            invert_bidegree_mirror_map(&mirror_first, &mirror_second, q_degree);
        let flat_cone_point =
            h_bidegree_compose(&gauged, &inverse_first, &inverse_second, q_degree);

        // A two-parameter small J-function may have no positive-degree
        // z^-1 coordinate after removing the unit and two divisor mirror
        // coordinates.  A surviving higher-cohomology component means that
        // the I-function has landed on a genuinely big-quantum path.  Along
        // that path q d/dq is insertion of the grading divisor plus additional
        // primary directions, so treating it as the small divisor direction
        // corrupts R/graph reconstruction (the divisor equation already
        // detects this in genus zero).  Recovering the small slice requires a
        // generalized mirror transformation, not another scalar gauge; fail
        // closed until that transformation is implemented.
        for (&grade, series) in &flat_cone_point {
            if grade == (0, 0) || grade.0 + grade.1 > q_degree {
                continue;
            }
            for power in 0..self.size() {
                let Some(coefficient) = series
                    .terms_at_h_power(power)
                    .and_then(|terms| terms.get(&-1))
                else {
                    continue;
                };
                if !coefficient.is_zero() {
                    return Err(GwError::UnsupportedFeature {
                        target: self.theory.theory_id(),
                        feature: "generalized mirror normalization".to_string(),
                        witness: format!(
                            "Birkhoff cone retains a higher-primary z^-1 mirror coordinate at shifted bidegree ({}, {}), D-power {}",
                            grade.0, grade.1, power
                        ),
                    });
                }
            }
        }

        Ok(flat_cone_point)
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
        static CACHE: OnceLock<
            Mutex<BoundedCache<(String, usize, usize), HLaurentBidegreeSeries>>,
        > = OnceLock::new();
        let key = (self.rayless_cache_key(), q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
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
        let fundamental = fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff(
            self.size() - 1,
            q_degree,
            cone_point,
            &self.h_power_relation(),
        );
        birkhoff_descendant_s_matrix_from_fundamental_coeff(
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
            Mutex<BoundedCache<(String, usize, usize), BidegreeLaurentFactor<Rational>>>,
        > = OnceLock::new();
        let key = (self.rayless_cache_key(), q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
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
        let bounds = self.bidegree_birkhoff_bounds(q_degree, z_order)?;
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
        let cone_point = self.i_container(q_degree, bounds.min_z)?;
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
        let negative = crate::reconstruction::birkhoff_negative_factor_by_bidegree_with_z_bounds(
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
        static CACHE: OnceLock<
            Mutex<BoundedCache<(String, usize, usize), SeriesSMatrix<Rational>>>,
        > = OnceLock::new();
        let target_key = self.cache_key();
        let key = (target_key.clone(), q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
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
            let truncated = cache
                .iter()
                .find(|((cached_key, cached_q_degree, cached_z_order), _)| {
                    cached_key == &target_key
                        && *cached_q_degree == q_degree
                        && *cached_z_order >= z_order
                })
                .map(|(_, fundamental_s)| fundamental_s.truncated(z_order));
            if let Some(fundamental_s) = truncated {
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
        metric_adjoint_descendant_s_matrix_coeff(fundamental_s, &self.flat_metric_series(q_degree))
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
    target: ProjectiveBundleRay,
}

impl BundleRayProvider {
    pub fn new(target: ProjectiveBundleRay) -> Self {
        Self { target }
    }

    pub fn canonical_theory(&self) -> &ProjectiveBundleTheory {
        self.target.canonical_theory()
    }

    /// Read-only access to the calibrated ray specialization.
    pub fn target(&self) -> &ProjectiveBundleRay {
        &self.target
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
            Mutex<BoundedCache<(String, usize, usize, usize), Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let key = (self.target.cache_key(), q_degree, r_order, graph_dimension);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
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
        for i in 0..=self.target.base_dimension() {
            for j in 0..self.target.rank() {
                let mut differences = Vec::new();
                for k in 0..=self.target.base_dimension() {
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
        let quantum_algebra = CyclicQuantumAlgebra::try_new(
            quantum_grading,
            "projective-bundle quantum grading multiplication",
        )?;
        let left_coordinates: CyclicCoordinates<Rational> =
            quantum_algebra.coordinates(&insertion_vectors[0])?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE bundle_direct_coordinates={:.3}s degree={} ray={}",
                coordinates_started.elapsed().as_secs_f64(),
                degree,
                self.target.ray
            );
        }
        let product_started = Instant::now();
        let product = quantum_algebra.left_product(&left_coordinates, &insertion_vectors[1])?;
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
    try_bundle_dimension_matches(n, twists, genus, d1, d2, insertions).unwrap_or(false)
}

/// Fallible form of [`bundle_dimension_matches`], preserving validation
/// errors for untrusted target dimensions and degrees.
pub fn try_bundle_dimension_matches(
    n: usize,
    twists: &[usize],
    genus: usize,
    d1: usize,
    d2: isize,
    insertions: &[BundleInsertion],
) -> Result<bool, GwError> {
    let theory = ProjectiveBundleTheory::new(n, twists.to_vec())?;
    bundle_dimension_matches_in_theory(&theory, genus, d1, d2, insertions)
}

/// Checked dimension match using an existing canonical bundle theory.
pub fn bundle_dimension_matches_in_theory(
    theory: &ProjectiveBundleTheory,
    genus: usize,
    d1: usize,
    d2: isize,
    insertions: &[BundleInsertion],
) -> Result<bool, GwError> {
    let insertion_degree = insertions
        .iter()
        .try_fold(0usize, |total, insertion| {
            total
                .checked_add(insertion.descendant_power)
                .and_then(|value| value.checked_add(insertion.h_power))
                .and_then(|value| value.checked_add(insertion.xi_power))
        })
        .ok_or_else(|| GwError::AlgebraFailure("bundle insertion degree overflow".to_string()))?;
    let d1 = i64::try_from(d1).map_err(|_| {
        GwError::AlgebraFailure("bundle base degree does not fit in i64".to_string())
    })?;
    let d2 = i64::try_from(d2).map_err(|_| {
        GwError::AlgebraFailure("bundle fiber degree does not fit in i64".to_string())
    })?;
    let virtual_dimension =
        theory.virtual_dimension(genus, &CurveClass::new(vec![d1, d2]), insertions.len())?;
    Ok(usize::try_from(virtual_dimension).ok() == Some(insertion_degree))
}

fn rayless_precompute_z_orders(genus: usize, insertions: &[BundleInsertion]) -> Vec<usize> {
    let mut z_orders = Vec::new();
    if genus == 0 && genus_zero_three_primary_bundle_layout(insertions) {
        z_orders.push(1);
    }
    if is_stable_cohft_range(genus, insertions.len()) {
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
    let input_twists = twists.to_vec();
    let theory = ProjectiveBundleTheory::new(n, input_twists.clone())?;
    let weights_fiber =
        theory.canonicalize_summand_payloads(input_twists, weights_fiber.to_vec())?;
    reconstruct_bundle_invariants_in_theory(
        &theory,
        weights_base,
        &weights_fiber,
        genus,
        total_degree,
        insertions,
    )
}

/// Reconstruct using an already validated canonical theory.  `weights_fiber`
/// must follow [`ProjectiveBundleTheory::twists`], whose summand order is
/// canonicalized, so evaluator caches and fingerprints share the same
/// geometry instance.
pub fn reconstruct_bundle_invariants_in_theory(
    theory: &ProjectiveBundleTheory,
    weights_base: &[Rational],
    weights_fiber: &[Rational],
    genus: usize,
    total_degree: usize,
    insertions: &[BundleInsertion],
) -> Result<Vec<(usize, isize, Rational)>, GwError> {
    reconstruct_bundle_invariants_in_theory_with_nodes(
        theory,
        weights_base,
        weights_fiber,
        genus,
        total_degree,
        insertions,
        None,
    )
}

fn reconstruct_bundle_invariants_in_theory_with_nodes(
    theory: &ProjectiveBundleTheory,
    weights_base: &[Rational],
    weights_fiber: &[Rational],
    genus: usize,
    total_degree: usize,
    insertions: &[BundleInsertion],
    ray_nodes: Option<&[Rational]>,
) -> Result<Vec<(usize, isize, Rational)>, GwError> {
    if is_stable_cohft_range(genus, insertions.len()) {
        crate::graphs::stable_graph_generation_bounds(genus, insertions.len())?;
    }
    let interpolation = match ray_nodes {
        Some(nodes) => ExactRayInterpolation::with_nodes("bundle", total_degree, nodes)?,
        None => ExactRayInterpolation::for_total_degree("bundle", total_degree)?,
    };
    let ray_count = interpolation.ray_count();

    let profile_enabled = crate::env_flag("GW_PROFILE");
    let started = Instant::now();

    let warm_target = ProjectiveBundleRay::from_theory(
        theory.clone(),
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

    let values = interpolation.reconstruct(|ray| {
        let target = ProjectiveBundleRay::from_theory(
            theory.clone(),
            weights_base.to_vec(),
            weights_fiber.to_vec(),
            ray.clone(),
        )?;
        let provider = BundleRayProvider::new(target);
        let value = compute_semisimple_graph_value(
            &provider,
            genus,
            interpolation.total_degree(),
            insertions,
            None,
        )?;
        value.as_rational().ok_or_else(|| {
            GwError::AlgebraFailure("bundle ray value did not specialize to a rational".to_string())
        })
    })?;
    if profile_enabled {
        eprintln!(
            "GW_PROFILE bundle_reconstruct_rays={:.3}s total_degree={} rays={}",
            started.elapsed().as_secs_f64(),
            total_degree,
            ray_count
        );
    }

    values
        .into_iter()
        .enumerate()
        .map(|(d2p, value)| {
            let d1 = total_degree - d2p;
            let curve = theory.curve_from_shifted(d1, d2p)?;
            let d2 = isize::try_from(curve.coordinate(1).expect("rank-two bundle class")).map_err(
                |_| {
                    GwError::AlgebraFailure("bundle fiber degree does not fit in isize".to_string())
                },
            )?;
            let value = if bundle_dimension_matches_in_theory(theory, genus, d1, d2, insertions)? {
                value
            } else {
                Rational::zero()
            };
            Ok((d1, d2, value))
        })
        .collect()
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
    fn bundle_ray_rejects_picard_rank_one_degenerations() {
        let rank_one = ProjectiveBundleRay::new(
            1,
            vec![0],
            base_weights(),
            vec![Rational::from(11)],
            Rational::one(),
        )
        .unwrap_err();
        assert!(matches!(rank_one, GwError::ConventionMismatch(_)));

        let point_base = ProjectiveBundleRay::new(
            0,
            vec![0, 0],
            vec![Rational::from(2)],
            fiber_weights(),
            Rational::one(),
        )
        .unwrap_err();
        assert!(matches!(point_base, GwError::ConventionMismatch(_)));
    }

    #[test]
    fn bundle_reconstruction_rejects_oversized_ray_families_before_warmup() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let error = reconstruct_bundle_invariants_in_theory(
            &theory,
            &base_weights(),
            &fiber_weights(),
            0,
            MAX_EXACT_RECONSTRUCTION_RAYS,
            &[],
        )
        .unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { .. }));
        assert!(error.to_string().contains("Novikov-ray"));
    }

    #[test]
    fn bundle_rayless_stability_preflight_accepts_extreme_stable_genus() {
        assert_eq!(rayless_precompute_z_orders(usize::MAX, &[]), vec![1]);
    }

    #[test]
    fn bundle_reconstruction_rejects_extreme_genus_before_warmup() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let error = reconstruct_bundle_invariants_in_theory(
            &theory,
            &base_weights(),
            &fiber_weights(),
            usize::MAX,
            0,
            &[],
        )
        .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
        assert!(error.to_string().contains("stable-graph"));
    }

    #[test]
    fn f2_native_reconstruction_vanishes_in_deformation_negative_degree() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap();
        // In shifted total degree one, (d1,d2)=(1,-2) maps to product
        // bidegree (-1,1), hence its on-dimension genus-one coefficient
        // vanishes.  This is reconstructed natively, without substituting
        // the deformation-equivalent product backend.
        let invariants = reconstruct_bundle_invariants_in_theory(
            &theory,
            &base_weights(),
            &fiber_weights(),
            1,
            1,
            &[BundleInsertion::new(0, 0, 1)],
        )
        .unwrap();
        assert_eq!(
            invariants,
            vec![(1, -2, Rational::zero()), (0, 1, Rational::zero())]
        );
    }

    #[test]
    fn negative_fiber_numerators_do_not_lose_laurent_boundary_terms() {
        let target = ProjectiveBundleRay::new(
            2,
            vec![0, 2],
            vec![Rational::from(2), Rational::from(5), Rational::from(7)],
            fiber_weights(),
            Rational::one(),
        )
        .unwrap();
        let data = target.classical_data();
        let fiber_degrees = target.shifted_fiber_degrees(1, 0).unwrap();
        assert_eq!(fiber_degrees, vec![-2, 0]);

        let min_z = -6;
        let mut saw_boundary_term = false;
        for i in 0..=target.base_dimension() {
            for j in 0..target.rank() {
                let shallow = target
                    .i_restriction_with_data(&data, i, j, 1, &fiber_degrees, min_z)
                    .unwrap();
                let mut one_term_deeper = target
                    .i_restriction_with_data(&data, i, j, 1, &fiber_degrees, min_z - 1)
                    .unwrap();
                one_term_deeper.retain(|z_power, _| *z_power >= min_z);
                saw_boundary_term |= shallow.contains_key(&min_z);
                assert_eq!(shallow, one_term_deeper, "fixed point ({i},{j})");
            }
        }
        assert!(saw_boundary_term, "the comparison must exercise z^-6");
    }

    #[test]
    fn multiple_negative_fiber_numerators_preserve_laurent_boundary_terms() {
        let target = ProjectiveBundleRay::new(
            1,
            vec![0, 1, 3],
            base_weights(),
            vec![Rational::from(11), Rational::from(23), Rational::from(41)],
            Rational::one(),
        )
        .unwrap();
        let data = target.classical_data();
        let fiber_degrees = target.shifted_fiber_degrees(1, 0).unwrap();
        assert_eq!(fiber_degrees, vec![-3, -2, 0]);
        assert_eq!(
            fiber_degrees.iter().filter(|degree| **degree < 0).count(),
            2
        );

        // The two negative factors can jointly raise z-degree by 2 + 1.
        // Expanding three terms deeper is therefore an independent retained-
        // window reference for the requested floor.
        let min_z = -6;
        let deep_min_z = min_z - 3;
        let mut saw_boundary_term = false;
        for i in 0..=target.base_dimension() {
            for j in 0..target.rank() {
                let shallow = target
                    .i_restriction_with_data(&data, i, j, 1, &fiber_degrees, min_z)
                    .unwrap();
                let mut deep = target
                    .i_restriction_with_data(&data, i, j, 1, &fiber_degrees, deep_min_z)
                    .unwrap();
                deep.retain(|z_power, _| *z_power >= min_z);
                saw_boundary_term |= shallow.contains_key(&min_z);
                assert_eq!(shallow, deep, "fixed point ({i},{j})");
            }
        }
        assert!(saw_boundary_term, "the comparison must exercise z^-6");
    }

    #[test]
    fn p2_o_plus_o2_negative_section_is_weight_independent() {
        let insertions = [
            BundleInsertion::new(0, 1, 1),
            BundleInsertion::new(0, 1, 0),
            BundleInsertion::new(0, 1, 0),
        ];
        let expected = vec![(1, -2, Rational::from(2)), (0, 1, Rational::zero())];

        for (base, fiber) in [
            (
                vec![Rational::from(2), Rational::from(5), Rational::from(7)],
                vec![Rational::from(11), Rational::from(23)],
            ),
            (
                vec![Rational::from(-3), Rational::from(4), Rational::from(13)],
                vec![Rational::from(-7), Rational::from(19)],
            ),
        ] {
            assert_eq!(
                reconstruct_bundle_invariants(2, &[0, 2], &base, &fiber, 0, 1, &insertions)
                    .unwrap(),
                expected
            );
        }
    }

    fn three_insertion_permutations(
        insertions: &[BundleInsertion; 3],
    ) -> Vec<Vec<BundleInsertion>> {
        const PERMUTATIONS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        PERMUTATIONS
            .into_iter()
            .map(|order| {
                order
                    .into_iter()
                    .map(|index| insertions[index].clone())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn fano_asymmetric_rank3_three_primary_shortcut_matches_graph_series() {
        let target = ProjectiveBundleRay::new(
            2,
            vec![0, 0, 1],
            vec![Rational::from(2), Rational::from(5), Rational::from(11)],
            vec![Rational::from(17), Rational::from(31), Rational::from(53)],
            Rational::from(2),
        )
        .unwrap();
        let provider = BundleRayProvider::new(target);

        // The degree-one triple has total codimension seven, matching the
        // fiber class in that ray coefficient.  The degree-two triple has
        // total codimension ten, matching the twice-fiber class.  Distinct
        // insertions make all six marked-leg orders observable.
        let cases = [
            (
                1,
                [
                    BundleInsertion::new(0, 2, 2),
                    BundleInsertion::new(0, 1, 1),
                    BundleInsertion::new(0, 1, 0),
                ],
            ),
            (
                2,
                [
                    BundleInsertion::new(0, 2, 2),
                    BundleInsertion::new(0, 1, 2),
                    BundleInsertion::new(0, 2, 1),
                ],
            ),
        ];

        for (degree, insertions) in cases {
            let permutations = three_insertion_permutations(&insertions);
            let generic =
                compute_semisimple_graph_series(&provider, 0, degree, &permutations[0], None)
                    .unwrap()
                    .coeff(degree)
                    .cloned()
                    .unwrap_or_else(RatFun::zero);
            assert!(!generic.is_zero(), "degree-{degree} probe must be nonzero");
            for permutation in &permutations {
                let direct = provider
                    .direct_value(0, degree, permutation, None)
                    .unwrap()
                    .expect("three primary insertions must use the shortcut");
                assert_eq!(
                    direct, generic,
                    "shortcut and graph series differ in degree {degree} for {permutation:?}"
                );
            }

            let insertions = permutations.into_iter().next().unwrap();
            let three_point = generic;
            let mut with_xi = insertions.clone();
            with_xi.push(BundleInsertion::new(0, 0, 1));
            let mut with_h = insertions;
            with_h.push(BundleInsertion::new(0, 1, 0));
            let xi_value = compute_semisimple_graph_series(&provider, 0, degree, &with_xi, None)
                .unwrap()
                .coeff(degree)
                .cloned()
                .unwrap_or_else(RatFun::zero);
            let h_value = compute_semisimple_graph_series(&provider, 0, degree, &with_h, None)
                .unwrap()
                .coeff(degree)
                .cloned()
                .unwrap_or_else(RatFun::zero);

            // D = xi + (A+1)H = xi + 2H grades the shifted Novikov
            // variable, so adding D multiplies its q^degree coefficient by
            // `degree`.
            let divisor_value = &xi_value + &(&h_value * &RatFun::from_rational(Rational::from(2)));
            let expected_divisor = &three_point * &RatFun::from_rational(Rational::from(degree));
            assert_eq!(
                divisor_value, expected_divisor,
                "grading-divisor equation failed in degree {degree}"
            );
        }
    }

    #[test]
    fn zero_twist_bundle_three_primary_divisor_equation() {
        let provider = BundleRayProvider::new(
            ProjectiveBundleRay::new(
                1,
                vec![0, 0],
                base_weights(),
                fiber_weights(),
                Rational::from(2),
            )
            .unwrap(),
        );
        let degree = 2;
        let point = BundleInsertion::new(0, 1, 1);
        let insertions = vec![point.clone(), point.clone(), point];
        let direct = provider
            .direct_value(0, degree, &insertions, None)
            .unwrap()
            .unwrap();
        let generic = compute_semisimple_graph_series(&provider, 0, degree, &insertions, None)
            .unwrap()
            .coeff(degree)
            .cloned()
            .unwrap_or_else(RatFun::zero);
        assert_eq!(direct, generic);

        let mut with_xi = insertions.clone();
        with_xi.push(BundleInsertion::new(0, 0, 1));
        let mut with_h = insertions;
        with_h.push(BundleInsertion::new(0, 1, 0));
        let xi_value = compute_semisimple_graph_series(&provider, 0, degree, &with_xi, None)
            .unwrap()
            .coeff(degree)
            .cloned()
            .unwrap_or_else(RatFun::zero);
        let h_value = compute_semisimple_graph_series(&provider, 0, degree, &with_h, None)
            .unwrap()
            .coeff(degree)
            .cloned()
            .unwrap_or_else(RatFun::zero);
        assert_eq!(
            &xi_value + &h_value,
            &generic * &RatFun::from_rational(Rational::from(degree))
        );
    }

    #[test]
    fn nonsemipositive_rank3_bundle_requires_generalized_mirror_normalization() {
        let target = ProjectiveBundleRay::new(
            2,
            vec![0, 1, 4],
            vec![Rational::from(2), Rational::from(5), Rational::from(11)],
            vec![Rational::from(17), Rational::from(31), Rational::from(53)],
            Rational::from(2),
        )
        .unwrap();
        let error = target
            .normalized_flat_bidegree_cone_point(1, 1)
            .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
        assert!(error.to_string().contains(
            "higher-primary z^-1 mirror coordinate at shifted bidegree (1, 0), D-power 2"
        ));
        assert!(error
            .to_string()
            .contains("unsupported generalized mirror normalization"));
    }

    #[test]
    fn bundle_ray_rejects_extreme_base_dimension_fallibly() {
        let error = ProjectiveBundleRay::new(
            usize::MAX,
            vec![0, 1],
            Vec::new(),
            Vec::new(),
            Rational::one(),
        )
        .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
    }

    #[test]
    fn bundle_ray_rejects_missing_fiber_weights_without_dropping_summands() {
        let error = ProjectiveBundleRay::new(
            1,
            vec![0, 2],
            base_weights(),
            vec![Rational::from(11)],
            Rational::one(),
        )
        .unwrap_err();
        assert!(matches!(error, GwError::ConventionMismatch(_)));
        assert!(error.to_string().contains("must have length 2"));
    }

    #[test]
    fn bundle_ray_reports_colliding_seeds_before_lagrange_division() {
        let error = ProjectiveBundleRay::new(
            1,
            vec![1, 0],
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(6), Rational::from(8)],
            Rational::one(),
        )
        .unwrap_err();
        assert_eq!(error, GwError::NonSemisimplePoint);
    }

    #[test]
    fn bundle_ray_owns_canonical_geometry_and_keeps_summand_weights_aligned() {
        let target = ProjectiveBundleRay::new(
            1,
            vec![1, 0, 1],
            vec![Rational::from(1), Rational::from(3)],
            vec![Rational::from(10), Rational::from(20), Rational::from(30)],
            Rational::from(2),
        )
        .unwrap();

        assert_eq!(target.twists(), &[0, 1, 1]);
        // Sorting is stable, so equal-twist summands retain their input order.
        assert_eq!(
            target.fiber_weights(),
            &[Rational::from(20), Rational::from(10), Rational::from(30)]
        );
        assert_eq!(target.base_dimension(), 1);
        assert_eq!(target.rank(), 3);
        assert_eq!(target.size(), 6);

        let expected = ProjectiveBundleTheory::new(1, vec![0, 1, 1]).unwrap();
        assert_eq!(
            target.canonical_theory().theory_fingerprint(),
            expected.theory_fingerprint()
        );

        let provider = BundleRayProvider::new(target);
        assert!(std::ptr::eq(
            provider.canonical_theory(),
            provider.target().canonical_theory()
        ));
    }

    #[test]
    fn reconstruction_canonicalizes_twists_with_their_fiber_weights() {
        let insertions = [
            BundleInsertion::new(0, 1, 0),
            BundleInsertion::new(0, 0, 1),
            BundleInsertion::new(0, 0, 0),
        ];
        // These weights are deliberately collision-sensitive: if `[1, 0]`
        // is sorted without carrying its paired fiber weights along, the
        // resulting `[0, 1]` specialization has two equal grading seeds.
        let base_weights = [Rational::from(1), Rational::from(2)];
        let left = reconstruct_bundle_invariants(
            1,
            &[0, 1],
            &base_weights,
            &[Rational::from(-5), Rational::from(-4)],
            0,
            0,
            &insertions,
        )
        .unwrap();
        let right = reconstruct_bundle_invariants(
            1,
            &[1, 0],
            &base_weights,
            &[Rational::from(-4), Rational::from(-5)],
            0,
            0,
            &insertions,
        )
        .unwrap();
        assert_eq!(left, right);
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
    fn bounded_bidegree_birkhoff_matches_deep_full_factorization_window() {
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
        let bounds = target.bidegree_birkhoff_bounds(q_degree, z_order).unwrap();
        let bounded_cone = target.i_container(q_degree, bounds.min_z).unwrap();
        let bounded_fundamental = target.fundamental_bidegree_matrix(q_degree, &bounded_cone);

        // Build the oracle from an independently fixed source window.  Using
        // the bounded fundamental on both sides would verify only the
        // factorization algorithm, not that `bounds.min_z` retained enough of
        // the I-function before factorization.
        // Through total degree two, the two negative summands can jointly
        // raise z-degree by ten.  This fixed floor is more than ten terms
        // below the adaptive floor for this target, so it also serves as an
        // oracle for the pre-factorization numerator tail.
        let deep_min_z = -24;
        assert!(bounds
            .min_z
            .checked_sub(deep_min_z)
            .is_some_and(|margin| margin > 10));
        let deep_cone = target.i_container(q_degree, deep_min_z).unwrap();
        let deep_fundamental = target.fundamental_bidegree_matrix(q_degree, &deep_cone);
        let (_, full_negative) = crate::reconstruction::birkhoff_factor_by_bidegree(
            target.size(),
            q_degree,
            &deep_fundamental,
        )
        .unwrap();
        let bounded_negative =
            crate::reconstruction::birkhoff_negative_factor_by_bidegree_with_z_bounds(
                target.size(),
                q_degree,
                &bounded_fundamental,
                &bounds.positive_z_windows,
                &bounds.negative_z_depths,
            )
            .unwrap();

        let matrix_at = |factor: &BidegreeLaurentFactor<Rational>, grade: Grade, z_power: i32| {
            factor
                .get(&grade)
                .and_then(|laurent| laurent.get(&z_power))
                .cloned()
                .unwrap_or_else(|| zero_coeff_matrix(target.size()))
        };
        for total in 1..=q_degree {
            for first in 0..=total {
                let grade = (first, total - first);
                // Only z^-1, ..., z^-z_order are observable in the requested
                // descendant S-matrix.  Deeper entries are internal working
                // coefficients and can legitimately differ at the adaptive
                // source boundary.
                for z_power in -(z_order as i32)..0 {
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
    ) -> Vec<(
        Rational,
        crate::spaces::product_projective::ProductInsertion,
    )> {
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
                crate::spaces::product_projective::ProductInsertion::new(
                    insertion.descendant_power,
                    a,
                    b,
                ),
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
            let invariants = crate::spaces::product_projective::reconstruct_bidegree_invariants(
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
    fn f4_deformation_presentation_requires_generalized_mirror_normalization() {
        // P(O(2) + O(-2)) normalizes to P(O(4) + O).  Its Birkhoff cone has
        // higher-primary mirror coordinates, so the old isolated agreement
        // with a P1 x P1 ruling did not certify the small GW theory.
        let target = ProjectiveBundleRay::new(
            1,
            vec![4, 0],
            base_weights(),
            fiber_weights(),
            Rational::one(),
        )
        .unwrap();
        let error = target
            .normalized_flat_bidegree_cone_point(1, 1)
            .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
    }

    #[test]
    fn rank3_deformation_presentation_requires_generalized_mirror_normalization() {
        // P(O(1) + O + O(-1)) normalizes to P(O(2) + O(1) + O).  Under
        // deformation it has isolated numerical coincidences with P1 x P2,
        // but a higher-primary mirror coordinate prevents interpreting this
        // two-parameter I-function path as the small GW slice.
        let target = ProjectiveBundleRay::new(
            1,
            vec![2, 1, 0],
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(0), Rational::from(10), Rational::from(30)],
            Rational::one(),
        )
        .unwrap();
        let error = target
            .normalized_flat_bidegree_cone_point(1, 1)
            .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
    }

    #[test]
    fn harder_rank3_deformation_requires_generalized_mirror_normalization() {
        // P(O(2) + O(1) + O(-3)) normalizes to P(O(5) + O(4) + O).  The
        // same higher-primary obstruction is already visible at the first
        // positive shifted bidegree, before any graph reconstruction.
        let target = ProjectiveBundleRay::new(
            1,
            vec![5, 4, 0],
            vec![Rational::from(1), Rational::from(2)],
            vec![Rational::from(0), Rational::from(10), Rational::from(30)],
            Rational::one(),
        )
        .unwrap();
        let error = target
            .normalized_flat_bidegree_cone_point(1, 1)
            .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
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
        let min_z = target.min_z_power(2, 2).unwrap();
        assert!(!target.i_coefficient(1, 0, min_z).unwrap().is_empty());
        assert!(!target.i_coefficient(0, 1, min_z).unwrap().is_empty());
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
    fn zero_twist_bundle_reconstruction_is_independent_of_ray_nodes() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let point = BundleInsertion::new(0, 1, 1);
        let insertions = [point.clone(), point.clone(), point];
        let canonical = reconstruct_bundle_invariants_in_theory(
            &theory,
            &base_weights(),
            &fiber_weights(),
            0,
            2,
            &insertions,
        )
        .unwrap();
        let alternate = reconstruct_bundle_invariants_in_theory_with_nodes(
            &theory,
            &base_weights(),
            &fiber_weights(),
            0,
            2,
            &insertions,
            Some(&[Rational::from(5), Rational::from(7), Rational::from(11)]),
        )
        .unwrap();
        assert_eq!(alternate, canonical);
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
        let product_target = crate::spaces::product_projective::ProductProjectiveRay::new(
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
        let product = crate::spaces::product_projective::ProductRayProvider::new(product_target)
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
        let product_target = crate::spaces::product_projective::ProductProjectiveRay::new(
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
        let product_s = crate::spaces::product_projective::ProductRayProvider::new(product_target)
            .descendant_s_matrix(q_degree, z_order)
            .unwrap();
        assert_eq!(bundle_s.coefficients(), product_s.coefficients());
    }
}
