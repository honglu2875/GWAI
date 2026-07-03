//! `P^n x P^m` through exact Novikov ray specialization.
//!
//! The engine's series layer carries one Novikov variable, while a product
//! has curve bidegrees `(d1, d2)`.  Specializing `(q1, q2) = (t, b t)` along
//! a rational ray `b` is a ring homomorphism on the Novikov ring, so every
//! calibration and every invariant of the specialized theory is the exact
//! specialization of the rank-two theory.  Along a generic ray the quantum
//! ring is cyclic over the graded divisor `D = H1 + H2` (its fixed-point
//! eigenvalues `lambda_i + mu_j` are pairwise distinct), so the ordinary
//! divisor-generated recipe applies verbatim with two corrections handled by
//! the recipe layer: metric norms come from the Atiyah-Bott flat metric (the
//! residue shortcut is wrong for products) and the classical `R`-asymptotics
//! use the true tangent weights of the product.
//!
//! A degree-`k` coefficient of the ray theory is
//! `sum_{d1+d2=k} b^{d2} N_{(d1,d2)}`; running `k+1` distinct rays and
//! solving the Vandermonde system recovers each bidegree exactly over the
//! rationals.  [`reconstruct_bidegree_invariants`] packages this.

use super::*;

/// Insertion `tau_k(H1^a H2^b)` on the product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductInsertion {
    pub descendant_power: usize,
    pub h1_power: usize,
    pub h2_power: usize,
}

impl ProductInsertion {
    pub fn new(descendant_power: usize, h1_power: usize, h2_power: usize) -> Self {
        Self {
            descendant_power,
            h1_power,
            h2_power,
        }
    }
}

/// The `P^n x P^m` theory specialized along the Novikov ray `(t, ray * t)`,
/// at rational equivariant weights for both factors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductProjectiveRay {
    pub n: usize,
    pub m: usize,
    pub weights_x: Vec<Rational>,
    pub weights_y: Vec<Rational>,
    pub ray: Rational,
}

impl ProductProjectiveRay {
    pub fn new(
        n: usize,
        m: usize,
        weights_x: Vec<Rational>,
        weights_y: Vec<Rational>,
        ray: Rational,
    ) -> Result<Self, GwError> {
        if weights_x.len() != n + 1 || weights_y.len() != m + 1 {
            return Err(GwError::ConventionMismatch(format!(
                "product weights must have lengths {} and {}",
                n + 1,
                m + 1
            )));
        }
        let target = Self {
            n,
            m,
            weights_x,
            weights_y,
            ray,
        };
        let seeds = target.classical_seeds();
        for left in 0..seeds.len() {
            for right in left + 1..seeds.len() {
                if seeds[left] == seeds[right] {
                    return Err(GwError::NonSemisimplePoint);
                }
            }
        }
        Ok(target)
    }

    fn size(&self) -> usize {
        (self.n + 1) * (self.m + 1)
    }

    fn point(&self, i: usize, j: usize) -> usize {
        i * (self.m + 1) + j
    }

    /// Classical eigenvalues of `D = H1 + H2`: `lambda_i + mu_j`.
    fn classical_seeds(&self) -> Vec<Rational> {
        let mut seeds = vec![Rational::zero(); self.size()];
        for i in 0..=self.n {
            for j in 0..=self.m {
                seeds[self.point(i, j)] = self.weights_x[i].clone() + self.weights_y[j].clone();
            }
        }
        seeds
    }

    /// Equivariant Euler class of the tangent space at fixed point `(i, j)`.
    fn euler(&self, i: usize, j: usize) -> Rational {
        let mut euler = Rational::one();
        for k in 0..=self.n {
            if k != i {
                euler = euler * (self.weights_x[i].clone() - self.weights_x[k].clone());
            }
        }
        for l in 0..=self.m {
            if l != j {
                euler = euler * (self.weights_y[j].clone() - self.weights_y[l].clone());
            }
        }
        euler
    }

    /// Atiyah-Bott flat metric in the `D`-power basis:
    /// `G_{rs} = sum_p d_p^{r+s} / Euler_p` over the classical eigenvalues.
    fn flat_metric(&self) -> Vec<Vec<RatFun>> {
        let size = self.size();
        let seeds = self.classical_seeds();
        let mut metric = vec![vec![RatFun::zero(); size]; size];
        for row in 0..size {
            for col in 0..size {
                let mut total = Rational::zero();
                for i in 0..=self.n {
                    for j in 0..=self.m {
                        let seed = &seeds[self.point(i, j)];
                        total += seed.pow_usize(row + col) / self.euler(i, j);
                    }
                }
                metric[row][col] = RatFun::from_rational(total);
            }
        }
        metric
    }

    /// Classical Lagrange transition: column `p` holds the coefficients of
    /// the classical idempotent `E_p(D)` in the classical `D`-power basis.
    fn classical_transition(&self) -> Vec<Vec<Rational>> {
        recipe::classical_lagrange_transition(&self.classical_seeds())
    }

    /// Quantum idempotents of the product expressed in the constant classical
    /// `D`-power basis, together with the quantum roots of `D`.
    ///
    /// The quantum idempotents tensor: `e_(i,j) = e_i^X(q1) (x) e_j^Y(q2)`.
    /// Their classical fixed-point restrictions multiply likewise, and any
    /// cohomology element is recovered from its restrictions through the
    /// classical Lagrange transition.  Working in a constant flat basis is
    /// essential: quantum powers of `D` are t-dependent elements, and both
    /// the Dubrovin connection and the descendant QDE differentiate the
    /// frame against a fixed basis.
    fn quantum_frame_matrices(
        &self,
        q_degree: usize,
    ) -> Result<(Vec<QSeries>, SeriesMatrix, SeriesMatrix), GwError> {
        let x_roots = factor_root_series(&self.weights_x, &Rational::one(), q_degree)?;
        let y_roots = factor_root_series(&self.weights_y, &self.ray, q_degree)?;
        let x_frame = recipe::divisor_lagrange_frame(x_roots.clone(), q_degree)?;
        let y_frame = recipe::divisor_lagrange_frame(y_roots.clone(), q_degree)?;

        // Restrictions of the factor quantum idempotents at the factor fixed
        // points: r[k][i] = e_i(q)|_k = sum_r T[r][i] w_k^r.
        let restrictions = |weights: &[Rational], frame: &recipe::CanonicalFrame| {
            let size = weights.len();
            let mut out = vec![vec![QSeries::zero(q_degree); size]; size];
            for (k, weight) in weights.iter().enumerate() {
                for i in 0..size {
                    let mut total = QSeries::zero(q_degree);
                    let mut power = Rational::one();
                    for r in 0..size {
                        total = total.add(
                            &frame
                                .transition_to_flat
                                .entry(r, i)
                                .scale(&RatFun::from_rational(power.clone())),
                        );
                        power = power * weight.clone();
                    }
                    out[k][i] = total;
                }
            }
            out
        };
        let x_restrictions = restrictions(&self.weights_x, &x_frame);
        let y_restrictions = restrictions(&self.weights_y, &y_frame);

        // R_e[(k,l)][(i,j)]: restriction of the product idempotent (i,j) at
        // the fixed point (k,l).  Equals the identity at q = 0.
        let size = self.size();
        let mut restriction_entries = vec![vec![QSeries::zero(q_degree); size]; size];
        for k in 0..=self.n {
            for l in 0..=self.m {
                for i in 0..=self.n {
                    for j in 0..=self.m {
                        restriction_entries[self.point(k, l)][self.point(i, j)] =
                            x_restrictions[k][i].mul(&y_restrictions[l][j]);
                    }
                }
            }
        }
        let restriction_matrix = SeriesMatrix::from_entries(restriction_entries);

        let classical_transition = SeriesMatrix::constant(
            self.classical_transition()
                .into_iter()
                .map(|row| row.into_iter().map(RatFun::from_rational).collect())
                .collect(),
            q_degree,
        );
        // W: quantum idempotents in the classical D-power basis, and its
        // inverse through V_cl and a Neumann series (R_e = I + O(q)).
        let transition_to_flat = classical_transition.mul(&restriction_matrix);
        let classical_vandermonde = SeriesMatrix::constant(
            self.classical_seeds()
                .iter()
                .map(|seed| {
                    (0..size)
                        .map(|power| RatFun::from_rational(seed.pow_usize(power)))
                        .collect()
                })
                .collect(),
            q_degree,
        );
        let flat_to_canonical =
            recipe::neumann_inverse(&restriction_matrix, q_degree)?.mul(&classical_vandermonde);

        let mut roots = vec![QSeries::zero(q_degree); size];
        for i in 0..=self.n {
            for j in 0..=self.m {
                roots[self.point(i, j)] = x_roots[i].add(&y_roots[j]);
            }
        }
        Ok((roots, transition_to_flat, flat_to_canonical))
    }

    fn canonical_frame(&self, q_degree: usize) -> Result<recipe::CanonicalFrame, GwError> {
        let (roots, transition_to_flat, flat_to_canonical) =
            self.quantum_frame_matrices(q_degree)?;
        let metric = SeriesMatrix::constant(self.flat_metric(), q_degree);
        let canonical_metric = transition_to_flat
            .transpose()
            .mul(&metric)
            .mul(&transition_to_flat);
        let size = self.size();
        let mut metric_norms = Vec::with_capacity(size);
        let mut inverse_metric_norms = Vec::with_capacity(size);
        for branch in 0..size {
            let norm = canonical_metric.entry(branch, branch).clone();
            inverse_metric_norms.push(norm.inverse()?);
            metric_norms.push(norm);
        }
        Ok(recipe::CanonicalFrame {
            roots,
            transition_to_flat,
            flat_to_canonical,
            metric_norms,
            inverse_metric_norms,
        })
    }

    /// Multiplication by `D` in the classical `D`-power basis:
    /// `W diag(u_p) W^{-1}` for the quantum product, the constant companion
    /// action of the classical characteristic polynomial otherwise.
    fn divisor_multiplication(
        &self,
        q_degree: usize,
        quantum: bool,
    ) -> Result<SeriesMatrix, GwError> {
        if quantum {
            let (roots, transition_to_flat, flat_to_canonical) =
                self.quantum_frame_matrices(q_degree)?;
            let diagonal = SeriesMatrix::diagonal(roots);
            Ok(transition_to_flat.mul(&diagonal).mul(&flat_to_canonical))
        } else {
            let size = self.size();
            let constants = self
                .classical_seeds()
                .iter()
                .map(|seed| QSeries::constant(RatFun::from_rational(seed.clone()), q_degree))
                .collect::<Vec<_>>();
            let charpoly = series_polynomial_from_roots(&constants, q_degree);
            let mut matrix = vec![vec![QSeries::zero(q_degree); size]; size];
            for col in 0..size.saturating_sub(1) {
                matrix[col + 1][col] = QSeries::one(q_degree);
            }
            for row in 0..size {
                matrix[row][size - 1] = charpoly[row].neg();
            }
            Ok(SeriesMatrix::from_entries(matrix))
        }
    }

    /// Classical `D`-power-basis coefficients of `H1^a H2^b`, via the
    /// classical Lagrange projectors: the class equals
    /// `sum_p (restriction at p) * E_p(D)`.
    fn insertion_class_vector(&self, h1_power: usize, h2_power: usize) -> Vec<Rational> {
        let size = self.size();
        let transition = self.classical_transition();
        let mut vector = vec![Rational::zero(); size];
        for i in 0..=self.n {
            for j in 0..=self.m {
                let restriction =
                    self.weights_x[i].pow_usize(h1_power) * self.weights_y[j].pow_usize(h2_power);
                for row in 0..size {
                    vector[row] += transition[row][self.point(i, j)].clone() * restriction.clone();
                }
            }
        }
        vector
    }

    fn cache_key(&self) -> String {
        format!(
            "p{}xp{}[{};{}]@{}",
            self.n,
            self.m,
            self.weights_x
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.weights_y
                .iter()
                .map(Rational::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.ray
        )
    }

    fn build_calibration(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SemisimpleCalibration, GwError> {
        let frame = self.canonical_frame(q_degree)?;
        let mut classical_diagonal = Vec::with_capacity(self.size());
        for i in 0..=self.n {
            for j in 0..=self.m {
                // Tangent weight differences of the product at (i, j): the
                // factor differences, matching R_{X x Y} = R_X (x) R_Y.
                let mut differences = Vec::new();
                for k in 0..=self.n {
                    if k != i {
                        differences.push(RatFun::from_rational(
                            self.weights_x[k].clone() - self.weights_x[i].clone(),
                        ));
                    }
                }
                for l in 0..=self.m {
                    if l != j {
                        differences.push(RatFun::from_rational(
                            self.weights_y[l].clone() - self.weights_y[j].clone(),
                        ));
                    }
                }
                classical_diagonal.push(classical_r_asymptotics_for_point(&differences, z_order));
            }
        }
        calibration_from_canonical_frame(
            &frame,
            &classical_diagonal,
            q_degree,
            z_order,
            CalibrationId(format!("product-ray-j:{}", self.cache_key())),
        )
    }
}

/// Root series of one factor's relation `prod_i (x - w_i) = scale * t`.
fn factor_root_series(
    weights: &[Rational],
    scale: &Rational,
    q_degree: usize,
) -> Result<Vec<QSeries>, GwError> {
    let constants = weights
        .iter()
        .map(|weight| QSeries::constant(RatFun::from_rational(weight.clone()), q_degree))
        .collect::<Vec<_>>();
    let mut charpoly = series_polynomial_from_roots(&constants, q_degree);
    charpoly[0] =
        charpoly[0].sub(&QSeries::q(q_degree).scale(&RatFun::from_rational(scale.clone())));
    weights
        .iter()
        .map(|weight| {
            recipe::newton_root_series(&charpoly, &RatFun::from_rational(weight.clone()), q_degree)
        })
        .collect()
}

/// Ascending coefficients of `prod_i (x - roots[i])` over `q`-series.
fn series_polynomial_from_roots(roots: &[QSeries], q_degree: usize) -> Vec<QSeries> {
    let mut coefficients = vec![QSeries::one(q_degree)];
    for root in roots {
        coefficients = multiply_qseries_polynomial_by_linear(&coefficients, &root.neg(), q_degree);
    }
    coefficients
}

/// Engine-facing provider for one ray of the product theory.
#[derive(Debug, Clone)]
pub struct ProductRayProvider {
    pub target: ProductProjectiveRay,
}

impl ProductRayProvider {
    pub fn new(target: ProductProjectiveRay) -> Self {
        Self { target }
    }
}

impl SemisimpleCohftProvider for ProductRayProvider {
    type Insertion = ProductInsertion;

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
            total = total.checked_add(insertion.h1_power)?;
            total = total.checked_add(insertion.h2_power)?;
        }
        Some(total)
    }

    // The virtual dimension depends on the bidegree, not on the ray degree
    // alone, so no per-degree pruning is available along a ray; bidegrees
    // that fail the dimension count reconstruct to zero instead.

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        static CACHE: OnceLock<Mutex<HashMap<(String, usize, usize), SeriesSMatrix>>> =
            OnceLock::new();
        let key = (self.target.cache_key(), q_degree, z_order);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(descendant_s) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(descendant_s);
        }

        let quantum = self.target.divisor_multiplication(q_degree, true)?;
        let classical = self.target.divisor_multiplication(q_degree, false)?;
        let descendant_s = descendant_s_from_divisor_qde(
            &quantum,
            &classical,
            z_order,
            CalibrationId(format!("product-ray-small-j:{}", self.target.cache_key())),
        )?;
        cache.lock().unwrap().insert(key, descendant_s.clone());
        Ok(descendant_s)
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

        let calibration = self.target.build_calibration(q_degree, r_order)?;
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
            .insertion_class_vector(insertion.h1_power, insertion.h2_power)
            .into_iter()
            .map(|coefficient| QSeries::constant(RatFun::from_rational(coefficient), q_degree))
            .collect())
    }
}

/// Computes all bidegree invariants `N_{(d1, d2)}` with `d1 + d2 =
/// total_degree` by running `total_degree + 1` rays and solving the
/// Vandermonde system exactly.
///
/// Returns the invariants ordered by `d2 = 0 ..= total_degree`.
pub fn reconstruct_bidegree_invariants(
    n: usize,
    m: usize,
    weights_x: &[Rational],
    weights_y: &[Rational],
    genus: usize,
    total_degree: usize,
    insertions: &[ProductInsertion],
) -> Result<Vec<Rational>, GwError> {
    let ray_count = total_degree + 1;
    let mut rays = Vec::with_capacity(ray_count);
    let mut values = Vec::with_capacity(ray_count);
    for step in 0..ray_count {
        let ray = Rational::from(step + 1);
        let target =
            ProductProjectiveRay::new(n, m, weights_x.to_vec(), weights_y.to_vec(), ray.clone())?;
        let provider = ProductRayProvider::new(target);
        let value =
            compute_semisimple_graph_value(&provider, genus, total_degree, insertions, None)?;
        let value = value.as_rational().ok_or_else(|| {
            GwError::AlgebraFailure(
                "product ray value did not specialize to a rational".to_string(),
            )
        })?;
        rays.push(ray);
        values.push(value);
    }

    // Solve sum_{d2} ray^{d2} N_{d2} = value for each ray (Vandermonde).
    let mut matrix = rays
        .iter()
        .map(|ray| {
            (0..ray_count)
                .map(|power| ray.pow_usize(power))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    recipe::solve_rational_system(&mut matrix, &mut values)?;

    // Equivariant invariants at dimension-mismatched bidegrees are nonzero
    // weight-dependent quantities whose non-equivariant limit vanishes;
    // filter them so the output is the list of non-equivariant numbers.
    // (For n != m the virtual dimension varies across the bidegrees of one
    // total degree, so this per-bidegree check cannot happen on a ray.)
    for (d2, value) in values.iter_mut().enumerate() {
        if !bidegree_dimension_matches(n, m, genus, total_degree - d2, d2, insertions) {
            *value = Rational::zero();
        }
    }
    Ok(values)
}

/// Whether the insertions match the virtual dimension of genus-`genus`
/// bidegree-`(d1, d2)` maps to `P^n x P^m`.
pub fn bidegree_dimension_matches(
    n: usize,
    m: usize,
    genus: usize,
    d1: usize,
    d2: usize,
    insertions: &[ProductInsertion],
) -> bool {
    let insertion_degree: usize = insertions
        .iter()
        .map(|insertion| insertion.descendant_power + insertion.h1_power + insertion.h2_power)
        .sum();
    let virtual_dimension = (1 - genus as isize) * ((n + m) as isize - 3)
        + (n as isize + 1) * d1 as isize
        + (m as isize + 1) * d2 as isize
        + insertions.len() as isize;
    insertion_degree as isize == virtual_dimension
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_x() -> Vec<Rational> {
        vec![Rational::from(2), Rational::from(5)]
    }

    fn weights_y() -> Vec<Rational> {
        vec![Rational::from(11), Rational::from(23)]
    }

    fn point_class(descendant_power: usize) -> ProductInsertion {
        ProductInsertion::new(descendant_power, 1, 1)
    }

    /// Substitutes `q -> scale * q` in every entry of a series matrix.
    fn scale_novikov(matrix: &SeriesMatrix, scale: &Rational) -> SeriesMatrix {
        SeriesMatrix::from_entries(
            matrix
                .entries()
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|series| {
                            QSeries::from_coeffs(
                                series
                                    .coeffs()
                                    .iter()
                                    .enumerate()
                                    .map(|(degree, coeff)| {
                                        coeff * &RatFun::from_rational(scale.pow_usize(degree))
                                    })
                                    .collect(),
                            )
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    fn product_calibration_is_tensor_of_factor_calibrations() {
        // Behrend's product formula: in matching canonical frames,
        // R_{X x Y}(z) = R_X(z) (x) R_Y(z) and S factors the same way, with
        // the factor Novikov variables specialized along the ray.
        let q_degree = 2;
        let z_order = 2;
        let ray = Rational::from(3);
        let target =
            ProductProjectiveRay::new(1, 1, weights_x(), weights_y(), ray.clone()).unwrap();
        let calibration = target.build_calibration(q_degree, z_order).unwrap();

        let factor_x =
            projective_space_j_calibration_at_lambda_weights(1, q_degree, z_order, &weights_x())
                .unwrap();
        let factor_y =
            projective_space_j_calibration_at_lambda_weights(1, q_degree, z_order, &weights_y())
                .unwrap();

        for order in 0..=z_order {
            // Tensor coefficient at z^order: sum over order splittings.
            for i in 0..2usize {
                for i_prime in 0..2usize {
                    for j in 0..2usize {
                        for j_prime in 0..2usize {
                            let row = target.point(i, j);
                            let col = target.point(i_prime, j_prime);
                            let mut expected = QSeries::zero(q_degree);
                            for split in 0..=order {
                                let left = scale_novikov(
                                    factor_x.r_matrix.coefficient(split).unwrap(),
                                    &Rational::one(),
                                );
                                let right = scale_novikov(
                                    factor_y.r_matrix.coefficient(order - split).unwrap(),
                                    &ray,
                                );
                                expected = expected
                                    .add(&left.entry(i, i_prime).mul(right.entry(j, j_prime)));
                            }
                            let actual = calibration
                                .r_matrix
                                .coefficient(order)
                                .unwrap()
                                .entry(row, col);
                            for degree in 0..=q_degree {
                                assert_eq!(
                                    actual.coeff(degree).unwrap().as_rational(),
                                    expected.coeff(degree).unwrap().as_rational(),
                                    "R tensor mismatch at z^{order} ({i}{j},{i_prime}{j_prime}) q^{degree}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn classical_three_point_integral() {
        // <H1, H2, 1> at bidegree (0,0) is the classical integral
        // int_{P1xP1} H1 H2 = 1; a single ray suffices at total degree zero.
        let invariants = reconstruct_bidegree_invariants(
            1,
            1,
            &weights_x(),
            &weights_y(),
            0,
            0,
            &[
                ProductInsertion::new(0, 1, 0),
                ProductInsertion::new(0, 0, 1),
                ProductInsertion::new(0, 0, 0),
            ],
        )
        .unwrap();
        assert_eq!(invariants, vec![Rational::one()]);

        // Asymmetric product P^1 x P^2: int H1 H2^2 = 1.
        let asymmetric = reconstruct_bidegree_invariants(
            1,
            2,
            &weights_x(),
            &[Rational::from(7), Rational::from(17), Rational::from(29)],
            0,
            0,
            &[
                ProductInsertion::new(0, 1, 0),
                ProductInsertion::new(0, 0, 2),
                ProductInsertion::new(0, 0, 0),
            ],
        )
        .unwrap();
        assert_eq!(asymmetric, vec![Rational::one()]);
    }

    #[test]
    fn three_points_lie_on_one_bidegree_one_one_curve() {
        // Genus 0, three point insertions, total degree 2: the only
        // dimension-valid bidegree with a nonzero count is (1,1), where the
        // unique curve through three general points contributes 1; the
        // reducible ruling classes (2,0) and (0,2) contribute 0.
        let invariants = reconstruct_bidegree_invariants(
            1,
            1,
            &weights_x(),
            &weights_y(),
            0,
            2,
            &[point_class(0), point_class(0), point_class(0)],
        )
        .unwrap();
        assert_eq!(
            invariants,
            vec![Rational::zero(), Rational::one(), Rational::zero()],
            "expected N_(2,0)=0, N_(1,1)=1, N_(0,2)=0"
        );
    }

    #[test]
    fn rulings_through_a_point_distinguish_the_factors() {
        // <pt, H1, H1> at total degree 1: the divisor axiom gives
        // d1^2 <pt>_beta, and the unique ruling through a point contributes
        // <pt>_{(1,0)} = 1, so the bidegree split is [1, 0]; swapping the
        // divisor to H2 flips it to [0, 1].  This pins the orientation of
        // the two Novikov directions through the whole pipeline.
        let first = reconstruct_bidegree_invariants(
            1,
            1,
            &weights_x(),
            &weights_y(),
            0,
            1,
            &[
                point_class(0),
                ProductInsertion::new(0, 1, 0),
                ProductInsertion::new(0, 1, 0),
            ],
        )
        .unwrap();
        assert_eq!(first, vec![Rational::one(), Rational::zero()]);

        let second = reconstruct_bidegree_invariants(
            1,
            1,
            &weights_x(),
            &weights_y(),
            0,
            1,
            &[
                point_class(0),
                ProductInsertion::new(0, 0, 1),
                ProductInsertion::new(0, 0, 1),
            ],
        )
        .unwrap();
        assert_eq!(second, vec![Rational::zero(), Rational::one()]);
    }

    #[test]
    fn dimension_mismatched_bidegrees_are_filtered_to_zero() {
        // <tau_2(pt), pt, pt> at total degree 1 matches no bidegree's
        // virtual dimension.  The raw ray values are legitimately nonzero
        // equivariant quantities; the reconstruction filters them so the
        // output is the vanishing non-equivariant invariant.
        let invariants = reconstruct_bidegree_invariants(
            1,
            1,
            &weights_x(),
            &weights_y(),
            0,
            1,
            &[point_class(2), point_class(0), point_class(0)],
        )
        .unwrap();
        assert_eq!(invariants, vec![Rational::zero(), Rational::zero()]);
    }
}
