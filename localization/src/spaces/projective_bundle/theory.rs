//! Canonical Gromov--Witten theory data for split projective bundles.

use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{
    canonicalize_line_summand_payloads, power_label, scan_bound_overflow, tex_power_label,
    two_ray_class_count, BasisElement, BasisId, CharacteristicNumbers, CurveClass, CurveClassSpace,
    CurveClassSplit, CurveEffectivity, GwTheory, NondegeneratePairing, Parity, StateSpace,
    StateSpaceMatrix, VirasoroOperatorKind,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveBundleTheory {
    n: usize,
    twists: Vec<usize>,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
    characteristic_numbers: CharacteristicNumbers,
}

impl ProjectiveBundleTheory {
    pub fn new(n: usize, twists: Vec<usize>) -> Result<Self, GwError> {
        let rank = twists.len();
        let (twists, _) =
            canonicalize_line_summand_payloads(twists, vec![(); rank], "projective-bundle")?;
        Self::from_canonical_twists(n, twists)
    }

    fn from_canonical_twists(n: usize, twists: Vec<usize>) -> Result<Self, GwError> {
        if n == 0 || twists.len() < 2 {
            return Err(GwError::ConventionMismatch(
                "projective-bundle theory requires a positive-dimensional base and rank at least two"
                    .to_string(),
            ));
        }
        if !twists.contains(&0) {
            return Err(GwError::ConventionMismatch(
                "projective-bundle twists must be normalized so their minimum is zero".to_string(),
            ));
        }
        let rank = twists.len();
        let n_plus_one = n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("projective-bundle dimension is too large".to_string())
        })?;
        let size = n_plus_one.checked_mul(rank).ok_or_else(|| {
            GwError::UnsupportedInvariant("projective-bundle state space is too large".to_string())
        })?;
        let twist_sum = twists.iter().try_fold(0usize, |sum, twist| {
            sum.checked_add(*twist).ok_or_else(|| {
                GwError::UnsupportedInvariant("projective-bundle twist sum overflow".to_string())
            })
        })?;
        let c1_h = n_plus_one.checked_add(twist_sum).ok_or_else(|| {
            GwError::UnsupportedInvariant("projective-bundle c1 coefficient overflow".to_string())
        })?;
        i64::try_from(c1_h).map_err(|_| {
            GwError::UnsupportedInvariant(
                "projective-bundle c1 coefficient does not fit the curve lattice".to_string(),
            )
        })?;
        let dimension = n.checked_add(rank - 1).ok_or_else(|| {
            GwError::UnsupportedInvariant("projective-bundle dimension is too large".to_string())
        })?;
        let id = |h: usize, xi: usize| BasisId(h * rank + xi);
        let mut basis = Vec::new();
        basis.try_reserve_exact(size).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate the {size}-element projective-bundle basis"
            ))
        })?;
        for h in 0..=n {
            for xi in 0..rank {
                let label = match (h, xi) {
                    (0, 0) => "1".to_string(),
                    (a, 0) => power_label("H", a),
                    (0, b) => power_label("xi", b),
                    (a, b) => format!("{} {}", power_label("H", a), power_label("xi", b)),
                };
                basis.push(BasisElement {
                    id: id(h, xi),
                    label,
                    hodge_p_degree: h + xi,
                    complex_codimension: h + xi,
                    parity: Parity::Even,
                });
            }
        }
        let mut metric = StateSpaceMatrix::try_zero(size)?;
        for left_h in 0..=n {
            for left_xi in 0..rank {
                for right_h in 0..=n {
                    for right_xi in 0..rank {
                        let value = bundle_monomial_integral(
                            n,
                            &twists,
                            left_h + right_h,
                            left_xi + right_xi,
                        );
                        metric.set_entry(id(left_h, left_xi).0, id(right_h, right_xi).0, value);
                    }
                }
            }
        }
        let pairing = NondegeneratePairing::from_metric(metric)?;
        let mut c1_action = StateSpaceMatrix::try_zero(size)?;
        for h in 0..=n {
            for xi in 0..rank {
                let input = id(h, xi).0;
                for ((out_h, out_xi), coefficient) in reduce_bundle_monomial(n, &twists, h + 1, xi)
                {
                    let old = c1_action.entry(id(out_h, out_xi).0, input).clone();
                    c1_action.set_entry(
                        id(out_h, out_xi).0,
                        input,
                        old + Rational::from(c1_h) * coefficient,
                    );
                }
                for ((out_h, out_xi), coefficient) in reduce_bundle_monomial(n, &twists, h, xi + 1)
                {
                    let old = c1_action.entry(id(out_h, out_xi).0, input).clone();
                    c1_action.set_entry(
                        id(out_h, out_xi).0,
                        input,
                        old + Rational::from(rank) * coefficient,
                    );
                }
            }
        }
        let state_space = StateSpace::try_new(basis, id(0, 0), Some(pairing), Some(c1_action))?;
        let characteristic_numbers = CharacteristicNumbers {
            top_chern_integral: (Rational::from(n) + Rational::one())
                * Rational::from(rank),
            c1_c_dim_minus_one_integral: bundle_c1_c_dim_minus_one_integral(
                n,
                &twists,
                dimension,
            ),
            convention: "line-projectivization P(E), xi=-c1(S), with prod_i(xi+a_i H)=0 and integral H^n xi^(r-1)=1"
                .to_string(),
            source: "relative Euler sequence: c(TX)=(1+H)^(n+1) prod_i(1+xi+a_i H)"
                .to_string(),
        };
        Ok(Self {
            n,
            twists,
            state_space,
            curve_space: CurveClassSpace {
                coordinate_names: vec!["H.beta".to_string(), "xi.beta".to_string()],
                effective_grading: "d1 + (d2 + max(a) d1)".to_string(),
            },
            characteristic_numbers,
        })
    }

    /// Reorder data attached to input summands into this theory's canonical
    /// direct-sum order.
    ///
    /// The input twists must describe this same bundle presentation. Keeping
    /// this permutation on the canonical theory prevents ray adapters, CLI
    /// weights, and reconstruction entry points from implementing their own
    /// potentially divergent sorting rules.
    pub fn canonicalize_summand_payloads<T>(
        &self,
        twists: Vec<usize>,
        payloads: Vec<T>,
    ) -> Result<Vec<T>, GwError> {
        let (canonical_twists, payloads) =
            canonicalize_line_summand_payloads(twists, payloads, "projective-bundle")?;
        if canonical_twists != self.twists {
            return Err(GwError::ConventionMismatch(
                "projective-bundle summand payloads do not describe the canonical theory"
                    .to_string(),
            ));
        }
        Ok(payloads)
    }

    pub fn base_dimension(&self) -> usize {
        self.n
    }

    pub fn twists(&self) -> &[usize] {
        &self.twists
    }

    pub fn rank(&self) -> usize {
        self.twists.len()
    }

    pub fn basis_powers(&self, basis: BasisId) -> Option<(usize, usize)> {
        (basis.0 < (self.n + 1) * self.rank())
            .then_some((basis.0 / self.rank(), basis.0 % self.rank()))
    }

    pub fn basis_id(&self, h_power: usize, xi_power: usize) -> Option<BasisId> {
        (h_power <= self.n && xi_power < self.rank())
            .then_some(BasisId(h_power * self.rank() + xi_power))
    }

    /// Multiply one canonical basis element by the tautological divisor
    /// `xi`, reducing the result through the projective-bundle relation.
    ///
    /// The output is expressed in the same canonical `H^h xi^j` basis and
    /// sorted by basis id.  An empty output represents the zero class.
    pub fn multiply_basis_by_xi(
        &self,
        basis: BasisId,
    ) -> Result<Vec<(BasisId, Rational)>, GwError> {
        let xi = self
            .basis_id(0, 1)
            .expect("projective-bundle rank is at least two");
        self.classical_product(xi, basis)
    }

    pub fn try_curve(&self, d1: usize, d2: i64) -> Result<CurveClass, GwError> {
        let d1 = i64::try_from(d1).map_err(|_| scan_bound_overflow())?;
        Ok(CurveClass::new(vec![d1, d2]))
    }

    /// Construct a geometric bundle bidegree.
    ///
    /// Panics when the base coordinate does not fit the canonical signed curve
    /// lattice; use [`Self::try_curve`] for untrusted input.
    pub fn curve(&self, d1: usize, d2: i64) -> CurveClass {
        self.try_curve(d1, d2)
            .expect("bundle base degree must fit in i64")
    }

    pub fn curve_from_shifted(
        &self,
        d1: usize,
        shifted_degree: usize,
    ) -> Result<CurveClass, GwError> {
        let d1_i64 = i64::try_from(d1).map_err(|_| scan_bound_overflow())?;
        let shifted_i64 = i64::try_from(shifted_degree).map_err(|_| scan_bound_overflow())?;
        let big_a = i64::try_from(*self.twists.iter().max().expect("nonempty"))
            .map_err(|_| scan_bound_overflow())?;
        let fiber_degree = big_a
            .checked_mul(d1_i64)
            .and_then(|offset| shifted_i64.checked_sub(offset))
            .ok_or_else(scan_bound_overflow)?;
        Ok(CurveClass::new(vec![d1_i64, fiber_degree]))
    }

    pub fn bidegree(&self, curve: &CurveClass) -> Option<(usize, i64)> {
        if curve.rank() != 2 {
            return None;
        }
        Some((
            usize::try_from(curve.coordinates()[0]).ok()?,
            curve.coordinates()[1],
        ))
    }

    pub fn shifted_bidegree(&self, curve: &CurveClass) -> Option<(usize, usize)> {
        let (d1, d2) = self.bidegree(curve)?;
        let big_a = i64::try_from(*self.twists.iter().max()?).ok()?;
        let d1_i64 = i64::try_from(d1).ok()?;
        let shifted = d2.checked_add(big_a.checked_mul(d1_i64)?)?;
        Some((d1, usize::try_from(shifted).ok()?))
    }
}

impl GwTheory for ProjectiveBundleTheory {
    fn theory_id(&self) -> String {
        let summands = self
            .twists
            .iter()
            .map(|twist| match twist {
                0 => "O".to_string(),
                degree => format!("O({degree})"),
            })
            .collect::<Vec<_>>()
            .join(" + ");
        format!("P({summands}) over P^{}", self.n)
    }

    fn theory_tex(&self) -> String {
        let summands = self
            .twists
            .iter()
            .map(|twist| format!("\\mathcal{{O}}({twist})"))
            .collect::<Vec<_>>()
            .join("\\oplus");
        format!("\\mathbb{{P}}({summands})\\to\\mathbb{{P}}^{{{}}}", self.n)
    }

    fn target_dimension(&self) -> usize {
        self.n + self.rank() - 1
    }

    fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
        VirasoroOperatorKind::StandardCompactGetzler
    }

    fn theory_fingerprint(&self) -> String {
        format!(
            "gw-theory-v1/standard-compact/projective-bundle/{}/{:?}",
            self.n, self.twists
        )
    }

    fn state_space(&self) -> &StateSpace {
        &self.state_space
    }

    fn curve_class_space(&self) -> &CurveClassSpace {
        &self.curve_space
    }

    fn basis_tex(&self, basis: BasisId) -> Option<String> {
        let (h, xi) = self.basis_powers(basis)?;
        Some(match (h, xi) {
            (0, 0) => "1".to_string(),
            (a, 0) => tex_power_label("H", a),
            (0, b) => tex_power_label("\\xi", b),
            (a, b) => format!(
                "{}\\,{}",
                tex_power_label("H", a),
                tex_power_label("\\xi", b)
            ),
        })
    }

    fn curve_coordinate_tex_names(&self) -> Vec<String> {
        vec![
            "H\\!\\cdot\\!\\beta".to_string(),
            "\\xi\\!\\cdot\\!\\beta".to_string(),
        ]
    }

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        let c1_h = (self.n + 1 + self.twists.iter().sum::<usize>()) as i64;
        c1_h.checked_mul(curve.coordinates()[0])
            .and_then(|value| {
                (self.rank() as i64)
                    .checked_mul(curve.coordinates()[1])
                    .and_then(|fiber| value.checked_add(fiber))
            })
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(match self.shifted_bidegree(curve) {
            // The shifted I-function cone is a conservative support cone; it
            // need not certify that every lattice point is represented by a
            // curve.  Unknown means "query the backend", never "force zero".
            Some(_) => CurveEffectivity::Unknown,
            None => CurveEffectivity::Ineffective,
        })
    }

    fn classical_product(
        &self,
        left: BasisId,
        right: BasisId,
    ) -> Result<Vec<(BasisId, Rational)>, GwError> {
        let (left_h, left_xi) = self.basis_powers(left).ok_or_else(|| {
            GwError::ConventionMismatch(
                "projective-bundle cup product received an invalid basis id".to_string(),
            )
        })?;
        let (right_h, right_xi) = self.basis_powers(right).ok_or_else(|| {
            GwError::ConventionMismatch(
                "projective-bundle cup product received an invalid basis id".to_string(),
            )
        })?;
        let h_power = left_h.checked_add(right_h).ok_or_else(|| {
            GwError::AlgebraFailure("projective-bundle cup-product degree overflow".to_string())
        })?;
        let xi_power = left_xi.checked_add(right_xi).ok_or_else(|| {
            GwError::AlgebraFailure("projective-bundle cup-product degree overflow".to_string())
        })?;
        reduce_bundle_monomial(self.n, &self.twists, h_power, xi_power)
            .into_iter()
            .map(|((out_h, out_xi), coefficient)| {
                self.basis_id(out_h, out_xi)
                    .map(|basis| (basis, coefficient))
                    .ok_or_else(|| {
                        GwError::AlgebraFailure(
                            "projective-bundle relation reduced outside its canonical basis"
                                .to_string(),
                        )
                    })
            })
            .collect()
    }

    fn stabilizing_divisor(&self, curve: &CurveClass) -> Result<Option<(BasisId, i64)>, GwError> {
        self.curve_space.validate(curve)?;
        if curve.coordinates()[0] > 0 {
            Ok(Some((
                self.basis_id(1, 0)
                    .expect("projective-bundle base divisor exists"),
                curve.coordinates()[0],
            )))
        } else if curve.coordinates()[1] > 0 {
            Ok(Some((
                self.basis_id(0, 1)
                    .expect("projective-bundle fiber divisor exists"),
                curve.coordinates()[1],
            )))
        } else {
            Ok(None)
        }
    }

    fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers> {
        Some(&self.characteristic_numbers)
    }

    fn admissible_decompositions(
        &self,
        total: &CurveClass,
    ) -> Result<Vec<CurveClassSplit>, GwError> {
        self.curve_space.validate(total)?;
        if self.effectivity(total)? == CurveEffectivity::Ineffective {
            return Ok(Vec::new());
        }
        let (d1, d2_shifted) = self.shifted_bidegree(total).ok_or_else(|| {
            GwError::ConventionMismatch(
                "bundle class is outside the canonical theory's admissible cone".to_string(),
            )
        })?;
        let split_count = self.admissible_decomposition_count(total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(split_count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {split_count} bundle curve-class decompositions"
            ))
        })?;
        for left_d1 in 0..=d1 {
            for left_shifted in 0..=d2_shifted {
                let right_d1 = d1 - left_d1;
                let right_shifted = d2_shifted - left_shifted;
                out.push(CurveClassSplit {
                    left: self.curve_from_shifted(left_d1, left_shifted)?,
                    right: self.curve_from_shifted(right_d1, right_shifted)?,
                });
            }
        }
        Ok(out)
    }

    fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError> {
        self.curve_space.validate(total)?;
        let Some((d1, shifted)) = self.shifted_bidegree(total) else {
            return Ok(0);
        };
        d1.checked_add(1)
            .and_then(|left| {
                shifted
                    .checked_add(1)
                    .and_then(|right| left.checked_mul(right))
            })
            .ok_or_else(scan_bound_overflow)
    }

    fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError> {
        let count = self.bounded_admissible_class_count(max_total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {count} projective-bundle curve classes"
            ))
        })?;
        for total in 0..=max_total {
            for d1 in 0..=total {
                let shifted = total - d1;
                out.push(self.curve_from_shifted(d1, shifted)?);
            }
        }
        Ok(out)
    }

    fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError> {
        let max_i64 = i64::try_from(max_total).map_err(|_| scan_bound_overflow())?;
        let big_a = i64::try_from(*self.twists.iter().max().expect("nonempty"))
            .map_err(|_| scan_bound_overflow())?;
        big_a.checked_mul(max_i64).ok_or_else(scan_bound_overflow)?;
        two_ray_class_count(max_total)
    }
}

fn elementary_symmetric_integers(values: &[usize]) -> Vec<Rational> {
    let mut elementary = vec![Rational::zero(); values.len() + 1];
    elementary[0] = Rational::one();
    for (seen, value) in values.iter().enumerate() {
        for degree in (1..=seen + 1).rev() {
            elementary[degree] = elementary[degree].clone()
                + elementary[degree - 1].clone() * Rational::from(*value);
        }
    }
    elementary
}

fn reduce_bundle_monomial(
    n: usize,
    twists: &[usize],
    h_power: usize,
    xi_power: usize,
) -> BTreeMap<(usize, usize), Rational> {
    let rank = twists.len();
    let elementary = elementary_symmetric_integers(twists);
    let mut pending = BTreeMap::from([((h_power, xi_power), Rational::one())]);
    let mut reduced = BTreeMap::new();
    while let Some((&(h, xi), coefficient)) =
        pending.iter().next_back().map(|(k, v)| (k, v.clone()))
    {
        pending.remove(&(h, xi));
        if coefficient.is_zero() || h > n {
            continue;
        }
        if xi < rank {
            let entry = reduced.entry((h, xi)).or_insert_with(Rational::zero);
            *entry += coefficient;
            continue;
        }
        for degree in 1..=rank {
            if elementary[degree].is_zero() {
                continue;
            }
            let next_h = h + degree;
            if next_h > n {
                continue;
            }
            let next_xi = xi - degree;
            let entry = pending
                .entry((next_h, next_xi))
                .or_insert_with(Rational::zero);
            *entry = entry.clone() - coefficient.clone() * elementary[degree].clone();
        }
    }
    reduced.retain(|_, coefficient| !coefficient.is_zero());
    reduced
}

fn bundle_monomial_integral(
    n: usize,
    twists: &[usize],
    h_power: usize,
    xi_power: usize,
) -> Rational {
    reduce_bundle_monomial(n, twists, h_power, xi_power)
        .get(&(n, twists.len() - 1))
        .cloned()
        .unwrap_or_else(Rational::zero)
}

fn bundle_c1_c_dim_minus_one_integral(n: usize, twists: &[usize], dimension: usize) -> Rational {
    let mut c = vec![BTreeMap::<(usize, usize), Rational>::new(); dimension];
    c[0].insert((0, 0), Rational::one());
    for _ in 0..=n {
        multiply_total_chern_factor(&mut c, dimension - 1, &[(1, 0, Rational::one())]);
    }
    for twist in twists {
        multiply_total_chern_factor(
            &mut c,
            dimension - 1,
            &[(0, 1, Rational::one()), (1, 0, Rational::from(*twist))],
        );
    }
    let c_dim_minus_one = &c[dimension - 1];
    let c1_h = Rational::from(n + 1 + twists.iter().sum::<usize>());
    let c1_xi = Rational::from(twists.len());
    let mut total = Rational::zero();
    for ((h, xi), coefficient) in c_dim_minus_one {
        total +=
            coefficient.clone() * c1_h.clone() * bundle_monomial_integral(n, twists, h + 1, *xi);
        total +=
            coefficient.clone() * c1_xi.clone() * bundle_monomial_integral(n, twists, *h, xi + 1);
    }
    total
}

fn multiply_total_chern_factor(
    classes: &mut [BTreeMap<(usize, usize), Rational>],
    max_degree: usize,
    degree_one_terms: &[(usize, usize, Rational)],
) {
    for degree in (1..=max_degree).rev() {
        let previous = classes[degree - 1].clone();
        for ((h, xi), coefficient) in previous {
            for (add_h, add_xi, factor) in degree_one_terms {
                let entry = classes[degree]
                    .entry((h + add_h, xi + add_xi))
                    .or_insert_with(Rational::zero);
                *entry += coefficient.clone() * factor.clone();
            }
        }
    }
}
