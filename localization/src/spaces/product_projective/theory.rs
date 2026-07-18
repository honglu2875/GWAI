//! Canonical Gromov--Witten theory data for products of projective spaces.

use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{
    ensure_curve_bound_fits_i64, power_label, scan_bound_overflow, tex_power_label,
    two_ray_class_count, BasisElement, BasisId, CharacteristicNumbers, CurveClass, CurveClassSpace,
    CurveClassSplit, CurveEffectivity, GwTheory, NondegeneratePairing, Parity, StateSpace,
    StateSpaceMatrix, VirasoroOperatorKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductProjectiveTheory {
    n: usize,
    m: usize,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
    characteristic_numbers: CharacteristicNumbers,
}

impl ProductProjectiveTheory {
    pub fn new(n: usize, m: usize) -> Result<Self, GwError> {
        if n == 0 || m == 0 {
            return Err(GwError::ConventionMismatch(
                "a P^0 product factor has no independent curve coordinate; reduce to projective space"
                    .to_string(),
            ));
        }
        let n_plus_one = n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("product dimension is too large".to_string())
        })?;
        let m_plus_one = m.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("product dimension is too large".to_string())
        })?;
        n.checked_add(m).ok_or_else(|| {
            GwError::UnsupportedInvariant("product target dimension overflow".to_string())
        })?;
        i64::try_from(n_plus_one).map_err(|_| {
            GwError::UnsupportedInvariant(
                "product c1 coefficient does not fit the curve lattice".to_string(),
            )
        })?;
        i64::try_from(m_plus_one).map_err(|_| {
            GwError::UnsupportedInvariant(
                "product c1 coefficient does not fit the curve lattice".to_string(),
            )
        })?;
        let size = n_plus_one.checked_mul(m_plus_one).ok_or_else(|| {
            GwError::UnsupportedInvariant("product state-space size overflow".to_string())
        })?;
        let id = |h1: usize, h2: usize| BasisId(h1 * m_plus_one + h2);
        let mut basis = Vec::new();
        basis.try_reserve_exact(size).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate the {size}-element product basis"
            ))
        })?;
        for h1 in 0..=n {
            for h2 in 0..=m {
                let label = match (h1, h2) {
                    (0, 0) => "1".to_string(),
                    (a, 0) => power_label("H1", a),
                    (0, b) => power_label("H2", b),
                    (a, b) => format!("{} {}", power_label("H1", a), power_label("H2", b)),
                };
                basis.push(BasisElement {
                    id: id(h1, h2),
                    label,
                    hodge_p_degree: h1 + h2,
                    complex_codimension: h1 + h2,
                    parity: Parity::Even,
                });
            }
        }
        let mut metric = StateSpaceMatrix::try_zero(size)?;
        for h1 in 0..=n {
            for h2 in 0..=m {
                metric.set_entry(id(n - h1, m - h2).0, id(h1, h2).0, Rational::one());
            }
        }
        let pairing = NondegeneratePairing::from_metric(metric)?;
        let mut c1_action = StateSpaceMatrix::try_zero(size)?;
        for h1 in 0..=n {
            for h2 in 0..=m {
                let input = id(h1, h2).0;
                if h1 < n {
                    c1_action.set_entry(id(h1 + 1, h2).0, input, Rational::from(n_plus_one));
                }
                if h2 < m {
                    c1_action.set_entry(id(h1, h2 + 1).0, input, Rational::from(m_plus_one));
                }
            }
        }
        let state_space = StateSpace::try_new(basis, id(0, 0), Some(pairing), Some(c1_action))?;
        let euler_n = Rational::from(n) + Rational::one();
        let euler_m = Rational::from(m) + Rational::one();
        let c1cn_n = Rational::from(n) * euler_n.pow_usize(2) / Rational::from(2);
        let c1cn_m = Rational::from(m) * euler_m.pow_usize(2) / Rational::from(2);
        Ok(Self {
            n,
            m,
            state_space,
            curve_space: CurveClassSpace {
                coordinate_names: vec!["d1".to_string(), "d2".to_string()],
                effective_grading: "d1+d2".to_string(),
            },
            characteristic_numbers: CharacteristicNumbers {
                top_chern_integral: euler_n.clone() * euler_m.clone(),
                c1_c_dim_minus_one_integral: c1cn_n * euler_m + euler_n * c1cn_m,
                convention: "product orientation with integral H1^n H2^m = 1".to_string(),
                source: "Whitney product c(T(XxY))=c(TX)c(TY)".to_string(),
            },
        })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    pub fn basis_powers(&self, basis: BasisId) -> Option<(usize, usize)> {
        (basis.0 < (self.n + 1) * (self.m + 1))
            .then_some((basis.0 / (self.m + 1), basis.0 % (self.m + 1)))
    }

    pub fn basis_id(&self, h1_power: usize, h2_power: usize) -> Option<BasisId> {
        (h1_power <= self.n && h2_power <= self.m)
            .then_some(BasisId(h1_power * (self.m + 1) + h2_power))
    }

    pub fn try_curve(&self, d1: usize, d2: usize) -> Result<CurveClass, GwError> {
        let d1 = i64::try_from(d1).map_err(|_| scan_bound_overflow())?;
        let d2 = i64::try_from(d2).map_err(|_| scan_bound_overflow())?;
        Ok(CurveClass::new(vec![d1, d2]))
    }

    /// Construct a nonnegative geometric bidegree.
    ///
    /// Panics when either coordinate does not fit the canonical signed curve
    /// lattice; use [`Self::try_curve`] for untrusted input.
    pub fn curve(&self, d1: usize, d2: usize) -> CurveClass {
        self.try_curve(d1, d2)
            .expect("product curve degrees must fit in i64")
    }

    pub fn bidegree(&self, curve: &CurveClass) -> Option<(usize, usize)> {
        if curve.rank() != 2 {
            return None;
        }
        Some((
            usize::try_from(curve.coordinates()[0]).ok()?,
            usize::try_from(curve.coordinates()[1]).ok()?,
        ))
    }
}

impl GwTheory for ProductProjectiveTheory {
    fn theory_id(&self) -> String {
        format!("P^{} x P^{}", self.n, self.m)
    }

    fn theory_tex(&self) -> String {
        format!(
            "\\mathbb{{P}}^{{{}}}\\times\\mathbb{{P}}^{{{}}}",
            self.n, self.m
        )
    }

    fn target_dimension(&self) -> usize {
        self.n + self.m
    }

    fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
        VirasoroOperatorKind::StandardCompactGetzler
    }

    fn theory_fingerprint(&self) -> String {
        format!(
            "gw-theory-v1/standard-compact/product-projective/{}/{}",
            self.n, self.m
        )
    }

    fn state_space(&self) -> &StateSpace {
        &self.state_space
    }

    fn curve_class_space(&self) -> &CurveClassSpace {
        &self.curve_space
    }

    fn basis_tex(&self, basis: BasisId) -> Option<String> {
        let (h1, h2) = self.basis_powers(basis)?;
        Some(match (h1, h2) {
            (0, 0) => "1".to_string(),
            (a, 0) => tex_power_label("H_1", a),
            (0, b) => tex_power_label("H_2", b),
            (a, b) => format!(
                "{}\\,{}",
                tex_power_label("H_1", a),
                tex_power_label("H_2", b)
            ),
        })
    }

    fn curve_coordinate_tex_names(&self) -> Vec<String> {
        vec!["d_1".to_string(), "d_2".to_string()]
    }

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        (self.n as i64 + 1)
            .checked_mul(curve.coordinates()[0])
            .and_then(|left| {
                (self.m as i64 + 1)
                    .checked_mul(curve.coordinates()[1])
                    .and_then(|right| left.checked_add(right))
            })
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(match self.bidegree(curve) {
            Some(_) => CurveEffectivity::Effective,
            None => CurveEffectivity::Ineffective,
        })
    }

    fn classical_product(
        &self,
        left: BasisId,
        right: BasisId,
    ) -> Result<Vec<(BasisId, Rational)>, GwError> {
        let (left_h1, left_h2) = self.basis_powers(left).ok_or_else(|| {
            GwError::ConventionMismatch(
                "product cup product received an invalid basis id".to_string(),
            )
        })?;
        let (right_h1, right_h2) = self.basis_powers(right).ok_or_else(|| {
            GwError::ConventionMismatch(
                "product cup product received an invalid basis id".to_string(),
            )
        })?;
        let h1 = left_h1.checked_add(right_h1).ok_or_else(|| {
            GwError::AlgebraFailure("product cup-product degree overflow".to_string())
        })?;
        let h2 = left_h2.checked_add(right_h2).ok_or_else(|| {
            GwError::AlgebraFailure("product cup-product degree overflow".to_string())
        })?;
        Ok(self
            .basis_id(h1, h2)
            .map(|basis| vec![(basis, Rational::one())])
            .unwrap_or_default())
    }

    fn stabilizing_divisor(&self, curve: &CurveClass) -> Result<Option<(BasisId, i64)>, GwError> {
        self.curve_space.validate(curve)?;
        if curve.coordinates()[0] > 0 {
            Ok(Some((
                self.basis_id(1, 0).expect("product first divisor exists"),
                curve.coordinates()[0],
            )))
        } else if curve.coordinates()[1] > 0 {
            Ok(Some((
                self.basis_id(0, 1).expect("product second divisor exists"),
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
        let (d1, d2) = self.bidegree(total).ok_or_else(|| {
            GwError::ConventionMismatch("product bidegree must be nonnegative".to_string())
        })?;
        let count = self.admissible_decomposition_count(total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {count} product curve-class decompositions"
            ))
        })?;
        for left_d1 in 0..=d1 {
            for left_d2 in 0..=d2 {
                out.push(CurveClassSplit {
                    left: self.curve(left_d1, left_d2),
                    right: self.curve(d1 - left_d1, d2 - left_d2),
                });
            }
        }
        Ok(out)
    }

    fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError> {
        self.curve_space.validate(total)?;
        let Some((d1, d2)) = self.bidegree(total) else {
            return Ok(0);
        };
        d1.checked_add(1)
            .and_then(|left| d2.checked_add(1).and_then(|right| left.checked_mul(right)))
            .ok_or_else(scan_bound_overflow)
    }

    fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError> {
        let count = self.bounded_admissible_class_count(max_total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!("cannot allocate {count} product curve classes"))
        })?;
        for total in 0..=max_total {
            for d1 in 0..=total {
                out.push(self.curve(d1, total - d1));
            }
        }
        Ok(out)
    }

    fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError> {
        ensure_curve_bound_fits_i64(max_total)?;
        two_ray_class_count(max_total)
    }
}
