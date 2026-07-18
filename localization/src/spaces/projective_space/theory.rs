//! Canonical Gromov--Witten theory data for ordinary projective space.

use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{
    ensure_curve_bound_fits_i64, scan_bound_overflow, tex_power_label, BasisElement, BasisId,
    CharacteristicNumbers, CurveClass, CurveClassSpace, CurveClassSplit, CurveEffectivity,
    GwTheory, NondegeneratePairing, Parity, StateSpace, StateSpaceMatrix, VirasoroOperatorKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveSpaceTheory {
    n: usize,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
    characteristic_numbers: CharacteristicNumbers,
}

impl ProjectiveSpaceTheory {
    pub fn new(n: usize) -> Self {
        Self::try_new(n).expect("projective-space canonical theory construction failed")
    }

    pub fn try_new(n: usize) -> Result<Self, GwError> {
        let size = n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("projective-space dimension is too large".to_string())
        })?;
        i64::try_from(size).map_err(|_| {
            GwError::UnsupportedInvariant(
                "projective-space c1 coefficient does not fit the curve lattice".to_string(),
            )
        })?;
        let mut basis = Vec::new();
        basis.try_reserve_exact(size).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate the {size}-element projective-space basis"
            ))
        })?;
        basis.extend((0..size).map(|power| BasisElement {
            id: BasisId(power),
            label: if power == 0 {
                "1".to_string()
            } else if power == 1 {
                "H".to_string()
            } else {
                format!("H^{power}")
            },
            hodge_p_degree: power,
            complex_codimension: power,
            parity: Parity::Even,
        }));
        let mut metric = StateSpaceMatrix::try_zero(size)?;
        for left in 0..size {
            metric.set_entry(left, n - left, Rational::one());
        }
        // In the monomial basis the P^n pairing is the anti-diagonal
        // permutation matrix, hence is its own inverse.  Recording that
        // closed form avoids cubic Gaussian elimination for data whose
        // inverse is known analytically.
        let pairing = NondegeneratePairing {
            metric: metric.clone(),
            inverse: metric,
        };
        let mut c1_action = StateSpaceMatrix::try_zero(size)?;
        let n_plus_one = Rational::from(n) + Rational::one();
        for input in 0..n {
            c1_action.set_entry(input + 1, input, n_plus_one.clone());
        }
        // Every id and matrix above is built from the same checked `size`, and
        // the analytic pairing inverse was just established.  Generic
        // extension providers still use `StateSpace::try_new` for validation.
        let state_space = StateSpace {
            basis,
            unit: BasisId(0),
            pairing: Some(pairing),
            c1_action: Some(c1_action),
        };
        let top_chern_integral = n_plus_one.clone();
        let c1_c_dim_minus_one_integral =
            Rational::from(n) * n_plus_one.pow_usize(2) / Rational::from(2);
        Ok(Self {
            n,
            state_space,
            curve_space: CurveClassSpace {
                coordinate_names: vec!["d".to_string()],
                effective_grading: "d".to_string(),
            },
            characteristic_numbers: CharacteristicNumbers {
                top_chern_integral,
                c1_c_dim_minus_one_integral,
                convention: "Euler sequence with integral_X H^n = 1".to_string(),
                source: "c(TP^n)=(1+H)^(n+1)".to_string(),
            },
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn try_curve(&self, degree: usize) -> Result<CurveClass, GwError> {
        let degree = i64::try_from(degree).map_err(|_| scan_bound_overflow())?;
        Ok(CurveClass::new(vec![degree]))
    }

    /// Construct a nonnegative curve class.
    ///
    /// Panics when `degree` does not fit the canonical signed curve lattice;
    /// use [`Self::try_curve`] for untrusted input.
    pub fn curve(&self, degree: usize) -> CurveClass {
        self.try_curve(degree)
            .expect("projective curve degree must fit in i64")
    }

    pub fn degree(&self, curve: &CurveClass) -> Option<usize> {
        usize::try_from(curve.coordinate(0)?)
            .ok()
            .filter(|_| curve.rank() == 1)
    }
}

impl GwTheory for ProjectiveSpaceTheory {
    fn theory_id(&self) -> String {
        format!("P^{}", self.n)
    }

    fn theory_tex(&self) -> String {
        format!("\\mathbb{{P}}^{{{}}}", self.n)
    }

    fn target_dimension(&self) -> usize {
        self.n
    }

    fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
        VirasoroOperatorKind::StandardCompactGetzler
    }

    fn theory_fingerprint(&self) -> String {
        format!("gw-theory-v1/standard-compact/projective-space/{}", self.n)
    }

    fn state_space(&self) -> &StateSpace {
        &self.state_space
    }

    fn curve_class_space(&self) -> &CurveClassSpace {
        &self.curve_space
    }

    fn basis_tex(&self, basis: BasisId) -> Option<String> {
        (basis.0 <= self.n).then(|| tex_power_label("H", basis.0))
    }

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        curve.coordinates()[0]
            .checked_mul((self.n + 1) as i64)
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(
            if curve.coordinates()[0] < 0 || (self.n == 0 && curve.coordinates()[0] != 0) {
                CurveEffectivity::Ineffective
            } else {
                CurveEffectivity::Effective
            },
        )
    }

    fn classical_product(
        &self,
        left: BasisId,
        right: BasisId,
    ) -> Result<Vec<(BasisId, Rational)>, GwError> {
        if left.0 > self.n || right.0 > self.n {
            return Err(GwError::ConventionMismatch(
                "projective-space cup product received an invalid basis id".to_string(),
            ));
        }
        let power = left.0.checked_add(right.0).ok_or_else(|| {
            GwError::AlgebraFailure("projective-space cup-product degree overflow".to_string())
        })?;
        Ok((power <= self.n)
            .then_some((BasisId(power), Rational::one()))
            .into_iter()
            .collect())
    }

    fn stabilizing_divisor(&self, curve: &CurveClass) -> Result<Option<(BasisId, i64)>, GwError> {
        self.curve_space.validate(curve)?;
        Ok((self.n > 0 && curve.coordinates()[0] > 0)
            .then_some((BasisId(1), curve.coordinates()[0])))
    }

    fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers> {
        Some(&self.characteristic_numbers)
    }

    fn admissible_decompositions(
        &self,
        total: &CurveClass,
    ) -> Result<Vec<CurveClassSplit>, GwError> {
        self.curve_space.validate(total)?;
        if self.effectivity(total)? != CurveEffectivity::Effective {
            return Ok(Vec::new());
        }
        let degree = self.degree(total).ok_or_else(|| {
            GwError::ConventionMismatch("projective degree must be nonnegative".to_string())
        })?;
        let count = self.admissible_decomposition_count(total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {count} projective curve-class decompositions"
            ))
        })?;
        out.extend((0..=degree).map(|left| CurveClassSplit {
            left: self.curve(left),
            right: self.curve(degree - left),
        }));
        Ok(out)
    }

    fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError> {
        self.curve_space.validate(total)?;
        if self.effectivity(total)? != CurveEffectivity::Effective {
            return Ok(0);
        }
        self.degree(total)
            .and_then(|degree| degree.checked_add(1))
            .ok_or_else(scan_bound_overflow)
    }

    fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError> {
        let count = self.bounded_admissible_class_count(max_total)?;
        if self.n == 0 {
            return Ok(vec![self.curve(0)]);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {count} projective curve classes"
            ))
        })?;
        out.extend((0..=max_total).map(|degree| self.curve(degree)));
        Ok(out)
    }

    fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError> {
        if self.n == 0 {
            Ok(1)
        } else {
            ensure_curve_bound_fits_i64(max_total)?;
            max_total.checked_add(1).ok_or_else(scan_bound_overflow)
        }
    }
}
