//! Canonical Gromov--Witten theory data for negative split total spaces.

use crate::core::error::GwError;
use crate::core::theory::{
    canonicalize_line_summand_payloads, ensure_curve_bound_fits_i64, power_label,
    scan_bound_overflow, tex_power_label, BasisElement, BasisId, CharacteristicNumbers, CurveClass,
    CurveClassSpace, CurveClassSplit, CurveEffectivity, GwTheory, Parity, StateSpace,
    VirasoroOperatorKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NegativeSplitDegreeIssue {
    NonPositive,
    SumOverflow,
}

/// Validate and canonicalize the unordered summands of a negative split bundle.
///
/// This is the sole normalization path shared by the geometric theory and the
/// hypergeometric twist recipe. An empty list is deliberately accepted here:
/// it denotes the untwisted recipe, while the local total-space theory rejects
/// it because that target would just be ordinary projective space.
pub(crate) fn canonicalize_negative_split_degrees(
    degrees: Vec<usize>,
) -> Result<(Vec<usize>, usize), NegativeSplitDegreeIssue> {
    if degrees.contains(&0) {
        return Err(NegativeSplitDegreeIssue::NonPositive);
    }
    let degree_sum = degrees.iter().try_fold(0usize, |sum, degree| {
        sum.checked_add(*degree)
            .ok_or(NegativeSplitDegreeIssue::SumOverflow)
    })?;
    let mut degrees = degrees;
    degrees.sort_unstable();
    Ok((degrees, degree_sum))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitTotalSpaceTheory {
    base_n: usize,
    degrees: Vec<usize>,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
}

impl NegativeSplitTotalSpaceTheory {
    pub fn new(base_n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let (degrees, degree_sum) =
            canonicalize_negative_split_degrees(degrees).map_err(|issue| match issue {
                NegativeSplitDegreeIssue::NonPositive => GwError::ConventionMismatch(
                    "negative split degrees are stored as positive absolute values".to_string(),
                ),
                NegativeSplitDegreeIssue::SumOverflow => {
                    GwError::UnsupportedInvariant("local twist degree sum overflow".to_string())
                }
            })?;
        if degrees.is_empty() {
            return Err(GwError::ConventionMismatch(
                "negative split degrees are stored as positive absolute values".to_string(),
            ));
        }
        let size = base_n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("local base dimension is too large".to_string())
        })?;
        base_n.checked_add(degrees.len()).ok_or_else(|| {
            GwError::UnsupportedInvariant("local target dimension overflow".to_string())
        })?;
        i64::try_from(size).map_err(|_| {
            GwError::UnsupportedInvariant(
                "local c1 coefficient does not fit the curve lattice".to_string(),
            )
        })?;
        i64::try_from(degree_sum).map_err(|_| {
            GwError::UnsupportedInvariant(
                "local twist degree sum does not fit the curve lattice".to_string(),
            )
        })?;
        let mut basis = Vec::new();
        basis.try_reserve_exact(size).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate the {size}-element local state-space basis"
            ))
        })?;
        basis.extend((0..size).map(|power| BasisElement {
            id: BasisId(power),
            label: power_label("H", power),
            hodge_p_degree: power,
            complex_codimension: power,
            parity: Parity::Even,
        }));
        let state_space = StateSpace::try_new(basis, BasisId(0), None, None)?;
        Ok(Self {
            base_n,
            degrees,
            state_space,
            curve_space: CurveClassSpace {
                coordinate_names: vec!["d".to_string()],
                effective_grading: "d".to_string(),
            },
        })
    }

    pub fn base_dimension(&self) -> usize {
        self.base_n
    }

    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// Construct the canonical one-parameter curve class of degree `degree`.
    pub fn try_curve(&self, degree: usize) -> Result<CurveClass, GwError> {
        degree_curve(degree)
    }

    /// Virtual dimension in the canonical degree-`degree` curve class.
    pub fn virtual_dimension_at_degree(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Result<isize, GwError> {
        virtual_dimension_at_degree(self, genus, degree, markings)
    }

    /// Whether the canonical degree-`degree` curve class is effective.
    pub fn degree_is_effective(&self, degree: usize) -> Result<bool, GwError> {
        degree_is_effective(self, degree)
    }

    /// Return the unique effective curve degree forced by dimension, if one
    /// exists. A degree-independent constraint deliberately returns `None`:
    /// callers which need every bounded solution should use
    /// [`Self::candidate_degrees_from_dimension`].
    pub fn expected_degree_from_dimension(
        &self,
        genus: usize,
        markings: usize,
        insertion_degree: usize,
    ) -> Result<Option<usize>, GwError> {
        expected_degree_from_dimension(self, genus, markings, insertion_degree)
    }

    /// Enumerate the bounded effective degrees allowed by the canonical
    /// virtual-dimension constraint. With no insertion degree, this returns
    /// the theory-owned bounded effective cone.
    pub fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Result<Vec<usize>, GwError> {
        candidate_degrees_from_dimension(self, genus, degree_max, markings, insertion_degree)
    }

    /// Reorder data attached to input line summands into this theory's
    /// canonical direct-sum order.
    ///
    /// Providers use this operation for custom equivariant weights so the
    /// target theory remains the sole owner of summand normalization.
    pub fn canonicalize_summand_payloads<T>(
        &self,
        degrees: Vec<usize>,
        payloads: Vec<T>,
    ) -> Result<Vec<T>, GwError> {
        let (canonical_degrees, payloads) =
            canonicalize_line_summand_payloads(degrees, payloads, "negative-split")?;
        if canonical_degrees != self.degrees {
            return Err(GwError::ConventionMismatch(
                "negative-split summand payloads do not describe the canonical theory".to_string(),
            ));
        }
        Ok(payloads)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DimensionDegreeConstraint {
    All,
    None,
    Unique(usize),
}

fn degree_curve(degree: usize) -> Result<CurveClass, GwError> {
    let degree = i64::try_from(degree).map_err(|_| {
        GwError::UnsupportedInvariant(
            "local curve degree does not fit the canonical signed lattice".to_string(),
        )
    })?;
    Ok(CurveClass::new(vec![degree]))
}

fn curve_degree(curve: &CurveClass) -> Result<usize, GwError> {
    if curve.rank() != 1 {
        return Err(GwError::ConventionMismatch(
            "negative-split degree queries require a rank-one curve lattice".to_string(),
        ));
    }
    usize::try_from(curve.coordinates()[0]).map_err(|_| {
        GwError::ConventionMismatch(
            "negative-split bounded support produced a negative degree".to_string(),
        )
    })
}

pub(crate) fn virtual_dimension_at_degree(
    theory: &dyn GwTheory,
    genus: usize,
    degree: usize,
    markings: usize,
) -> Result<isize, GwError> {
    theory.virtual_dimension(genus, &degree_curve(degree)?, markings)
}

pub(crate) fn degree_is_effective(theory: &dyn GwTheory, degree: usize) -> Result<bool, GwError> {
    Ok(theory.effectivity(&degree_curve(degree)?)? == CurveEffectivity::Effective)
}

fn dimension_degree_constraint(
    theory: &dyn GwTheory,
    genus: usize,
    markings: usize,
    insertion_degree: usize,
) -> Result<DimensionDegreeConstraint, GwError> {
    let insertion_degree = i128::try_from(insertion_degree).map_err(|_| {
        GwError::UnsupportedInvariant("local insertion degree does not fit in i128".to_string())
    })?;
    let constant_dimension =
        i128::try_from(virtual_dimension_at_degree(theory, genus, 0, markings)?).map_err(|_| {
            GwError::AlgebraFailure("local virtual dimension does not fit in i128".to_string())
        })?;
    let slope = i128::from(theory.c1_pairing(&degree_curve(1)?)?);
    let numerator = insertion_degree - constant_dimension;
    if slope == 0 {
        return Ok(if numerator == 0 {
            DimensionDegreeConstraint::All
        } else {
            DimensionDegreeConstraint::None
        });
    }
    if numerator % slope != 0 {
        return Ok(DimensionDegreeConstraint::None);
    }
    Ok(match usize::try_from(numerator / slope) {
        Ok(degree) => DimensionDegreeConstraint::Unique(degree),
        Err(_) => DimensionDegreeConstraint::None,
    })
}

pub(crate) fn expected_degree_from_dimension(
    theory: &dyn GwTheory,
    genus: usize,
    markings: usize,
    insertion_degree: usize,
) -> Result<Option<usize>, GwError> {
    let DimensionDegreeConstraint::Unique(degree) =
        dimension_degree_constraint(theory, genus, markings, insertion_degree)?
    else {
        return Ok(None);
    };
    Ok(degree_is_effective(theory, degree)?.then_some(degree))
}

fn bounded_effective_degrees(
    theory: &dyn GwTheory,
    degree_max: usize,
) -> Result<Vec<usize>, GwError> {
    // Validate the public one-parameter bound even for targets whose bounded
    // cone happens to collapse to degree zero (for example P^0).
    degree_curve(degree_max)?;
    theory
        .bounded_admissible_classes(degree_max)?
        .into_iter()
        .filter_map(|curve| match theory.effectivity(&curve) {
            Ok(CurveEffectivity::Effective) => Some(curve_degree(&curve)),
            Ok(CurveEffectivity::Ineffective | CurveEffectivity::Unknown) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

pub(crate) fn candidate_degrees_from_dimension(
    theory: &dyn GwTheory,
    genus: usize,
    degree_max: usize,
    markings: usize,
    insertion_degree: Option<usize>,
) -> Result<Vec<usize>, GwError> {
    degree_curve(degree_max)?;
    let Some(insertion_degree) = insertion_degree else {
        return bounded_effective_degrees(theory, degree_max);
    };
    match dimension_degree_constraint(theory, genus, markings, insertion_degree)? {
        DimensionDegreeConstraint::All => bounded_effective_degrees(theory, degree_max),
        DimensionDegreeConstraint::None => Ok(Vec::new()),
        DimensionDegreeConstraint::Unique(degree)
            if degree <= degree_max && degree_is_effective(theory, degree)? =>
        {
            Ok(vec![degree])
        }
        DimensionDegreeConstraint::Unique(_) => Ok(Vec::new()),
    }
}

impl GwTheory for NegativeSplitTotalSpaceTheory {
    fn theory_id(&self) -> String {
        format!("Tot(O(-{:?})) over P^{}", self.degrees, self.base_n)
    }

    fn theory_tex(&self) -> String {
        let summands = self
            .degrees
            .iter()
            .map(|degree| format!("\\mathcal{{O}}(-{degree})"))
            .collect::<Vec<_>>()
            .join("\\oplus");
        format!(
            "\\operatorname{{Tot}}({summands}\\to\\mathbb{{P}}^{{{}}})",
            self.base_n
        )
    }

    fn target_dimension(&self) -> usize {
        self.base_n + self.degrees.len()
    }

    fn virasoro_operator_kind(&self) -> VirasoroOperatorKind {
        VirasoroOperatorKind::QrrConjugatedRequired
    }

    fn theory_fingerprint(&self) -> String {
        format!(
            "gw-theory-v1/qrr-required/negative-split/{}/{:?}",
            self.base_n, self.degrees
        )
    }

    fn state_space(&self) -> &StateSpace {
        &self.state_space
    }

    fn curve_class_space(&self) -> &CurveClassSpace {
        &self.curve_space
    }

    fn basis_tex(&self, basis: BasisId) -> Option<String> {
        (basis.0 <= self.base_n).then(|| tex_power_label("H", basis.0))
    }

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        let slope = self.base_n as i64 + 1 - self.degrees.iter().sum::<usize>() as i64;
        slope
            .checked_mul(curve.coordinates()[0])
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(
            if curve.coordinates()[0] >= 0 && (self.base_n > 0 || curve.is_zero()) {
                CurveEffectivity::Effective
            } else {
                CurveEffectivity::Ineffective
            },
        )
    }

    fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers> {
        None
    }

    fn admissible_decompositions(
        &self,
        total: &CurveClass,
    ) -> Result<Vec<CurveClassSplit>, GwError> {
        self.curve_space.validate(total)?;
        if self.effectivity(total)? != CurveEffectivity::Effective {
            return Ok(Vec::new());
        }
        let degree = usize::try_from(total.coordinates()[0]).map_err(|_| {
            GwError::ConventionMismatch("local degree must be nonnegative".to_string())
        })?;
        let count = self.admissible_decomposition_count(total)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {count} local curve-class decompositions"
            ))
        })?;
        out.extend((0..=degree).map(|left| CurveClassSplit {
            left: CurveClass::new(vec![left as i64]),
            right: CurveClass::new(vec![(degree - left) as i64]),
        }));
        Ok(out)
    }

    fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError> {
        self.curve_space.validate(total)?;
        if self.effectivity(total)? != CurveEffectivity::Effective {
            return Ok(0);
        }
        usize::try_from(total.coordinates()[0])
            .ok()
            .and_then(|degree| degree.checked_add(1))
            .ok_or_else(scan_bound_overflow)
    }

    fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError> {
        let count = self.bounded_admissible_class_count(max_total)?;
        if self.base_n == 0 {
            return Ok(vec![CurveClass::zero(1)]);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| {
            GwError::UnsupportedInvariant(format!("cannot allocate {count} local curve classes"))
        })?;
        out.extend((0..=max_total).map(|degree| CurveClass::new(vec![degree as i64])));
        Ok(out)
    }

    fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError> {
        if self.base_n == 0 {
            Ok(1)
        } else {
            ensure_curve_bound_fits_i64(max_total)?;
            max_total.checked_add(1).ok_or_else(scan_bound_overflow)
        }
    }
}
