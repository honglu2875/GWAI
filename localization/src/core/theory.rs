//! Canonical geometric data for Gromov--Witten theories.
//!
//! Reconstruction engines are algorithms, not independent descriptions of a
//! target.  The types in this module are the shared source of geometric data
//! used by constraint generators and by backend adapters: the homogeneous
//! state space, Poincare pairing, classical cup product, first-Chern action,
//! numerical curve lattice, effective splittings, stabilizing divisors, and
//! characteristic numbers.

use super::algebra::Rational;
use super::error::GwError;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BasisId(pub usize);

impl fmt::Display for BasisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e_{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Parity {
    Even,
    Odd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisElement {
    pub id: BasisId,
    pub label: String,
    /// Hodge `p`-degree used in Getzler's grading operator `mu`.
    pub hodge_p_degree: usize,
    /// Half the real cohomological degree, used for virtual-dimension pruning.
    /// This differs from `hodge_p_degree` away from Hodge--Tate state spaces.
    pub complex_codimension: usize,
    pub parity: Parity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSpaceMatrix {
    /// `entries[output][input]`; matrices act on column vectors.
    entries: Vec<Vec<Rational>>,
}

impl StateSpaceMatrix {
    pub fn try_new(entries: Vec<Vec<Rational>>) -> Result<Self, GwError> {
        let size = entries.len();
        if entries.iter().any(|row| row.len() != size) {
            return Err(GwError::ConventionMismatch(
                "state-space matrix must be square".to_string(),
            ));
        }
        Ok(Self { entries })
    }

    pub fn zero(size: usize) -> Self {
        Self::try_zero(size).expect("state-space matrix allocation failed")
    }

    pub fn try_zero(size: usize) -> Result<Self, GwError> {
        let cells = size.checked_mul(size).ok_or_else(|| {
            GwError::UnsupportedInvariant("state-space matrix size overflow".to_string())
        })?;
        if cells > (isize::MAX as usize) / std::mem::size_of::<Rational>().max(1) {
            return Err(GwError::UnsupportedInvariant(
                "state-space matrix exceeds the addressable allocation size".to_string(),
            ));
        }
        let mut entries = Vec::new();
        entries.try_reserve_exact(size).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {size}-dimensional state-space matrix"
            ))
        })?;
        for _ in 0..size {
            let mut row = Vec::new();
            row.try_reserve_exact(size).map_err(|_| {
                GwError::UnsupportedInvariant(format!(
                    "cannot allocate {size}-dimensional state-space matrix row"
                ))
            })?;
            row.resize(size, Rational::zero());
            entries.push(row);
        }
        Ok(Self { entries })
    }

    pub fn identity(size: usize) -> Self {
        Self::try_identity(size).expect("identity matrix allocation failed")
    }

    pub fn try_identity(size: usize) -> Result<Self, GwError> {
        let mut matrix = Self::try_zero(size)?;
        for index in 0..size {
            matrix.entries[index][index] = Rational::one();
        }
        Ok(matrix)
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn entry(&self, output: usize, input: usize) -> &Rational {
        &self.entries[output][input]
    }

    pub fn set_entry(&mut self, output: usize, input: usize, value: Rational) {
        self.entries[output][input] = value;
    }

    pub fn rows(&self) -> &[Vec<Rational>] {
        &self.entries
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, GwError> {
        if self.size() != rhs.size() {
            return Err(GwError::ConventionMismatch(
                "cannot multiply state-space matrices of different sizes".to_string(),
            ));
        }
        let size = self.size();
        let mut out = Self::try_zero(size)?;
        for output in 0..size {
            for middle in 0..size {
                let left = self.entry(output, middle);
                if left.is_zero() {
                    continue;
                }
                for input in 0..size {
                    let right = rhs.entry(middle, input);
                    if !right.is_zero() {
                        out.entries[output][input] += left.clone() * right.clone();
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn pow(&self, exponent: usize) -> Result<Self, GwError> {
        let mut result = Self::try_identity(self.size())?;
        let mut base = self.clone();
        let mut exponent = exponent;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.multiply(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.multiply(&base)?;
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NondegeneratePairing {
    pub metric: StateSpaceMatrix,
    pub inverse: StateSpaceMatrix,
}

impl NondegeneratePairing {
    pub fn from_metric(metric: StateSpaceMatrix) -> Result<Self, GwError> {
        let inverse = invert_matrix(&metric)?;
        Ok(Self { metric, inverse })
    }

    pub fn try_new(metric: StateSpaceMatrix, inverse: StateSpaceMatrix) -> Result<Self, GwError> {
        if metric.size() != inverse.size() {
            return Err(GwError::ConventionMismatch(
                "pairing and inverse pairing have different sizes".to_string(),
            ));
        }
        let product = metric.multiply(&inverse)?;
        if product != StateSpaceMatrix::try_identity(metric.size())? {
            return Err(GwError::ConventionMismatch(
                "claimed inverse Poincare pairing is not an inverse".to_string(),
            ));
        }
        Ok(Self { metric, inverse })
    }
}

fn invert_matrix(matrix: &StateSpaceMatrix) -> Result<StateSpaceMatrix, GwError> {
    let size = matrix.size();
    let mut left = matrix.rows().to_vec();
    let mut right = StateSpaceMatrix::try_identity(size)?.entries;
    for column in 0..size {
        let pivot = (column..size)
            .find(|row| !left[*row][column].is_zero())
            .ok_or_else(|| {
                GwError::ConventionMismatch("Poincare pairing is degenerate".to_string())
            })?;
        left.swap(column, pivot);
        right.swap(column, pivot);
        let scale = left[column][column].clone();
        for entry in 0..size {
            left[column][entry] = left[column][entry].clone() / scale.clone();
            right[column][entry] = right[column][entry].clone() / scale.clone();
        }
        for row in 0..size {
            if row == column || left[row][column].is_zero() {
                continue;
            }
            let factor = left[row][column].clone();
            for entry in 0..size {
                left[row][entry] =
                    left[row][entry].clone() - factor.clone() * left[column][entry].clone();
                right[row][entry] =
                    right[row][entry].clone() - factor.clone() * right[column][entry].clone();
            }
        }
    }
    StateSpaceMatrix::try_new(right)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSpace {
    pub basis: Vec<BasisElement>,
    pub unit: BasisId,
    /// `None` is meaningful: for example, a local theory must supply its
    /// twisted pairing before the compact Virasoro generator may be used.
    pub pairing: Option<NondegeneratePairing>,
    /// Cup product by `c_1(TX)`, in output-row/input-column convention.
    pub c1_action: Option<StateSpaceMatrix>,
}

impl StateSpace {
    pub fn try_new(
        basis: Vec<BasisElement>,
        unit: BasisId,
        pairing: Option<NondegeneratePairing>,
        c1_action: Option<StateSpaceMatrix>,
    ) -> Result<Self, GwError> {
        if basis.is_empty() {
            return Err(GwError::ConventionMismatch(
                "state space must contain the unit".to_string(),
            ));
        }
        for (index, element) in basis.iter().enumerate() {
            if element.id != BasisId(index) {
                return Err(GwError::ConventionMismatch(
                    "basis ids must be dense and agree with basis order".to_string(),
                ));
            }
        }
        if unit.0 >= basis.len() {
            return Err(GwError::ConventionMismatch(
                "unit basis id is outside the state space".to_string(),
            ));
        }
        if let Some(pairing) = &pairing {
            if pairing.metric.size() != pairing.inverse.size()
                || pairing.metric.multiply(&pairing.inverse)?
                    != StateSpaceMatrix::try_identity(pairing.metric.size())?
            {
                return Err(GwError::ConventionMismatch(
                    "state-space pairing does not contain a valid inverse".to_string(),
                ));
            }
        }
        for size in pairing
            .iter()
            .map(|pairing| pairing.metric.size())
            .chain(c1_action.iter().map(StateSpaceMatrix::size))
        {
            if size != basis.len() {
                return Err(GwError::ConventionMismatch(
                    "geometric matrix size does not match the basis".to_string(),
                ));
            }
        }
        Ok(Self {
            basis,
            unit,
            pairing,
            c1_action,
        })
    }

    pub fn element(&self, id: BasisId) -> Option<&BasisElement> {
        self.basis.get(id.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CurveClass {
    coordinates: Vec<i64>,
}

impl CurveClass {
    pub fn new(coordinates: Vec<i64>) -> Self {
        Self { coordinates }
    }

    pub fn zero(rank: usize) -> Self {
        Self::new(vec![0; rank])
    }

    pub fn coordinates(&self) -> &[i64] {
        &self.coordinates
    }

    pub fn coordinate(&self, index: usize) -> Option<i64> {
        self.coordinates.get(index).copied()
    }

    pub fn rank(&self) -> usize {
        self.coordinates.len()
    }

    pub fn checked_add(&self, rhs: &Self) -> Option<Self> {
        if self.rank() != rhs.rank() {
            return None;
        }
        self.coordinates
            .iter()
            .zip(rhs.coordinates.iter())
            .map(|(left, right)| left.checked_add(*right))
            .collect::<Option<Vec<_>>>()
            .map(Self::new)
    }

    pub fn is_zero(&self) -> bool {
        self.coordinates.iter().all(|coordinate| *coordinate == 0)
    }
}

impl fmt::Display for CurveClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.coordinates.as_slice() {
            [degree] => write!(f, "{degree}"),
            coordinates => {
                write!(f, "(")?;
                for (index, coordinate) in coordinates.iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{coordinate}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveClassSpace {
    pub coordinate_names: Vec<String>,
    pub effective_grading: String,
}

impl CurveClassSpace {
    pub fn rank(&self) -> usize {
        self.coordinate_names.len()
    }

    pub fn validate(&self, curve: &CurveClass) -> Result<(), GwError> {
        if curve.rank() != self.rank() {
            return Err(GwError::ConventionMismatch(format!(
                "curve class has rank {}, but theory expects rank {}",
                curve.rank(),
                self.rank()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CurveEffectivity {
    Effective,
    Ineffective,
    Unknown,
}

/// Which Virasoro operator family a theory is allowed to request.
///
/// This capability is explicit and required: the presence of matrices alone
/// must never cause a twisted or equivariant theory to receive the ordinary
/// compact Getzler operator accidentally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VirasoroOperatorKind {
    StandardCompactGetzler,
    QrrConjugatedRequired,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CurveClassSplit {
    pub left: CurveClass,
    pub right: CurveClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacteristicNumbers {
    pub top_chern_integral: Rational,
    pub c1_c_dim_minus_one_integral: Rational,
    pub convention: String,
    pub source: String,
}

impl CharacteristicNumbers {
    pub fn virasoro_anomaly(&self, dimension: usize) -> Rational {
        (Rational::from(3i128 - dimension as i128) * self.top_chern_integral.clone()
            - Rational::from(2) * self.c1_c_dim_minus_one_integral.clone())
            / Rational::from(48)
    }
}

/// Complete target data needed by universal identities.
pub trait GwTheory: Send + Sync {
    fn theory_id(&self) -> String;
    fn theory_tex(&self) -> String {
        self.theory_id()
    }
    fn target_dimension(&self) -> usize;
    fn virasoro_operator_kind(&self) -> VirasoroOperatorKind;
    /// Stable identity for all geometry and conventions that affect a
    /// generated coefficient equation.  Backends must expose the fingerprint
    /// of the exact canonical theory they evaluate.
    fn theory_fingerprint(&self) -> String;
    fn state_space(&self) -> &StateSpace;
    fn curve_class_space(&self) -> &CurveClassSpace;
    /// TeX spelling of a canonical basis element. Concrete theories override
    /// this when their presentation uses mathematical symbols not represented
    /// faithfully by the plain-text basis label.
    fn basis_tex(&self, basis: BasisId) -> Option<String> {
        self.state_space()
            .element(basis)
            .map(|element| element.label.clone())
    }
    /// TeX spellings of the numerical curve coordinates, in canonical lattice
    /// order. The default preserves the plain-text coordinate names.
    fn curve_coordinate_tex_names(&self) -> Vec<String> {
        self.curve_class_space().coordinate_names.clone()
    }
    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError>;
    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError>;
    /// Classical cup product of two canonical basis elements.
    ///
    /// Universal identities use this for divisor and topological-recursion
    /// corrections.  Returning a sparse canonical-basis expansion keeps the
    /// target's ring relation in the same geometric source of truth as its
    /// pairing and curve lattice.
    fn classical_product(
        &self,
        _left: BasisId,
        _right: BasisId,
    ) -> Result<Vec<(BasisId, Rational)>, GwError> {
        Err(GwError::UnsupportedInvariant(format!(
            "{} does not expose its classical cup product",
            self.theory_id()
        )))
    }
    /// Choose a divisor with strictly positive pairing against `curve` for
    /// unstable divisor-equation stabilization.
    ///
    /// `None` means no such canonical divisor is available (in particular for
    /// the zero class).  The returned integer is the exact divisor pairing.
    fn stabilizing_divisor(&self, curve: &CurveClass) -> Result<Option<(BasisId, i64)>, GwError> {
        self.curve_class_space().validate(curve)?;
        Ok(None)
    }
    fn characteristic_numbers(&self) -> Option<&CharacteristicNumbers>;
    /// Ordered decompositions in the canonical theory's admissible support cone,
    /// including `0+beta` and `beta+0`.  Unknown-effectivity summands must be
    /// evaluated by a backend rather than silently treated as nonzero or zero.
    fn admissible_decompositions(
        &self,
        total: &CurveClass,
    ) -> Result<Vec<CurveClassSplit>, GwError>;
    /// Number of ordered splits returned by
    /// [`Self::admissible_decompositions`].  Universal formula generators use
    /// this to enforce their term budget before materializing the splits.
    fn admissible_decomposition_count(&self, total: &CurveClass) -> Result<usize, GwError>;
    /// Number of classes returned by [`Self::bounded_admissible_classes`].
    /// Scanners must query this before materializing a theory-owned cone so
    /// that their global equation budget remains an allocation guard.
    fn bounded_admissible_class_count(&self, max_total: usize) -> Result<usize, GwError>;
    /// Candidate classes whose theory-defined nonnegative grading is at
    /// most `max_total`.  For a reconstruction cone this can be a conservative
    /// superset of the actual effective cone.  The canonical theory, not a CLI, owns
    /// this enumeration.
    fn bounded_admissible_classes(&self, max_total: usize) -> Result<Vec<CurveClass>, GwError>;

    fn virasoro_anomaly(&self) -> Option<Rational> {
        self.characteristic_numbers()
            .map(|numbers| numbers.virasoro_anomaly(self.target_dimension()))
    }

    fn virtual_dimension(
        &self,
        genus: usize,
        curve: &CurveClass,
        markings: usize,
    ) -> Result<isize, GwError> {
        self.curve_class_space().validate(curve)?;
        let dimension = i128::try_from(self.target_dimension()).map_err(|_| {
            GwError::AlgebraFailure("target dimension does not fit in i128".to_string())
        })?;
        let genus = i128::try_from(genus)
            .map_err(|_| GwError::AlgebraFailure("genus does not fit in i128".to_string()))?;
        let markings = i128::try_from(markings).map_err(|_| {
            GwError::AlgebraFailure("marking count does not fit in i128".to_string())
        })?;
        let c1_pairing = i128::from(self.c1_pairing(curve)?);
        let value = (1i128 - genus)
            .checked_mul(dimension - 3)
            .and_then(|term| term.checked_add(c1_pairing))
            .and_then(|term| term.checked_add(markings))
            .ok_or_else(|| GwError::AlgebraFailure("virtual dimension overflow".to_string()))?;
        isize::try_from(value).map_err(|_| {
            GwError::AlgebraFailure("virtual dimension does not fit in isize".to_string())
        })
    }
}

// Shared checked helpers used by more than one concrete target theory.
pub(crate) fn canonicalize_line_summand_payloads<T>(
    degrees: Vec<usize>,
    payloads: Vec<T>,
    presentation: &str,
) -> Result<(Vec<usize>, Vec<T>), GwError> {
    if degrees.len() != payloads.len() {
        return Err(GwError::ConventionMismatch(format!(
            "{presentation} summand payloads must have length {}",
            degrees.len()
        )));
    }
    let mut summands = degrees.into_iter().zip(payloads).collect::<Vec<_>>();
    // Stable sorting keeps caller-supplied payloads deterministic when
    // several isomorphic summands have the same twist.
    summands.sort_by_key(|(twist, _)| *twist);
    Ok(summands.into_iter().unzip())
}

pub(crate) fn scan_bound_overflow() -> GwError {
    GwError::UnsupportedInvariant("curve-class scan bound is too large".to_string())
}

pub(crate) fn ensure_curve_bound_fits_i64(max_total: usize) -> Result<(), GwError> {
    i64::try_from(max_total)
        .map(|_| ())
        .map_err(|_| scan_bound_overflow())
}

pub(crate) fn two_ray_class_count(max_total: usize) -> Result<usize, GwError> {
    let first = max_total.checked_add(1).ok_or_else(scan_bound_overflow)?;
    let second = max_total.checked_add(2).ok_or_else(scan_bound_overflow)?;
    let (first, second) = if first % 2 == 0 {
        (first / 2, second)
    } else {
        (first, second / 2)
    };
    first.checked_mul(second).ok_or_else(scan_bound_overflow)
}

pub(crate) fn power_label(symbol: &str, power: usize) -> String {
    match power {
        0 => "1".to_string(),
        1 => symbol.to_string(),
        _ => format!("{symbol}^{power}"),
    }
}

pub(crate) fn tex_power_label(symbol: &str, power: usize) -> String {
    match power {
        0 => "1".to_string(),
        1 => symbol.to_string(),
        _ => format!("{symbol}^{{{power}}}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn characteristic_anomaly_uses_wide_dimension_arithmetic() {
        let numbers = CharacteristicNumbers {
            top_chern_integral: Rational::one(),
            c1_c_dim_minus_one_integral: Rational::zero(),
            convention: "test".to_string(),
            source: "test".to_string(),
        };
        assert_eq!(
            numbers.virasoro_anomaly(usize::MAX),
            Rational::new(3i128 - usize::MAX as i128, 48)
        );
        assert_eq!(
            StateSpaceMatrix::try_identity(2).unwrap(),
            StateSpaceMatrix::identity(2)
        );
    }
}
