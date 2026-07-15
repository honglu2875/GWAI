//! Canonical geometric data for Gromov--Witten theories.
//!
//! Reconstruction engines are algorithms, not independent descriptions of a
//! target.  The types in this module are the shared source of geometric data
//! used by constraint generators and by backend adapters: the homogeneous
//! state space, Poincare pairing, first-Chern action, numerical curve lattice,
//! effective splittings, and characteristic numbers.

use crate::algebra::Rational;
use crate::error::GwError;
use std::collections::BTreeMap;
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
    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError>;
    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError>;
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

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        curve.coordinates[0]
            .checked_mul((self.n + 1) as i64)
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(
            if curve.coordinates[0] < 0 || (self.n == 0 && curve.coordinates[0] != 0) {
                CurveEffectivity::Ineffective
            } else {
                CurveEffectivity::Effective
            },
        )
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
            usize::try_from(curve.coordinates[0]).ok()?,
            usize::try_from(curve.coordinates[1]).ok()?,
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

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        (self.n as i64 + 1)
            .checked_mul(curve.coordinates[0])
            .and_then(|left| {
                (self.m as i64 + 1)
                    .checked_mul(curve.coordinates[1])
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveBundleTheory {
    n: usize,
    twists: Vec<usize>,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
    characteristic_numbers: CharacteristicNumbers,
}

impl ProjectiveBundleTheory {
    pub fn new(n: usize, mut twists: Vec<usize>) -> Result<Self, GwError> {
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
        // Direct-sum order is not geometric data.  Canonicalize it so
        // isomorphic presentations have identical theory fingerprints and
        // backend compatibility checks.
        twists.sort_unstable();
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
            usize::try_from(curve.coordinates[0]).ok()?,
            curve.coordinates[1],
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

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        let c1_h = (self.n + 1 + self.twists.iter().sum::<usize>()) as i64;
        c1_h.checked_mul(curve.coordinates[0])
            .and_then(|value| {
                (self.rank() as i64)
                    .checked_mul(curve.coordinates[1])
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitTotalSpaceTheory {
    base_n: usize,
    degrees: Vec<usize>,
    state_space: StateSpace,
    curve_space: CurveClassSpace,
}

impl NegativeSplitTotalSpaceTheory {
    pub fn new(base_n: usize, mut degrees: Vec<usize>) -> Result<Self, GwError> {
        if degrees.is_empty() || degrees.contains(&0) {
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
        let degree_sum = degrees.iter().try_fold(0usize, |sum, degree| {
            sum.checked_add(*degree).ok_or_else(|| {
                GwError::UnsupportedInvariant("local twist degree sum overflow".to_string())
            })
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
        // Direct-sum order is not geometric data.  Keep theory identity,
        // formula rendering, and backend compatibility canonical under a
        // permutation of the line-bundle summands.
        degrees.sort_unstable();
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

    fn c1_pairing(&self, curve: &CurveClass) -> Result<i64, GwError> {
        self.curve_space.validate(curve)?;
        let slope = self.base_n as i64 + 1 - self.degrees.iter().sum::<usize>() as i64;
        slope
            .checked_mul(curve.coordinates[0])
            .ok_or_else(|| GwError::AlgebraFailure("c1 pairing overflow".to_string()))
    }

    fn effectivity(&self, curve: &CurveClass) -> Result<CurveEffectivity, GwError> {
        self.curve_space.validate(curve)?;
        Ok(
            if curve.coordinates[0] >= 0 && (self.base_n > 0 || curve.is_zero()) {
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
        let degree = usize::try_from(total.coordinates[0]).map_err(|_| {
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
        usize::try_from(total.coordinates[0])
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

fn scan_bound_overflow() -> GwError {
    GwError::UnsupportedInvariant("curve-class scan bound is too large".to_string())
}

fn ensure_curve_bound_fits_i64(max_total: usize) -> Result<(), GwError> {
    i64::try_from(max_total)
        .map(|_| ())
        .map_err(|_| scan_bound_overflow())
}

fn two_ray_class_count(max_total: usize) -> Result<usize, GwError> {
    let first = max_total.checked_add(1).ok_or_else(scan_bound_overflow)?;
    let second = max_total.checked_add(2).ok_or_else(scan_bound_overflow)?;
    let (first, second) = if first % 2 == 0 {
        (first / 2, second)
    } else {
        (first, second / 2)
    };
    first.checked_mul(second).ok_or_else(scan_bound_overflow)
}

fn power_label(symbol: &str, power: usize) -> String {
    match power {
        0 => "1".to_string(),
        1 => symbol.to_string(),
        _ => format!("{symbol}^{power}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projective_space_data_and_anomaly_are_exact() {
        let p2 = ProjectiveSpaceTheory::new(2);
        assert_eq!(p2.target_dimension(), 2);
        assert_eq!(
            p2.state_space()
                .pairing
                .as_ref()
                .unwrap()
                .metric
                .entry(0, 2),
            &Rational::one()
        );
        assert_eq!(
            p2.state_space().c1_action.as_ref().unwrap().entry(1, 0),
            &Rational::from(3)
        );
        assert_eq!(
            p2.characteristic_numbers().unwrap().virasoro_anomaly(2),
            Rational::new(-5, 16)
        );
    }

    #[test]
    fn high_dimensional_projective_pairing_has_its_declared_analytic_inverse() {
        let projective = ProjectiveSpaceTheory::new(100);
        let pairing = projective.state_space().pairing.as_ref().unwrap();

        assert_eq!(pairing.inverse, pairing.metric);
        assert_eq!(
            pairing.metric.multiply(&pairing.inverse).unwrap(),
            StateSpaceMatrix::identity(101)
        );
    }

    #[test]
    fn extreme_target_dimensions_are_rejected_fallibly() {
        assert!(matches!(
            ProjectiveSpaceTheory::try_new(usize::MAX),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            ProductProjectiveTheory::new(usize::MAX, 1),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            NegativeSplitTotalSpaceTheory::new(usize::MAX, vec![1]),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            ProjectiveBundleTheory::new(usize::MAX, vec![0, 1]),
            Err(GwError::UnsupportedInvariant(_))
        ));

        let p1 = ProjectiveSpaceTheory::new(1);
        let product = ProductProjectiveTheory::new(1, 1).unwrap();
        let bundle = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        assert!(matches!(
            p1.try_curve(usize::MAX),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            product.try_curve(usize::MAX, 0),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            bundle.try_curve(usize::MAX, 0),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    #[test]
    fn permutation_equivalent_bundle_presentations_share_one_fingerprint() {
        let left = ProjectiveBundleTheory::new(1, vec![0, 3, 1]).unwrap();
        let right = ProjectiveBundleTheory::new(1, vec![1, 0, 3]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.theory_fingerprint(), right.theory_fingerprint());
        assert_eq!(left.theory_id(), "P(O + O(1) + O(3)) over P^1");
    }

    #[test]
    fn permutation_equivalent_local_splits_share_one_fingerprint() {
        let left = NegativeSplitTotalSpaceTheory::new(2, vec![3, 1, 2]).unwrap();
        let right = NegativeSplitTotalSpaceTheory::new(2, vec![2, 3, 1]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.degrees(), &[1, 2, 3]);
        assert_eq!(left.theory_fingerprint(), right.theory_fingerprint());
        assert_eq!(left.theory_id(), "Tot(O(-[1, 2, 3])) over P^2");
    }

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

    #[test]
    fn projective_point_has_only_the_zero_curve_class() {
        let point = ProjectiveSpaceTheory::new(0);
        assert_eq!(
            point.bounded_admissible_classes(4).unwrap(),
            vec![point.curve(0)]
        );
        assert_eq!(
            point.effectivity(&point.curve(1)).unwrap(),
            CurveEffectivity::Ineffective
        );
        assert_eq!(
            point.characteristic_numbers().unwrap().virasoro_anomaly(0),
            Rational::new(1, 16)
        );
    }

    #[test]
    fn product_splits_are_geometric_bidegree_splits() {
        let theory = ProductProjectiveTheory::new(1, 2).unwrap();
        let splits = theory
            .admissible_decompositions(&theory.curve(1, 2))
            .unwrap();
        assert_eq!(splits.len(), 6);
        assert_eq!(splits[0].left, theory.curve(0, 0));
        assert_eq!(splits[5].right, theory.curve(0, 0));
    }

    #[test]
    fn ineffective_classes_have_zero_decomposition_count_and_no_splits() {
        let cases: Vec<(Box<dyn GwTheory>, CurveClass)> = vec![
            (
                Box::new(ProjectiveSpaceTheory::new(2)),
                CurveClass::new(vec![-1]),
            ),
            (
                Box::new(ProductProjectiveTheory::new(1, 1).unwrap()),
                CurveClass::new(vec![-1, 0]),
            ),
            (
                Box::new(ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap()),
                CurveClass::new(vec![0, -1]),
            ),
            (
                Box::new(NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap()),
                CurveClass::new(vec![-1]),
            ),
        ];
        for (theory, curve) in cases {
            assert_eq!(theory.admissible_decomposition_count(&curve).unwrap(), 0);
            assert!(theory.admissible_decompositions(&curve).unwrap().is_empty());
        }
    }

    #[test]
    fn product_decomposition_count_overflow_is_fallible() {
        let product = ProductProjectiveTheory::new(1, 1).unwrap();
        let huge = CurveClass::new(vec![i64::MAX, i64::MAX]);
        assert!(matches!(
            product.admissible_decomposition_count(&huge),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            product.admissible_decompositions(&huge),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    #[test]
    fn bounded_class_counts_match_canonical_theory_enumerations() {
        let theories: Vec<Box<dyn GwTheory>> = vec![
            Box::new(ProjectiveSpaceTheory::new(0)),
            Box::new(ProjectiveSpaceTheory::new(3)),
            Box::new(ProductProjectiveTheory::new(1, 2).unwrap()),
            Box::new(ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap()),
            Box::new(NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap()),
        ];
        for theory in theories {
            for bound in 0..=20 {
                let classes = theory.bounded_admissible_classes(bound).unwrap();
                assert_eq!(
                    theory.bounded_admissible_class_count(bound).unwrap(),
                    classes.len(),
                    "{} at bound {bound}",
                    theory.theory_id()
                );
                assert!(classes
                    .iter()
                    .all(|curve| curve.rank() == theory.curve_class_space().rank()));
                assert_eq!(
                    classes
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                    classes.len(),
                    "{} returned duplicate classes at bound {bound}",
                    theory.theory_id()
                );
            }
        }
    }

    #[test]
    fn point_class_count_accepts_an_irrelevant_large_bound() {
        let point = ProjectiveSpaceTheory::new(0);
        assert_eq!(point.bounded_admissible_class_count(usize::MAX).unwrap(), 1);
        assert_eq!(
            point.bounded_admissible_classes(usize::MAX).unwrap(),
            vec![point.curve(0)]
        );
    }

    #[test]
    fn bundle_scan_rejects_shifted_degree_arithmetic_overflow() {
        let twist = (i64::MAX as usize) / 2 + 1;
        let bundle = ProjectiveBundleTheory::new(1, vec![0, twist]).unwrap();
        let error = bundle.bounded_admissible_class_count(2).unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
        let error = bundle.bounded_admissible_classes(2).unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
    }

    #[test]
    fn trivial_bundle_recovers_product_pairing_and_characteristics() {
        let bundle = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let product = ProductProjectiveTheory::new(1, 1).unwrap();
        assert_eq!(bundle.state_space().pairing, product.state_space().pairing);
        assert_eq!(
            bundle.characteristic_numbers.top_chern_integral,
            product.characteristic_numbers.top_chern_integral
        );
        assert_eq!(
            bundle.characteristic_numbers.c1_c_dim_minus_one_integral,
            product.characteristic_numbers.c1_c_dim_minus_one_integral
        );
    }

    #[test]
    fn bundle_splits_use_shifted_effective_coordinates() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap();
        let exceptional = theory.curve(1, -2);
        assert_eq!(
            theory.effectivity(&exceptional).unwrap(),
            CurveEffectivity::Unknown
        );
        let classes = theory.bounded_admissible_classes(1).unwrap();
        assert!(classes.contains(&exceptional));
        let splits = theory.admissible_decompositions(&exceptional).unwrap();
        assert_eq!(splits.len(), 2);
        for split in splits {
            assert_eq!(
                split.left.checked_add(&split.right),
                Some(exceptional.clone())
            );
        }
    }

    #[test]
    fn bundle_characteristic_numbers_do_not_depend_on_splitting_twists() {
        for twist in [0, 1, 2, 5] {
            let hirzebruch = ProjectiveBundleTheory::new(1, vec![0, twist]).unwrap();
            let numbers = hirzebruch.characteristic_numbers().unwrap();
            assert_eq!(numbers.top_chern_integral, Rational::from(4));
            assert_eq!(numbers.c1_c_dim_minus_one_integral, Rational::from(8));

            let threefold = ProjectiveBundleTheory::new(1, vec![0, 0, twist]).unwrap();
            let numbers = threefold.characteristic_numbers().unwrap();
            assert_eq!(numbers.top_chern_integral, Rational::from(6));
            assert_eq!(numbers.c1_c_dim_minus_one_integral, Rational::from(24));
        }
    }

    #[test]
    fn local_theory_does_not_fabricate_compact_pairing_or_anomaly() {
        let local = NegativeSplitTotalSpaceTheory::new(2, vec![3]).unwrap();
        assert!(local.state_space().pairing.is_none());
        assert!(local.state_space().c1_action.is_none());
        assert!(local.characteristic_numbers().is_none());
    }
}
