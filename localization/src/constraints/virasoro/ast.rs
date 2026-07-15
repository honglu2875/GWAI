use std::collections::BTreeMap;

/// A descendant insertion `tau_psi_power(class)`.
///
/// Ordering first by descendant power and then by the canonical theory's basis key
/// gives deterministic coefficient monomials and correlator keys.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Descendant<B> {
    pub psi_power: usize,
    pub class: B,
}

impl<B> Descendant<B> {
    pub fn new(psi_power: usize, class: B) -> Self {
        Self { psi_power, class }
    }
}

/// A canonical monomial in descendant time variables.
///
/// Zero multiplicities are discarded.  Repeated factors are combined, so two
/// monomials constructed in different orders compare equal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeMonomial<B> {
    factors: BTreeMap<Descendant<B>, usize>,
}

impl<B> Default for TimeMonomial<B> {
    fn default() -> Self {
        Self {
            factors: BTreeMap::new(),
        }
    }
}

impl<B: Ord> TimeMonomial<B> {
    pub fn one() -> Self {
        Self::default()
    }

    /// Construct a monomial from individual time variables.
    pub fn from_descendants(descendants: impl IntoIterator<Item = Descendant<B>>) -> Self {
        let mut factors = BTreeMap::new();
        for descendant in descendants {
            let multiplicity = factors.entry(descendant).or_insert(0usize);
            *multiplicity = multiplicity
                .checked_add(1)
                .expect("time-monomial multiplicity overflow");
        }
        Self { factors }
    }

    /// Construct a monomial from `(time variable, multiplicity)` pairs.
    ///
    /// A pair with multiplicity zero has no effect.  Multiplicity overflow is
    /// reported instead of silently changing the coefficient being extracted.
    pub fn try_from_factors(
        factors: impl IntoIterator<Item = (Descendant<B>, usize)>,
    ) -> Result<Self, &'static str> {
        let mut canonical = BTreeMap::new();
        for (descendant, count) in factors {
            if count == 0 {
                continue;
            }
            let multiplicity = canonical.entry(descendant).or_insert(0usize);
            *multiplicity = multiplicity
                .checked_add(count)
                .ok_or("time-monomial multiplicity overflow")?;
        }
        Ok(Self { factors: canonical })
    }
}

impl<B> TimeMonomial<B> {
    pub fn factors(&self) -> impl ExactSizeIterator<Item = (&Descendant<B>, usize)> {
        self.factors
            .iter()
            .map(|(descendant, multiplicity)| (descendant, *multiplicity))
    }

    pub fn total_degree(&self) -> Option<usize> {
        self.factors
            .values()
            .try_fold(0usize, |total, count| total.checked_add(*count))
    }

    pub fn is_one(&self) -> bool {
        self.factors.is_empty()
    }
}

/// A canonical connected correlator requested from an evaluation backend.
///
/// This key assumes the current even-cohomology scope, where insertions can be
/// sorted without Koszul signs.  Super state spaces must not use it until the
/// ordering/sign convention is represented explicitly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CorrelatorKey<D, B> {
    pub genus: usize,
    pub degree: D,
    insertions: Vec<Descendant<B>>,
}

impl<D, B: Ord> CorrelatorKey<D, B> {
    pub fn new(genus: usize, degree: D, mut insertions: Vec<Descendant<B>>) -> Self {
        insertions.sort();
        Self {
            genus,
            degree,
            insertions,
        }
    }
}

impl<D, B> CorrelatorKey<D, B> {
    pub fn insertions(&self) -> &[Descendant<B>] {
        &self.insertions
    }
}

/// The finite genus/curve-class sector of a coefficient constraint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintSector<D> {
    pub genus: usize,
    pub degree: D,
}

impl<D> ConstraintSector<D> {
    pub fn new(genus: usize, degree: D) -> Self {
        Self { genus, degree }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirasoroOperator {
    pub index: i32,
}

impl VirasoroOperator {
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Why a term occurs in the coefficient equation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TermOrigin {
    LinearOperator,
    DilatonShift,
    GenusReduction,
    DegreeSplitting,
    UnstableCorrection,
    Other(String),
}

impl TermOrigin {
    pub fn label(&self) -> &str {
        match self {
            Self::LinearOperator => "linear operator",
            Self::DilatonShift => "dilaton shift",
            Self::GenusReduction => "genus reduction",
            Self::DegreeSplitting => "degree splitting",
            Self::UnstableCorrection => "unstable correction",
            Self::Other(label) => label,
        }
    }
}

/// A single-correlator contribution.  Genus-reduction terms are linear after
/// coefficient extraction and are represented by this same type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearTerm<D, B, C> {
    pub coefficient: C,
    pub correlator: CorrelatorKey<D, B>,
    pub origin: TermOrigin,
}

impl<D, B, C> LinearTerm<D, B, C> {
    pub fn new(coefficient: C, correlator: CorrelatorKey<D, B>, origin: TermOrigin) -> Self {
        Self {
            coefficient,
            correlator,
            origin,
        }
    }
}

/// A product of two connected correlators arising from the nonlinear form of
/// a Virasoro equation for the connected potential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticTerm<D, B, C> {
    pub coefficient: C,
    pub left: CorrelatorKey<D, B>,
    pub right: CorrelatorKey<D, B>,
    pub origin: TermOrigin,
}

impl<D: Ord, B: Ord, C> QuadraticTerm<D, B, C> {
    /// Construct a canonical product in the even-state-space convention.
    pub fn new(
        coefficient: C,
        mut left: CorrelatorKey<D, B>,
        mut right: CorrelatorKey<D, B>,
        origin: TermOrigin,
    ) -> Self {
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        Self {
            coefficient,
            left,
            right,
            origin,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintTerm<D, B, C> {
    Constant { coefficient: C, origin: TermOrigin },
    Linear(LinearTerm<D, B, C>),
    Quadratic(QuadraticTerm<D, B, C>),
}

impl<D, B, C> ConstraintTerm<D, B, C> {
    pub fn origin(&self) -> &TermOrigin {
        match self {
            Self::Constant { origin, .. } => origin,
            Self::Linear(term) => &term.origin,
            Self::Quadratic(term) => &term.origin,
        }
    }
}

/// Text and TeX names for the target theory.  The canonical theory creates
/// this label; the constraint subsystem does not reconstruct target geometry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TheoryLabel {
    pub text: String,
    pub tex: String,
}

impl TheoryLabel {
    pub fn new(text: impl Into<String>, tex: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tex: tex.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PotentialConvention {
    ConnectedDescendant,
    TotalDescendantPartitionFunction,
    /// Connected-correlator expansion of `Z^{-1} L_k Z = 0`.
    LogarithmicPartitionFunctionEquation,
}

impl PotentialConvention {
    pub fn label(self) -> &'static str {
        match self {
            Self::ConnectedDescendant => "connected descendant potential",
            Self::TotalDescendantPartitionFunction => "total descendant partition function",
            Self::LogarithmicPartitionFunctionEquation => {
                "connected-correlator expansion of Z^{-1} L_k Z"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeNormalization {
    /// Coefficients use the usual `1/n!` exponential-generating normalization.
    Exponential,
    /// Markings are ordered and no `1/n!` is built into the potential.
    OrderedMarkings,
}

impl TimeNormalization {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exponential => "exponential generating series (1/n!)",
            Self::OrderedMarkings => "ordered markings (no 1/n!)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DilatonShift {
    None,
    /// The standard shift `q_k^a = t_k^a - delta_{k,1} delta_{a,unit}`.
    StandardUnit,
    Explicit(String),
}

impl DilatonShift {
    pub fn label(&self) -> &str {
        match self {
            Self::None => "none",
            Self::StandardUnit => "q_k^a = t_k^a - delta_(k,1) delta_(a,unit)",
            Self::Explicit(description) => description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CohomologicalGrading {
    Complex,
    Real,
    Explicit(String),
}

impl CohomologicalGrading {
    pub fn label(&self) -> &str {
        match self {
            Self::Complex => "complex cohomological degree",
            Self::Real => "real cohomological degree",
            Self::Explicit(description) => description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnstableConvention {
    Excluded,
    IncludedByStandardExtension,
    Explicit(String),
}

impl UnstableConvention {
    pub fn label(&self) -> &str {
        match self {
            Self::Excluded => "unstable correlators excluded",
            Self::IncludedByStandardExtension => "standard unstable extension included",
            Self::Explicit(description) => description,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateSpaceConvention {
    EvenOnly,
    SuperWithKoszulSigns,
}

impl StateSpaceConvention {
    pub fn label(self) -> &'static str {
        match self {
            Self::EvenOnly => "even state space",
            Self::SuperWithKoszulSigns => "super state space with Koszul signs",
        }
    }
}

/// All choices that can change the coefficient of a displayed constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirasoroConventions {
    pub potential: PotentialConvention,
    pub time_normalization: TimeNormalization,
    pub dilaton_shift: DilatonShift,
    pub grading: CohomologicalGrading,
    pub unstable: UnstableConvention,
    pub state_space: StateSpaceConvention,
    pub novikov_variables: Vec<String>,
    pub equivariant_parameters: Vec<String>,
    pub notes: Vec<String>,
}

impl VirasoroConventions {
    pub fn connected_even() -> Self {
        Self {
            potential: PotentialConvention::ConnectedDescendant,
            time_normalization: TimeNormalization::Exponential,
            dilaton_shift: DilatonShift::StandardUnit,
            grading: CohomologicalGrading::Complex,
            unstable: UnstableConvention::Excluded,
            state_space: StateSpaceConvention::EvenOnly,
            novikov_variables: vec!["q".to_string()],
            equivariant_parameters: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// Bibliographic and derivational provenance for an equation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaSource {
    pub title: String,
    pub citation: Option<String>,
    pub locator: Option<String>,
    pub derivation: Option<String>,
    pub notes: Vec<String>,
}

impl FormulaSource {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            citation: None,
            locator: None,
            derivation: None,
            notes: Vec::new(),
        }
    }
}

/// A finite coefficient equation generated from `L_m`.
///
/// Coefficients are deliberately generic: ordinary and equivariant theories
/// can use different exact coefficient rings without changing this AST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirasoroConstraint<D, B, C> {
    pub theory: TheoryLabel,
    /// Exact canonical-theory identity checked again before evaluation.
    pub theory_fingerprint: String,
    pub operator: VirasoroOperator,
    pub sector: ConstraintSector<D>,
    pub time_coefficient: TimeMonomial<B>,
    pub terms: Vec<ConstraintTerm<D, B, C>>,
    pub conventions: VirasoroConventions,
    pub source: FormulaSource,
}

impl<D, B, C> VirasoroConstraint<D, B, C> {
    pub fn correlator_count(&self) -> usize {
        self.terms
            .iter()
            .map(|term| match term {
                ConstraintTerm::Constant { .. } => 0,
                ConstraintTerm::Linear(_) => 1,
                ConstraintTerm::Quadratic(_) => 2,
            })
            .sum()
    }
}
