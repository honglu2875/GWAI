//! Named atoms used in the explainable Givental graph formula.
//!
//! These names deliberately match the mathematical objects, not the optimized
//! Rust storage layout.  The renderer in `skeleton` uses this glossary as the
//! source of truth for human-readable output.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomKind {
    DescendantS,
    Psi,
    PsiInverse,
    RInverse,
    Edge,
    Translation,
    Delta,
    InverseDelta,
    RelativeSqrtDelta,
    EtaInverse,
    Unit,
    PsiIntegral,
}

const ATOMS: &[AtomKind] = &[
    AtomKind::DescendantS,
    AtomKind::Psi,
    AtomKind::PsiInverse,
    AtomKind::RInverse,
    AtomKind::Edge,
    AtomKind::Translation,
    AtomKind::Delta,
    AtomKind::InverseDelta,
    AtomKind::RelativeSqrtDelta,
    AtomKind::EtaInverse,
    AtomKind::Unit,
    AtomKind::PsiIntegral,
];

pub fn all_atom_kinds() -> &'static [AtomKind] {
    ATOMS
}

impl AtomKind {
    pub fn symbol(self) -> &'static str {
        match self {
            AtomKind::DescendantS => "S_s[a,b]",
            AtomKind::Psi => "Psi[a,i]",
            AtomKind::PsiInverse => "PsiInv[i,a]",
            AtomKind::RInverse => "RInv_r[i,j]",
            AtomKind::Edge => "Edge_{i,j}^{a,b}",
            AtomKind::Translation => "T_i^p",
            AtomKind::Delta => "Delta_i",
            AtomKind::InverseDelta => "DeltaInv_i",
            AtomKind::RelativeSqrtDelta => "RelSqrtDelta_i",
            AtomKind::EtaInverse => "EtaInv_i",
            AtomKind::Unit => "Unit_i",
            AtomKind::PsiIntegral => "PsiInt(h; p_1,...,p_N)",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            AtomKind::DescendantS => "descendant S",
            AtomKind::Psi => "Psi",
            AtomKind::PsiInverse => "Psi inverse",
            AtomKind::RInverse => "inverse R",
            AtomKind::Edge => "edge propagator",
            AtomKind::Translation => "translation",
            AtomKind::Delta => "Delta",
            AtomKind::InverseDelta => "inverse Delta",
            AtomKind::RelativeSqrtDelta => "relative square-root Delta",
            AtomKind::EtaInverse => "canonical metric inverse",
            AtomKind::Unit => "unit",
            AtomKind::PsiIntegral => "point-theory psi integral",
        }
    }

    pub fn definition(self) -> &'static str {
        match self {
            AtomKind::DescendantS => {
                "Coefficient of z^{-s} in the descendant-to-ancestor S-calibration. It first acts on a flat-basis insertion before the graph R-action."
            }
            AtomKind::Psi => {
                "Transition matrix from canonical idempotents to the flat basis. The column i is the flat-basis expression of the i-th canonical idempotent."
            }
            AtomKind::PsiInverse => {
                "Transition matrix from the flat basis to canonical colors. It is applied after S so graph legs live in the idempotent color basis."
            }
            AtomKind::RInverse => {
                "Coefficient of z^r in R(z)^{-1}. External ancestor legs and internal edge propagators are built from these coefficients."
            }
            AtomKind::Edge => {
                "Coefficient of psi_left^a psi_right^b in the regular part of (eta^{-1} - R^{-1}(psi_left) eta^{-1} R^{-1}(-psi_right)^T)/(psi_left + psi_right)."
            }
            AtomKind::Translation => {
                "Coefficient of psi^p in T(psi)=psi(1-R^{-1}(psi))1. In the current convention T^0 and T^1 vanish, so p starts at 2."
            }
            AtomKind::Delta => {
                "Canonical metric norm factor used by positive-genus TFT vertices. In the stored relative frame, genus h>0 contributes Delta_i^{h-1}."
            }
            AtomKind::InverseDelta => {
                "Genus-zero canonical TFT factor. In the stored relative frame, h=0 vertices use DeltaInv_i."
            }
            AtomKind::RelativeSqrtDelta => {
                "Relative normalization factor attached once per incident half-edge or marking at a vertex."
            }
            AtomKind::EtaInverse => {
                "Diagonal inverse canonical metric entry. It appears in the edge propagator before the regular quotient by psi_left+psi_right."
            }
            AtomKind::Unit => {
                "The CohFT unit expressed in canonical colors. It is the input vector for the translation T(psi)."
            }
            AtomKind::PsiIntegral => {
                "Pure Witten-Kontsevich intersection number on Mbar_{h,N}. Vertex psi powers and translation psi powers are integrated here."
            }
        }
    }
}

pub fn atom_glossary() -> String {
    let mut out = String::new();
    out.push_str("Atom glossary\n");
    out.push_str("-------------\n");
    out.push_str("Indices: a,b are flat-basis indices; i,j are canonical colors; r,s,p are z/psi orders.\n\n");
    for atom in all_atom_kinds() {
        out.push_str("- ");
        out.push_str(atom.symbol());
        out.push_str(" (");
        out.push_str(atom.short_name());
        out.push_str("): ");
        out.push_str(atom.definition());
        out.push('\n');
    }
    out
}
