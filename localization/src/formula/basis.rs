//! Named basis elements used in the explainable Givental graph formula.
//!
//! These names deliberately match the mathematical objects, not the optimized
//! Rust storage layout.  The renderer in `skeleton` uses this glossary as the
//! source of truth for human-readable output.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisKind {
    DescendantS,
    PsiInverse,
    RInverse,
    Translation,
    Delta,
    InverseDelta,
    RelativeSqrtDelta,
    EtaInverse,
    PsiIntegral,
}

const BASIS: &[BasisKind] = &[
    BasisKind::DescendantS,
    BasisKind::PsiInverse,
    BasisKind::RInverse,
    BasisKind::Translation,
    BasisKind::Delta,
    BasisKind::InverseDelta,
    BasisKind::RelativeSqrtDelta,
    BasisKind::EtaInverse,
    BasisKind::PsiIntegral,
];

pub fn all_basis_kinds() -> &'static [BasisKind] {
    BASIS
}

impl BasisKind {
    pub fn symbol(self) -> &'static str {
        match self {
            BasisKind::DescendantS => "S_s[a,b]",
            BasisKind::PsiInverse => "PsiInv[i,a]",
            BasisKind::RInverse => "RInv_r[i,j]",
            BasisKind::Translation => "T_i^p",
            BasisKind::Delta => "Delta_i",
            BasisKind::InverseDelta => "DeltaInv_i",
            BasisKind::RelativeSqrtDelta => "RelSqrtDelta_i",
            BasisKind::EtaInverse => "EtaInv_i",
            BasisKind::PsiIntegral => "PsiInt(h; p_1,...,p_N)",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            BasisKind::DescendantS => "descendant S",
            BasisKind::PsiInverse => "Psi inverse",
            BasisKind::RInverse => "inverse R",
            BasisKind::Translation => "translation",
            BasisKind::Delta => "Delta",
            BasisKind::InverseDelta => "inverse Delta",
            BasisKind::RelativeSqrtDelta => "relative square-root Delta",
            BasisKind::EtaInverse => "canonical metric inverse",
            BasisKind::PsiIntegral => "point-theory psi integral",
        }
    }

    pub fn definition(self) -> &'static str {
        match self {
            BasisKind::DescendantS => {
                "Coefficient of z^{-s} in the descendant-to-ancestor S-calibration. It first acts on a flat-basis insertion before the graph R-action."
            }
            BasisKind::PsiInverse => {
                "Transition matrix from the flat basis to canonical colors. It is applied after S so graph legs live in the idempotent color basis."
            }
            BasisKind::RInverse => {
                "Coefficient of z^r in R(z)^{-1}. External ancestor legs and internal edge propagators are built from these coefficients."
            }
            BasisKind::Translation => {
                "Coefficient of psi^p in T(psi)=psi(1-R^{-1}(psi))1. In the current graph convention T^0 and T^1 vanish, so p starts at 2."
            }
            BasisKind::Delta => {
                "Canonical metric norm factor used by positive-genus TFT vertices. In the stored relative frame, genus h>0 contributes Delta_i^{h-1}."
            }
            BasisKind::InverseDelta => {
                "Genus-zero canonical TFT factor. In the stored relative frame, h=0 vertices use DeltaInv_i."
            }
            BasisKind::RelativeSqrtDelta => {
                "Relative normalization factor attached once per incident half-edge or marking at a vertex."
            }
            BasisKind::EtaInverse => {
                "Diagonal inverse canonical metric entry. It appears in the edge propagator before the regular quotient by psi_left+psi_right."
            }
            BasisKind::PsiIntegral => {
                "Pure Witten-Kontsevich intersection number on Mbar_{h,N}. Vertex psi powers and translation psi powers are integrated here."
            }
        }
    }
}

pub fn basis_glossary() -> String {
    let mut out = String::new();
    out.push_str("Raw basis glossary\n");
    out.push_str("------------------\n");
    out.push_str("Indices: a,b are flat-basis indices; i,j are canonical colors; r,s,p are z/psi orders.\n\n");
    for basis in all_basis_kinds() {
        out.push_str("- ");
        out.push_str(basis.symbol());
        out.push_str(" (");
        out.push_str(basis.short_name());
        out.push_str("): ");
        out.push_str(basis.definition());
        out.push('\n');
    }
    out
}
