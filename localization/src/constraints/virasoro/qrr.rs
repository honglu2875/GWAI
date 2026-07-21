//! Quantum Riemann--Roch conjugation data.
//!
//! This module represents the operator before any backend is queried.  The
//! finite term table is a reviewable truncation of Coates--Givental's exact
//! formula; [`QrrConjugationFormula::render_text`] and
//! [`QrrConjugationFormula::render_tex`] also display the untruncated formula
//! and all coordinate conventions.

use crate::core::algebra::Rational;
use crate::core::error::GwError;

pub const MAX_QRR_CHERN_INDEX: usize = 64;
pub const MAX_QRR_POSITIVE_Z_POWER: usize = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QrrFactor {
    Negative,
    Positive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrrHamiltonianTerm {
    pub factor: QrrFactor,
    pub z_power: i32,
    pub chern_character_index: usize,
    pub characteristic_parameter_index: usize,
    pub rational_multiplier: Rational,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrrConjugationFormula {
    pub base_theory_text: String,
    pub base_theory_tex: String,
    pub bundle_text: String,
    pub bundle_tex: String,
    pub characteristic_class_text: String,
    pub max_chern_character_index: usize,
    pub max_positive_z_power: usize,
    terms: Vec<QrrHamiltonianTerm>,
}

impl QrrConjugationFormula {
    /// Build a finite, exact table of the Hamiltonians in Theorem 1 of
    /// Coates--Givental.
    ///
    /// The table contains every `ch_l(E)` through `max_chern_character_index`
    /// and every positive odd `z` power through `max_positive_z_power`.  The
    /// represented formula itself remains the formal, untruncated QRR
    /// conjugation displayed by the renderers.
    pub fn new(
        base_theory_text: impl Into<String>,
        base_theory_tex: impl Into<String>,
        bundle_text: impl Into<String>,
        bundle_tex: impl Into<String>,
        characteristic_class_text: impl Into<String>,
        max_chern_character_index: usize,
        max_positive_z_power: usize,
    ) -> Result<Self, GwError> {
        if max_chern_character_index > MAX_QRR_CHERN_INDEX {
            return Err(GwError::ResourceLimit {
                operation: "QRR Chern-character table".to_string(),
                requested: max_chern_character_index,
                limit: MAX_QRR_CHERN_INDEX,
            });
        }
        if max_positive_z_power > MAX_QRR_POSITIVE_Z_POWER {
            return Err(GwError::ResourceLimit {
                operation: "QRR positive z-power table".to_string(),
                requested: max_positive_z_power,
                limit: MAX_QRR_POSITIVE_Z_POWER,
            });
        }

        let mut terms = Vec::new();
        terms
            .try_reserve_exact(
                max_chern_character_index
                    .checked_add(1)
                    .and_then(|width| width.checked_mul(max_positive_z_power.saturating_add(3) / 2))
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant("QRR term-table size overflow".to_string())
                    })?,
            )
            .map_err(|_| {
                GwError::UnsupportedInvariant("cannot allocate QRR term table".to_string())
            })?;

        for chern_character_index in 1..=max_chern_character_index {
            terms.push(QrrHamiltonianTerm {
                factor: QrrFactor::Negative,
                z_power: -1,
                chern_character_index,
                characteristic_parameter_index: chern_character_index - 1,
                rational_multiplier: Rational::one(),
            });
        }
        for z_power in (1..=max_positive_z_power).step_by(2) {
            let bernoulli_index = z_power + 1;
            let multiplier = bernoulli_over_factorial(bernoulli_index)?;
            for chern_character_index in 0..=max_chern_character_index {
                terms.push(QrrHamiltonianTerm {
                    factor: QrrFactor::Positive,
                    z_power: i32::try_from(z_power).map_err(|_| {
                        GwError::UnsupportedInvariant("QRR z power does not fit i32".to_string())
                    })?,
                    chern_character_index,
                    characteristic_parameter_index: z_power
                        .checked_add(chern_character_index)
                        .ok_or_else(|| {
                            GwError::UnsupportedInvariant(
                                "QRR characteristic-parameter index overflow".to_string(),
                            )
                        })?,
                    rational_multiplier: multiplier.clone(),
                });
            }
        }

        Ok(Self {
            base_theory_text: base_theory_text.into(),
            base_theory_tex: base_theory_tex.into(),
            bundle_text: bundle_text.into(),
            bundle_tex: bundle_tex.into(),
            characteristic_class_text: characteristic_class_text.into(),
            max_chern_character_index,
            max_positive_z_power,
            terms,
        })
    }

    pub fn terms(&self) -> &[QrrHamiltonianTerm] {
        &self.terms
    }

    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str("QRR-conjugated Virasoro operator\n");
        out.push_str("=================================\n");
        out.push_str(&format!("Base theory: {}\n", self.base_theory_text));
        out.push_str(&format!("Bundle: {}\n", self.bundle_text));
        out.push_str(&format!(
            "Characteristic class: {} = exp(sum_(k>=0) s_k ch_k)\n\n",
            self.characteristic_class_text
        ));
        out.push_str("Twisted pairing: (a,b)_tw = integral_X c(E) a b\n");
        out.push_str(
            "Dilaton coordinate under the CG Fock-space isometry: q_CG(z) = sqrt(c(E)) (t(z)-z)\n",
        );
        out.push_str("A_- = sum_(l>0) s_(l-1) ch_l(E) z^(-1)\n");
        out.push_str("A_+ = sum_(m>0,l>=0) s_(2m-1+l) B_(2m)/(2m)! ch_l(E) z^(2m-1)\n");
        out.push_str("Hat convention: hat_CG(A) quantizes Omega(Af,f)/2 as in Coates--Givental.\n");
        out.push_str("The crate's Getzler normal-order map has hat_G(A) = -hat_CG(A).\n");
        out.push_str("U_(c,E) = exp(hat_CG(A_+)) exp(hat_CG(A_-))\n");
        out.push_str("          = exp(-hat_G(A_+)) exp(-hat_G(A_-))\n");
        out.push_str("L_m^(c,E) = U_(c,E) L_m^X U_(c,E)^(-1)\n\n");
        out.push_str("The QRR genus-one determinant scalar is omitted here because every central scalar cancels in operator conjugation.\n\n");
        out.push_str(&format!(
            "Finite exact term table: ch_l through l={}, positive z powers through {}.\n",
            self.max_chern_character_index, self.max_positive_z_power
        ));
        out.push_str("The table bounds an expansion, not the formal identity above.\n");
        out.push_str("Source: Coates--Givental, Quantum Riemann--Roch, Lefschetz and Serre, Theorem 1 (7).\n");
        out.push_str("https://arxiv.org/pdf/math/0110142\n");
        out
    }

    pub fn render_tex(&self) -> String {
        format!(
            "\\subsection*{{QRR-conjugated Virasoro operator}}\n\\textbf{{Base:}} \\({}\\), \\textbf{{bundle:}} \\({}\\).\\\\\n\\[ \\mathbf c(V)=\\exp\\!\\left(\\sum_{{k\\ge0}}s_k\\operatorname{{ch}}_k(V)\\right),\\qquad (a,b)_{{\\mathbf c(E)}}=\\int_X\\mathbf c(E)ab, \\]\n\\[ q_{{\\rm CG}}(z)=\\sqrt{{\\mathbf c(E)}}(t(z)-z). \\]\n\\[ A_- = \\sum_{{l>0}}s_{{l-1}}\\operatorname{{ch}}_l(E)z^{{-1}}, \\]\n\\[ A_+ = \\sum_{{m>0,\\,l\\ge0}}s_{{2m-1+l}}\\frac{{B_{{2m}}}}{{(2m)!}}\\operatorname{{ch}}_l(E)z^{{2m-1}}. \\]\n\\[ U_{{\\mathbf c,E}}=e^{{\\widehat A_+^{{\\rm CG}}}}e^{{\\widehat A_-^{{\\rm CG}}}}=e^{{-\\widehat A_+^{{G}}}}e^{{-\\widehat A_-^{{G}}}},\\qquad \\mathbb L_m^{{\\mathbf c,E}}=U_{{\\mathbf c,E}}\\mathbb L_m^XU_{{\\mathbf c,E}}^{{-1}}. \\]\n\\noindent Here $\\widehat A^{{\\rm CG}}$ quantizes $\\Omega(Af,f)/2$ as in Coates--Givental, whereas the crate's Getzler normal ordering is $\\widehat A^G=-\\widehat A^{{\\rm CG}}$. The QRR genus-one determinant scalar is omitted because it cancels from operator conjugation. The finite term table retains $\\operatorname{{ch}}_l$ through $l={}$ and positive powers through $z^{{{}}}$.\\\\\n\\noindent Source: Coates--Givental, \\emph{{Quantum Riemann--Roch, Lefschetz and Serre}}, Theorem~1, equation~(7), arXiv:math/0110142.\n",
            self.base_theory_tex,
            self.bundle_tex,
            self.max_chern_character_index,
            self.max_positive_z_power
        )
    }
}

fn bernoulli_over_factorial(index: usize) -> Result<Rational, GwError> {
    let mut value = qrr_bernoulli_number(index)?;
    for factor in 2..=index {
        value = value / Rational::from(factor);
    }
    Ok(value)
}

pub fn qrr_bernoulli_number(index: usize) -> Result<Rational, GwError> {
    let maximum = MAX_QRR_POSITIVE_Z_POWER + 1;
    if index > maximum {
        return Err(GwError::ResourceLimit {
            operation: "QRR Bernoulli-number table".to_string(),
            requested: index,
            limit: maximum,
        });
    }
    let mut numbers = Vec::new();
    numbers
        .try_reserve_exact(index.saturating_add(1))
        .map_err(|_| {
            GwError::UnsupportedInvariant("cannot allocate Bernoulli-number table".to_string())
        })?;
    numbers.push(Rational::one());
    for n in 1..=index {
        let mut sum = Rational::zero();
        let mut binomial = 1u128;
        for (k, number) in numbers.iter().enumerate().take(n) {
            if k > 0 {
                binomial = binomial
                    .checked_mul((n + 2 - k) as u128)
                    .and_then(|value| value.checked_div(k as u128))
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant(
                            "Bernoulli binomial coefficient overflow".to_string(),
                        )
                    })?;
            }
            let coefficient = i128::try_from(binomial).map_err(|_| {
                GwError::UnsupportedInvariant(
                    "Bernoulli binomial coefficient does not fit i128".to_string(),
                )
            })?;
            sum += Rational::from(coefficient) * number.clone();
        }
        numbers.push(-sum / Rational::from(n + 1));
    }
    Ok(numbers[index].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qrr_table_has_b2_and_b4_coefficients() {
        let formula = QrrConjugationFormula::new(
            "P^2",
            "\\mathbb P^2",
            "O(-3)",
            "\\mathcal O(-3)",
            "inverse Euler",
            2,
            3,
        )
        .unwrap();
        let positive = formula
            .terms()
            .iter()
            .filter(|term| term.factor == QrrFactor::Positive)
            .collect::<Vec<_>>();
        assert_eq!(positive[0].rational_multiplier, Rational::new(1, 12));
        assert_eq!(positive[3].rational_multiplier, Rational::new(-1, 720));
    }

    #[test]
    fn public_bernoulli_helper_obeys_the_qrr_mode_cap() {
        assert!(matches!(
            qrr_bernoulli_number(MAX_QRR_POSITIVE_Z_POWER + 2),
            Err(GwError::ResourceLimit { .. })
        ));
    }

    #[test]
    fn rendered_formula_records_pairing_shift_order_and_source() {
        let formula = QrrConjugationFormula::new(
            "P^1",
            "\\mathbb P^1",
            "O(-1)+O(-1)",
            "\\mathcal O(-1)^{\\oplus2}",
            "inverse Euler",
            1,
            1,
        )
        .unwrap();
        let text = formula.render_text();
        assert!(text.contains("q_CG(z) = sqrt(c(E)) (t(z)-z)"));
        assert!(text.contains("exp(hat_CG(A_+)) exp(hat_CG(A_-))"));
        assert!(text.contains("exp(-hat_G(A_+)) exp(-hat_G(A_-))"));
        assert!(text.contains("math/0110142"));
        let tex = formula.render_tex();
        assert!(tex.contains("\\mathbf c(V)=\\exp"));
        assert!(tex.contains("math/0110142"));
    }
}
