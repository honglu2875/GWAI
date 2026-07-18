//! Canonical-correlator adapter for Virasoro audits of products of projective spaces.

use crate::constraints::virasoro::{
    evaluate_with_divisor_recursion, CanonicalCorrelatorEvaluator, CorrelatorKey, Descendant,
};
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{reconstruct_bidegree_invariants_in_theory, ProductInsertion, ProductProjectiveTheory};

/// Exact ray-reconstruction adapter used to substitute product invariants into
/// canonical Virasoro constraints.
pub struct ProductProjectiveEvaluator {
    theory: ProductProjectiveTheory,
    weights_x: Vec<Rational>,
    weights_y: Vec<Rational>,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl ProductProjectiveEvaluator {
    pub fn new(n: usize, m: usize) -> Result<Self, GwError> {
        let theory = ProductProjectiveTheory::new(n, m)?;
        let weights_x = (0..=n).map(|index| Rational::from(index + 1)).collect();
        let weight_stride = n.checked_add(2).ok_or_else(|| {
            GwError::UnsupportedInvariant("product default weight overflow".to_string())
        })?;
        let weights_y = (0..=m)
            .map(|index| {
                weight_stride
                    .checked_mul(index + 1)
                    .map(Rational::from)
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant("product default weight overflow".to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_weights(theory, weights_x, weights_y)
    }

    pub fn with_weights(
        theory: ProductProjectiveTheory,
        weights_x: Vec<Rational>,
        weights_y: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let (n, m) = theory.dimensions();
        if weights_x.len() != n + 1 || weights_y.len() != m + 1 {
            return Err(GwError::ConventionMismatch(
                "product evaluator weights do not match its canonical theory".to_string(),
            ));
        }
        Ok(Self {
            theory,
            weights_x,
            weights_y,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn product_theory(&self) -> &ProductProjectiveTheory {
        &self.theory
    }
}

impl CanonicalCorrelatorEvaluator for ProductProjectiveEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        &self.theory
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let (d1, d2) = self
            .theory
            .bidegree(&correlator.degree)
            .ok_or_else(|| GwError::ConventionMismatch("invalid product bidegree".to_string()))?;
        let insertions = correlator
            .insertions()
            .iter()
            .map(|insertion| {
                let (h1, h2) = self.theory.basis_powers(insertion.class).ok_or_else(|| {
                    GwError::ConventionMismatch("invalid product basis id".to_string())
                })?;
                Ok(ProductInsertion::new(insertion.psi_power, h1, h2))
            })
            .collect::<Result<Vec<_>, GwError>>()?;
        let value = evaluate_product_with_divisor_recursion(
            &self.theory,
            &self.weights_x,
            &self.weights_y,
            correlator.genus,
            d1,
            d2,
            insertions,
        )?;
        let value = RatFun::from_rational(value);
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

fn reconstruct_requested_product_value(
    theory: &ProductProjectiveTheory,
    weights_x: &[Rational],
    weights_y: &[Rational],
    genus: usize,
    d1: usize,
    d2: usize,
    insertions: &[ProductInsertion],
) -> Result<Rational, GwError> {
    let total_degree = d1
        .checked_add(d2)
        .ok_or_else(|| GwError::AlgebraFailure("total bidegree overflow".to_string()))?;
    let values = reconstruct_bidegree_invariants_in_theory(
        theory,
        weights_x,
        weights_y,
        genus,
        total_degree,
        insertions,
    )?;
    values.get(d2).cloned().ok_or_else(|| {
        GwError::AlgebraFailure("product reconstruction omitted requested bidegree".to_string())
    })
}

fn evaluate_product_with_divisor_recursion(
    theory: &ProductProjectiveTheory,
    weights_x: &[Rational],
    weights_y: &[Rational],
    genus: usize,
    d1: usize,
    d2: usize,
    insertions: Vec<ProductInsertion>,
) -> Result<Rational, GwError> {
    let curve = theory.try_curve(d1, d2)?;
    let canonical = insertions
        .iter()
        .map(|insertion| {
            theory
                .basis_id(insertion.h1_power, insertion.h2_power)
                .map(|basis| Descendant::new(insertion.descendant_power, basis))
                .ok_or_else(|| {
                    GwError::ConventionMismatch(
                        "product insertion is outside the canonical basis".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let evaluate_stable = |canonical: &[Descendant<BasisId>]| {
        let backend_insertions = canonical
            .iter()
            .map(|insertion| {
                theory
                    .basis_powers(insertion.class)
                    .map(|(h1_power, h2_power)| {
                        ProductInsertion::new(insertion.psi_power, h1_power, h2_power)
                    })
                    .ok_or_else(|| {
                        GwError::AlgebraFailure(
                            "product cup product returned an invalid basis id".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        reconstruct_requested_product_value(
            theory,
            weights_x,
            weights_y,
            genus,
            d1,
            d2,
            &backend_insertions,
        )
    };
    evaluate_with_divisor_recursion(
        theory,
        genus,
        &curve,
        canonical,
        "product",
        &evaluate_stable,
    )
}
