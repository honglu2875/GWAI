use super::solve_rational_system;
use crate::core::algebra::Rational;
use crate::core::error::GwError;

/// Maximum number of one-parameter Novikov rays materialized by an exact
/// homogeneous two-degree reconstruction.
///
/// The current evaluator runs one scoped worker per ray and solves a dense
/// Vandermonde system.  The limit makes allocation and thread creation
/// fallible at the public API boundary; the implementation can later switch
/// to bounded parallelism without changing target providers.
pub const MAX_EXACT_RECONSTRUCTION_RAYS: usize = 64;

/// A checked sampling plan for exact recovery of
/// `f(b) = c_0 + b c_1 + ... + b^D c_D`.
///
/// Returned coefficients are ordered by increasing exponent.  Target code is
/// responsible only for evaluating its one-parameter specialization at a
/// supplied rational ray; worker management and exact interpolation live
/// here so products and projective bundles cannot drift apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactRayInterpolation {
    target: String,
    total_degree: usize,
    nodes: Vec<Rational>,
}

impl ExactRayInterpolation {
    /// Use the canonical nonzero rays `1, ..., D + 1`.
    pub(crate) fn for_total_degree(
        target: impl Into<String>,
        total_degree: usize,
    ) -> Result<Self, GwError> {
        let target = target.into();
        let ray_count = checked_reconstruction_ray_count(&target, total_degree)?;
        let nodes = (1..=ray_count).map(Rational::from).collect();
        Ok(Self {
            target,
            total_degree,
            nodes,
        })
    }

    /// Use caller-supplied distinct rays, primarily for independent
    /// interpolation and cache-sensitivity checks.
    pub(crate) fn with_nodes(
        target: impl Into<String>,
        total_degree: usize,
        nodes: &[Rational],
    ) -> Result<Self, GwError> {
        let target = target.into();
        let ray_count = checked_reconstruction_ray_count(&target, total_degree)?;
        if nodes.len() != ray_count {
            return Err(GwError::ConventionMismatch(format!(
                "{target} reconstruction of total degree {total_degree} requires exactly {ray_count} rays, received {}",
                nodes.len()
            )));
        }
        for left in 0..nodes.len() {
            for right in left + 1..nodes.len() {
                if nodes[left] == nodes[right] {
                    return Err(GwError::ConventionMismatch(format!(
                        "{target} reconstruction rays must be pairwise distinct"
                    )));
                }
            }
        }
        Ok(Self {
            target,
            total_degree,
            nodes: nodes.to_vec(),
        })
    }

    pub(crate) fn ray_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn total_degree(&self) -> usize {
        self.total_degree
    }

    /// Evaluate every ray in a named scoped worker and recover the exact
    /// monomial coefficients in deterministic node order.
    pub(crate) fn reconstruct<F>(&self, evaluate: F) -> Result<Vec<Rational>, GwError>
    where
        F: Fn(&Rational) -> Result<Rational, GwError> + Sync,
    {
        let ray_count = self.ray_count();
        let target = self.target.as_str();
        let values = std::thread::scope(|scope| -> Result<Vec<_>, GwError> {
            let mut handles = Vec::new();
            handles.try_reserve_exact(ray_count).map_err(|_| {
                GwError::UnsupportedInvariant(format!(
                    "cannot allocate {ray_count} {target} reconstruction workers"
                ))
            })?;
            for (step, ray) in self.nodes.iter().enumerate() {
                let evaluate = &evaluate;
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("gw-{target}-ray-{step}"))
                        .spawn_scoped(scope, move || evaluate(ray))
                        .map_err(|error| {
                            GwError::AlgebraFailure(format!(
                                "cannot spawn {target} reconstruction ray {step}: {error}"
                            ))
                        })?,
                );
            }

            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        GwError::AlgebraFailure(format!("{target} ray worker panicked"))
                    })?
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        self.interpolate_values(values)
    }

    fn interpolate_values(&self, mut values: Vec<Rational>) -> Result<Vec<Rational>, GwError> {
        let ray_count = self.ray_count();
        let mut matrix = self
            .nodes
            .iter()
            .map(|ray| {
                (0..ray_count)
                    .map(|power| ray.pow_usize(power))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        solve_rational_system(&mut matrix, &mut values)?;
        Ok(values)
    }
}

pub(crate) fn checked_reconstruction_ray_count(
    target: &str,
    total_degree: usize,
) -> Result<usize, GwError> {
    let ray_count = total_degree.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant(format!("{target} reconstruction degree is too large"))
    })?;
    if ray_count > MAX_EXACT_RECONSTRUCTION_RAYS {
        return Err(GwError::ResourceLimit {
            operation: format!("{target} exact Novikov-ray reconstruction"),
            requested: ray_count,
            limit: MAX_EXACT_RECONSTRUCTION_RAYS,
        });
    }
    Ok(ray_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(coefficients: &[Rational], ray: &Rational) -> Rational {
        coefficients
            .iter()
            .enumerate()
            .fold(Rational::zero(), |value, (power, coefficient)| {
                value + coefficient.clone() * ray.pow_usize(power)
            })
    }

    #[test]
    fn default_rays_recover_coefficients_in_monomial_order() {
        let expected = vec![
            Rational::from(7),
            Rational::from(-3),
            Rational::from(11),
            Rational::from(2),
        ];
        let plan = ExactRayInterpolation::for_total_degree("test", 3).unwrap();
        let actual = plan
            .reconstruct(|ray| Ok(evaluate(&expected, ray)))
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn disjoint_ray_families_recover_the_same_coefficients() {
        let expected = vec![
            Rational::from(-5),
            Rational::from(13),
            Rational::zero(),
            Rational::from(4),
        ];
        let first = ExactRayInterpolation::with_nodes(
            "test",
            3,
            &[
                Rational::from(1),
                Rational::from(2),
                Rational::from(3),
                Rational::from(4),
            ],
        )
        .unwrap()
        .reconstruct(|ray| Ok(evaluate(&expected, ray)))
        .unwrap();
        let second = ExactRayInterpolation::with_nodes(
            "test",
            3,
            &[
                Rational::from(7),
                Rational::from(11),
                Rational::from(17),
                Rational::from(23),
            ],
        )
        .unwrap()
        .reconstruct(|ray| Ok(evaluate(&expected, ray)))
        .unwrap();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn custom_ray_plan_rejects_wrong_count_and_collisions() {
        let wrong_count =
            ExactRayInterpolation::with_nodes("test", 2, &[Rational::one()]).unwrap_err();
        assert!(matches!(wrong_count, GwError::ConventionMismatch(_)));

        let collision =
            ExactRayInterpolation::with_nodes("test", 1, &[Rational::from(3), Rational::from(3)])
                .unwrap_err();
        assert!(matches!(collision, GwError::ConventionMismatch(_)));
    }

    #[test]
    fn oversized_plan_is_rejected_before_evaluation() {
        let error = ExactRayInterpolation::for_total_degree("test", MAX_EXACT_RECONSTRUCTION_RAYS)
            .unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested: 65,
                limit: 64,
                ..
            }
        ));

        let error = ExactRayInterpolation::for_total_degree("test", usize::MAX).unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
    }

    #[test]
    fn evaluator_errors_are_propagated() {
        let plan = ExactRayInterpolation::for_total_degree("test", 0).unwrap();
        let error = plan
            .reconstruct(|_| {
                Err(GwError::AlgebraFailure(
                    "sentinel evaluator failure".to_string(),
                ))
            })
            .unwrap_err();
        assert!(error.to_string().contains("sentinel evaluator failure"));
    }
}
