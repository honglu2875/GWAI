//! Validation-only genus-zero primary localization for split projective bundles.
//!
//! This module deliberately stops at shifted total degree two.  It evaluates
//! the stable-map fixed trees directly from the toric one-skeleton and the
//! moving `H^0-H^1` weights of the normal bundles.  It does not use the bundle
//! I-function, Birkhoff factorization, Novikov-ray reconstruction, or the
//! Givental `S`/`R` graph engine.

use super::{labelled_leg_assignments, labelled_trees, LocEdge, LocGraph, LocVertex, MarkedLeg};
use crate::algebra::Rational;
use crate::error::GwError;
use crate::givental::BundleInsertion;
use crate::theory::{CurveClass, GwTheory, ProjectiveBundleTheory};
use std::collections::BTreeSet;

/// Largest theory-owned shifted total degree accepted by this direct oracle.
pub const MAX_PROJECTIVE_BUNDLE_LOCALIZATION_DEGREE: usize = 2;

/// Conservative guard on the labelled fixed-tree candidates materialized
/// before canonical graph deduplication.
const MAX_FIXED_TREE_CANDIDATES: usize = 5_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalWeight {
    degree: i64,
    at_from: Rational,
    at_to: Rational,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OrbitData {
    tangent_at_from: Rational,
    normals: Vec<NormalWeight>,
}

#[derive(Debug, Clone)]
struct BundleTorus<'a> {
    theory: &'a ProjectiveBundleTheory,
    base_weights: &'a [Rational],
    fiber_weights: &'a [Rational],
}

impl<'a> BundleTorus<'a> {
    fn new(
        theory: &'a ProjectiveBundleTheory,
        base_weights: &'a [Rational],
        fiber_weights: &'a [Rational],
    ) -> Result<Self, GwError> {
        if base_weights.len() != theory.base_dimension() + 1 || fiber_weights.len() != theory.rank()
        {
            return Err(GwError::ConventionMismatch(format!(
                "projective-bundle localization weights must have lengths {} and {}",
                theory.base_dimension() + 1,
                theory.rank()
            )));
        }
        let torus = Self {
            theory,
            base_weights,
            fiber_weights,
        };
        torus.validate_isolated_fixed_points()?;
        Ok(torus)
    }

    fn rank(&self) -> usize {
        self.theory.rank()
    }

    fn fixed_point_count(&self) -> usize {
        (self.theory.base_dimension() + 1) * self.rank()
    }

    #[cfg(test)]
    fn point(&self, base: usize, summand: usize) -> usize {
        base * self.rank() + summand
    }

    fn point_indices(&self, point: usize) -> (usize, usize) {
        (point / self.rank(), point % self.rank())
    }

    /// `c_ij = a_j lambda_i + mu_j`; the tautological class restricts as
    /// `xi|_(i,j) = -c_ij`.
    fn fiber_coordinate_weight(&self, base: usize, summand: usize) -> Rational {
        Rational::from(self.theory.twists()[summand]) * self.base_weights[base].clone()
            + self.fiber_weights[summand].clone()
    }

    fn fixed_point_euler(&self, point: usize) -> Rational {
        let (base, summand) = self.point_indices(point);
        let mut euler = Rational::one();
        for other_base in 0..=self.theory.base_dimension() {
            if other_base != base {
                euler = euler
                    * (self.base_weights[base].clone() - self.base_weights[other_base].clone());
            }
        }
        let selected = self.fiber_coordinate_weight(base, summand);
        for other_summand in 0..self.rank() {
            if other_summand != summand {
                euler =
                    euler * (self.fiber_coordinate_weight(base, other_summand) - selected.clone());
            }
        }
        euler
    }

    fn insertion_restriction(&self, point: usize, insertion: &BundleInsertion) -> Rational {
        let (base, summand) = self.point_indices(point);
        let h = self.base_weights[base].pow_usize(insertion.h_power);
        let xi = (-self.fiber_coordinate_weight(base, summand)).pow_usize(insertion.xi_power);
        h * xi
    }

    fn validate_isolated_fixed_points(&self) -> Result<(), GwError> {
        for left in 0..self.base_weights.len() {
            for right in left + 1..self.base_weights.len() {
                if self.base_weights[left] == self.base_weights[right] {
                    return Err(GwError::NonSemisimplePoint);
                }
            }
        }
        for base in 0..self.base_weights.len() {
            for left in 0..self.rank() {
                for right in left + 1..self.rank() {
                    if self.fiber_coordinate_weight(base, left)
                        == self.fiber_coordinate_weight(base, right)
                    {
                        return Err(GwError::NonSemisimplePoint);
                    }
                }
            }
        }
        Ok(())
    }

    fn orbit_class(&self, from: usize, to: usize) -> Option<(i64, i64)> {
        let (from_base, from_summand) = self.point_indices(from);
        let (to_base, to_summand) = self.point_indices(to);
        if from_base == to_base && from_summand != to_summand {
            Some((0, 1))
        } else if from_base != to_base && from_summand == to_summand {
            let twist = i64::try_from(self.theory.twists()[from_summand]).ok()?;
            Some((1, -twist))
        } else {
            None
        }
    }

    fn orbit_data(&self, from: usize, to: usize) -> Result<Option<OrbitData>, GwError> {
        let (from_base, from_summand) = self.point_indices(from);
        let (to_base, to_summand) = self.point_indices(to);
        let mut normals = Vec::new();
        let tangent_at_from = if from_base == to_base && from_summand != to_summand {
            let from_coordinate = self.fiber_coordinate_weight(from_base, from_summand);
            let tangent =
                self.fiber_coordinate_weight(from_base, to_summand) - from_coordinate.clone();
            for other_base in 0..=self.theory.base_dimension() {
                if other_base != from_base {
                    let weight = self.base_weights[from_base].clone()
                        - self.base_weights[other_base].clone();
                    normals.push(NormalWeight {
                        degree: 0,
                        at_from: weight.clone(),
                        at_to: weight,
                    });
                }
            }
            for other_summand in 0..self.rank() {
                if other_summand != from_summand && other_summand != to_summand {
                    normals.push(NormalWeight {
                        degree: 1,
                        at_from: self.fiber_coordinate_weight(from_base, other_summand)
                            - from_coordinate.clone(),
                        at_to: self.fiber_coordinate_weight(from_base, other_summand)
                            - self.fiber_coordinate_weight(from_base, to_summand),
                    });
                }
            }
            tangent
        } else if from_base != to_base && from_summand == to_summand {
            let tangent = self.base_weights[from_base].clone() - self.base_weights[to_base].clone();
            for other_base in 0..=self.theory.base_dimension() {
                if other_base != from_base && other_base != to_base {
                    normals.push(NormalWeight {
                        degree: 1,
                        at_from: self.base_weights[from_base].clone()
                            - self.base_weights[other_base].clone(),
                        at_to: self.base_weights[to_base].clone()
                            - self.base_weights[other_base].clone(),
                    });
                }
            }
            let selected_twist = i64::try_from(self.theory.twists()[from_summand])
                .map_err(|_| localization_overflow("bundle twist"))?;
            let from_coordinate = self.fiber_coordinate_weight(from_base, from_summand);
            let to_coordinate = self.fiber_coordinate_weight(to_base, to_summand);
            for other_summand in 0..self.rank() {
                if other_summand == from_summand {
                    continue;
                }
                let other_twist = i64::try_from(self.theory.twists()[other_summand])
                    .map_err(|_| localization_overflow("bundle twist"))?;
                normals.push(NormalWeight {
                    degree: other_twist - selected_twist,
                    at_from: self.fiber_coordinate_weight(from_base, other_summand)
                        - from_coordinate.clone(),
                    at_to: self.fiber_coordinate_weight(to_base, other_summand)
                        - to_coordinate.clone(),
                });
            }
            tangent
        } else {
            return Ok(None);
        };

        if tangent_at_from.is_zero() {
            return Err(GwError::NonSemisimplePoint);
        }
        for normal in &normals {
            let expected_to =
                normal.at_from.clone() - Rational::from(normal.degree) * tangent_at_from.clone();
            if normal.at_to != expected_to {
                return Err(GwError::ConventionMismatch(
                    "projective-bundle orbit normal weights do not match their line-bundle degree"
                        .to_string(),
                ));
            }
        }
        Ok(Some(OrbitData {
            tangent_at_from,
            normals,
        }))
    }
}

/// Direct fixed-tree localization of an ordinary genus-zero primary invariant.
///
/// The supplied rational weights are only an exact generic torus
/// specialization.  A dimension-matched ordinary invariant is independent of
/// them after summing all fixed loci.  Requests outside positive shifted total
/// degree at most two, descendants, or the canonical curve cone fail closed.
pub fn genus_zero_primary_projective_bundle_localization(
    theory: &ProjectiveBundleTheory,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    curve: &CurveClass,
    insertions: &[BundleInsertion],
) -> Result<Rational, GwError> {
    if insertions
        .iter()
        .any(|insertion| insertion.descendant_power != 0)
    {
        return Err(unsupported(
            "direct projective-bundle localization only supports primary insertions",
        ));
    }
    for insertion in insertions {
        if insertion.h_power > theory.base_dimension() || insertion.xi_power >= theory.rank() {
            return Err(GwError::ConventionMismatch(
                "projective-bundle localization insertion lies outside the canonical basis"
                    .to_string(),
            ));
        }
    }
    let (d1, shifted) = theory.shifted_bidegree(curve).ok_or_else(|| {
        unsupported("projective-bundle localization curve is outside the shifted cone")
    })?;
    let shifted_total = d1
        .checked_add(shifted)
        .ok_or_else(|| localization_overflow("shifted total degree"))?;
    if shifted_total == 0 || shifted_total > MAX_PROJECTIVE_BUNDLE_LOCALIZATION_DEGREE {
        return Err(unsupported(
            "direct projective-bundle localization supports positive shifted total degree at most two",
        ));
    }

    let insertion_degree = insertions.iter().try_fold(0usize, |total, insertion| {
        total
            .checked_add(insertion.h_power)
            .and_then(|value| value.checked_add(insertion.xi_power))
    });
    let Some(insertion_degree) = insertion_degree else {
        return Err(localization_overflow("insertion degree"));
    };
    let virtual_dimension = theory.virtual_dimension(0, curve, insertions.len())?;
    if usize::try_from(virtual_dimension).ok() != Some(insertion_degree) {
        return Ok(Rational::zero());
    }

    let torus = BundleTorus::new(theory, base_weights, fiber_weights)?;
    let graphs = bounded_fixed_trees(&torus, curve, insertions.len(), shifted_total)?;
    let mut total = Rational::zero();
    for graph in &graphs {
        total += graph_contribution(&torus, graph, insertions)?;
    }
    Ok(total)
}

fn bounded_fixed_trees(
    torus: &BundleTorus<'_>,
    curve: &CurveClass,
    markings: usize,
    shifted_total: usize,
) -> Result<Vec<LocGraph>, GwError> {
    let (target_d1, target_d2) = torus
        .theory
        .bidegree(curve)
        .ok_or_else(|| GwError::ConventionMismatch("invalid bundle curve class".to_string()))?;
    let target_d1 =
        i64::try_from(target_d1).map_err(|_| localization_overflow("bundle base degree"))?;
    check_candidate_budget(torus.fixed_point_count(), markings, shifted_total)?;

    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for edge_count in 1..=shifted_total {
        let vertex_count = edge_count + 1;
        for abstract_edges in labelled_trees(vertex_count) {
            for fixed_points in bundle_colorings(
                vertex_count,
                torus.fixed_point_count(),
                &abstract_edges,
                torus,
            )? {
                for covers in bounded_edge_covers(edge_count, shifted_total) {
                    let mut graph_d1 = 0i64;
                    let mut graph_d2 = 0i64;
                    let mut valid = true;
                    for ((from, to), cover) in abstract_edges.iter().zip(&covers) {
                        let Some((edge_d1, edge_d2)) =
                            torus.orbit_class(fixed_points[*from], fixed_points[*to])
                        else {
                            valid = false;
                            break;
                        };
                        let cover = i64::try_from(*cover)
                            .map_err(|_| localization_overflow("edge cover degree"))?;
                        let covered_d1 = edge_d1
                            .checked_mul(cover)
                            .ok_or_else(|| localization_overflow("covered base degree"))?;
                        let covered_d2 = edge_d2
                            .checked_mul(cover)
                            .ok_or_else(|| localization_overflow("covered fiber degree"))?;
                        graph_d1 = graph_d1
                            .checked_add(covered_d1)
                            .ok_or_else(|| localization_overflow("graph base degree"))?;
                        graph_d2 = graph_d2
                            .checked_add(covered_d2)
                            .ok_or_else(|| localization_overflow("graph fiber degree"))?;
                    }
                    if !valid || (graph_d1, graph_d2) != (target_d1, target_d2) {
                        continue;
                    }
                    for leg_vertices in labelled_leg_assignments(markings, vertex_count) {
                        let graph = LocGraph {
                            vertices: fixed_points
                                .iter()
                                .map(|fixed_point| LocVertex {
                                    fixed_point: *fixed_point,
                                    genus: 0,
                                })
                                .collect(),
                            edges: abstract_edges
                                .iter()
                                .zip(&covers)
                                .map(|(&(from, to), &degree)| LocEdge::new(from, to, degree))
                                .collect(),
                            legs: leg_vertices
                                .iter()
                                .enumerate()
                                .map(|(marking, &vertex)| MarkedLeg { vertex, marking })
                                .collect(),
                        };
                        let label = graph.canonical_label();
                        if seen.insert(label) {
                            out.push(graph);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn bundle_colorings(
    vertex_count: usize,
    fixed_point_count: usize,
    edges: &[(usize, usize)],
    torus: &BundleTorus<'_>,
) -> Result<Vec<Vec<usize>>, GwError> {
    fn rec(
        vertex_count: usize,
        fixed_point_count: usize,
        edges: &[(usize, usize)],
        torus: &BundleTorus<'_>,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) -> Result<(), GwError> {
        if current.len() == vertex_count {
            out.push(current.clone());
            return Ok(());
        }
        let vertex = current.len();
        for point in 0..fixed_point_count {
            let valid = edges.iter().all(|&(left, right)| {
                let neighbor = if left == vertex && right < current.len() {
                    Some(right)
                } else if right == vertex && left < current.len() {
                    Some(left)
                } else {
                    None
                };
                neighbor
                    .is_none_or(|neighbor| torus.orbit_class(point, current[neighbor]).is_some())
            });
            if valid {
                current.push(point);
                rec(vertex_count, fixed_point_count, edges, torus, current, out)?;
                current.pop();
            }
        }
        Ok(())
    }

    let mut out = Vec::new();
    rec(
        vertex_count,
        fixed_point_count,
        edges,
        torus,
        &mut Vec::new(),
        &mut out,
    )?;
    Ok(out)
}

fn bounded_edge_covers(edge_count: usize, max_total: usize) -> Vec<Vec<usize>> {
    fn rec(
        edge_count: usize,
        max_total: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == edge_count {
            out.push(current.clone());
            return;
        }
        for degree in 1..=max_total {
            current.push(degree);
            rec(edge_count, max_total, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(edge_count, max_total, &mut Vec::new(), &mut out);
    out
}

fn graph_contribution(
    torus: &BundleTorus<'_>,
    graph: &LocGraph,
    insertions: &[BundleInsertion],
) -> Result<Rational, GwError> {
    let mut contribution = Rational::one();
    for edge in &graph.edges {
        contribution = contribution * edge_factor(torus, graph, edge)?;
    }
    contribution = checked_div(
        contribution,
        Rational::from(graph.cover_automorphism_order()),
        "fixed-tree automorphism order",
    )?;
    for vertex in 0..graph.vertices.len() {
        contribution = contribution * vertex_factor(torus, graph, vertex, insertions)?;
    }
    Ok(contribution)
}

/// Edge factor after separating the tangent `O(2)` summand.  For a degree-`d`
/// cover of an invariant orbit with tangent weight `alpha` and endpoint tangent
/// Euler classes `e_p,e_q`, the tangent summand contributes
///
/// `(-1)^d d^(2d) e_p e_q / ((d!)^2 alpha^(2d))`.
///
/// This is the standard multiple-cover factor; in particular, its factorial
/// ratio is the reciprocal of the historical expression in the adjacent
/// degree-one-only projective-space validator.  For a normal line `O(m)` with
/// endpoint weight `w` and flag weight `omega=alpha/d`, the pullback contributes
///
/// - `prod_{r=0}^{md} (w-r omega)^(-1)` when `md >= 0` (`H^0`),
/// - `prod_{r=1}^{-md-1} (w+r omega)` when `md < 0` (`H^1`).
fn edge_factor(
    torus: &BundleTorus<'_>,
    graph: &LocGraph,
    edge: &LocEdge,
) -> Result<Rational, GwError> {
    let from = graph.vertices[edge.from].fixed_point;
    let to = graph.vertices[edge.to].fixed_point;
    let orbit = torus.orbit_data(from, to)?.ok_or_else(|| {
        GwError::ConventionMismatch("localization edge is not a toric orbit".to_string())
    })?;
    let degree = edge.degree;
    let degree_i64 =
        i64::try_from(degree).map_err(|_| localization_overflow("edge cover degree"))?;
    let degree_rational = Rational::from(degree);
    let alpha = orbit.tangent_at_from;
    let omega = checked_div(alpha.clone(), degree_rational.clone(), "edge flag weight")?;

    let factorial = (1..=degree).product::<usize>().max(1);
    let sign = if degree.is_multiple_of(2) {
        1i64
    } else {
        -1i64
    };
    let mut factor = Rational::from(sign) * degree_rational.pow_usize(2 * degree);
    factor = checked_div(
        factor,
        Rational::from(factorial) * Rational::from(factorial),
        "multiple-cover tangent factor",
    )?;
    factor = factor * torus.fixed_point_euler(from) * torus.fixed_point_euler(to);
    factor = checked_div(factor, alpha.pow_usize(2 * degree), "orbit tangent weight")?;

    for normal in orbit.normals {
        let pulled_degree = normal
            .degree
            .checked_mul(degree_i64)
            .ok_or_else(|| localization_overflow("pulled-back normal degree"))?;
        if pulled_degree >= 0 {
            for step in 0..=usize::try_from(pulled_degree)
                .map_err(|_| localization_overflow("normal H0 dimension"))?
            {
                let weight = normal.at_from.clone() - Rational::from(step) * omega.clone();
                factor = checked_div(factor, weight, "normal H0 moving weight")?;
            }
        } else {
            let obstruction_rank = usize::try_from(-pulled_degree)
                .map_err(|_| localization_overflow("normal H1 dimension"))?;
            for step in 1..obstruction_rank {
                factor = factor * (normal.at_from.clone() + Rational::from(step) * omega.clone());
            }
        }
    }
    Ok(factor)
}

fn vertex_factor(
    torus: &BundleTorus<'_>,
    graph: &LocGraph,
    vertex: usize,
    insertions: &[BundleInsertion],
) -> Result<Rational, GwError> {
    let point = graph.vertices[vertex].fixed_point;
    let mut flags = Vec::new();
    for edge in &graph.edges {
        let neighbor = if edge.from == vertex {
            Some(edge.to)
        } else if edge.to == vertex {
            Some(edge.from)
        } else {
            None
        };
        if let Some(neighbor) = neighbor {
            let neighbor_point = graph.vertices[neighbor].fixed_point;
            let tangent = torus
                .orbit_data(point, neighbor_point)?
                .ok_or_else(|| {
                    GwError::ConventionMismatch(
                        "localization flag is not a toric orbit".to_string(),
                    )
                })?
                .tangent_at_from;
            flags.push(checked_div(
                tangent,
                Rational::from(edge.degree),
                "vertex flag weight",
            )?);
        }
    }

    let legs = graph
        .legs
        .iter()
        .filter(|leg| leg.vertex == vertex)
        .collect::<Vec<_>>();
    let mut marking_factor = Rational::one();
    for leg in &legs {
        marking_factor =
            marking_factor * torus.insertion_restriction(point, &insertions[leg.marking]);
    }
    let euler = torus.fixed_point_euler(point);
    let valence = flags.len() + legs.len();
    let normal_factor = match (valence, flags.len(), legs.len()) {
        (1, 1, 0) => checked_div(flags[0].clone(), euler, "unstable one-flag vertex")?,
        (2, 1, 1) => checked_div(Rational::one(), euler, "unstable marked vertex")?,
        (2, 2, 0) => {
            let denominator = euler * (flags[0].clone() + flags[1].clone());
            checked_div(Rational::one(), denominator, "unstable two-flag vertex")?
        }
        (stable, _, _) if stable >= 3 => stable_vertex_factor(euler, &flags, stable)?,
        _ => {
            return Err(GwError::ConventionMismatch(
                "unsupported unstable vertex appeared in a positive-degree fixed tree".to_string(),
            ))
        }
    };
    Ok(normal_factor * marking_factor)
}

fn stable_vertex_factor(
    euler: Rational,
    flags: &[Rational],
    valence: usize,
) -> Result<Rational, GwError> {
    let mut reciprocal_sum = Rational::zero();
    let mut reciprocal_product = Rational::one();
    for flag in flags {
        let reciprocal = checked_div(Rational::one(), flag.clone(), "stable vertex flag")?;
        reciprocal_sum += reciprocal.clone();
        reciprocal_product = reciprocal_product * reciprocal;
    }
    checked_div(
        reciprocal_product * reciprocal_sum.pow_usize(valence - 3),
        euler,
        "stable vertex Euler class",
    )
}

fn check_candidate_budget(
    fixed_points: usize,
    markings: usize,
    shifted_total: usize,
) -> Result<(), GwError> {
    let markings = u32::try_from(markings)
        .map_err(|_| unsupported("too many markings for bounded fixed-tree localization"))?;
    let mut candidates = 0usize;
    for edges in 1..=shifted_total {
        let vertices = edges + 1;
        let colorings = fixed_points
            .checked_pow(u32::try_from(vertices).expect("degree-two vertex count"))
            .ok_or_else(|| localization_overflow("fixed-tree coloring count"))?;
        let leg_assignments = vertices
            .checked_pow(markings)
            .ok_or_else(|| localization_overflow("fixed-tree leg assignment count"))?;
        let covers = shifted_total
            .checked_pow(u32::try_from(edges).expect("degree-two edge count"))
            .ok_or_else(|| localization_overflow("fixed-tree cover count"))?;
        let labelled_tree_bound = if vertices == 2 { 1 } else { 3 };
        let term = colorings
            .checked_mul(leg_assignments)
            .and_then(|value| value.checked_mul(covers))
            .and_then(|value| value.checked_mul(labelled_tree_bound))
            .ok_or_else(|| localization_overflow("fixed-tree candidate count"))?;
        candidates = candidates
            .checked_add(term)
            .ok_or_else(|| localization_overflow("fixed-tree candidate count"))?;
    }
    if candidates > MAX_FIXED_TREE_CANDIDATES {
        return Err(GwError::ResourceLimit {
            operation: "direct projective-bundle localization fixed-tree candidates".to_string(),
            requested: candidates,
            limit: MAX_FIXED_TREE_CANDIDATES,
        });
    }
    Ok(())
}

fn checked_div(
    numerator: Rational,
    denominator: Rational,
    _context: &'static str,
) -> Result<Rational, GwError> {
    if denominator.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    Ok(numerator / denominator)
}

fn localization_overflow(quantity: &str) -> GwError {
    GwError::AlgebraFailure(format!(
        "projective-bundle localization {quantity} overflow"
    ))
}

fn unsupported(message: &str) -> GwError {
    GwError::UnsupportedInvariant(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holdout_theory() -> ProjectiveBundleTheory {
        ProjectiveBundleTheory::new(2, vec![0, 3, 3]).unwrap()
    }

    fn base_weights() -> Vec<Rational> {
        vec![Rational::from(1), Rational::from(2), Rational::from(4)]
    }

    fn fiber_weights() -> Vec<Rational> {
        vec![Rational::from(0), Rational::from(17), Rational::from(43)]
    }

    fn value(curve: CurveClass, insertions: &[BundleInsertion]) -> Result<Rational, GwError> {
        genus_zero_primary_projective_bundle_localization(
            &holdout_theory(),
            &base_weights(),
            &fiber_weights(),
            &curve,
            insertions,
        )
    }

    #[test]
    fn orbit_normal_degrees_reproduce_endpoint_weights() {
        let theory = holdout_theory();
        let base_weights = base_weights();
        let fiber_weights = fiber_weights();
        let torus = BundleTorus::new(&theory, &base_weights, &fiber_weights).unwrap();
        let base_orbit = torus
            .orbit_data(torus.point(0, 1), torus.point(2, 1))
            .unwrap()
            .unwrap();
        assert_eq!(
            base_orbit
                .normals
                .iter()
                .map(|normal| normal.degree)
                .collect::<Vec<_>>(),
            vec![1, -3, 0]
        );
        let fiber_orbit = torus
            .orbit_data(torus.point(0, 0), torus.point(0, 1))
            .unwrap()
            .unwrap();
        assert_eq!(
            fiber_orbit
                .normals
                .iter()
                .map(|normal| normal.degree)
                .collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
    }

    #[test]
    fn bounded_tree_enumerator_has_the_expected_primitive_orbits() {
        let theory = holdout_theory();
        let base_weights = base_weights();
        let fiber_weights = fiber_weights();
        let torus = BundleTorus::new(&theory, &base_weights, &fiber_weights).unwrap();
        let fiber = bounded_fixed_trees(&torus, &theory.curve(0, 1), 0, 1).unwrap();
        let section = bounded_fixed_trees(&torus, &theory.curve(1, -3), 0, 1).unwrap();
        assert_eq!(fiber.len(), 9);
        assert_eq!(section.len(), 6);
        assert!(fiber.iter().all(|graph| graph.edges.len() == 1));
        assert!(section.iter().all(|graph| graph.edges.len() == 1));
    }

    #[test]
    fn fiber_line_through_two_fiber_points_is_one() {
        let theory = holdout_theory();
        let top = BundleInsertion::new(0, 2, 2);
        let fiber_point = BundleInsertion::new(0, 0, 2);
        assert_eq!(
            value(theory.curve(0, 1), &[top, fiber_point]).unwrap(),
            Rational::one()
        );
    }

    #[test]
    fn degree_two_fiber_conic_through_five_points_is_one() {
        let theory = holdout_theory();
        let mut insertions = vec![BundleInsertion::new(0, 2, 2)];
        insertions.extend(std::iter::repeat_n(BundleInsertion::new(0, 0, 2), 4));
        assert_eq!(
            value(theory.curve(0, 2), &insertions).unwrap(),
            Rational::one()
        );
    }

    #[test]
    fn holdout_section_and_mixed_rows_match_the_known_numbers() {
        let theory = holdout_theory();
        let h_squared = BundleInsertion::new(0, 2, 0);
        let xi = BundleInsertion::new(0, 0, 1);
        let top = BundleInsertion::new(0, 2, 2);
        let fiber_point = BundleInsertion::new(0, 0, 2);

        // The one-point section value is -3.  Applying the divisor equation
        // twice with xi.beta=-3 gives -27; localization obtains the same number
        // directly from the marked fixed trees.
        assert_eq!(
            value(theory.curve(1, -3), &[h_squared, xi.clone(), xi]).unwrap(),
            Rational::from(-27)
        );
        assert_eq!(
            value(theory.curve(1, -2), &[top, fiber_point]).unwrap(),
            Rational::from(19)
        );
    }

    #[test]
    fn holdout_rows_are_independent_of_the_generic_rational_weights() {
        let theory = holdout_theory();
        let alternate_base = vec![Rational::from(-2), Rational::from(5), Rational::from(11)];
        let alternate_fiber = vec![Rational::from(7), Rational::from(29), Rational::from(61)];
        let cases = [
            (
                theory.curve(0, 1),
                vec![BundleInsertion::new(0, 2, 2), BundleInsertion::new(0, 0, 2)],
                Rational::one(),
            ),
            (
                theory.curve(1, -3),
                vec![
                    BundleInsertion::new(0, 2, 0),
                    BundleInsertion::new(0, 0, 1),
                    BundleInsertion::new(0, 0, 1),
                ],
                Rational::from(-27),
            ),
            (
                theory.curve(1, -2),
                vec![BundleInsertion::new(0, 2, 2), BundleInsertion::new(0, 0, 2)],
                Rational::from(19),
            ),
        ];
        for (curve, insertions, expected) in cases {
            let alternate = genus_zero_primary_projective_bundle_localization(
                &theory,
                &alternate_base,
                &alternate_fiber,
                &curve,
                &insertions,
            )
            .unwrap();
            assert_eq!(alternate, expected);
            assert_eq!(
                genus_zero_primary_projective_bundle_localization(
                    &theory,
                    &base_weights(),
                    &fiber_weights(),
                    &curve,
                    &insertions,
                )
                .unwrap(),
                alternate
            );
        }
    }

    #[test]
    fn f1_and_product_rows_recover_elementary_curve_counts() {
        let f1 = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        let base = vec![Rational::from(2), Rational::from(5)];
        let fiber = vec![Rational::from(11), Rational::from(23)];
        let xi = BundleInsertion::new(0, 0, 1);
        assert_eq!(
            genus_zero_primary_projective_bundle_localization(
                &f1,
                &base,
                &fiber,
                &f1.curve(1, -1),
                &[xi.clone(), xi.clone(), xi],
            )
            .unwrap(),
            -Rational::one()
        );

        let product = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let point = BundleInsertion::new(0, 1, 1);
        assert_eq!(
            genus_zero_primary_projective_bundle_localization(
                &product,
                &base,
                &fiber,
                &product.curve(1, 1),
                &[point.clone(), point.clone(), point],
            )
            .unwrap(),
            Rational::one()
        );
    }

    #[test]
    fn rejects_descendants_and_degrees_beyond_the_foothold() {
        let theory = holdout_theory();
        let descendant = BundleInsertion::new(1, 0, 0);
        assert!(matches!(
            value(theory.curve(0, 1), &[descendant]),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            value(
                theory.curve(0, 3),
                &[BundleInsertion::new(0, 2, 2), BundleInsertion::new(0, 0, 2)]
            ),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    #[test]
    fn fixed_tree_work_guard_is_machine_readable() {
        let error = check_candidate_budget(100, 10, 2).unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested,
                limit: MAX_FIXED_TREE_CANDIDATES,
                ..
            } if requested > MAX_FIXED_TREE_CANDIDATES
        ));
    }
}
