//! Target-neutral Laurent-window planning for graded Birkhoff factorization.
//!
//! A positive Laurent term at one grade can propagate into higher grades
//! during inversion of the positive factor.  Conversely, recovering a fixed
//! negative Laurent depth at a target grade can require deeper input at each
//! proper left summand.  This module records both dependency closures without
//! knowing whether the Novikov grading is one-dimensional, a bidegree, or a
//! target-specific effective-monoid element.

use crate::error::GwError;
use std::collections::BTreeMap;

/// Laurent windows sufficient for a graded Birkhoff recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BirkhoffWindowPlan<G> {
    pub(crate) positive_windows: BTreeMap<G, usize>,
    pub(crate) negative_depths: BTreeMap<G, usize>,
}

/// Close raw positive Laurent windows and requested negative depths under all
/// proper decompositions of the supplied grades.
///
/// `grades` must be in dependency order: whenever `grade = left + right` is a
/// proper split, `right` must occur before `grade`.  Reversing that same order
/// then propagates negative-depth requirements from targets to their proper
/// left summands.  The caller owns the grading law through `proper_splits`.
pub(crate) fn plan_birkhoff_windows<G, F>(
    grades: &[G],
    raw_positive_windows: &BTreeMap<G, usize>,
    base_negative_depth: usize,
    proper_splits: F,
    depth_overflow_message: &'static str,
) -> Result<BirkhoffWindowPlan<G>, GwError>
where
    G: Clone + Ord,
    F: Fn(&G) -> Vec<(G, G)>,
{
    let mut positive_windows: BTreeMap<G, usize> = BTreeMap::new();
    for grade in grades {
        let mut window = raw_positive_windows.get(grade).copied().unwrap_or(0);
        for (_, right_grade) in proper_splits(grade) {
            if let Some(right_window) = positive_windows.get(&right_grade).copied() {
                window = window.max(right_window.saturating_sub(1));
            }
        }
        positive_windows.insert(grade.clone(), window);
    }

    let mut negative_depths = grades
        .iter()
        .cloned()
        .map(|grade| (grade, base_negative_depth))
        .collect::<BTreeMap<_, _>>();
    for grade in grades.iter().rev() {
        let target_depth = negative_depths
            .get(grade)
            .copied()
            .unwrap_or(base_negative_depth);
        for (left_grade, right_grade) in proper_splits(grade) {
            let right_window = positive_windows.get(&right_grade).copied().unwrap_or(0);
            let needed_depth = target_depth
                .checked_add(right_window)
                .ok_or_else(|| GwError::UnsupportedInvariant(depth_overflow_message.to_string()))?;
            negative_depths
                .entry(left_grade)
                .and_modify(|depth| *depth = (*depth).max(needed_depth))
                .or_insert(needed_depth);
        }
    }

    Ok(BirkhoffWindowPlan {
        positive_windows,
        negative_depths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive_degrees(max_degree: usize) -> Vec<usize> {
        (1..=max_degree).collect()
    }

    fn degree_splits(degree: &usize) -> Vec<(usize, usize)> {
        (1..*degree).map(|left| (left, degree - left)).collect()
    }

    fn positive_bidegrees(max_total_degree: usize) -> Vec<(usize, usize)> {
        (1..=max_total_degree)
            .flat_map(|total| (0..=total).map(move |first| (first, total - first)))
            .collect()
    }

    fn bidegree_splits(grade: &(usize, usize)) -> Vec<((usize, usize), (usize, usize))> {
        let mut splits = Vec::new();
        for left_first in 0..=grade.0 {
            for left_second in 0..=grade.1 {
                let left = (left_first, left_second);
                if left == (0, 0) || left == *grade {
                    continue;
                }
                splits.push((left, (grade.0 - left_first, grade.1 - left_second)));
            }
        }
        splits
    }

    #[test]
    fn q_degree_plan_closes_positive_windows_and_negative_depths() {
        let plan = plan_birkhoff_windows(
            &positive_degrees(4),
            &BTreeMap::from([(3, 2)]),
            3,
            degree_splits,
            "q-depth overflow",
        )
        .unwrap();

        assert_eq!(
            plan.positive_windows,
            BTreeMap::from([(1, 0), (2, 0), (3, 2), (4, 1)])
        );
        assert_eq!(
            plan.negative_depths,
            BTreeMap::from([(1, 5), (2, 3), (3, 3), (4, 3)])
        );
    }

    #[test]
    fn bidegree_plan_uses_the_same_generic_dependency_closure() {
        let plan = plan_birkhoff_windows(
            &positive_bidegrees(3),
            &BTreeMap::from([((1, 1), 2)]),
            4,
            bidegree_splits,
            "bidegree-depth overflow",
        )
        .unwrap();

        assert_eq!(plan.positive_windows[&(1, 1)], 2);
        assert_eq!(plan.positive_windows[&(1, 2)], 1);
        assert_eq!(plan.positive_windows[&(2, 1)], 1);
        assert_eq!(plan.negative_depths[&(1, 0)], 6);
        assert_eq!(plan.negative_depths[&(0, 1)], 6);
        assert!(plan
            .negative_depths
            .iter()
            .filter(|(grade, _)| **grade != (1, 0) && **grade != (0, 1))
            .all(|(_, depth)| *depth == 4));
    }

    #[test]
    fn depth_overflow_keeps_the_callers_error_context() {
        let error = plan_birkhoff_windows(
            &positive_degrees(2),
            &BTreeMap::from([(1, 1)]),
            usize::MAX,
            degree_splits,
            "exact caller overflow message",
        )
        .unwrap_err();

        assert_eq!(
            error,
            GwError::UnsupportedInvariant("exact caller overflow message".to_string())
        );
    }
}
