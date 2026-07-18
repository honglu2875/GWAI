//! Target-neutral stability facts and finite recursion envelopes for pointed
//! curves.

/// Work bound for the branching descendant divisor recursion used when the
/// underlying pointed curve is unstable.
///
/// Stable graph evaluation has its own, larger bounds; this limit controls
/// only the lattice of correction terms in unstable divisor recursion.
pub const MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI: usize = 8;

/// Stability of a complete connected pointed curve (or a stable-graph
/// vertex), without evaluating the potentially overflowing expression
/// `2g + n > 2`.
pub const fn pointed_curve_is_stable(genus: usize, markings: usize) -> bool {
    match genus {
        0 => markings >= 3,
        1 => markings >= 1,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::pointed_curve_is_stable;

    #[test]
    fn pointed_curve_stability_covers_the_boundary() {
        assert!(!pointed_curve_is_stable(0, 2));
        assert!(pointed_curve_is_stable(0, 3));
        assert!(!pointed_curve_is_stable(1, 0));
        assert!(pointed_curve_is_stable(1, 1));
        assert!(pointed_curve_is_stable(2, 0));
    }
}
