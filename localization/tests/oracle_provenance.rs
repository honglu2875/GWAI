use std::collections::{BTreeMap, BTreeSet};

const MANIFEST: &str = include_str!("../docs/oracle-provenance.tsv");
const HEADER: [&str; 13] = [
    "id",
    "target_family",
    "evidence",
    "production_exercised",
    "default_ci",
    "production_path",
    "oracle_path",
    "cargo_target",
    "cargo_test",
    "source_locator",
    "source",
    "shared_code",
    "claim",
];

#[derive(Debug)]
struct Row<'a> {
    id: &'a str,
    target_family: &'a str,
    evidence: &'a str,
    production_exercised: &'a str,
    default_ci: &'a str,
    cargo_target: &'a str,
    cargo_test: &'a str,
    source_locator: &'a str,
    shared_code: &'a str,
}

fn rows() -> Vec<Row<'static>> {
    let mut lines = MANIFEST.lines();
    assert_eq!(
        lines.next().unwrap().split('\t').collect::<Vec<_>>(),
        HEADER,
        "oracle provenance schema changed without updating its assertions"
    );
    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(
                fields.len(),
                HEADER.len(),
                "oracle row must have {} tab-separated fields: {line}",
                HEADER.len()
            );
            assert!(
                fields.iter().all(|field| !field.trim().is_empty()),
                "oracle fields must be nonempty: {line}"
            );
            Row {
                id: fields[0],
                target_family: fields[1],
                evidence: fields[2],
                production_exercised: fields[3],
                default_ci: fields[4],
                cargo_target: fields[7],
                cargo_test: fields[8],
                source_locator: fields[9],
                shared_code: fields[11],
            }
        })
        .collect()
}

#[test]
fn oracle_provenance_manifest_is_well_formed_and_gaps_are_explicit() {
    const EVIDENCE: [&str; 10] = [
        "independent_algorithm",
        "independent_formula",
        "external_golden",
        "closed_form",
        "oracle_inventory",
        "shared_production",
        "deformation_cross_backend",
        "shared_cross_path",
        "coverage_gap",
        "unsourced_golden",
    ];
    const TARGET_FAMILIES: [&str; 5] = [
        "projective_space",
        "negative_split_projective",
        "projective_bundle",
        "product_projective",
        "point_theory",
    ];
    const SHARED_CODE: [&str; 5] = [
        "none",
        "conventions_only",
        "target_theory_data",
        "same_seed",
        "universal_graph",
    ];
    let rows = rows();
    let mut ids = BTreeSet::new();
    for row in &rows {
        assert!(ids.insert(row.id), "duplicate oracle id {}", row.id);
        assert!(
            TARGET_FAMILIES.contains(&row.target_family),
            "unknown target family {} for {}",
            row.target_family,
            row.id
        );
        assert!(
            EVIDENCE.contains(&row.evidence),
            "unknown evidence kind {} for {}",
            row.evidence,
            row.id
        );
        assert!(
            matches!(row.production_exercised, "yes" | "no"),
            "invalid production_exercised value for {}",
            row.id
        );
        assert!(
            matches!(row.default_ci, "yes" | "no"),
            "invalid default_ci value for {}",
            row.id
        );
        assert!(
            SHARED_CODE.contains(&row.shared_code),
            "unknown shared-code label {} for {}",
            row.shared_code,
            row.id
        );
        assert!(
            row.cargo_target == "lib"
                || ["test:", "bin:", "example:", "bench:"]
                    .iter()
                    .any(|prefix| {
                        row.cargo_target
                            .strip_prefix(prefix)
                            .is_some_and(|name| !name.is_empty())
                    }),
            "invalid cargo_target {} for {}",
            row.cargo_target,
            row.id
        );
        assert!(
            !row.cargo_test.trim().is_empty(),
            "empty cargo_test for {}",
            row.id
        );
        assert!(
            row.source_locator.starts_with("src/")
                || row.source_locator.starts_with("tests/")
                || row.source_locator.starts_with("docs/"),
            "source locator for {} must be package-relative",
            row.id
        );
        if matches!(
            row.evidence,
            "independent_algorithm" | "independent_formula"
        ) {
            assert_eq!(row.production_exercised, "yes", "{}", row.id);
            assert!(
                matches!(row.shared_code, "conventions_only" | "target_theory_data"),
                "algorithmic independence for {} may share conventions or canonical target data, not a reconstruction engine",
                row.id,
            );
        }
        if row.evidence == "coverage_gap" {
            assert_eq!(row.production_exercised, "no", "{}", row.id);
        }
    }

    let independent_default = rows
        .iter()
        .filter(|row| {
            row.production_exercised == "yes"
                && row.default_ci == "yes"
                && matches!(
                    row.evidence,
                    "independent_algorithm"
                        | "independent_formula"
                        | "external_golden"
                        | "closed_form"
                )
        })
        .fold(BTreeMap::<&str, usize>::new(), |mut counts, row| {
            *counts.entry(row.target_family).or_default() += 1;
            counts
        });
    for family in [
        "projective_space",
        "negative_split_projective",
        "projective_bundle",
        "point_theory",
    ] {
        assert!(
            independent_default.get(family).copied().unwrap_or(0) > 0,
            "lost default independent oracle coverage for {family}"
        );
    }

    let product_rows = rows
        .iter()
        .filter(|row| row.target_family == "product_projective")
        .collect::<Vec<_>>();
    let product_has_independent_default = product_rows.iter().any(|row| {
        row.production_exercised == "yes"
            && row.default_ci == "yes"
            && matches!(
                row.evidence,
                "independent_algorithm" | "independent_formula" | "external_golden" | "closed_form"
            )
    });
    assert!(
        product_has_independent_default
            || product_rows
                .iter()
                .any(|row| row.evidence == "coverage_gap"),
        "product-space independent coverage must exist or remain an explicit gap"
    );
}
