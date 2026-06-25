use gw_pn::error::GwError;
use gw_pn::formula::{build_formula_skeleton, FormulaBasisMode, FormulaExpansion, FormulaRequest};
use gw_pn::geometry::CohomologyClass;
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::testsuite::run_builtin_tests;
use gw_pn::twisted::{
    compute_negative_split_twisted, NegativeSplitBundleTwist, TwistedInvariantRequest,
};
use gw_pn::{
    algebra::Rational, compute, compute_series, tau, ComputeMode, InvariantRequest, SeriesRequest,
};
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), GwError> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return Ok(());
    }

    let command = args.remove(0);
    match command.as_str() {
        "psi" => run_psi(&args),
        "compute" => run_compute(&args),
        "twisted" => run_twisted(&args),
        "degree-series" => run_degree_series(&args),
        "genus-series" => run_genus_series(&args),
        "formula" => run_formula(&args),
        "series" => run_series(&args),
        "tests" | "test" => run_tests(&args),
        _ => Err(GwError::ParseError(format!(
            "unknown command `{command}`; try --help"
        ))),
    }
}

fn run_tests(args: &[String]) -> Result<(), GwError> {
    validate_flags(args, &[])?;
    let report = run_builtin_tests();
    for case in &report.cases {
        let status = if case.passed { "ok" } else { "FAILED" };
        println!("{status} {}", case.name);
        if !case.passed {
            println!("  {}", case.message);
        }
    }
    println!(
        "summary: {} passed, {} failed",
        report.passed(),
        report.failed()
    );
    if report.is_success() {
        Ok(())
    } else {
        Err(GwError::ValidationFailure(format!(
            "{} built-in tests failed",
            report.failed()
        )))
    }
}

fn run_psi(args: &[String]) -> Result<(), GwError> {
    validate_flags(
        args,
        &[
            CliFlag::value("--g"),
            CliFlag::value("--genus"),
            CliFlag::value("--powers"),
        ],
    )?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let powers_raw = parse_string_flag(args, "--powers")?
        .ok_or_else(|| GwError::ParseError("missing --powers".to_string()))?;
    let powers = if powers_raw.trim().is_empty() {
        Vec::new()
    } else {
        powers_raw
            .split(',')
            .map(|part| {
                part.trim().parse::<usize>().map_err(|_| {
                    GwError::ParseError(format!("invalid psi power `{}`", part.trim()))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let value = WittenKontsevich::new().psi_integral(genus, &powers);
    println!("{value}");
    Ok(())
}

fn run_series(args: &[String]) -> Result<(), GwError> {
    validate_flags(
        args,
        &[
            CliFlag::value("--n"),
            CliFlag::value("--g"),
            CliFlag::value("--genus"),
            CliFlag::value("--d-max"),
            CliFlag::value("--degree-max"),
            CliFlag::value("--max-markings"),
            CliFlag::value("--m-max"),
            CliFlag::value("--max-descendant"),
            CliFlag::value("--k-max"),
            CliFlag::value("--mode"),
            CliFlag::switch("--include-zero"),
            CliFlag::switch("--equivariant"),
        ],
    )?;
    let n = required_usize(args, "--n")?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let degree_max = first_usize_flag(args, &["--d-max", "--degree-max"])?
        .ok_or_else(|| GwError::ParseError("missing --d-max".to_string()))?;
    let max_markings = first_usize_flag(args, &["--max-markings", "--m-max"])?
        .ok_or_else(|| GwError::ParseError("missing --max-markings".to_string()))?;
    let max_descendant_power =
        first_usize_flag(args, &["--max-descendant", "--k-max"])?.unwrap_or(0);
    let include_zero = has_flag(args, "--include-zero");
    let mode = match parse_string_flag(args, "--mode")?.as_deref() {
        None | Some("givental") => ComputeMode::Givental,
        Some(other) => {
            return Err(GwError::ParseError(format!(
                "invalid --mode `{other}`; expected givental"
            )))
        }
    };
    let req = SeriesRequest {
        n,
        genus,
        degree_max,
        max_markings,
        max_descendant_power,
        include_zero,
        equivariant: has_flag(args, "--equivariant"),
        mode,
        truncation: None,
    };
    let result = compute_series(req)?;
    for coefficient in result.coefficients {
        println!(
            "q^{} [{}] = {}",
            coefficient.degree,
            coefficient.insertion_label(),
            coefficient.value
        );
    }
    if let Some(path) = write_warnings_file("series", &result.notes)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn run_twisted(args: &[String]) -> Result<(), GwError> {
    validate_flags(
        args,
        &[
            CliFlag::value("--n"),
            CliFlag::value("--twist"),
            CliFlag::value("--g"),
            CliFlag::value("--genus"),
            CliFlag::value("--d"),
            CliFlag::value("--degree"),
            CliFlag::value("--insert"),
            CliFlag::switch("--equivariant"),
        ],
    )?;
    let n = required_usize(args, "--n")?;
    let twist = parse_negative_twist_flag(args, "--twist")?
        .ok_or_else(|| GwError::ParseError("missing --twist".to_string()))?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let degree = first_usize_flag(args, &["--d", "--degree"])?
        .ok_or_else(|| GwError::ParseError("missing --d".to_string()))?;
    let insertions = repeated_string_flag(args, "--insert")
        .into_iter()
        .map(|raw| parse_insertion(n, &raw))
        .collect::<Result<Vec<_>, _>>()?;

    let mut req = TwistedInvariantRequest::new(n, twist, genus, degree, insertions)?;
    req.equivariant = has_flag(args, "--equivariant");
    let result = compute_negative_split_twisted(&req)?;
    println!("{}", result.value);
    for note in result.notes {
        println!("note: {note}");
    }
    Ok(())
}

fn run_formula(args: &[String]) -> Result<(), GwError> {
    validate_flags(
        args,
        &[
            CliFlag::value("--g"),
            CliFlag::value("--genus"),
            CliFlag::value("--markings"),
            CliFlag::value("--m"),
            CliFlag::value("--n"),
            CliFlag::value("--colors"),
            CliFlag::value("--max-descendant"),
            CliFlag::value("--k-max"),
            CliFlag::value("--d"),
            CliFlag::value("--degree"),
            CliFlag::value("--format"),
            CliFlag::value("--twist"),
            CliFlag::value("--basis"),
            CliFlag::switch("--equivariant"),
            CliFlag::switch("--expand"),
            CliFlag::switch("--no-glossary"),
            CliFlag::switch("--tex"),
        ],
    )?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let markings = first_usize_flag(args, &["--markings", "--m"])?
        .ok_or_else(|| GwError::ParseError("missing --markings".to_string()))?;
    let colors_flag = first_usize_flag(args, &["--colors"])?;
    let n_flag = first_usize_flag(args, &["--n"])?;
    let colors = match (colors_flag, n_flag) {
        (Some(_), Some(_)) => {
            return Err(GwError::ParseError(
                "pass either --colors or --n, not both".to_string(),
            ))
        }
        (Some(colors), None) => colors,
        (None, Some(n)) => n + 1,
        (None, None) => return Err(GwError::ParseError("missing --colors or --n".to_string())),
    };
    let mut request = FormulaRequest::new(genus, markings, colors);
    request.max_descendant_power =
        first_usize_flag(args, &["--max-descendant", "--k-max"])?.unwrap_or(0);
    request.q_degree = first_usize_flag(args, &["--d", "--degree"])?;
    request.include_glossary = !has_flag(args, "--no-glossary");
    request.basis = parse_formula_basis(args)?;
    request.expansion =
        formula_expansion_from_args(args, n_flag, colors, request.basis.requires_expansion())?;
    let skeleton = build_formula_skeleton(request)?;
    let format = match (
        parse_string_flag(args, "--format")?,
        has_flag(args, "--tex"),
    ) {
        (Some(_), true) => {
            return Err(GwError::ParseError(
                "pass either --format tex or --tex, not both".to_string(),
            ))
        }
        (Some(format), false) => format,
        (None, true) => "tex".to_string(),
        (None, false) => "text".to_string(),
    };
    match format.as_str() {
        "text" => println!("{}", skeleton.render_text()),
        "tex" | "tex-document" => println!("{}", skeleton.render_tex_document()),
        "tex-fragment" => println!("{}", skeleton.render_tex()),
        other => {
            return Err(GwError::ParseError(format!(
                "invalid --format `{other}`; expected text, tex, tex-document, or tex-fragment"
            )))
        }
    }
    Ok(())
}

fn formula_expansion_from_args(
    args: &[String],
    n_flag: Option<usize>,
    colors: usize,
    force: bool,
) -> Result<Option<FormulaExpansion>, GwError> {
    if !force && !has_flag(args, "--expand") {
        return Ok(None);
    }
    let equivariant = has_flag(args, "--equivariant");
    if let Some(degrees) = parse_negative_twist_flag(args, "--twist")? {
        let n = n_flag.ok_or_else(|| {
            GwError::ParseError("twisted formula expansion needs --n".to_string())
        })?;
        return Ok(Some(FormulaExpansion::NegativeSplitTwisted {
            n,
            degrees,
            equivariant,
        }));
    }
    let n = match n_flag {
        Some(n) => n,
        None => colors.checked_sub(1).ok_or_else(|| {
            GwError::ParseError(
                "projective formula expansion needs --n or positive --colors".to_string(),
            )
        })?,
    };
    Ok(Some(FormulaExpansion::ProjectiveSpace { n, equivariant }))
}

fn parse_formula_basis(args: &[String]) -> Result<FormulaBasisMode, GwError> {
    match parse_string_flag(args, "--basis")?.as_deref() {
        None => Ok(FormulaBasisMode::Raw),
        Some("coefficients") | Some("coefficient") => Ok(FormulaBasisMode::Coefficients),
        Some("raw") => Ok(FormulaBasisMode::Raw),
        Some("rational") => Ok(FormulaBasisMode::Rational),
        Some("resolvent") => Err(GwError::ParseError(
            "invalid --basis `resolvent`; the resolvent display was folded into --basis rational"
                .to_string(),
        )),
        Some(other) => Err(GwError::ParseError(format!(
            "invalid --basis `{other}`; expected coefficients, raw, or rational"
        ))),
    }
}

fn run_degree_series(args: &[String]) -> Result<(), GwError> {
    validate_flags(args, degree_series_flags())?;
    let n = required_usize(args, "--n")?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let degree_max = first_usize_flag(args, &["--d-max", "--degree-max"])?
        .ok_or_else(|| GwError::ParseError("missing --d-max".to_string()))?;
    let twist = parse_negative_twist_flag(args, "--twist")?;
    let degree_min = first_usize_flag(args, &["--d-min", "--degree-min"])?
        .unwrap_or(if twist.is_some() { 1 } else { 0 });
    if degree_min > degree_max {
        return Err(GwError::ParseError(format!(
            "--d-min ({degree_min}) cannot exceed --d-max ({degree_max})"
        )));
    }
    let mode = parse_compute_mode(args)?;
    let equivariant = has_flag(args, "--equivariant");
    let insertion_selection = parse_series_insertions(n, args)?;
    let twist_model = parse_twist_model(twist.as_ref())?;

    let mut warnings = Vec::new();
    match insertion_selection {
        InsertionSelection::Fixed(insertions) => {
            let label = insertion_list_label(&insertions);
            for degree in degree_min..=degree_max {
                match compute_series_point(
                    n,
                    twist.as_deref(),
                    genus,
                    degree,
                    &insertions,
                    equivariant,
                    mode,
                ) {
                    Ok(result) => println!("q^{degree} [{label}] = {}", result.value),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        warnings.push(format!("skipped q^{degree} [{label}]: {msg}"))
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        InsertionSelection::Bounded(scan) => {
            for degree in degree_min..=degree_max {
                for insertions in &scan.profiles {
                    if !dimension_compatible(n, twist_model.as_ref(), genus, degree, insertions) {
                        continue;
                    }
                    let label = insertion_list_label(insertions);
                    match compute_series_point(
                        n,
                        twist.as_deref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                        mode,
                    ) {
                        Ok(result) => {
                            if scan.include_zero || !result.value.is_zero() {
                                println!("q^{degree} [{label}] = {}", result.value);
                            }
                        }
                        Err(GwError::UnsupportedInvariant(msg)) => {
                            warnings.push(format!("skipped q^{degree} [{label}]: {msg}"))
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
        }
    }
    if let Some(path) = write_warnings_file("degree-series", &warnings)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn run_genus_series(args: &[String]) -> Result<(), GwError> {
    validate_flags(args, genus_series_flags())?;
    let n = required_usize(args, "--n")?;
    let degree = first_usize_flag(args, &["--d", "--degree"])?
        .ok_or_else(|| GwError::ParseError("missing --d".to_string()))?;
    let genus_max = first_usize_flag(args, &["--g-max", "--genus-max"])?
        .ok_or_else(|| GwError::ParseError("missing --g-max".to_string()))?;
    let genus_min = first_usize_flag(args, &["--g-min", "--genus-min"])?.unwrap_or(0);
    if genus_min > genus_max {
        return Err(GwError::ParseError(format!(
            "--g-min ({genus_min}) cannot exceed --g-max ({genus_max})"
        )));
    }
    let twist = parse_negative_twist_flag(args, "--twist")?;
    let mode = parse_compute_mode(args)?;
    let equivariant = has_flag(args, "--equivariant");
    let insertion_selection = parse_series_insertions(n, args)?;
    let twist_model = parse_twist_model(twist.as_ref())?;

    let mut warnings = Vec::new();
    match insertion_selection {
        InsertionSelection::Fixed(insertions) => {
            let label = insertion_list_label(&insertions);
            for genus in genus_min..=genus_max {
                match compute_series_point(
                    n,
                    twist.as_deref(),
                    genus,
                    degree,
                    &insertions,
                    equivariant,
                    mode,
                ) {
                    Ok(result) => println!("g={genus} q^{degree} [{label}] = {}", result.value),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        warnings.push(format!("skipped g={genus} q^{degree} [{label}]: {msg}"))
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        InsertionSelection::Bounded(scan) => {
            for genus in genus_min..=genus_max {
                for insertions in &scan.profiles {
                    if !dimension_compatible(n, twist_model.as_ref(), genus, degree, insertions) {
                        continue;
                    }
                    let label = insertion_list_label(insertions);
                    match compute_series_point(
                        n,
                        twist.as_deref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                        mode,
                    ) {
                        Ok(result) => {
                            if scan.include_zero || !result.value.is_zero() {
                                println!("g={genus} q^{degree} [{label}] = {}", result.value);
                            }
                        }
                        Err(GwError::UnsupportedInvariant(msg)) => {
                            warnings.push(format!("skipped g={genus} q^{degree} [{label}]: {msg}"))
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
        }
    }
    if let Some(path) = write_warnings_file("genus-series", &warnings)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn run_compute(args: &[String]) -> Result<(), GwError> {
    validate_flags(
        args,
        &[
            CliFlag::value("--n"),
            CliFlag::value("--g"),
            CliFlag::value("--genus"),
            CliFlag::value("--d"),
            CliFlag::value("--degree"),
            CliFlag::value("--mode"),
            CliFlag::value("--insert"),
            CliFlag::switch("--equivariant"),
            CliFlag::switch("--nonequivariant-limit"),
        ],
    )?;
    let n = required_usize(args, "--n")?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let degree = first_usize_flag(args, &["--d", "--degree"])?
        .ok_or_else(|| GwError::ParseError("missing --d".to_string()))?;
    let mode = parse_compute_mode(args)?;
    let insertions = parse_insertions(n, args)?;

    let req = InvariantRequest {
        n,
        genus,
        degree,
        insertions,
        equivariant: has_flag(args, "--equivariant"),
        mode,
        truncation: None,
    };
    let result = compute(req)?;
    println!("{}", result.value);
    if has_flag(args, "--nonequivariant-limit") {
        let weights = default_lambda_line_weights(n);
        let limit = result.nonequivariant_limit_line(n, &weights)?;
        println!("nonequivariant_limit: {limit}");
    }
    for note in result.notes {
        println!("note: {note}");
    }
    Ok(())
}

fn compute_series_point(
    n: usize,
    twist: Option<&[usize]>,
    genus: usize,
    degree: usize,
    insertions: &[gw_pn::Insertion],
    equivariant: bool,
    mode: ComputeMode,
) -> Result<gw_pn::InvariantResult, GwError> {
    if let Some(twist) = twist {
        let mut req =
            TwistedInvariantRequest::new(n, twist.to_vec(), genus, degree, insertions.to_vec())?;
        req.equivariant = equivariant;
        return compute_negative_split_twisted(&req);
    }

    let req = InvariantRequest {
        n,
        genus,
        degree,
        insertions: insertions.to_vec(),
        equivariant,
        mode,
        truncation: None,
    };
    compute(req)
}

#[derive(Debug, Clone)]
enum InsertionSelection {
    Fixed(Vec<gw_pn::Insertion>),
    Bounded(BoundedInsertionScan),
}

#[derive(Debug, Clone)]
struct BoundedInsertionScan {
    profiles: Vec<Vec<gw_pn::Insertion>>,
    include_zero: bool,
}

#[derive(Debug, Clone, Copy)]
struct CliFlag {
    name: &'static str,
    takes_value: bool,
}

impl CliFlag {
    const fn value(name: &'static str) -> Self {
        Self {
            name,
            takes_value: true,
        }
    }

    const fn switch(name: &'static str) -> Self {
        Self {
            name,
            takes_value: false,
        }
    }
}

const DEGREE_SERIES_FLAGS: &[CliFlag] = &[
    CliFlag::value("--n"),
    CliFlag::value("--g"),
    CliFlag::value("--genus"),
    CliFlag::value("--d-max"),
    CliFlag::value("--degree-max"),
    CliFlag::value("--d-min"),
    CliFlag::value("--degree-min"),
    CliFlag::value("--twist"),
    CliFlag::value("--mode"),
    CliFlag::value("--insert"),
    CliFlag::value("--max-markings"),
    CliFlag::value("--m-max"),
    CliFlag::value("--max-descendant"),
    CliFlag::value("--k-max"),
    CliFlag::switch("--equivariant"),
    CliFlag::switch("--include-zero"),
];

const GENUS_SERIES_FLAGS: &[CliFlag] = &[
    CliFlag::value("--n"),
    CliFlag::value("--d"),
    CliFlag::value("--degree"),
    CliFlag::value("--g-max"),
    CliFlag::value("--genus-max"),
    CliFlag::value("--g-min"),
    CliFlag::value("--genus-min"),
    CliFlag::value("--twist"),
    CliFlag::value("--mode"),
    CliFlag::value("--insert"),
    CliFlag::value("--max-markings"),
    CliFlag::value("--m-max"),
    CliFlag::value("--max-descendant"),
    CliFlag::value("--k-max"),
    CliFlag::switch("--equivariant"),
    CliFlag::switch("--include-zero"),
];

fn degree_series_flags() -> &'static [CliFlag] {
    DEGREE_SERIES_FLAGS
}

fn genus_series_flags() -> &'static [CliFlag] {
    GENUS_SERIES_FLAGS
}

fn validate_flags(args: &[String], allowed: &[CliFlag]) -> Result<(), GwError> {
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(flag) = allowed.iter().find(|candidate| candidate.name == arg) {
            if flag.takes_value {
                let value = args
                    .get(idx + 1)
                    .ok_or_else(|| GwError::ParseError(format!("{arg} requires a value")))?;
                if allowed
                    .iter()
                    .any(|candidate| candidate.name == value.as_str())
                {
                    return Err(GwError::ParseError(format!("{arg} requires a value")));
                }
                idx += 2;
            } else {
                idx += 1;
            }
            continue;
        }

        if arg.starts_with('-') {
            let suggestion = suggest_flag(arg, allowed)
                .map(|candidate| format!("; maybe you meant `{candidate}`"))
                .unwrap_or_default();
            return Err(GwError::ParseError(format!(
                "unknown flag `{arg}`{suggestion}"
            )));
        }

        return Err(GwError::ParseError(format!("unexpected argument `{arg}`")));
    }
    Ok(())
}

fn suggest_flag(arg: &str, allowed: &[CliFlag]) -> Option<&'static str> {
    let (name, distance) = allowed
        .iter()
        .map(|flag| (flag.name, levenshtein(arg, flag.name)))
        .min_by_key(|(_, distance)| *distance)?;
    (distance <= flag_suggestion_threshold(arg)).then_some(name)
}

fn flag_suggestion_threshold(arg: &str) -> usize {
    match arg.len() {
        0..=6 => 1,
        7..=12 => 2,
        _ => 3,
    }
}

fn levenshtein(left: &str, right: &str) -> usize {
    let right_len = right.chars().count();
    let mut previous = (0..=right_len).collect::<Vec<_>>();
    let mut current = vec![0; right_len + 1];

    for (left_idx, left_char) in left.chars().enumerate() {
        current[0] = left_idx + 1;
        for (right_idx, right_char) in right.chars().enumerate() {
            let substitution = previous[right_idx] + usize::from(left_char != right_char);
            let insertion = current[right_idx] + 1;
            let deletion = previous[right_idx + 1] + 1;
            current[right_idx + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_len]
}

fn write_warnings_file(command: &str, warnings: &[String]) -> Result<Option<PathBuf>, GwError> {
    if warnings.is_empty() {
        return Ok(None);
    }

    let mut hasher = DefaultHasher::new();
    command.hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);
    warnings.hash(&mut hasher);
    let hash = hasher.finish();

    let path = env::temp_dir().join(format!("gw-pn-{command}-warnings-{hash:016x}.txt"));
    let mut contents = String::new();
    for warning in warnings {
        contents.push_str("warning: ");
        contents.push_str(warning);
        contents.push('\n');
    }
    fs::write(&path, contents).map_err(|err| {
        GwError::AlgebraFailure(format!(
            "failed to write warnings to {}: {err}",
            path.display()
        ))
    })?;
    Ok(Some(path))
}

fn parse_series_insertions(n: usize, args: &[String]) -> Result<InsertionSelection, GwError> {
    let explicit = parse_insertions(n, args)?;
    if !explicit.is_empty() {
        return Ok(InsertionSelection::Fixed(explicit));
    }

    let max_markings = first_usize_flag(args, &["--max-markings", "--m-max"])?;
    let max_descendant = first_usize_flag(args, &["--max-descendant", "--k-max"])?;
    if max_markings.is_none() {
        if max_descendant.is_some() {
            return Err(GwError::ParseError(
                "--max-descendant requires --max-markings".to_string(),
            ));
        }
        return Ok(InsertionSelection::Fixed(Vec::new()));
    }

    Ok(InsertionSelection::Bounded(BoundedInsertionScan {
        profiles: bounded_insertion_profiles(n, max_markings.unwrap(), max_descendant.unwrap_or(0)),
        include_zero: has_flag(args, "--include-zero"),
    }))
}

fn parse_insertions(n: usize, args: &[String]) -> Result<Vec<gw_pn::Insertion>, GwError> {
    repeated_string_flag(args, "--insert")
        .into_iter()
        .map(|raw| parse_insertion(n, &raw))
        .collect()
}

fn parse_insertion(n: usize, raw: &str) -> Result<gw_pn::Insertion, GwError> {
    let compact = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let open = compact
        .find('(')
        .ok_or_else(|| GwError::ParseError(format!("invalid insertion `{raw}`")))?;
    let close = compact
        .strip_suffix(')')
        .ok_or_else(|| GwError::ParseError(format!("invalid insertion `{raw}`")))?;
    let tau_part = &close[..open];
    let class_part = &close[open + 1..];
    let descendant_power = tau_part
        .strip_prefix("tau")
        .ok_or_else(|| GwError::ParseError(format!("invalid insertion `{raw}`")))?
        .parse::<usize>()
        .map_err(|_| GwError::ParseError(format!("invalid insertion `{raw}`")))?;
    let class = parse_class(n, class_part)?;
    Ok(tau(descendant_power, class))
}

fn parse_negative_twist_flag(args: &[String], flag: &str) -> Result<Option<Vec<usize>>, GwError> {
    let Some(raw) = parse_string_flag(args, flag)? else {
        return Ok(None);
    };
    let degrees = raw
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(GwError::ParseError(format!(
                    "empty twist degree in {flag} value `{raw}`"
                )));
            }
            let degree = part.parse::<isize>().map_err(|_| {
                GwError::ParseError(format!("invalid twist degree `{part}` in {flag}"))
            })?;
            if degree >= 0 {
                return Err(GwError::ParseError(format!(
                    "negative split bundles must be written with negative degrees, e.g. `{flag} -3` or `{flag} -1,-1`; got `{part}`"
                )));
            }
            degree
                .checked_abs()
                .map(|value| value as usize)
                .ok_or_else(|| {
                    GwError::ParseError(format!("invalid twist degree `{part}` in {flag}"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(degrees))
}

fn parse_twist_model(
    twist: Option<&Vec<usize>>,
) -> Result<Option<NegativeSplitBundleTwist>, GwError> {
    twist
        .map(|degrees| NegativeSplitBundleTwist::new(degrees.clone()))
        .transpose()
}

fn parse_compute_mode(args: &[String]) -> Result<ComputeMode, GwError> {
    match parse_string_flag(args, "--mode")?.as_deref() {
        None | Some("givental") => Ok(ComputeMode::Givental),
        Some(other) => Err(GwError::ParseError(format!(
            "invalid --mode `{other}`; expected givental"
        ))),
    }
}

fn parse_class(n: usize, raw: &str) -> Result<CohomologyClass, GwError> {
    match raw {
        "1" => Ok(CohomologyClass::one(n)),
        "H" => Ok(CohomologyClass::h_power(n, 1)),
        _ => {
            let power = raw
                .strip_prefix("H^")
                .ok_or_else(|| GwError::ParseError(format!("invalid class `{raw}`")))?
                .parse::<usize>()
                .map_err(|_| GwError::ParseError(format!("invalid class `{raw}`")))?;
            Ok(CohomologyClass::h_power(n, power))
        }
    }
}

fn required_usize(args: &[String], flag: &str) -> Result<usize, GwError> {
    parse_usize_flag(args, flag)?.ok_or_else(|| GwError::ParseError(format!("missing {flag}")))
}

fn first_usize_flag(args: &[String], flags: &[&str]) -> Result<Option<usize>, GwError> {
    for flag in flags {
        if let Some(value) = parse_usize_flag(args, flag)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn parse_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>, GwError> {
    parse_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| GwError::ParseError(format!("invalid value `{value}` for {flag}")))
        })
        .transpose()
}

fn parse_string_flag(args: &[String], flag: &str) -> Result<Option<String>, GwError> {
    let mut idx = 0;
    while idx < args.len() {
        if args[idx] == flag {
            return args
                .get(idx + 1)
                .cloned()
                .map(Some)
                .ok_or_else(|| GwError::ParseError(format!("{flag} requires a value")));
        }
        idx += 1;
    }
    Ok(None)
}

fn repeated_string_flag(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                out.push(value.clone());
            }
        }
        idx += 1;
    }
    out
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn default_lambda_line_weights(n: usize) -> Vec<Rational> {
    let mut weights = Vec::with_capacity(n + 1);
    let mut value = 1usize;
    for _ in 0..=n {
        weights.push(Rational::from(value));
        value = value.saturating_mul(2);
    }
    weights
}

fn bounded_insertion_profiles(
    n: usize,
    max_markings: usize,
    max_descendant_power: usize,
) -> Vec<Vec<gw_pn::Insertion>> {
    let mut basis = Vec::new();
    for descendant_power in 0..=max_descendant_power {
        for h_power in 0..=n {
            basis.push(tau(descendant_power, CohomologyClass::h_power(n, h_power)));
        }
    }

    let mut profiles = Vec::new();
    for markings in 0..=max_markings {
        collect_insertion_profiles(&basis, markings, 0, &mut Vec::new(), &mut profiles);
    }
    profiles
}

fn collect_insertion_profiles(
    basis: &[gw_pn::Insertion],
    markings: usize,
    start: usize,
    current: &mut Vec<gw_pn::Insertion>,
    out: &mut Vec<Vec<gw_pn::Insertion>>,
) {
    if current.len() == markings {
        out.push(current.clone());
        return;
    }
    for idx in start..basis.len() {
        current.push(basis[idx].clone());
        collect_insertion_profiles(basis, markings, idx, current, out);
        current.pop();
    }
}

fn dimension_compatible(
    n: usize,
    twist: Option<&NegativeSplitBundleTwist>,
    genus: usize,
    degree: usize,
    insertions: &[gw_pn::Insertion],
) -> bool {
    let Some(total_degree) = insertion_degree(insertions) else {
        return true;
    };
    let virtual_dimension = match twist {
        Some(twist) => twist.virtual_dimension(n, genus, degree, insertions.len()),
        None => ordinary_virtual_dimension(n, genus, degree, insertions.len()),
    };
    virtual_dimension >= 0 && total_degree as isize == virtual_dimension
}

fn ordinary_virtual_dimension(n: usize, genus: usize, degree: usize, markings: usize) -> isize {
    (1 - genus as isize) * (n as isize - 3) + (n + 1) as isize * degree as isize + markings as isize
}

fn insertion_degree(insertions: &[gw_pn::Insertion]) -> Option<usize> {
    let mut total = 0usize;
    for insertion in insertions {
        total = total.checked_add(insertion.descendant_power)?;
        total = total.checked_add(insertion.class.pure_power()?)?;
    }
    Some(total)
}

fn insertion_list_label(insertions: &[gw_pn::Insertion]) -> String {
    if insertions.is_empty() {
        return "1".to_string();
    }
    insertions
        .iter()
        .map(|insertion| {
            let class = match insertion.class.pure_power() {
                Some(0) => "1".to_string(),
                Some(1) => "H".to_string(),
                Some(power) => format!("H^{power}"),
                None => "class".to_string(),
            };
            format!("tau{}({class})", insertion.descendant_power)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_help() {
    println!(
        "gw-pn\n\
\n\
Commands:\n\
  gw-pn tests\n\
  gw-pn psi --g 2 --powers 4\n\
  gw-pn compute --n 2 --g 0 --d 1 --insert 'tau0(H^2)' --insert 'tau0(H^2)' --insert 'tau0(H)' --mode givental\n\
  gw-pn twisted --n 2 --twist -1 --g 2 --d 2 --insert 'tau4(H)'\n\
  gw-pn twisted --n 2 --twist -3 --g 2 --d 3\n\
  gw-pn formula --n 2 --g 2 --markings 1 --max-descendant 5 --d 3\n\
  gw-pn formula --n 2 --g 2 --markings 1 --max-descendant 5 --format tex\n\
  gw-pn formula --n 2 --g 2 --markings 1 --basis rational --format tex-fragment\n\
  gw-pn formula --n 2 --g 2 --markings 1 --basis raw --format tex\n\
  gw-pn formula --n 2 --g 2 --markings 1 --twist -3 --basis raw --format tex\n\
  gw-pn formula --n 2 --g 2 --markings 1 --max-descendant 5 --format tex-fragment\n\
  gw-pn degree-series --n 2 --twist -3 --g 2 --d-max 3\n\
  gw-pn degree-series --n 2 --twist -1 --g 2 --d-max 2 --max-markings 1 --max-descendant 5\n\
  gw-pn genus-series --n 2 --twist -3 --d 1 --g-max 3\n\
  gw-pn genus-series --n 2 --twist -1 --d 2 --g-max 2 --max-markings 1 --max-descendant 5\n\
  gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3 --mode givental\n\
\n\
Supported compute seed cases:\n\
  P^0 point-theory psi integrals, genus-zero degree-zero constants,\n\
  and genus-zero three-point primary small quantum products."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn flag_validation_suggests_close_match() {
        let err = validate_flags(
            &args(&[
                "--n",
                "2",
                "--g",
                "2",
                "--d-max",
                "3",
                "--max-descendants",
                "5",
            ]),
            degree_series_flags(),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("maybe you meant `--max-descendant`"));
    }

    #[test]
    fn flag_validation_accepts_negative_twist_value() {
        validate_flags(
            &args(&["--n", "1", "--twist", "-1,-1", "--g", "2", "--d", "1"]),
            &[
                CliFlag::value("--n"),
                CliFlag::value("--twist"),
                CliFlag::value("--g"),
                CliFlag::value("--d"),
            ],
        )
        .unwrap();
    }

    #[test]
    fn flag_validation_rejects_unused_positional_arguments() {
        let err = validate_flags(&args(&["extra"]), &[]).unwrap_err();
        assert!(err.to_string().contains("unexpected argument `extra`"));
    }

    #[test]
    fn formula_expansion_ignores_twist_without_expand() {
        let expansion =
            formula_expansion_from_args(&args(&["--twist", "-3"]), Some(2), 3, false).unwrap();
        assert_eq!(expansion, None);
    }

    #[test]
    fn formula_expansion_infers_projective_with_expand() {
        let expansion =
            formula_expansion_from_args(&args(&["--expand"]), Some(2), 3, false).unwrap();
        assert_eq!(
            expansion,
            Some(FormulaExpansion::ProjectiveSpace {
                n: 2,
                equivariant: false,
            })
        );
    }

    #[test]
    fn formula_expansion_infers_twisted_with_expand_and_twist() {
        let expansion =
            formula_expansion_from_args(&args(&["--expand", "--twist", "-3"]), Some(2), 3, false)
                .unwrap();
        assert_eq!(
            expansion,
            Some(FormulaExpansion::NegativeSplitTwisted {
                n: 2,
                degrees: vec![3],
                equivariant: false,
            })
        );
    }

    #[test]
    fn formula_basis_raw_forces_expansion() {
        assert_eq!(
            parse_formula_basis(&args(&["--basis", "raw"])).unwrap(),
            FormulaBasisMode::Raw
        );
        let expansion =
            formula_expansion_from_args(&args(&["--basis", "raw"]), Some(2), 3, true).unwrap();
        assert!(matches!(
            expansion,
            Some(FormulaExpansion::ProjectiveSpace {
                n: 2,
                equivariant: false
            })
        ));
    }

    #[test]
    fn formula_basis_coefficients_is_legacy_unrolled_mode() {
        assert_eq!(
            parse_formula_basis(&args(&["--basis", "coefficients"])).unwrap(),
            FormulaBasisMode::Coefficients
        );
    }

    #[test]
    fn formula_basis_resolvent_points_to_rational() {
        let err = parse_formula_basis(&args(&["--basis", "resolvent"])).unwrap_err();
        assert!(err
            .to_string()
            .contains("resolvent display was folded into --basis rational"));
    }
}
