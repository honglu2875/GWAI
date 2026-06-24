use gw_pn::error::GwError;
use gw_pn::geometry::CohomologyClass;
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::testsuite::run_builtin_tests;
use gw_pn::twisted::{compute_negative_split_twisted, TwistedInvariantRequest};
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
        "series" => run_series(&args),
        "tests" | "test" => run_tests(),
        _ => Err(GwError::ParseError(format!(
            "unknown command `{command}`; try --help"
        ))),
    }
}

fn run_tests() -> Result<(), GwError> {
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

fn run_degree_series(args: &[String]) -> Result<(), GwError> {
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
    let insertions = parse_insertions(n, args)?;
    let label = insertion_list_label(&insertions);

    let mut warnings = Vec::new();
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
    if let Some(path) = write_warnings_file("degree-series", &warnings)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn run_genus_series(args: &[String]) -> Result<(), GwError> {
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
    let insertions = parse_insertions(n, args)?;
    let label = insertion_list_label(&insertions);

    let mut warnings = Vec::new();
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
    if let Some(path) = write_warnings_file("genus-series", &warnings)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn run_compute(args: &[String]) -> Result<(), GwError> {
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
  gw-pn degree-series --n 2 --twist -3 --g 2 --d-max 3\n\
  gw-pn genus-series --n 2 --twist -3 --d 1 --g-max 3\n\
  gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3 --mode givental\n\
\n\
Supported compute seed cases:\n\
  P^0 point-theory psi integrals, genus-zero degree-zero constants,\n\
  and genus-zero three-point primary small quantum products."
    );
}
