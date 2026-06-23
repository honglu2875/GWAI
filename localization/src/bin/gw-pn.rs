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
    let twist = parse_degrees_flag(args, "--twist")?
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

fn run_compute(args: &[String]) -> Result<(), GwError> {
    let n = required_usize(args, "--n")?;
    let genus = first_usize_flag(args, &["--g", "--genus"])?
        .ok_or_else(|| GwError::ParseError("missing --g".to_string()))?;
    let degree = first_usize_flag(args, &["--d", "--degree"])?
        .ok_or_else(|| GwError::ParseError("missing --d".to_string()))?;
    let mode = match parse_string_flag(args, "--mode")?.as_deref() {
        None | Some("givental") => ComputeMode::Givental,
        Some(other) => {
            return Err(GwError::ParseError(format!(
                "invalid --mode `{other}`; expected givental"
            )))
        }
    };

    let insertions = repeated_string_flag(args, "--insert")
        .into_iter()
        .map(|raw| parse_insertion(n, &raw))
        .collect::<Result<Vec<_>, _>>()?;

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

fn parse_degrees_flag(args: &[String], flag: &str) -> Result<Option<Vec<usize>>, GwError> {
    let Some(raw) = parse_string_flag(args, flag)? else {
        return Ok(None);
    };
    let degrees = raw
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(GwError::ParseError(format!(
                    "empty degree in {flag} value `{raw}`"
                )));
            }
            part.parse::<usize>()
                .map_err(|_| GwError::ParseError(format!("invalid degree `{part}` in {flag}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(degrees))
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

fn print_help() {
    println!(
        "gw-pn\n\
\n\
Commands:\n\
  gw-pn tests\n\
  gw-pn psi --g 2 --powers 4\n\
  gw-pn compute --n 2 --g 0 --d 1 --insert 'tau0(H^2)' --insert 'tau0(H^2)' --insert 'tau0(H)' --mode givental\n\
  gw-pn twisted --n 2 --twist 1 --g 2 --d 2 --insert 'tau4(H)'\n\
  gw-pn twisted --n 2 --twist 3 --g 2 --d 3\n\
  gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3 --mode givental\n\
\n\
Supported compute seed cases:\n\
  P^0 point-theory psi integrals, genus-zero degree-zero constants,\n\
  and genus-zero three-point primary small quantum products."
    );
}
