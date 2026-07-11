use clap::{Args, Parser, Subcommand};
use gw_pn::error::GwError;
use gw_pn::formula::{build_formula_skeleton, FormulaBasisMode, FormulaExpansion, FormulaRequest};
use gw_pn::geometry::CohomologyClass;
use gw_pn::givental::{
    bidegree_dimension_matches, reconstruct_bidegree_invariants, reconstruct_bundle_invariants,
    BundleInsertion, ProductInsertion,
};
use gw_pn::resolvent::{compute_resolvent_generating_function, ResolventRequest};
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::testsuite::run_builtin_tests;
use gw_pn::twisted::{
    compute_negative_split_twisted, compute_negative_split_twisted_factored,
    compute_negative_split_twisted_resolvent_packed,
    compute_negative_split_twisted_resolvent_packed_factored, NegativeSplitBundleTwist,
    TwistedInvariantRequest,
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

const EXAMPLES: &str = "Examples:
  gw-pn tests
  gw-pn psi --g 2 --powers 4
  gw-pn compute --n 2 --g 0 --d 1 --insert 'tau0(H^2)' --insert 'tau0(H^2)' --insert 'tau0(H)'\n  gw-pn compute --n 2 --g 0 --insert H^2 --insert H^2 --insert H
  gw-pn twisted --n 2 --twist -1 --g 2 --d 2 --insert 'tau4(H)'
  gw-pn twisted --n 2 --twist -3 --g 2 --d 3
  gw-pn twisted --n 2 --twist -1 --g 0 --d 1 --insert 'tau1(H^2)' --insert 'tau0(H)' --equivariant
  gw-pn formula --n 2 --g 2 --markings 1 --max-descendant 5 --d 3
  gw-pn formula --n 2 --g 2 --markings 1 --basis raw --format tex
  gw-pn resolvent --n 2 --g 0 --d 1 --markings 3
  gw-pn degree-series --n 2 --twist -3 --g 2 --d-max 3
  gw-pn genus-series --n 2 --twist -3 --d 1 --g-max 3
  gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3";

#[derive(Debug, Parser)]
#[command(
    name = "gw-pn",
    about = "Exact computations for Gromov-Witten invariants of projective space",
    arg_required_else_help = true,
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Witten-Kontsevich psi (descendant) intersection numbers
    Psi(PsiArgs),
    /// One ordinary P^n invariant through the Givental/S/R path
    Compute(ComputeArgs),
    /// Negative split-bundle twisted invariants
    Twisted(TwistedArgs),
    /// Vary the degree for a fixed or bounded insertion profile
    DegreeSeries(DegreeSeriesArgs),
    /// Vary the genus for a fixed or bounded insertion profile
    GenusSeries(GenusSeriesArgs),
    /// Human-readable stable-graph formula skeleton (text or TeX)
    Formula(FormulaArgs),
    /// Fixed-degree labelled resolvent generating function
    Resolvent(ResolventArgs),
    /// Bounded sparse descendant potential for ordinary P^n
    Series(SeriesArgs),
    /// P^n x P^m invariants by exact Novikov ray reconstruction
    Product(ProductArgs),
    /// Projective bundle P(O(a_1)+...+O(a_m)) over P^n invariants
    Bundle(BundleArgs),
    /// Run the built-in validation suite
    #[command(alias = "test")]
    Tests,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:
  gw-pn product --n 1 --m 1 --g 0 --d 2 --insert 'tau0(H1*H2)' --insert 'tau0(H1*H2)' --insert 'tau0(H1*H2)'
  gw-pn product --n 1 --m 1 --g 0 --d 1 --insert H1*H2 --insert H1 --insert H1
  gw-pn product --n 1 --m 2 --g 0 --d 0 --insert H1 --insert H2^2 --insert 1

Insertions are tauK(CLASS) with CLASS a product of H1^a and H2^b (or 1);
a bare CLASS means tau0.  --d is the total degree d1 + d2; the command
reports every bidegree.  Values are the non-equivariant invariants,
computed at the given (or default) rational equivariant weights and
reconstructed exactly from d + 1 Novikov rays.")]
struct ProductArgs {
    /// Dimension of the first factor P^n
    #[arg(long)]
    n: usize,
    /// Dimension of the second factor P^m
    #[arg(long)]
    m: usize,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    /// Total curve degree d1 + d2
    #[arg(long, visible_alias = "degree")]
    d: usize,
    /// Insertion tauK(CLASS) with CLASS a product of H1^a, H2^b; repeat for
    /// multiple markings
    #[arg(long)]
    insert: Vec<String>,
    /// Comma-separated integer equivariant weights for the first factor
    /// (n + 1 values, pairwise distinct)
    #[arg(long, allow_hyphen_values = true)]
    weights_x: Option<String>,
    /// Comma-separated integer equivariant weights for the second factor
    #[arg(long, allow_hyphen_values = true)]
    weights_y: Option<String>,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:
  gw-pn bundle --n 1 --twists 0,1 --g 0 --d 0 --insert H --insert xi --insert 1
  gw-pn bundle --n 1 --twists 0,1 --g 0 --d 1 --insert xi --insert xi --insert xi

The bundle is P(O(a_1)+...+O(a_m)) over P^n; --twists are the a_l (any
integers, internally normalized so min a_l = 0 since P(E) = P(E tensor L)).
Insertions are tauK(CLASS) with CLASS a *-product of H^p and xi^q factors
(or 1); a bare CLASS means tau0.  --d is the SHIFTED total degree
d1 + (d2 + (max a) d1); the command reports every curve class (d1, d2) in
that slice, with d2 = xi . beta possibly negative.  Values are the
non-equivariant invariants, reconstructed exactly from d + 1 Novikov rays.")]
struct BundleArgs {
    /// Dimension of the base P^n
    #[arg(long)]
    n: usize,
    /// Comma-separated bundle twists a_1,...,a_m (integers)
    #[arg(long, allow_hyphen_values = true)]
    twists: String,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    /// Shifted total degree d1 + (d2 + (max a) d1)
    #[arg(long, visible_alias = "degree")]
    d: usize,
    /// Insertion tauK(CLASS) with CLASS a *-product of H^p, xi^q; repeat for
    /// multiple markings
    #[arg(long)]
    insert: Vec<String>,
    /// Comma-separated integer equivariant weights for the base (n+1 values)
    #[arg(long, allow_hyphen_values = true)]
    weights_base: Option<String>,
    /// Comma-separated integer equivariant weights for the fiber summands
    /// (m values)
    #[arg(long, allow_hyphen_values = true)]
    weights_fiber: Option<String>,
}

#[derive(Debug, Args)]
struct PsiArgs {
    #[arg(long, visible_alias = "genus")]
    g: usize,
    /// Comma-separated psi powers, e.g. 4 or 0,0,0
    #[arg(long)]
    powers: String,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:
  gw-pn compute --n 2 --g 0 --d 1 --insert 'tau0(H^2)' --insert 'tau0(H^2)' --insert 'tau0(H)'\n  gw-pn compute --n 2 --g 0 --insert H^2 --insert H^2 --insert H
  gw-pn compute --n 2 --g 0 --insert H^2 --insert H^2 --insert H     (degree inferred, tau0 implied)
  gw-pn compute --n 1 --g 2 --insert 'tau4(H)'                       (degree inferred: 1)")]
struct ComputeArgs {
    #[arg(long)]
    n: usize,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    /// Curve degree; omit to infer it from the dimension constraint
    #[arg(long, visible_alias = "degree")]
    d: Option<usize>,
    #[arg(long)]
    mode: Option<String>,
    /// Insertion tauK(CLASS) with CLASS one of 1, H, H^p; a bare CLASS means
    /// tau0(CLASS); repeat for multiple markings
    #[arg(long)]
    insert: Vec<String>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long = "nonequivariant-limit")]
    nonequivariant_limit: bool,
}

#[derive(Debug, Args)]
struct TwistedArgs {
    #[arg(long)]
    n: usize,
    /// Negative split degrees, e.g. -1 or -1,-1
    #[arg(long, allow_hyphen_values = true)]
    twist: String,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "degree")]
    d: usize,
    #[arg(long)]
    insert: Vec<String>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long)]
    factored: bool,
}

#[derive(Debug, Args)]
struct DegreeSeriesArgs {
    #[arg(long)]
    n: usize,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "degree-max")]
    d_max: usize,
    #[arg(long, visible_alias = "degree-min")]
    d_min: Option<usize>,
    #[arg(long, allow_hyphen_values = true)]
    twist: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    insert: Vec<String>,
    #[arg(long, visible_alias = "m-max")]
    max_markings: Option<usize>,
    #[arg(long, visible_alias = "k-max")]
    max_descendant: Option<usize>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long)]
    include_zero: bool,
}

#[derive(Debug, Args)]
struct GenusSeriesArgs {
    #[arg(long)]
    n: usize,
    #[arg(long, visible_alias = "degree")]
    d: usize,
    #[arg(long, visible_alias = "genus-max")]
    g_max: usize,
    #[arg(long, visible_alias = "genus-min")]
    g_min: Option<usize>,
    #[arg(long, allow_hyphen_values = true)]
    twist: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    insert: Vec<String>,
    #[arg(long, visible_alias = "m-max")]
    max_markings: Option<usize>,
    #[arg(long, visible_alias = "k-max")]
    max_descendant: Option<usize>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long)]
    include_zero: bool,
}

#[derive(Debug, Args)]
struct FormulaArgs {
    #[arg(long, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "m")]
    markings: usize,
    #[arg(long)]
    n: Option<usize>,
    #[arg(long)]
    colors: Option<usize>,
    #[arg(long, visible_alias = "k-max")]
    max_descendant: Option<usize>,
    #[arg(long, visible_alias = "degree")]
    d: Option<usize>,
    #[arg(long)]
    format: Option<String>,
    #[arg(long, allow_hyphen_values = true)]
    twist: Option<String>,
    #[arg(long)]
    basis: Option<String>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long)]
    expand: bool,
    #[arg(long)]
    no_glossary: bool,
    #[arg(long)]
    tex: bool,
}

#[derive(Debug, Args)]
struct ResolventArgs {
    #[arg(long)]
    n: usize,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "degree")]
    d: usize,
    #[arg(long, visible_alias = "m")]
    markings: usize,
    #[arg(long, allow_hyphen_values = true)]
    twist: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    equivariant: bool,
    #[arg(long)]
    validate: bool,
}

#[derive(Debug, Args)]
struct SeriesArgs {
    #[arg(long)]
    n: usize,
    #[arg(long, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "degree-max")]
    d_max: usize,
    #[arg(long, visible_alias = "m-max")]
    max_markings: usize,
    #[arg(long, visible_alias = "k-max")]
    max_descendant: Option<usize>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    include_zero: bool,
    #[arg(long)]
    equivariant: bool,
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli.command) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run(command: Commands) -> Result<(), GwError> {
    match command {
        Commands::Psi(args) => run_psi(args),
        Commands::Compute(args) => run_compute(args),
        Commands::Twisted(args) => run_twisted(args),
        Commands::DegreeSeries(args) => run_degree_series(args),
        Commands::GenusSeries(args) => run_genus_series(args),
        Commands::Formula(args) => run_formula(args),
        Commands::Resolvent(args) => run_resolvent(args),
        Commands::Series(args) => run_series(args),
        Commands::Product(args) => run_product(args),
        Commands::Bundle(args) => run_bundle(args),
        Commands::Tests => run_tests(),
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

fn run_psi(args: PsiArgs) -> Result<(), GwError> {
    let powers = if args.powers.trim().is_empty() {
        Vec::new()
    } else {
        args.powers
            .split(',')
            .map(|part| {
                part.trim().parse::<usize>().map_err(|_| {
                    GwError::ParseError(format!("invalid psi power `{}`", part.trim()))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let value = WittenKontsevich::new().psi_integral(args.g, &powers);
    println!("{value}");
    Ok(())
}

fn run_series(args: SeriesArgs) -> Result<(), GwError> {
    let req = SeriesRequest {
        n: args.n,
        genus: args.g,
        degree_max: args.d_max,
        max_markings: args.max_markings,
        max_descendant_power: args.max_descendant.unwrap_or(0),
        include_zero: args.include_zero,
        equivariant: args.equivariant,
        mode: parse_compute_mode(args.mode.as_deref())?,
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

fn run_twisted(args: TwistedArgs) -> Result<(), GwError> {
    let twist = parse_negative_twist(&args.twist)?;
    let insertions = parse_insertions(args.n, &args.insert)?;

    let mut req = TwistedInvariantRequest::new(args.n, twist, args.g, args.d, insertions)?;
    req.equivariant = args.equivariant;
    if args.factored && !req.equivariant {
        return Err(GwError::ParseError(
            "--factored for twisted computations currently requires --equivariant".to_string(),
        ));
    }
    if req.equivariant {
        print_fiber_parameters(&req.twist);
        let value = compute_negative_split_twisted_factored(&req)?;
        println!("{value}");
        return Ok(());
    }
    let result = compute_negative_split_twisted(&req)?;
    println!("{}", result.value);
    Ok(())
}

fn print_fiber_parameters(twist: &NegativeSplitBundleTwist) {
    let parameters = twist
        .degrees()
        .iter()
        .enumerate()
        .map(|(idx, degree)| format!("mu_{idx}=fiber weight of O(-{degree})"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("parameters: {parameters}");
}

fn run_formula(args: FormulaArgs) -> Result<(), GwError> {
    let colors = match (args.colors, args.n) {
        (Some(_), Some(_)) => {
            return Err(GwError::ParseError(
                "pass either --colors or --n, not both".to_string(),
            ))
        }
        (Some(colors), None) => colors,
        (None, Some(n)) => n + 1,
        (None, None) => return Err(GwError::ParseError("missing --colors or --n".to_string())),
    };
    let twist = parse_negative_twist_opt(args.twist.as_deref())?;
    let mut request = FormulaRequest::new(args.g, args.markings, colors);
    request.max_descendant_power = args.max_descendant.unwrap_or(0);
    request.q_degree = args.d;
    request.include_glossary = !args.no_glossary;
    request.basis = parse_formula_basis(args.basis.as_deref())?;
    request.expansion = formula_expansion(
        args.expand,
        args.equivariant,
        twist.as_deref(),
        args.n,
        colors,
        request.basis.requires_expansion(),
    )?;
    let skeleton = build_formula_skeleton(request)?;
    let format = match (args.format, args.tex) {
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

fn run_resolvent(args: ResolventArgs) -> Result<(), GwError> {
    let n = args.n;
    let genus = args.g;
    let degree = args.d;
    let markings = args.markings;
    let twist = parse_negative_twist_opt(args.twist.as_deref())?;
    let twist_model = parse_twist_model(twist.as_ref())?;
    let mode = parse_compute_mode(args.mode.as_deref())?;
    let equivariant = args.equivariant;
    let validate = args.validate;
    let virtual_dimension = match twist_model.as_ref() {
        Some(twist) => twist.virtual_dimension(n, genus, degree, markings),
        None => ordinary_virtual_dimension(n, genus, degree, markings),
    };

    let req = ResolventRequest {
        target_n: n,
        genus,
        degree,
        markings,
        virtual_dimension,
    };
    let compute_invariant_wise = || {
        compute_resolvent_generating_function(&req, |insertions| {
            compute_series_point(
                n,
                twist.as_deref(),
                genus,
                degree,
                insertions,
                equivariant,
                mode,
            )
        })
    };
    if equivariant {
        if let Some(degrees) = twist.as_ref() {
            let packed =
                compute_negative_split_twisted_resolvent_packed_factored(n, degrees.clone(), &req);
            let (result, used_packed) = match packed {
                Ok(result) => (result, true),
                Err(GwError::UnsupportedInvariant(message)) => {
                    let mut result = compute_invariant_wise()?;
                    result.notes.insert(
                        0,
                        format!(
                            "factored packed resolvent unavailable: {message}; fell back to invariant-wise resolver"
                        ),
                    );
                    println!("Resolvent generating function");
                    println!("target: P^{n}");
                    println!("twist: {}", twist_bundle_label(degrees));
                    println!("genus: {genus}");
                    println!("degree: {degree}");
                    println!("markings: {markings}");
                    println!("virtual_dimension: {virtual_dimension}");
                    print!(
                        "{}",
                        resolvent_definition_text(n, genus, degree, markings, twist.as_deref())
                    );
                    println!("engine: {}", result.engine);
                    println!(
                        "terms: {} candidate, {} nonzero",
                        result.candidate_terms, result.nonzero_terms
                    );
                    println!("F = {}", result.value);
                    if let Some(path) = write_warnings_file("resolvent", &result.notes)? {
                        eprintln!(
                            "warnings written to {}; inspect this file if needed",
                            path.display()
                        );
                    }
                    return Ok(());
                }
                Err(err) => return Err(err),
            };

            let validation = if validate && used_packed {
                let invariant_wise = compute_invariant_wise()?;
                let packed_ratfun = result.value.to_ratfun_polynomial();
                if packed_ratfun != invariant_wise.value {
                    return Err(GwError::ValidationFailure(format!(
                        "packed resolvent output does not match invariant-wise resolver: packed `{packed_ratfun}`, invariant-wise `{}`",
                        invariant_wise.value
                    )));
                }
                Some(format!(
                    "matched invariant-wise resolver ({} candidate, {} nonzero)",
                    invariant_wise.candidate_terms, invariant_wise.nonzero_terms
                ))
            } else {
                None
            };

            println!("Resolvent generating function");
            println!("target: P^{n}");
            println!("twist: {}", twist_bundle_label(degrees));
            println!("genus: {genus}");
            println!("degree: {degree}");
            println!("markings: {markings}");
            println!("virtual_dimension: {virtual_dimension}");
            print!(
                "{}",
                resolvent_definition_text(n, genus, degree, markings, twist.as_deref())
            );
            println!("engine: {}", result.engine);
            println!(
                "terms: {} candidate, {} nonzero",
                result.candidate_terms, result.nonzero_terms
            );
            if let Some(validation) = validation {
                println!("validation: {validation}");
            }
            println!("F = {}", result.value);

            if let Some(path) = write_warnings_file("resolvent", &result.notes)? {
                eprintln!(
                    "warnings written to {}; inspect this file if needed",
                    path.display()
                );
            }
            return Ok(());
        }
    }
    let packed = match twist.as_ref() {
        Some(degrees) => {
            compute_negative_split_twisted_resolvent_packed(n, degrees.clone(), &req, equivariant)
        }
        None => gw_pn::givental::compute_projective_resolvent_packed(&req, equivariant),
    };
    let (result, used_packed) = match packed {
        Ok(result) => (result, true),
        Err(GwError::UnsupportedInvariant(message)) => {
            let mut result = compute_invariant_wise()?;
            result.notes.insert(
                0,
                format!(
                    "packed resolvent unavailable: {message}; fell back to invariant-wise resolver"
                ),
            );
            (result, false)
        }
        Err(err) => return Err(err),
    };

    let validation = if validate && used_packed {
        let invariant_wise = compute_invariant_wise()?;
        if result.value != invariant_wise.value {
            return Err(GwError::ValidationFailure(format!(
                "packed resolvent output does not match invariant-wise resolver: packed `{}`, invariant-wise `{}`",
                result.value, invariant_wise.value
            )));
        }
        Some(format!(
            "matched invariant-wise resolver ({} candidate, {} nonzero)",
            invariant_wise.candidate_terms, invariant_wise.nonzero_terms
        ))
    } else if validate {
        Some("packed resolver unavailable; invariant-wise fallback was used".to_string())
    } else {
        None
    };

    println!("Resolvent generating function");
    println!("target: P^{n}");
    if let Some(twist) = twist.as_ref() {
        println!("twist: {}", twist_bundle_label(twist));
    }
    println!("genus: {genus}");
    println!("degree: {degree}");
    println!("markings: {markings}");
    println!("virtual_dimension: {virtual_dimension}");
    print!(
        "{}",
        resolvent_definition_text(n, genus, degree, markings, twist.as_deref())
    );
    println!("engine: {}", result.engine);
    println!(
        "terms: {} candidate, {} nonzero",
        result.candidate_terms, result.nonzero_terms
    );
    if let Some(validation) = validation {
        println!("validation: {validation}");
    }
    println!("F = {}", result.value);

    if let Some(path) = write_warnings_file("resolvent", &result.notes)? {
        eprintln!(
            "warnings written to {}; inspect this file if needed",
            path.display()
        );
    }
    Ok(())
}

fn resolvent_definition_text(
    n: usize,
    genus: usize,
    degree: usize,
    markings: usize,
    twist: Option<&[usize]>,
) -> String {
    let target = resolvent_target_label(n, twist);
    let mut out = String::new();
    out.push_str("definition:\n");
    let arguments = resolvent_arguments(markings);
    if arguments.is_empty() {
        out.push_str(&format!("  F_{{{genus},{degree}}}^{{{target}}}\n"));
    } else {
        out.push_str(&format!(
            "  F_{{{genus},{degree}}}^{{{target}}}({arguments})\n"
        ));
    }
    out.push_str("    =\n");
    if markings == 0 {
        out.push_str(&format!("      < 1 >_{{{genus},{degree}}}^{{{target}}}\n"));
    } else {
        out.push_str(&format!(
            "      < {} >_{{{genus},{degree}}}^{{{target}}}\n",
            resolvent_insertion_range(markings)
        ));
        out.push_str("  where, for each marking ell,\n");
        out.push_str("      I_ell(z_ell,t_ell)\n");
        out.push_str("        =\n");
        out.push_str(&format!("          sum_{{a=0}}^{{{n}}} t_ell^a H^a/a!\n"));
        out.push_str("          -------------------------------\n");
        out.push_str("                  z_ell - psi_ell\n");
        out.push_str(
            "  expansion convention: 1/(z_ell - psi_ell) = sum_{k>=0} psi_ell^k z_ell^{-k-1}\n",
        );
    }
    out
}

fn resolvent_target_label(n: usize, twist: Option<&[usize]>) -> String {
    let base = format!("P^{n}");
    match twist {
        Some(degrees) if !degrees.is_empty() => format!("{base}, {}", twist_bundle_label(degrees)),
        _ => base,
    }
}

fn twist_bundle_label(degrees: &[usize]) -> String {
    degrees
        .iter()
        .map(|degree| format!("O(-{degree})"))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn resolvent_arguments(markings: usize) -> String {
    if markings == 0 {
        String::new()
    } else {
        format!(
            "{}; {}",
            indexed_variable_range("z", markings),
            indexed_variable_range("t", markings)
        )
    }
}

fn indexed_variable_range(prefix: &str, count: usize) -> String {
    match count {
        0 => String::new(),
        1 => format!("{prefix}_0"),
        _ => format!("{prefix}_0,...,{prefix}_{}", count - 1),
    }
}

fn resolvent_insertion_range(markings: usize) -> String {
    match markings {
        0 => String::new(),
        1 => "I_0(z_0,t_0)".to_string(),
        _ => format!(
            "I_0(z_0,t_0), ..., I_{last}(z_{last},t_{last})",
            last = markings - 1
        ),
    }
}

fn formula_expansion(
    expand: bool,
    equivariant: bool,
    twist: Option<&[usize]>,
    n_flag: Option<usize>,
    colors: usize,
    force: bool,
) -> Result<Option<FormulaExpansion>, GwError> {
    if !force && !expand {
        return Ok(None);
    }
    if let Some(degrees) = twist {
        let n = n_flag.ok_or_else(|| {
            GwError::ParseError("twisted formula expansion needs --n".to_string())
        })?;
        return Ok(Some(FormulaExpansion::NegativeSplitTwisted {
            n,
            degrees: degrees.to_vec(),
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

fn parse_formula_basis(basis: Option<&str>) -> Result<FormulaBasisMode, GwError> {
    match basis {
        None => Ok(FormulaBasisMode::Raw),
        Some("coefficients") | Some("coefficient") => Ok(FormulaBasisMode::Coefficients),
        Some("raw") => Ok(FormulaBasisMode::Raw),
        Some("rational") => Err(GwError::ParseError(
            "invalid --basis `rational`; formula rational basis was removed, use --basis raw for formulas or the resolvent subcommand for fixed-degree resolvent generating functions"
                .to_string(),
        )),
        Some("resolvent") => Err(GwError::ParseError(
            "invalid --basis `resolvent`; use --basis raw for formulas or the resolvent subcommand for fixed-degree resolvent generating functions"
                .to_string(),
        )),
        Some(other) => Err(GwError::ParseError(format!(
            "invalid --basis `{other}`; expected coefficients or raw"
        ))),
    }
}

fn run_degree_series(args: DegreeSeriesArgs) -> Result<(), GwError> {
    let n = args.n;
    let genus = args.g;
    let degree_max = args.d_max;
    let twist = parse_negative_twist_opt(args.twist.as_deref())?;
    let degree_min = args.d_min.unwrap_or(if twist.is_some() { 1 } else { 0 });
    if degree_min > degree_max {
        return Err(GwError::ParseError(format!(
            "--d-min ({degree_min}) cannot exceed --d-max ({degree_max})"
        )));
    }
    let mode = parse_compute_mode(args.mode.as_deref())?;
    let equivariant = args.equivariant;
    let insertion_selection = parse_series_insertions(
        n,
        &args.insert,
        args.max_markings,
        args.max_descendant,
        args.include_zero,
    )?;
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
                    if !dimension_compatible(
                        n,
                        twist_model.as_ref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                    ) {
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

fn run_genus_series(args: GenusSeriesArgs) -> Result<(), GwError> {
    let n = args.n;
    let degree = args.d;
    let genus_max = args.g_max;
    let genus_min = args.g_min.unwrap_or(0);
    if genus_min > genus_max {
        return Err(GwError::ParseError(format!(
            "--g-min ({genus_min}) cannot exceed --g-max ({genus_max})"
        )));
    }
    let twist = parse_negative_twist_opt(args.twist.as_deref())?;
    let mode = parse_compute_mode(args.mode.as_deref())?;
    let equivariant = args.equivariant;
    let insertion_selection = parse_series_insertions(
        n,
        &args.insert,
        args.max_markings,
        args.max_descendant,
        args.include_zero,
    )?;
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
                    if !dimension_compatible(
                        n,
                        twist_model.as_ref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                    ) {
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

fn run_compute(args: ComputeArgs) -> Result<(), GwError> {
    let n = args.n;
    let insertions = parse_insertions(n, &args.insert)?;
    let degree = match args.d {
        Some(degree) => degree,
        None => {
            let probe = InvariantRequest::new(n, args.g, 0, insertions.clone());
            let degree = probe.expected_degree_from_dimension().ok_or_else(|| {
                GwError::ParseError(
                    "cannot infer --d: no degree makes the virtual dimension match these \
                     insertions; pass --d explicitly"
                        .to_string(),
                )
            })?;
            println!("note: degree inferred from the dimension constraint: --d {degree}");
            degree
        }
    };
    let req = InvariantRequest {
        n,
        genus: args.g,
        degree,
        insertions,
        equivariant: args.equivariant,
        mode: parse_compute_mode(args.mode.as_deref())?,
        truncation: None,
    };
    let result = compute(req)?;
    println!("{}", result.value);
    if args.nonequivariant_limit {
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

fn parse_series_insertions(
    n: usize,
    insert: &[String],
    max_markings: Option<usize>,
    max_descendant: Option<usize>,
    include_zero: bool,
) -> Result<InsertionSelection, GwError> {
    let explicit = parse_insertions(n, insert)?;
    if !explicit.is_empty() {
        return Ok(InsertionSelection::Fixed(explicit));
    }

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
        include_zero,
    }))
}

fn run_product(args: ProductArgs) -> Result<(), GwError> {
    let weights_x = match args.weights_x.as_deref() {
        Some(raw) => parse_integer_weights(raw, args.n + 1)?,
        // Defaults chosen so all pairwise sums lambda_i + mu_j are distinct.
        None => (0..=args.n).map(|i| Rational::from(i + 1)).collect(),
    };
    let weights_y = match args.weights_y.as_deref() {
        Some(raw) => parse_integer_weights(raw, args.m + 1)?,
        None => (0..=args.m)
            .map(|j| Rational::from((args.n + 2) * (j + 1)))
            .collect(),
    };
    let insertions = args
        .insert
        .iter()
        .map(|raw| parse_product_insertion(raw))
        .collect::<Result<Vec<_>, _>>()?;

    let invariants = reconstruct_bidegree_invariants(
        args.n,
        args.m,
        &weights_x,
        &weights_y,
        args.g,
        args.d,
        &insertions,
    )?;
    for (d2, value) in invariants.iter().enumerate() {
        let d1 = args.d - d2;
        if bidegree_dimension_matches(args.n, args.m, args.g, d1, d2, &insertions) {
            println!("N[({d1},{d2})] = {value}");
        } else {
            println!("N[({d1},{d2})] = 0 (dimension mismatch)");
        }
    }
    println!(
        "note: reconstructed exactly from {} Novikov rays at rational equivariant weights",
        args.d + 1
    );
    Ok(())
}

fn run_bundle(args: BundleArgs) -> Result<(), GwError> {
    let raw_twists = args
        .twists
        .split(',')
        .map(|part| {
            part.trim().parse::<i128>().map_err(|_| {
                GwError::ParseError(format!("invalid twist `{part}` in `{}`", args.twists))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if raw_twists.is_empty() {
        return Err(GwError::ParseError("--twists must be nonempty".to_string()));
    }
    // Normalize min a_l = 0: P(E) = P(E tensor L).
    let minimum = *raw_twists.iter().min().expect("nonempty");
    let twists = raw_twists
        .iter()
        .map(|value| (value - minimum) as usize)
        .collect::<Vec<_>>();

    let weights_base = match args.weights_base.as_deref() {
        Some(raw) => parse_integer_weights(raw, args.n + 1)?,
        None => (0..=args.n).map(|i| Rational::from(i + 1)).collect(),
    };
    // Fiber weight defaults must keep all grading eigenvalues distinct; a
    // wide, twist-aware spacing suffices for small cases.
    let weights_fiber = match args.weights_fiber.as_deref() {
        Some(raw) => parse_integer_weights(raw, twists.len())?,
        None => twists
            .iter()
            .enumerate()
            .map(|(l, &a)| Rational::from((args.n + 3) * (l + 1) + a * (args.n + 1)))
            .collect(),
    };

    let insertions = args
        .insert
        .iter()
        .map(|raw| parse_bundle_insertion(raw))
        .collect::<Result<Vec<_>, _>>()?;

    let invariants = reconstruct_bundle_invariants(
        args.n,
        &twists,
        &weights_base,
        &weights_fiber,
        args.g,
        args.d,
        &insertions,
    )?;
    if minimum != 0 {
        println!(
            "note: twists normalized to {twists:?} (subtracted min a_l = {minimum}); \
             invariants are unchanged"
        );
    }
    for (d1, d2, value) in &invariants {
        if value.to_string() == "0"
            && !gw_pn::givental::bundle_dimension_matches(
                args.n,
                &twists,
                args.g,
                *d1,
                *d2,
                &insertions,
            )
        {
            println!("N[({d1},{d2})] = 0 (dimension mismatch)");
        } else {
            println!("N[({d1},{d2})] = {value}");
        }
    }
    println!(
        "note: reconstructed exactly from {} Novikov rays at rational equivariant weights",
        args.d + 1
    );
    Ok(())
}

fn parse_bundle_insertion(raw: &str) -> Result<BundleInsertion, GwError> {
    let invalid = || {
        GwError::ParseError(format!(
            "invalid bundle insertion `{raw}`: expected `tauK(CLASS)` or a bare `CLASS` \
             (meaning tau0), with CLASS a `*`-product of `H^p` and `xi^q` factors or `1` — \
             for example `tau1(H*xi^2)` or `xi`"
        ))
    };
    let compact = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let (descendant_power, class_part) = match compact.find('(') {
        Some(open) => {
            let inner = compact.strip_suffix(')').ok_or_else(invalid)?;
            let descendant_power = inner[..open]
                .strip_prefix("tau")
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            (descendant_power, inner[open + 1..].to_string())
        }
        None => {
            if compact.starts_with("tau") {
                return Err(invalid());
            }
            (0, compact.clone())
        }
    };

    let mut h_power = 0usize;
    let mut xi_power = 0usize;
    for factor in class_part.split('*') {
        if factor == "1" {
            continue;
        }
        let (base, power) = match factor.split_once('^') {
            Some((base, power)) => (base, power.parse::<usize>().map_err(|_| invalid())?),
            None => (factor, 1),
        };
        match base {
            "H" => h_power += power,
            "xi" => xi_power += power,
            _ => return Err(invalid()),
        }
    }
    Ok(BundleInsertion::new(descendant_power, h_power, xi_power))
}

fn parse_integer_weights(raw: &str, expected: usize) -> Result<Vec<Rational>, GwError> {
    let weights = raw
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<i128>()
                .map(Rational::from)
                .map_err(|_| {
                    GwError::ParseError(format!("invalid integer weight `{part}` in `{raw}`"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != expected {
        return Err(GwError::ParseError(format!(
            "expected {expected} comma-separated weights, got {}",
            weights.len()
        )));
    }
    Ok(weights)
}

fn parse_product_insertion(raw: &str) -> Result<ProductInsertion, GwError> {
    let invalid = || {
        GwError::ParseError(format!(
            "invalid product insertion `{raw}`: expected `tauK(CLASS)` or a bare `CLASS` \
             (meaning tau0), with CLASS a `*`-product of `H1^a` and `H2^b` factors or `1` — \
             for example `tau1(H1*H2^2)` or `H1`"
        ))
    };
    let compact = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let (descendant_power, class_part) = match compact.find('(') {
        Some(open) => {
            let inner = compact.strip_suffix(')').ok_or_else(invalid)?;
            let descendant_power = inner[..open]
                .strip_prefix("tau")
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            (descendant_power, inner[open + 1..].to_string())
        }
        None => {
            if compact.starts_with("tau") {
                return Err(invalid());
            }
            (0, compact.clone())
        }
    };

    let mut h1_power = 0usize;
    let mut h2_power = 0usize;
    for factor in class_part.split('*') {
        if factor == "1" {
            continue;
        }
        let (base, power) = match factor.split_once('^') {
            Some((base, power)) => (base, power.parse::<usize>().map_err(|_| invalid())?),
            None => (factor, 1),
        };
        match base {
            "H1" => h1_power += power,
            "H2" => h2_power += power,
            _ => return Err(invalid()),
        }
    }
    Ok(ProductInsertion::new(descendant_power, h1_power, h2_power))
}

fn parse_insertions(n: usize, insert: &[String]) -> Result<Vec<gw_pn::Insertion>, GwError> {
    insert.iter().map(|raw| parse_insertion(n, raw)).collect()
}

fn parse_insertion(n: usize, raw: &str) -> Result<gw_pn::Insertion, GwError> {
    let invalid = || {
        GwError::ParseError(format!(
            "invalid insertion `{raw}`: expected `tauK(CLASS)` or a bare `CLASS` (meaning \
             tau0), with CLASS one of `1`, `H`, `H^p` — for example `tau2(H^2)` or `H`"
        ))
    };
    let compact = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let Some(open) = compact.find('(') else {
        if compact.starts_with("tau") {
            return Err(invalid());
        }
        // Bare class shorthand: `H^2` means `tau0(H^2)`.
        return Ok(tau(0, parse_class(n, &compact)?));
    };
    let close = compact.strip_suffix(')').ok_or_else(invalid)?;
    let tau_part = &close[..open];
    let class_part = &close[open + 1..];
    let descendant_power = tau_part
        .strip_prefix("tau")
        .ok_or_else(invalid)?
        .parse::<usize>()
        .map_err(|_| invalid())?;
    let class = parse_class(n, class_part)?;
    Ok(tau(descendant_power, class))
}

fn parse_negative_twist(raw: &str) -> Result<Vec<usize>, GwError> {
    raw.split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(GwError::ParseError(format!(
                    "empty twist degree in --twist value `{raw}`"
                )));
            }
            let degree = part.parse::<isize>().map_err(|_| {
                GwError::ParseError(format!("invalid twist degree `{part}` in --twist"))
            })?;
            if degree >= 0 {
                return Err(GwError::ParseError(format!(
                    "negative split bundles must be written with negative degrees, e.g. `--twist -3` or `--twist -1,-1`; got `{part}`"
                )));
            }
            degree
                .checked_abs()
                .map(|value| value as usize)
                .ok_or_else(|| {
                    GwError::ParseError(format!("invalid twist degree `{part}` in --twist"))
                })
        })
        .collect()
}

fn parse_negative_twist_opt(raw: Option<&str>) -> Result<Option<Vec<usize>>, GwError> {
    raw.map(parse_negative_twist).transpose()
}

fn parse_twist_model(
    twist: Option<&Vec<usize>>,
) -> Result<Option<NegativeSplitBundleTwist>, GwError> {
    twist
        .map(|degrees| NegativeSplitBundleTwist::new(degrees.clone()))
        .transpose()
}

fn parse_compute_mode(mode: Option<&str>) -> Result<ComputeMode, GwError> {
    match mode {
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
            let invalid = || {
                GwError::ParseError(format!(
                    "invalid class `{raw}`: expected `1`, `H`, or `H^p` with 0 <= p <= n"
                ))
            };
            let power = raw
                .strip_prefix("H^")
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            Ok(CohomologyClass::h_power(n, power))
        }
    }
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
    equivariant: bool,
) -> bool {
    let Some(total_degree) = insertion_degree(insertions) else {
        return true;
    };
    let virtual_dimension = match twist {
        Some(twist) => twist.virtual_dimension(n, genus, degree, insertions.len()),
        None => ordinary_virtual_dimension(n, genus, degree, insertions.len()),
    };
    if equivariant {
        // Fiber-equivariant twists are represented over a localized
        // coefficient ring, so retain every bounded profile.  Ordinary
        // equivariant pushforwards only vanish in negative parameter degree.
        return twist.is_some()
            || usize::try_from(virtual_dimension)
                .ok()
                .is_none_or(|dimension| total_degree >= dimension);
    }
    usize::try_from(virtual_dimension).ok() == Some(total_degree)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_scan_dimension_filter_respects_coefficient_ring() {
        let insertions = vec![
            tau(0, CohomologyClass::one(1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ];
        assert!(!dimension_compatible(1, None, 0, 0, &insertions, false));
        assert!(dimension_compatible(1, None, 0, 0, &insertions, true));

        assert!(!dimension_compatible(5, None, 2, 0, &[], false));
        assert!(dimension_compatible(5, None, 2, 0, &[], true));
    }

    #[test]
    fn cli_parses_negative_twist_value() {
        let cli = Cli::try_parse_from([
            "gw-pn", "twisted", "--n", "1", "--twist", "-1,-1", "--g", "2", "--d", "3",
        ])
        .unwrap();
        match cli.command {
            Commands::Twisted(args) => {
                assert_eq!(parse_negative_twist(&args.twist).unwrap(), vec![1, 1]);
            }
            _ => panic!("expected twisted subcommand"),
        }
    }

    #[test]
    fn cli_accepts_long_aliases() {
        let cli = Cli::try_parse_from([
            "gw-pn", "compute", "--n", "2", "--genus", "0", "--degree", "1",
        ])
        .unwrap();
        match cli.command {
            Commands::Compute(args) => {
                assert_eq!((args.n, args.g, args.d), (2, 0, Some(1)));
            }
            _ => panic!("expected compute subcommand"),
        }
    }

    #[test]
    fn cli_compute_degree_is_optional() {
        let cli = Cli::try_parse_from(["gw-pn", "compute", "--n", "2", "--g", "0"]).unwrap();
        match cli.command {
            Commands::Compute(args) => assert_eq!(args.d, None),
            _ => panic!("expected compute subcommand"),
        }
    }

    #[test]
    fn bundle_insertions_parse() {
        let insertion = parse_bundle_insertion("tau1(H*xi^2)").unwrap();
        assert_eq!(
            (
                insertion.descendant_power,
                insertion.h_power,
                insertion.xi_power
            ),
            (1, 1, 2)
        );
        let bare = parse_bundle_insertion("xi").unwrap();
        assert_eq!(
            (bare.descendant_power, bare.h_power, bare.xi_power),
            (0, 0, 1)
        );
        let unit = parse_bundle_insertion("1").unwrap();
        assert_eq!(
            (unit.descendant_power, unit.h_power, unit.xi_power),
            (0, 0, 0)
        );
        assert!(parse_bundle_insertion("tau1(H1)").is_err());
        assert!(parse_bundle_insertion("tau1[xi]").is_err());
        let cli = Cli::try_parse_from([
            "gw-pn", "bundle", "--n", "1", "--twists", "0,1", "--g", "0", "--d", "1", "--insert",
            "xi",
        ])
        .unwrap();
        match cli.command {
            Commands::Bundle(args) => {
                assert_eq!((args.n, args.twists.as_str(), args.d), (1, "0,1", 1))
            }
            _ => panic!("expected bundle subcommand"),
        }
    }

    #[test]
    fn product_insertions_parse() {
        let insertion = parse_product_insertion("tau1(H1*H2^2)").unwrap();
        assert_eq!(
            (
                insertion.descendant_power,
                insertion.h1_power,
                insertion.h2_power
            ),
            (1, 1, 2)
        );
        let bare = parse_product_insertion("H1*H2").unwrap();
        assert_eq!(
            (bare.descendant_power, bare.h1_power, bare.h2_power),
            (0, 1, 1)
        );
        let unit = parse_product_insertion("1").unwrap();
        assert_eq!(
            (unit.descendant_power, unit.h1_power, unit.h2_power),
            (0, 0, 0)
        );
        assert!(parse_product_insertion("tau1(H^2)").is_err());
        assert!(parse_product_insertion("tau1[H1]").is_err());
        let cli = Cli::try_parse_from([
            "gw-pn", "product", "--n", "1", "--m", "1", "--g", "0", "--d", "2", "--insert", "H1*H2",
        ])
        .unwrap();
        match cli.command {
            Commands::Product(args) => assert_eq!((args.n, args.m, args.d), (1, 1, 2)),
            _ => panic!("expected product subcommand"),
        }
    }

    #[test]
    fn insertion_shorthand_means_tau_zero() {
        assert_eq!(
            parse_insertion(2, "H^2").unwrap(),
            tau(0, CohomologyClass::h_power(2, 2))
        );
        assert_eq!(
            parse_insertion(2, "1").unwrap(),
            tau(0, CohomologyClass::one(2))
        );
        assert_eq!(
            parse_insertion(2, "tau3(H)").unwrap(),
            tau(3, CohomologyClass::h_power(2, 1))
        );
        // A malformed tau insertion must report the insertion format, not
        // fall through to the bare-class parser.
        let err = parse_insertion(2, "tau3[H]").unwrap_err().to_string();
        assert!(err.contains("tauK(CLASS)"), "unexpected error: {err}");
    }

    #[test]
    fn cli_collects_repeated_insertions() {
        let cli = Cli::try_parse_from([
            "gw-pn",
            "compute",
            "--n",
            "2",
            "--g",
            "0",
            "--d",
            "1",
            "--insert",
            "tau0(H^2)",
            "--insert",
            "tau0(H)",
        ])
        .unwrap();
        match cli.command {
            Commands::Compute(args) => assert_eq!(args.insert.len(), 2),
            _ => panic!("expected compute subcommand"),
        }
    }

    #[test]
    fn cli_rejects_unknown_flag() {
        let err = Cli::try_parse_from([
            "gw-pn",
            "degree-series",
            "--n",
            "2",
            "--g",
            "2",
            "--d-max",
            "3",
            "--max-descendants",
            "5",
        ])
        .unwrap_err();
        // clap reports the unknown flag (and suggests --max-descendant).
        assert!(err.to_string().contains("--max-descendants"));
    }

    #[test]
    fn parse_negative_twist_rejects_nonnegative() {
        let err = parse_negative_twist("3").unwrap_err();
        assert!(err.to_string().contains("negative degrees"));
    }

    #[test]
    fn formula_expansion_ignores_twist_without_expand() {
        let expansion = formula_expansion(false, false, Some(&[3]), Some(2), 3, false).unwrap();
        assert_eq!(expansion, None);
    }

    #[test]
    fn formula_expansion_infers_projective_with_expand() {
        let expansion = formula_expansion(true, false, None, Some(2), 3, false).unwrap();
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
        let expansion = formula_expansion(true, false, Some(&[3]), Some(2), 3, false).unwrap();
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
            parse_formula_basis(Some("raw")).unwrap(),
            FormulaBasisMode::Raw
        );
        let expansion = formula_expansion(false, false, None, Some(2), 3, true).unwrap();
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
            parse_formula_basis(Some("coefficients")).unwrap(),
            FormulaBasisMode::Coefficients
        );
    }

    #[test]
    fn formula_basis_resolvent_is_not_a_formula_basis() {
        let err = parse_formula_basis(Some("resolvent")).unwrap_err();
        assert!(err.to_string().contains("resolvent subcommand"));
    }

    #[test]
    fn formula_basis_rational_was_removed() {
        let err = parse_formula_basis(Some("rational")).unwrap_err();
        assert!(err
            .to_string()
            .contains("formula rational basis was removed"));
    }

    #[test]
    fn resolvent_definition_names_ordinary_target_and_fraction() {
        let rendered = resolvent_definition_text(2, 0, 1, 3, None);
        assert!(rendered.contains("F_{0,1}^{P^2}(z_0,...,z_2; t_0,...,t_2)"));
        assert!(rendered.contains("sum_{a=0}^{2} t_ell^a H^a/a!"));
        assert!(rendered.contains("z_ell - psi_ell"));
        assert!(rendered.contains("1/(z_ell - psi_ell)"));
    }

    #[test]
    fn resolvent_definition_names_twisted_target() {
        let rendered = resolvent_definition_text(2, 2, 1, 1, Some(&[1, 2]));
        assert!(rendered.contains("F_{2,1}^{P^2, O(-1) + O(-2)}(z_0; t_0)"));
        assert!(rendered.contains(">_{2,1}^{P^2, O(-1) + O(-2)}"));
    }

    #[test]
    fn resolvent_definition_handles_zero_markings() {
        let rendered = resolvent_definition_text(1, 2, 0, 0, None);
        assert!(rendered.contains("F_{2,0}^{P^1}"));
        assert!(rendered.contains("< 1 >_{2,0}^{P^1}"));
    }
}
