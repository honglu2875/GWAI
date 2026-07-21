use clap::{Args, Parser, Subcommand};
use gw_pn::constraints::virasoro::{
    evaluate_constraint_with_bounds as evaluate_virasoro_constraint,
    evaluate_symbolic_constraint_with_bounds, generate_constraint_with_term_limit,
    scan_constraints, CanonicalCorrelatorEvaluator, CorrelatorEvaluationBounds, Descendant,
    NegativeSplitCompletionEvaluator, ProductProjectiveEvaluator, ProjectiveBundleEvaluator,
    ProjectiveSpaceEvaluator, ResidualOutcome, ResidualReport, ResidualStatus, TimeMonomial,
    VirasoroScanBounds,
};
use gw_pn::error::GwError;
use gw_pn::formula::{build_formula_skeleton, FormulaBasisMode, FormulaExpansion, FormulaRequest};
use gw_pn::resolvent::{compute_resolvent_generating_function, ResolventRequest};
use gw_pn::spaces::negative_split_projective::{
    compute_negative_split_twisted, compute_negative_split_twisted_factored,
    compute_negative_split_twisted_resolvent_packed,
    compute_negative_split_twisted_resolvent_packed_factored,
    inverse_euler_qrr_l0_operator_for_constraint, NegativeSplitBundleTwist,
    NegativeSplitFixedFiberQrrEvaluator, NegativeSplitQrrEvaluator, NegativeSplitTotalSpaceTheory,
    TwistedInvariantRequest,
};
use gw_pn::spaces::product_projective::{
    bidegree_dimension_matches_in_theory, reconstruct_bidegree_invariants_in_theory,
    ProductInsertion, ProductProjectiveTheory,
};
use gw_pn::spaces::projective_bundle::{
    bundle_dimension_matches_in_theory, reconstruct_bundle_invariants_in_theory, BundleInsertion,
    ProjectiveBundleTheory,
};
use gw_pn::spaces::projective_space::{CohomologyClass, ProjectiveSpaceTheory};
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::testsuite::run_builtin_tests;
use gw_pn::theory::{BasisId, CurveClass, GwTheory};
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
  gw-pn virasoro formula --n 0 --k 1 --g 0 --d 0 --insert 1 --insert 1 --insert 1 --insert 1
  gw-pn virasoro formula --n 2 --local-twist -2 --k 0 --g 2 --d 1
  gw-pn virasoro check --n 2 --local-twist -2 --k 0 --g 2 --d 1 --fiber-weights 7
  gw-pn virasoro check --n 0 --k 0 --g 1 --d 0
  gw-pn virasoro scan --n 0 --k-max 1 --g-max 1 --d-max 0 --markings-max 4
  gw-pn resolvent --n 2 --g 0 --d 1 --markings 3
  gw-pn degree-series --n 2 --twist -3 --g 2 --d-max 3
  gw-pn genus-series --n 2 --twist -3 --d 1 --g-max 3
  gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3";

#[derive(Debug, Parser)]
#[command(
    name = "gw-pn",
    about = "Exact Gromov-Witten computations and Virasoro audits for projective targets",
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
    /// Generate and exactly audit Virasoro coefficient constraints
    Virasoro(VirasoroArgs),
}

#[derive(Debug, Args)]
struct VirasoroArgs {
    #[command(subcommand)]
    command: VirasoroCommands,
}

#[derive(Debug, Subcommand)]
enum VirasoroCommands {
    /// Render one canonical-theory-derived coefficient equation without evaluating it
    Formula(VirasoroFormulaArgs),
    /// Evaluate every correlator in one equation and compute its exact residual
    Check(VirasoroCheckArgs),
    /// Audit a bounded family of coefficient equations
    Scan(VirasoroScanArgs),
}

#[derive(Debug, Args, Clone)]
struct VirasoroTargetArgs {
    /// Dimension of P^n, or of the base P^n for a product/bundle/local target
    #[arg(long)]
    n: usize,
    /// Select P^n x P^m and give m
    #[arg(
        long,
        conflicts_with_all = ["bundle_twists", "local_twist", "local_completion_twist"]
    )]
    product_m: Option<usize>,
    /// Select P(O(a_1)+...+O(a_r)) over P^n; require nonnegative twists with min 0
    #[arg(
        long,
        allow_hyphen_values = true,
        conflicts_with_all = ["local_twist", "local_completion_twist"]
    )]
    bundle_twists: Option<String>,
    /// Select the negative-split local theory, e.g. -3 or -1,-1
    #[arg(
        long,
        allow_hyphen_values = true,
        conflicts_with = "local_completion_twist"
    )]
    local_twist: Option<String>,
    /// Audit a negative-split theory through its compact projective completion
    #[arg(long, allow_hyphen_values = true)]
    local_completion_twist: Option<String>,
}

#[derive(Debug, Args)]
struct VirasoroFormulaArgs {
    #[command(flatten)]
    target: VirasoroTargetArgs,
    /// Operator index -1..=64
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    k: i32,
    #[arg(long, default_value_t = 0, visible_alias = "genus")]
    g: usize,
    /// Geometric degree: d for P^n/local, d1,d2 for products/bundles
    #[arg(long, visible_alias = "degree")]
    d: Option<String>,
    /// Target-specific insertion syntax; repeat for multiple markings
    #[arg(long)]
    insert: Vec<String>,
    /// text or tex
    #[arg(long, default_value = "text")]
    format: String,
    /// Maximum unaggregated terms in the generated coefficient equation
    #[arg(long, default_value_t = 1_000_000)]
    term_limit: usize,
}

#[derive(Debug, Args)]
struct VirasoroCheckArgs {
    #[command(flatten)]
    target: VirasoroTargetArgs,
    /// Operator index -1..=64
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    k: i32,
    #[arg(long, default_value_t = 0, visible_alias = "genus")]
    g: usize,
    #[arg(long, visible_alias = "degree")]
    d: Option<String>,
    #[arg(long)]
    insert: Vec<String>,
    /// Exact nonzero rational fiber weights used to specialize a symbolic local QRR constraint
    /// (one comma-separated value per summand, e.g. 7 or 3/2,5)
    #[arg(long, allow_hyphen_values = true)]
    fiber_weights: Option<String>,
    /// Print the generated human-readable equation before its residual
    #[arg(long)]
    show_formula: bool,
    /// Maximum unaggregated terms in the generated coefficient equation
    #[arg(long, default_value_t = 1_000_000)]
    term_limit: usize,
    /// Maximum unique correlator dependencies evaluated
    #[arg(long, default_value_t = 100_000)]
    dependency_limit: usize,
    /// Maximum missing-dependency diagnostics to print
    #[arg(long, default_value_t = 20)]
    show_missing: usize,
}

#[derive(Debug, Args)]
struct VirasoroScanArgs {
    #[command(flatten)]
    target: VirasoroTargetArgs,
    /// Minimum operator index, in -1..=64
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    k_min: i32,
    /// Maximum operator index, in -1..=64
    #[arg(long, default_value_t = 1, allow_hyphen_values = true)]
    k_max: i32,
    #[arg(long, default_value_t = 1)]
    g_max: usize,
    /// Bound in the canonical theory's effective/admissible grading
    #[arg(long, default_value_t = 1)]
    d_max: usize,
    /// Maximum external markings in scanned profiles (hard cap: 20)
    #[arg(long, default_value_t = 2)]
    markings_max: usize,
    #[arg(long, default_value_t = 1)]
    descendant_max: usize,
    #[arg(long, default_value_t = 10_000)]
    equation_limit: usize,
    /// Maximum unaggregated terms in any generated coefficient equation
    #[arg(long, default_value_t = 1_000_000)]
    term_limit: usize,
    /// Maximum generated AST terms retained across the complete scan
    #[arg(long, default_value_t = 1_000_000)]
    total_term_limit: usize,
    /// Maximum markings in any correlator dependency (default: markings-max + 2)
    #[arg(long)]
    dependency_markings_max: Option<usize>,
    /// Maximum individual psi power in any dependency (default derived from k/descendant bounds)
    #[arg(long)]
    dependency_descendant_max: Option<usize>,
    /// Maximum unique correlator dependencies evaluated per equation
    #[arg(long, default_value_t = 100_000)]
    dependency_limit: usize,
    /// Maximum non-passing equations to print
    #[arg(long, default_value_t = 20)]
    show_failures: usize,
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

The bundle is P(O(a_1)+...+O(a_m)) over P^n.  --twists must be nonnegative
and include 0.  Tensor-normalize other presentations explicitly: although
P(E)=P(E tensor L), that operation changes xi and its curve coordinate.
Insertions are tauK(CLASS) with CLASS a *-product of H^p and xi^q factors
(or 1); a bare CLASS means tau0.  --d is the SHIFTED total degree
d1 + (d2 + (max a) d1); the command reports every curve class (d1, d2) in
that slice, with d2 = xi . beta possibly negative.  Values are the
non-equivariant invariants, reconstructed exactly from d + 1 Novikov rays.")]
struct BundleArgs {
    /// Dimension of the base P^n
    #[arg(long)]
    n: usize,
    /// Comma-separated nonnegative bundle twists a_1,...,a_m, including 0
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
        Commands::Virasoro(args) => run_virasoro(args),
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

enum CliVirasoroTheory {
    Projective(ProjectiveSpaceEvaluator),
    Product(ProductProjectiveEvaluator),
    Bundle(ProjectiveBundleEvaluator),
    Local(NegativeSplitTotalSpaceTheory),
    LocalCompletion(Box<NegativeSplitCompletionEvaluator>),
}

impl CliVirasoroTheory {
    fn theory(&self) -> &dyn GwTheory {
        match self {
            Self::Projective(evaluator) => evaluator.theory(),
            Self::Product(evaluator) => evaluator.theory(),
            Self::Bundle(evaluator) => evaluator.theory(),
            Self::Local(theory) => theory,
            Self::LocalCompletion(evaluator) => evaluator.theory(),
        }
    }

    fn evaluator(&self) -> Result<&dyn CanonicalCorrelatorEvaluator, GwError> {
        match self {
            Self::Projective(evaluator) => Ok(evaluator),
            Self::Product(evaluator) => Ok(evaluator),
            Self::Bundle(evaluator) => Ok(evaluator),
            Self::LocalCompletion(evaluator) => Ok(evaluator.as_ref()),
            Self::Local(_) => Err(GwError::UnsupportedInvariant(
                "negative-split/local Virasoro checking requires the QRR-conjugated operator, twisted pairing, and degree-zero twisted sector; the ordinary compact operator is deliberately not substituted"
                    .to_string(),
            )),
        }
    }
}

fn run_virasoro(args: VirasoroArgs) -> Result<(), GwError> {
    match args.command {
        VirasoroCommands::Formula(args) => run_virasoro_formula(args),
        VirasoroCommands::Check(args) => run_virasoro_check(args),
        VirasoroCommands::Scan(args) => run_virasoro_scan(args),
    }
}

fn run_virasoro_formula(args: VirasoroFormulaArgs) -> Result<(), GwError> {
    let target = build_virasoro_target(&args.target)?;
    let degree = parse_virasoro_degree(&target, args.d.as_deref())?;
    let time = parse_virasoro_time(&target, &args.insert)?;
    if let CliVirasoroTheory::Local(theory) = &target {
        if args.k != 0 {
            return Err(GwError::UnsupportedInvariant(
                "the exact inverse-Euler QRR generator currently implements L_0; other local Virasoro operators require full differential-operator conjugation"
                    .to_string(),
            ));
        }
        let operator =
            inverse_euler_qrr_l0_operator_for_constraint(theory, args.g, &degree, &time)?;
        let constraint = operator.generate_constraint_with_term_limit(
            theory,
            args.g,
            degree,
            time,
            args.term_limit,
        )?;
        match args.format.trim().to_ascii_lowercase().as_str() {
            "text" | "txt" => {
                print!("{}\nCoefficient equation:\n", operator.render_text()?);
                print!(
                    "{}",
                    constraint.render_symbolic_text_for_theory(target.theory())?
                );
            }
            "tex" | "latex" => {
                println!("{}", operator.render_tex()?);
                print!(
                    "{}",
                    constraint.render_symbolic_tex_for_theory(target.theory())?
                );
            }
            other => {
                return Err(GwError::ParseError(format!(
                    "unknown Virasoro formula format `{other}`; expected text or tex"
                )))
            }
        }
        return Ok(());
    }
    let constraint = generate_constraint_with_term_limit(
        target.theory(),
        args.k,
        args.g,
        degree,
        time,
        args.term_limit,
    )?;
    match args.format.trim().to_ascii_lowercase().as_str() {
        "text" | "txt" => print!("{}", constraint.render_text_for_theory(target.theory())?),
        "tex" | "latex" => print!("{}", constraint.render_tex_for_theory(target.theory())?),
        other => {
            return Err(GwError::ParseError(format!(
                "unknown Virasoro formula format `{other}`; expected text or tex"
            )))
        }
    }
    Ok(())
}

fn run_virasoro_check(args: VirasoroCheckArgs) -> Result<(), GwError> {
    let target = build_virasoro_target(&args.target)?;
    if args.fiber_weights.is_some() && !matches!(&target, CliVirasoroTheory::Local(_)) {
        return Err(GwError::ParseError(
            "--fiber-weights is only valid with --local-twist".to_string(),
        ));
    }
    let degree = parse_virasoro_degree(&target, args.d.as_deref())?;
    let time = parse_virasoro_time(&target, &args.insert)?;
    if let CliVirasoroTheory::Local(theory) = &target {
        if args.k != 0 {
            return Err(GwError::UnsupportedInvariant(
                "the exact inverse-Euler QRR checker currently implements L_0".to_string(),
            ));
        }
        let operator =
            inverse_euler_qrr_l0_operator_for_constraint(theory, args.g, &degree, &time)?;
        let constraint = operator.generate_constraint_with_term_limit(
            theory,
            args.g,
            degree,
            time,
            args.term_limit,
        )?;
        if args.show_formula {
            println!("{}\nCoefficient equation:", operator.render_text()?);
            println!(
                "{}",
                constraint.render_symbolic_text_for_theory(target.theory())?
            );
        }
        let bounds = CorrelatorEvaluationBounds {
            dependency_limit: args.dependency_limit,
            ..CorrelatorEvaluationBounds::unbounded()
        };
        let report = if let Some(raw_weights) = args.fiber_weights.as_deref() {
            let raw_degrees = parse_negative_twist(
                args.target
                    .local_twist
                    .as_deref()
                    .expect("local target was matched above"),
            )?;
            let weights = parse_rational_weights(raw_weights, raw_degrees.len())?;
            let evaluator = NegativeSplitFixedFiberQrrEvaluator::new(
                theory.base_dimension(),
                raw_degrees,
                weights,
            )?;
            let specialized = evaluator.specialize_constraint(&constraint)?;
            let assignments = specialized
                .assignments()
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("specialization: {assignments}");
            evaluate_virasoro_constraint(&evaluator, specialized.constraint(), bounds)
        } else {
            let evaluator =
                NegativeSplitQrrEvaluator::new(theory.base_dimension(), theory.degrees().to_vec())?;
            evaluate_symbolic_constraint_with_bounds(&evaluator, &constraint, bounds)
        };
        return finish_virasoro_report(target.theory(), report, args.show_missing);
    }
    let constraint = generate_constraint_with_term_limit(
        target.theory(),
        args.k,
        args.g,
        degree,
        time,
        args.term_limit,
    )?;
    if args.show_formula {
        println!("{}", constraint.render_text_for_theory(target.theory())?);
    }
    let report = evaluate_virasoro_constraint(
        target.evaluator()?,
        &constraint,
        CorrelatorEvaluationBounds {
            dependency_limit: args.dependency_limit,
            ..CorrelatorEvaluationBounds::unbounded()
        },
    );
    finish_virasoro_report(target.theory(), report, args.show_missing)
}

fn finish_virasoro_report(
    theory: &dyn GwTheory,
    report: ResidualReport<CurveClass, BasisId, gw_pn::algebra::RatFun>,
    show_missing: usize,
) -> Result<(), GwError> {
    print_virasoro_outcome(report.outcome());
    println!(
        "terms: {}/{} evaluated; dependencies: backend={} structural-zero={} missing={}",
        report.evaluated_term_count(),
        report.total_term_count(),
        report.backend_correlator_count(),
        report.structural_zero_correlator_count(),
        report.missing_correlator_count()
    );
    for missing in report.missing_correlators().iter().take(show_missing) {
        println!(
            "missing: g={} beta={} {} ({:?})",
            missing.correlator.genus,
            missing.correlator.degree,
            virasoro_insertion_label(theory, missing.correlator.insertions()),
            missing.reason
        );
    }
    if report.missing_correlator_count() > show_missing {
        println!(
            "missing: ... {} additional retained diagnostics not shown",
            report.missing_correlator_count() - show_missing
        );
    }
    for note in report.notes() {
        println!("note: {note}");
    }
    match report.status() {
        ResidualStatus::VerifiedZero => Ok(()),
        ResidualStatus::Nonzero => Err(GwError::ValidationFailure(
            "Virasoro constraint has a nonzero exact residual".to_string(),
        )),
        ResidualStatus::Incomplete => Err(GwError::ValidationFailure(
            "Virasoro constraint is incomplete; unsupported dependencies are not zeros".to_string(),
        )),
    }
}

fn run_virasoro_scan(args: VirasoroScanArgs) -> Result<(), GwError> {
    let target = build_virasoro_target(&args.target)?;
    let evaluator = target.evaluator()?;
    let dependency_markings_max = match args.dependency_markings_max {
        Some(bound) => bound,
        None => args.markings_max.checked_add(2).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "dependency marking bound derived from --markings-max overflowed".to_string(),
            )
        })?,
    };
    let dependency_descendant_max = match args.dependency_descendant_max {
        Some(bound) => bound,
        None if args.k_max < 0 => args.descendant_max,
        None => {
            let k_max = usize::try_from(args.k_max).map_err(|_| {
                GwError::UnsupportedInvariant("invalid Virasoro operator bound".to_string())
            })?;
            let shifted_external = args.descendant_max.checked_add(k_max).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "dependency descendant bound derived from scan bounds overflowed".to_string(),
                )
            })?;
            let pure_descendant = k_max.checked_add(1).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "dependency descendant bound derived from --k-max overflowed".to_string(),
                )
            })?;
            shifted_external.max(pure_descendant)
        }
    };
    let report = scan_constraints(
        evaluator,
        VirasoroScanBounds {
            operator_min: args.k_min,
            operator_max: args.k_max,
            genus_max: args.g_max,
            degree_grading_max: args.d_max,
            markings_max: args.markings_max,
            descendant_max: args.descendant_max,
            equation_limit: args.equation_limit,
            generated_term_limit: args.term_limit,
            total_generated_term_limit: args.total_term_limit,
            correlator_bounds: CorrelatorEvaluationBounds {
                max_genus: Some(args.g_max),
                max_markings: Some(dependency_markings_max),
                max_descendant_power: Some(dependency_descendant_max),
                dependency_limit: args.dependency_limit,
            },
        },
    )?;
    println!(
        "Virasoro scan: total={} terms={} verified-zero={} nonzero={} incomplete={}",
        report.total(),
        report.generated_term_count(),
        report.verified_zero_count(),
        report.nonzero_count(),
        report.incomplete_count()
    );
    println!(
        "coverage: backend-exercised={} structural-only={} vacuous={} unresolved-only={}",
        report.backend_exercised_count(),
        report.structural_only_count(),
        report.vacuous_count(),
        report.unresolved_only_count()
    );
    let mut shown = 0usize;
    for entry in report.entries() {
        if entry.report.status() == ResidualStatus::VerifiedZero || shown >= args.show_failures {
            continue;
        }
        println!(
            "{:?}: L_{} g={} beta={} time-degree={} missing={}",
            entry.report.status(),
            entry.constraint.operator.index,
            entry.constraint.sector.genus,
            entry.constraint.sector.degree,
            entry
                .constraint
                .time_coefficient
                .total_degree()
                .unwrap_or(usize::MAX),
            entry.report.missing_correlator_count()
        );
        shown += 1;
    }
    if report.is_success() {
        Ok(())
    } else {
        Err(GwError::ValidationFailure(format!(
            "Virasoro scan did not close: {} nonzero and {} incomplete equations",
            report.nonzero_count(),
            report.incomplete_count()
        )))
    }
}

fn print_virasoro_outcome(outcome: &ResidualOutcome<gw_pn::algebra::RatFun>) {
    match outcome {
        ResidualOutcome::VerifiedZero { exact_residual } => {
            println!("status: verified-zero\nexact residual: {exact_residual}")
        }
        ResidualOutcome::Nonzero { exact_residual } => {
            println!("status: NONZERO\nexact residual: {exact_residual}")
        }
        ResidualOutcome::Incomplete { exact_partial_sum } => {
            println!("status: INCOMPLETE");
            if let Some(partial) = exact_partial_sum {
                println!("exact partial sum: {partial}");
            }
        }
    }
}

fn build_virasoro_target(args: &VirasoroTargetArgs) -> Result<CliVirasoroTheory, GwError> {
    if let Some(m) = args.product_m {
        return Ok(CliVirasoroTheory::Product(ProductProjectiveEvaluator::new(
            args.n, m,
        )?));
    }
    if let Some(raw) = &args.bundle_twists {
        let twists = parse_canonical_bundle_twists(raw)?;
        let theory = ProjectiveBundleTheory::new(args.n, twists)?;
        return Ok(CliVirasoroTheory::Bundle(ProjectiveBundleEvaluator::new(
            theory,
        )?));
    }
    if let Some(raw) = &args.local_twist {
        let degrees = parse_negative_twist(raw)?;
        return Ok(CliVirasoroTheory::Local(
            NegativeSplitTotalSpaceTheory::new(args.n, degrees)?,
        ));
    }
    if let Some(raw) = &args.local_completion_twist {
        let degrees = parse_negative_twist(raw)?;
        return Ok(CliVirasoroTheory::LocalCompletion(Box::new(
            NegativeSplitCompletionEvaluator::new(args.n, degrees)?,
        )));
    }
    Ok(CliVirasoroTheory::Projective(
        ProjectiveSpaceEvaluator::try_new(args.n)?,
    ))
}

fn parse_canonical_bundle_twists(raw: &str) -> Result<Vec<usize>, GwError> {
    let values = raw
        .split(',')
        .map(|part| {
            part.trim().parse::<i128>().map_err(|_| {
                GwError::ParseError(format!("invalid bundle twist `{part}` in `{raw}`"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    canonical_bundle_twist_values(&values)
}

fn canonical_bundle_twist_values(values: &[i128]) -> Result<Vec<usize>, GwError> {
    if values.is_empty() {
        return Err(GwError::ParseError(
            "bundle twists must be nonempty".to_string(),
        ));
    }
    let twists = values
        .iter()
        .copied()
        .map(|value| {
            if value < 0 {
                return Err(GwError::ParseError(
                    "bundle twists must already be normalized as nonnegative integers with minimum 0; tensoring E changes the labelled xi class and xi.beta coordinate, so normalize the presentation explicitly"
                        .to_string(),
                ));
            }
            usize::try_from(value).map_err(|_| {
                GwError::ParseError(
                    "bundle twist does not fit in usize".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !twists.contains(&0) {
        return Err(GwError::ParseError(
            "bundle twists must already be normalized with minimum 0; tensoring E changes the labelled xi class and xi.beta coordinate, so normalize the presentation explicitly"
                .to_string(),
        ));
    }
    Ok(twists)
}

fn parse_virasoro_degree(
    target: &CliVirasoroTheory,
    raw: Option<&str>,
) -> Result<CurveClass, GwError> {
    let expected_rank = target.theory().curve_class_space().rank();
    let raw = raw.unwrap_or(if expected_rank == 1 { "0" } else { "0,0" });
    let coordinates = raw
        .split(',')
        .map(|part| {
            part.trim().parse::<i64>().map_err(|_| {
                GwError::ParseError(format!("invalid curve coordinate `{part}` in `{raw}`"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if coordinates.len() != expected_rank {
        return Err(GwError::ParseError(format!(
            "theory {} expects {expected_rank} curve coordinate(s), got {} in `{raw}`",
            target.theory().theory_id(),
            coordinates.len()
        )));
    }
    let curve = CurveClass::new(coordinates);
    target.theory().curve_class_space().validate(&curve)?;
    Ok(curve)
}

fn parse_virasoro_time(
    target: &CliVirasoroTheory,
    raw_insertions: &[String],
) -> Result<TimeMonomial<BasisId>, GwError> {
    let descendants = match target {
        CliVirasoroTheory::Projective(evaluator) => {
            parse_insertions(evaluator.projective_theory().n(), raw_insertions)?
                .into_iter()
                .map(|insertion| {
                    let power = insertion.class.pure_power().ok_or_else(|| {
                        GwError::ConventionMismatch(
                            "Virasoro CLI requires homogeneous basis insertions".to_string(),
                        )
                    })?;
                    Ok(Descendant::new(insertion.descendant_power, BasisId(power)))
                })
                .collect::<Result<Vec<_>, GwError>>()?
        }
        CliVirasoroTheory::Local(theory) => {
            parse_insertions(theory.base_dimension(), raw_insertions)?
                .into_iter()
                .map(|insertion| {
                    let power = insertion.class.pure_power().ok_or_else(|| {
                        GwError::ConventionMismatch(
                            "Virasoro CLI requires homogeneous basis insertions".to_string(),
                        )
                    })?;
                    Ok(Descendant::new(insertion.descendant_power, BasisId(power)))
                })
                .collect::<Result<Vec<_>, GwError>>()?
        }
        CliVirasoroTheory::Product(evaluator) => raw_insertions
            .iter()
            .map(|raw| {
                let insertion = parse_product_insertion(raw)?;
                let basis = evaluator
                    .product_theory()
                    .basis_id(insertion.h1_power, insertion.h2_power)
                    .ok_or_else(|| {
                        GwError::ParseError(format!(
                            "product insertion `{raw}` is outside the canonical basis"
                        ))
                    })?;
                Ok(Descendant::new(insertion.descendant_power, basis))
            })
            .collect::<Result<Vec<_>, GwError>>()?,
        CliVirasoroTheory::Bundle(evaluator) => raw_insertions
            .iter()
            .map(|raw| {
                let insertion = parse_bundle_insertion(raw)?;
                let basis = evaluator
                    .bundle_theory()
                    .basis_id(insertion.h_power, insertion.xi_power)
                    .ok_or_else(|| {
                        GwError::ParseError(format!(
                            "bundle insertion `{raw}` is outside the canonical basis"
                        ))
                    })?;
                Ok(Descendant::new(insertion.descendant_power, basis))
            })
            .collect::<Result<Vec<_>, GwError>>()?,
        CliVirasoroTheory::LocalCompletion(evaluator) => raw_insertions
            .iter()
            .map(|raw| {
                let insertion = parse_bundle_insertion(raw)?;
                let basis = evaluator
                    .compact_theory()
                    .basis_id(insertion.h_power, insertion.xi_power)
                    .ok_or_else(|| {
                        GwError::ParseError(format!(
                            "compact-completion insertion `{raw}` is outside the canonical bundle basis"
                        ))
                    })?;
                Ok(Descendant::new(insertion.descendant_power, basis))
            })
            .collect::<Result<Vec<_>, GwError>>()?,
    };
    Ok(TimeMonomial::from_descendants(descendants))
}

fn virasoro_insertion_label(theory: &dyn GwTheory, insertions: &[Descendant<BasisId>]) -> String {
    if insertions.is_empty() {
        return "<no insertions>".to_string();
    }
    insertions
        .iter()
        .map(|insertion| {
            let class = theory
                .state_space()
                .element(insertion.class)
                .map(|element| element.label.as_str())
                .unwrap_or("<unknown class>");
            format!("tau{}({class})", insertion.psi_power)
        })
        .collect::<Vec<_>>()
        .join(" ")
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
    if let Some(path) = write_warnings_file("series", &result.notes) {
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
        (None, Some(n)) => n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("formula state-space size is too large".to_string())
        })?,
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
    let dimension_theory = canonical_dimension_theory(n, twist_model.as_ref())?;
    let virtual_dimension =
        checked_virtual_dimension(dimension_theory.as_ref(), genus, degree, markings)?;

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
    let fallback_after_packed_error = |packed_error: GwError| {
        let restore_limit = matches!(packed_error, GwError::ResourceLimit { .. });
        let packed_message = match &packed_error {
            GwError::UnsupportedInvariant(message) => message.clone(),
            other => other.to_string(),
        };
        match compute_invariant_wise() {
            Ok(result) => Ok((result, packed_message)),
            Err(GwError::UnsupportedInvariant(_)) if restore_limit => Err(packed_error),
            Err(error) => Err(error),
        }
    };
    if equivariant {
        if let Some(degrees) = twist.as_ref() {
            let packed =
                compute_negative_split_twisted_resolvent_packed_factored(n, degrees.clone(), &req);
            let (result, used_packed) = match packed {
                Ok(result) => (result, true),
                Err(error @ GwError::UnsupportedInvariant(_))
                | Err(error @ GwError::ResourceLimit { .. }) => {
                    let (mut result, message) = fallback_after_packed_error(error)?;
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
                    if let Some(path) = write_warnings_file("resolvent", &result.notes) {
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
                if !packed_ratfun.equivalent(&invariant_wise.value) {
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

            if let Some(path) = write_warnings_file("resolvent", &result.notes) {
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
        Err(error @ GwError::UnsupportedInvariant(_))
        | Err(error @ GwError::ResourceLimit { .. }) => {
            let (mut result, message) = fallback_after_packed_error(error)?;
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
        if !result.value.equivalent(&invariant_wise.value) {
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

    if let Some(path) = write_warnings_file("resolvent", &result.notes) {
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

fn batch_skip_reason(error: GwError) -> Result<String, GwError> {
    match error {
        GwError::UnsupportedInvariant(message) => Ok(message),
        limit @ GwError::ResourceLimit { .. } => Ok(limit.to_string()),
        other => Err(other),
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
                    Ok(result) => {
                        if args.include_zero || !result.value.is_zero() {
                            println!("q^{degree} [{label}] = {}", result.value);
                        }
                    }
                    Err(error) => {
                        let msg = batch_skip_reason(error)?;
                        warnings.push(format!("skipped q^{degree} [{label}]: {msg}"))
                    }
                }
            }
        }
        InsertionSelection::Bounded(scan) => {
            let dimension_theory = canonical_dimension_theory(n, twist_model.as_ref())?;
            for degree in degree_min..=degree_max {
                for insertions in &scan.profiles {
                    if !dimension_compatible(
                        dimension_theory.as_ref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                        twist_model.is_some(),
                    )? {
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
                        Err(error) => {
                            let msg = batch_skip_reason(error)?;
                            warnings.push(format!("skipped q^{degree} [{label}]: {msg}"))
                        }
                    }
                }
            }
        }
    }
    if let Some(path) = write_warnings_file("degree-series", &warnings) {
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
                    Ok(result) => {
                        if args.include_zero || !result.value.is_zero() {
                            println!("g={genus} q^{degree} [{label}] = {}", result.value);
                        }
                    }
                    Err(error) => {
                        let msg = batch_skip_reason(error)?;
                        warnings.push(format!("skipped g={genus} q^{degree} [{label}]: {msg}"))
                    }
                }
            }
        }
        InsertionSelection::Bounded(scan) => {
            let dimension_theory = canonical_dimension_theory(n, twist_model.as_ref())?;
            for genus in genus_min..=genus_max {
                for insertions in &scan.profiles {
                    if !dimension_compatible(
                        dimension_theory.as_ref(),
                        genus,
                        degree,
                        insertions,
                        equivariant,
                        twist_model.is_some(),
                    )? {
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
                        Err(error) => {
                            let msg = batch_skip_reason(error)?;
                            warnings.push(format!("skipped g={genus} q^{degree} [{label}]: {msg}"))
                        }
                    }
                }
            }
        }
    }
    if let Some(path) = write_warnings_file("genus-series", &warnings) {
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
        let weights = default_lambda_line_weights(n)?;
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

fn write_warnings_file(command: &str, warnings: &[String]) -> Option<PathBuf> {
    let warnings = warnings
        .iter()
        .filter(|warning| {
            warning.starts_with("skipped ")
                || warning.contains(" unavailable:")
                || warning.contains("fell back")
        })
        .collect::<Vec<_>>();
    if warnings.is_empty() {
        return None;
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
    for warning in &warnings {
        contents.push_str("warning: ");
        contents.push_str(warning);
        contents.push('\n');
    }
    match fs::write(&path, contents) {
        Ok(()) => Some(path),
        Err(err) => {
            eprintln!(
                "warning: failed to write diagnostics to {}: {err}",
                path.display()
            );
            for warning in warnings {
                eprintln!("warning: {warning}");
            }
            None
        }
    }
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
    let theory = ProductProjectiveTheory::new(args.n, args.m)?;
    let (n, m) = theory.dimensions();
    let weights_x = match args.weights_x.as_deref() {
        Some(raw) => parse_integer_weights(raw, n + 1)?,
        // Defaults chosen so all pairwise sums lambda_i + mu_j are distinct.
        None => (0..=n).map(|i| Rational::from(i + 1)).collect(),
    };
    let weights_y = match args.weights_y.as_deref() {
        Some(raw) => parse_integer_weights(raw, m + 1)?,
        None => {
            let stride = n.checked_add(2).ok_or_else(|| {
                GwError::UnsupportedInvariant("product default weight overflow".to_string())
            })?;
            (0..=m)
                .map(|j| {
                    stride
                        .checked_mul(j + 1)
                        .map(Rational::from)
                        .ok_or_else(|| {
                            GwError::UnsupportedInvariant(
                                "product default weight overflow".to_string(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let insertions = args
        .insert
        .iter()
        .map(|raw| parse_product_insertion(raw))
        .collect::<Result<Vec<_>, _>>()?;

    let invariants = reconstruct_bidegree_invariants_in_theory(
        &theory,
        &weights_x,
        &weights_y,
        args.g,
        args.d,
        &insertions,
    )?;
    for (d2, value) in invariants.iter().enumerate() {
        let d1 = args.d - d2;
        if bidegree_dimension_matches_in_theory(&theory, args.g, d1, d2, &insertions)? {
            println!("N[({d1},{d2})] = {value}");
        } else {
            println!("N[({d1},{d2})] = 0 (dimension mismatch)");
        }
    }
    println!(
        "note: reconstructed exactly from {} Novikov rays at rational equivariant weights",
        args.d.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("product reconstruction degree is too large".to_string())
        })?
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
    let twists_in_input_order = canonical_bundle_twist_values(&raw_twists)?;
    let theory = ProjectiveBundleTheory::new(args.n, twists_in_input_order.clone())?;
    let twists = theory.twists();
    let base_size = theory.base_dimension().checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("bundle base dimension is too large".to_string())
    })?;

    let weights_base = match args.weights_base.as_deref() {
        Some(raw) => parse_integer_weights(raw, base_size)?,
        None => (0..base_size).map(|i| Rational::from(i + 1)).collect(),
    };
    let weights_fiber = match args.weights_fiber.as_deref() {
        Some(raw) => {
            let raw_weights = parse_integer_weights(raw, twists_in_input_order.len())?;
            theory.canonicalize_summand_payloads(twists_in_input_order.clone(), raw_weights)?
        }
        None => {
            let maximum = twists.iter().copied().max().expect("nonempty twists");
            let stride = maximum
                .checked_add(1)
                .and_then(|value| value.checked_mul(base_size))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant("bundle default weight overflow".to_string())
                })?;
            (0..twists.len())
                .map(|index| {
                    stride
                        .checked_mul(index)
                        .map(Rational::from)
                        .ok_or_else(|| {
                            GwError::UnsupportedInvariant(
                                "bundle default weight overflow".to_string(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    let insertions = args
        .insert
        .iter()
        .map(|raw| parse_bundle_insertion(raw))
        .collect::<Result<Vec<_>, _>>()?;

    let invariants = reconstruct_bundle_invariants_in_theory(
        &theory,
        &weights_base,
        &weights_fiber,
        args.g,
        args.d,
        &insertions,
    )?;
    for (d1, d2, value) in &invariants {
        if value.to_string() == "0"
            && !bundle_dimension_matches_in_theory(&theory, args.g, *d1, *d2, &insertions)?
        {
            println!("N[({d1},{d2})] = 0 (dimension mismatch)");
        } else {
            println!("N[({d1},{d2})] = {value}");
        }
    }
    println!(
        "note: reconstructed exactly from {} Novikov rays at rational equivariant weights",
        args.d.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("bundle reconstruction degree is too large".to_string())
        })?
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

fn parse_rational_weights(raw: &str, expected: usize) -> Result<Vec<Rational>, GwError> {
    let weights = raw
        .split(',')
        .map(|part| {
            let part = part.trim();
            let (numerator, denominator) = match part.split_once('/') {
                Some((numerator, denominator)) => (numerator.trim(), denominator.trim()),
                None => (part, "1"),
            };
            let numerator = numerator.parse::<i128>().map_err(|_| {
                GwError::ParseError(format!("invalid rational weight `{part}` in `{raw}`"))
            })?;
            let denominator = denominator.parse::<i128>().map_err(|_| {
                GwError::ParseError(format!("invalid rational weight `{part}` in `{raw}`"))
            })?;
            if denominator == 0 || numerator == 0 {
                return Err(GwError::ParseError(format!(
                    "fiber weight `{part}` must be a nonzero rational number"
                )));
            }
            Ok(Rational::new(numerator, denominator))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != expected {
        return Err(GwError::ParseError(format!(
            "expected {expected} comma-separated fiber weights, got {}",
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
    let checked_h_power = |power| {
        CohomologyClass::try_h_power(n, power).map_err(|_| {
            GwError::ParseError(format!(
                "invalid class `{raw}`: expected `1`, `H`, or `H^p` with 0 <= p <= n={n}"
            ))
        })
    };
    match raw {
        "1" => Ok(CohomologyClass::one(n)),
        "H" => checked_h_power(1),
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
            checked_h_power(power)
        }
    }
}

fn default_lambda_line_weights(n: usize) -> Result<Vec<Rational>, GwError> {
    let size = n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("projective default weight count overflow".to_string())
    })?;
    let mut weights = Vec::new();
    weights.try_reserve_exact(size).map_err(|_| {
        GwError::UnsupportedInvariant(format!("cannot allocate {size} projective default weights"))
    })?;
    weights.extend((1..=size).map(Rational::from));
    Ok(weights)
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
    theory: &dyn GwTheory,
    genus: usize,
    degree: usize,
    insertions: &[gw_pn::Insertion],
    equivariant: bool,
    localized_twist: bool,
) -> Result<bool, GwError> {
    let Some(total_degree) = insertion_degree(insertions) else {
        return Ok(true);
    };
    let virtual_dimension = checked_virtual_dimension(theory, genus, degree, insertions.len())?;
    if equivariant {
        // Fiber-equivariant twists are represented over a localized
        // coefficient ring, so retain every bounded profile.  Ordinary
        // equivariant pushforwards only vanish in negative parameter degree.
        return Ok(localized_twist
            || usize::try_from(virtual_dimension)
                .ok()
                .is_none_or(|dimension| total_degree >= dimension));
    }
    Ok(usize::try_from(virtual_dimension).ok() == Some(total_degree))
}

fn checked_virtual_dimension(
    theory: &dyn GwTheory,
    genus: usize,
    degree: usize,
    markings: usize,
) -> Result<isize, GwError> {
    let degree = i64::try_from(degree)
        .map_err(|_| GwError::UnsupportedInvariant("curve degree is too large".to_string()))?;
    let curve = CurveClass::new(vec![degree]);
    theory.virtual_dimension(genus, &curve, markings)
}

fn canonical_dimension_theory(
    n: usize,
    twist: Option<&NegativeSplitBundleTwist>,
) -> Result<Box<dyn GwTheory>, GwError> {
    match twist {
        Some(twist) if twist.rank() > 0 => Ok(Box::new(NegativeSplitTotalSpaceTheory::new(
            n,
            twist.degrees().to_vec(),
        )?)),
        _ => Ok(Box::new(ProjectiveSpaceTheory::try_new(n)?)),
    }
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
    fn virasoro_cli_parses_formula_check_and_scan() {
        for argv in [
            vec![
                "gw-pn", "virasoro", "formula", "--n", "0", "--k", "1", "--d", "0",
            ],
            vec![
                "gw-pn",
                "virasoro",
                "check",
                "--n",
                "1",
                "--product-m",
                "1",
                "--d",
                "0,0",
            ],
            vec![
                "gw-pn",
                "virasoro",
                "scan",
                "--n",
                "1",
                "--bundle-twists",
                "0,2",
                "--d-max",
                "0",
            ],
            vec![
                "gw-pn",
                "virasoro",
                "formula",
                "--n",
                "1",
                "--local-completion-twist",
                "-1,-1",
                "--d",
                "1,-1",
                "--insert",
                "tau1(H*xi)",
            ],
        ] {
            let cli = Cli::try_parse_from(argv).unwrap();
            assert!(matches!(cli.command, Commands::Virasoro(_)));
        }

        let cli = Cli::try_parse_from([
            "gw-pn",
            "virasoro",
            "formula",
            "--n",
            "0",
            "--term-limit",
            "17",
        ])
        .unwrap();
        let Commands::Virasoro(VirasoroArgs {
            command: VirasoroCommands::Formula(formula),
        }) = cli.command
        else {
            panic!("expected Virasoro formula command");
        };
        assert_eq!(formula.term_limit, 17);
    }

    #[test]
    fn bundle_twists_require_canonical_normalization_and_permute_canonically() {
        let extreme = format!("{},{}", i128::MIN, i128::MAX);
        assert!(matches!(
            parse_canonical_bundle_twists(&extreme),
            Err(GwError::ParseError(_))
        ));
        assert!(parse_canonical_bundle_twists("-2,0").is_err());
        assert!(parse_canonical_bundle_twists("2,3").is_err());

        let target = |twists: &str| {
            build_virasoro_target(&VirasoroTargetArgs {
                n: 1,
                product_m: None,
                bundle_twists: Some(twists.to_string()),
                local_twist: None,
                local_completion_twist: None,
            })
            .unwrap()
        };
        let left = target("0,1");
        let right = target("1,0");
        assert_eq!(
            left.theory().theory_fingerprint(),
            right.theory().theory_fingerprint()
        );
    }

    #[test]
    fn virasoro_target_selection_uses_physical_multidegrees() {
        let product = build_virasoro_target(&VirasoroTargetArgs {
            n: 1,
            product_m: Some(2),
            bundle_twists: None,
            local_twist: None,
            local_completion_twist: None,
        })
        .unwrap();
        assert_eq!(
            parse_virasoro_degree(&product, Some("2,3"))
                .unwrap()
                .coordinates(),
            &[2, 3]
        );

        let bundle = build_virasoro_target(&VirasoroTargetArgs {
            n: 1,
            product_m: None,
            bundle_twists: Some("0,2".to_string()),
            local_twist: None,
            local_completion_twist: None,
        })
        .unwrap();
        assert_eq!(
            parse_virasoro_degree(&bundle, Some("1,-2"))
                .unwrap()
                .coordinates(),
            &[1, -2]
        );
    }

    #[test]
    fn local_completion_virasoro_uses_bundle_coordinates_and_notation() {
        let completion = build_virasoro_target(&VirasoroTargetArgs {
            n: 1,
            product_m: None,
            bundle_twists: None,
            local_twist: None,
            local_completion_twist: Some("-1,-1".to_string()),
        })
        .unwrap();
        assert!(completion.evaluator().is_ok());
        assert_eq!(
            parse_virasoro_degree(&completion, Some("2,-2"))
                .unwrap()
                .coordinates(),
            &[2, -2]
        );

        let time = parse_virasoro_time(&completion, &["tau1(H*xi)".to_string()]).unwrap();
        let CliVirasoroTheory::LocalCompletion(evaluator) = &completion else {
            panic!("expected compact-completion evaluator");
        };
        let expected_basis = evaluator.compact_theory().basis_id(1, 1).unwrap();
        let factors = time.factors().collect::<Vec<_>>();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (&Descendant::new(1, expected_basis), 1));

        let formula = gw_pn::constraints::virasoro::generate_constraint(
            completion.theory(),
            0,
            2,
            CurveClass::new(vec![1, -1]),
            TimeMonomial::one(),
        )
        .unwrap();
        assert_eq!(formula.sector.degree.coordinates(), &[1, -1]);
    }

    #[test]
    fn ordinary_local_virasoro_generator_fails_before_using_compact_operator() {
        let local = build_virasoro_target(&VirasoroTargetArgs {
            n: 2,
            product_m: None,
            bundle_twists: None,
            local_twist: Some("-3".to_string()),
            local_completion_twist: None,
        })
        .unwrap();
        let error = gw_pn::constraints::virasoro::generate_constraint(
            local.theory(),
            0,
            1,
            CurveClass::new(vec![0]),
            TimeMonomial::one(),
        )
        .unwrap_err();
        assert!(matches!(error, GwError::UnsupportedFeature { .. }));
        assert!(error.to_string().contains("QRR"));
    }

    #[test]
    fn local_virasoro_formula_routes_l0_through_qrr() {
        run_virasoro_formula(VirasoroFormulaArgs {
            target: VirasoroTargetArgs {
                n: 0,
                product_m: None,
                bundle_twists: None,
                local_twist: Some("-1".to_string()),
                local_completion_twist: None,
            },
            k: 0,
            g: 1,
            d: Some("0".to_string()),
            insert: vec!["1".to_string()],
            format: "text".to_string(),
            term_limit: 100,
        })
        .unwrap();
    }

    #[test]
    fn local_virasoro_check_routes_l0_through_qrr() {
        run_virasoro_check(VirasoroCheckArgs {
            target: VirasoroTargetArgs {
                n: 0,
                product_m: None,
                bundle_twists: None,
                local_twist: Some("-1".to_string()),
                local_completion_twist: None,
            },
            k: 0,
            g: 1,
            d: Some("0".to_string()),
            insert: vec!["1".to_string()],
            fiber_weights: None,
            show_formula: false,
            term_limit: 100,
            dependency_limit: 100,
            show_missing: 10,
        })
        .unwrap();
    }

    #[test]
    fn local_virasoro_check_accepts_an_exact_fixed_fiber_specialization() {
        run_virasoro_check(VirasoroCheckArgs {
            target: VirasoroTargetArgs {
                n: 0,
                product_m: None,
                bundle_twists: None,
                local_twist: Some("-1".to_string()),
                local_completion_twist: None,
            },
            k: 0,
            g: 1,
            d: Some("0".to_string()),
            insert: vec!["1".to_string()],
            fiber_weights: Some("7/2".to_string()),
            show_formula: false,
            term_limit: 100,
            dependency_limit: 100,
            show_missing: 10,
        })
        .unwrap();
    }

    #[test]
    fn equivariant_p1_resolvent_validation_uses_semantic_coefficients() {
        run_resolvent(ResolventArgs {
            n: 1,
            g: 1,
            d: 0,
            markings: 1,
            twist: None,
            mode: None,
            equivariant: true,
            validate: true,
        })
        .unwrap();
    }

    #[test]
    fn point_resolvent_falls_back_past_the_packed_graph_work_limit() {
        run_resolvent(ResolventArgs {
            n: 0,
            g: 5,
            d: 0,
            markings: 1,
            twist: None,
            mode: None,
            equivariant: false,
            validate: false,
        })
        .unwrap();
    }

    #[test]
    fn batch_scans_skip_structured_work_limits_but_not_real_failures() {
        let message = batch_skip_reason(GwError::ResourceLimit {
            operation: "stable-graph complexity".to_string(),
            requested: 9,
            limit: 8,
        })
        .unwrap();
        assert!(message.contains("requested 9, limit 8"));
        assert!(matches!(
            batch_skip_reason(GwError::AlgebraFailure("deliberate".to_string())),
            Err(GwError::AlgebraFailure(message)) if message == "deliberate"
        ));
    }

    #[test]
    fn bounded_scan_dimension_filter_respects_coefficient_ring() {
        let insertions = vec![
            tau(0, CohomologyClass::one(1)),
            tau(0, CohomologyClass::h_power(1, 1)),
            tau(0, CohomologyClass::h_power(1, 1)),
        ];
        let p1 = ProjectiveSpaceTheory::new(1);
        assert!(!dimension_compatible(&p1, 0, 0, &insertions, false, false).unwrap());
        assert!(dimension_compatible(&p1, 0, 0, &insertions, true, false).unwrap());

        let p5 = ProjectiveSpaceTheory::new(5);
        assert!(!dimension_compatible(&p5, 2, 0, &[], false, false).unwrap());
        assert!(dimension_compatible(&p5, 2, 0, &[], true, false).unwrap());
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
    fn insertion_parser_rejects_out_of_basis_h_power() {
        let err = parse_insertion(1, "H^2").unwrap_err().to_string();
        assert!(err.contains("0 <= p <= n=1"), "unexpected error: {err}");
        assert!(parse_insertion(0, "H").is_err());
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
