use gw_pn::graphs::stable_graphs;
use std::env;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Args {
    g_min: usize,
    g_max: usize,
    markings_min: usize,
    markings_max: usize,
    include_unstable: bool,
    csv: bool,
    warn_ms: u128,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            g_min: 0,
            g_max: 3,
            markings_min: 0,
            markings_max: 2,
            include_unstable: false,
            csv: false,
            warn_ms: 1_000,
        }
    }
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("error: {err}");
        print_usage();
        std::process::exit(2);
    });

    if args.csv {
        println!("genus,markings,stable,count,elapsed_ms");
    } else {
        println!(
            "stable graph counts for genus {}..={} and markings {}..={}",
            args.g_min, args.g_max, args.markings_min, args.markings_max
        );
    }

    for genus in args.g_min..=args.g_max {
        for markings in args.markings_min..=args.markings_max {
            let stable = 2 * genus + markings > 2;
            if !stable && !args.include_unstable {
                print_row(&args, genus, markings, stable, 0, 0);
                continue;
            }

            let started = Instant::now();
            let count = if stable {
                stable_graphs(genus, markings).len()
            } else {
                0
            };
            let elapsed_ms = started.elapsed().as_millis();
            print_row(&args, genus, markings, stable, count, elapsed_ms);
        }
    }
}

fn print_row(
    args: &Args,
    genus: usize,
    markings: usize,
    stable: bool,
    count: usize,
    elapsed_ms: u128,
) {
    if args.csv {
        println!("{genus},{markings},{stable},{count},{elapsed_ms}");
        return;
    }

    let warning = if elapsed_ms >= args.warn_ms {
        " slow"
    } else {
        ""
    };
    println!(
        "g={genus:<2} markings={markings:<2} stable={stable:<5} graphs={count:<8} elapsed={elapsed_ms}ms{warning}"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--g-min" => args.g_min = parse_value(&arg, iter.next())?,
            "--g-max" => args.g_max = parse_value(&arg, iter.next())?,
            "--markings-min" | "--n-min" => args.markings_min = parse_value(&arg, iter.next())?,
            "--markings-max" | "--n-max" => args.markings_max = parse_value(&arg, iter.next())?,
            "--warn-ms" => args.warn_ms = parse_value(&arg, iter.next())?,
            "--include-unstable" => args.include_unstable = true,
            "--csv" => args.csv = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    if args.g_min > args.g_max {
        return Err("--g-min cannot exceed --g-max".to_string());
    }
    if args.markings_min > args.markings_max {
        return Err("--markings-min cannot exceed --markings-max".to_string());
    }
    Ok(args)
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, String>
where
    T: std::str::FromStr,
{
    let value = value.ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse()
        .map_err(|_| format!("invalid value `{value}` for {flag}"))
}

fn print_usage() {
    eprintln!(
        "usage: scripts/run-graph-stats.sh [options]\n\
         \n\
         options:\n\
           --g-min N             first genus, default 0\n\
           --g-max N             last genus, default 3\n\
           --markings-min N      first marking count, default 0\n\
           --markings-max N      last marking count, default 2\n\
           --include-unstable    print unstable ranges as zero-count rows\n\
           --csv                 print CSV rows\n\
           --warn-ms N           mark rows at least this slow, default 1000\n\
           -h, --help            show this help"
    );
}
