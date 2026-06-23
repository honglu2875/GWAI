use gw_pn::algebra::RatFun;
use gw_pn::givental::compute_semisimple_graph_coefficient_range;
use gw_pn::twisted::TwistedProjectiveSpaceProvider;
use gw_pn::validation_backends::local_cy::{local_p2_gw, resolved_conifold_gw};
use std::env;
use std::io::{self, Write};

fn main() {
    let config = ScriptConfig::from_args();
    match config.target {
        ScriptTarget::Conifold => {
            print_conifold(config.genus, config.degree_min, config.degree_max)
        }
        ScriptTarget::LocalP2 => print_local_p2(config.genus, config.degree_min, config.degree_max),
        ScriptTarget::Both => {
            print_conifold(config.genus, config.degree_min, config.degree_max);
            print_local_p2(config.genus, config.degree_min, config.degree_max);
        }
    }
}

#[derive(Clone, Copy)]
enum ScriptTarget {
    Conifold,
    LocalP2,
    Both,
}

struct ScriptConfig {
    target: ScriptTarget,
    genus: usize,
    degree_min: usize,
    degree_max: usize,
}

impl ScriptConfig {
    fn from_args() -> Self {
        let mut target = ScriptTarget::Both;
        let mut genus = 2usize;
        let mut degree_min = 1usize;
        let mut degree_max = 4usize;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--target" => {
                    target = match args.next().as_deref() {
                        Some("conifold") => ScriptTarget::Conifold,
                        Some("local-p2") => ScriptTarget::LocalP2,
                        Some("both") => ScriptTarget::Both,
                        other => panic!("expected --target conifold|local-p2|both, got {other:?}"),
                    };
                }
                "--genus" => {
                    genus = args
                        .next()
                        .expect("missing --genus value")
                        .parse()
                        .expect("invalid --genus value");
                }
                "--d-min" => {
                    degree_min = args
                        .next()
                        .expect("missing --d-min value")
                        .parse()
                        .expect("invalid --d-min value");
                }
                "--d-max" => {
                    degree_max = args
                        .next()
                        .expect("missing --d-max value")
                        .parse()
                        .expect("invalid --d-max value");
                }
                other => panic!("unknown argument {other}"),
            }
        }

        Self {
            target,
            genus,
            degree_min,
            degree_max,
        }
    }
}

fn print_conifold(genus: usize, degree_min: usize, degree_max: usize) {
    let provider = TwistedProjectiveSpaceProvider::new(1, vec![1, 1], false).unwrap();
    println!("resolved conifold raw graph values, genus {genus}");
    println!("degree,value-equals-oracle,lambda-line-limit,oracle,limit-ratio");
    print_rows_from_one_series(
        &provider,
        genus,
        degree_min,
        degree_max,
        resolved_conifold_gw,
    );
}

fn print_local_p2(genus: usize, degree_min: usize, degree_max: usize) {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    println!();
    println!("local P2 raw graph values, genus {genus}");
    println!("degree,value-equals-oracle,lambda-line-limit,oracle,limit-ratio");
    print_rows_from_one_series(&provider, genus, degree_min, degree_max, local_p2_gw);
}

fn print_rows_from_one_series(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree_min: usize,
    degree_max: usize,
    oracle_fn: fn(usize, usize) -> Option<gw_pn::algebra::Rational>,
) {
    let raw_series = std::panic::catch_unwind(|| {
        compute_semisimple_graph_coefficient_range(
            provider,
            genus,
            degree_min,
            degree_max,
            &[],
            None,
        )
    });

    for degree in degree_min..=degree_max {
        let oracle = RatFun::from_rational(oracle_fn(genus, degree).unwrap());
        match &raw_series {
            Ok(Ok(values)) => {
                print_row_from_raw(degree, values[degree - degree_min].clone(), oracle)
            }
            Ok(Err(err)) => println!("{degree},error:{err},,{oracle},"),
            Err(_) => println!("{degree},panic,,{oracle},"),
        }
        io::stdout().flush().unwrap();
    }
}

fn print_row_from_raw(degree: usize, raw: RatFun, oracle: RatFun) {
    let value = match raw.as_rational() {
        Some(value) => Ok(value),
        None => raw.nonequivariant_limit_line(0, &[gw_pn::algebra::Rational::one()]),
    };
    match value {
        Ok(limit) => {
            let limit = RatFun::from_rational(limit);
            let ratio = if oracle.is_zero() {
                "undefined".to_string()
            } else {
                (limit.clone() / oracle.clone()).to_string()
            };
            println!("{degree},{},{limit},{oracle},{ratio}", limit == oracle);
        }
        Err(err) => {
            println!("{degree},false,limit-error:{err},{oracle},");
        }
    }
}
