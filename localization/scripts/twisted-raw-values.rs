use gw_pn::algebra::RatFun;
use gw_pn::givental::compute_semisimple_graph_value;
use gw_pn::local_oracle::{local_p2_gw, resolved_conifold_gw};
use gw_pn::twisted::TwistedProjectiveSpaceProvider;
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
    println!("degree,raw-equals-oracle,lambda-line-limit,oracle,limit-ratio");
    for degree in degree_min..=degree_max {
        let oracle = RatFun::from_rational(resolved_conifold_gw(genus, degree).unwrap());
        print_row(&provider, genus, degree, oracle);
    }
}

fn print_local_p2(genus: usize, degree_min: usize, degree_max: usize) {
    let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    println!();
    println!("local P2 raw graph values, genus {genus}");
    println!("degree,raw-equals-oracle,lambda-line-limit,oracle,limit-ratio");
    for degree in degree_min..=degree_max {
        let oracle = RatFun::from_rational(local_p2_gw(genus, degree).unwrap());
        print_row(&provider, genus, degree, oracle);
    }
}

fn print_row(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree: usize,
    oracle: RatFun,
) {
    let raw = std::panic::catch_unwind(|| {
        compute_semisimple_graph_value(provider, genus, degree, &[], None)
    });
    match raw {
        Ok(Ok(raw)) => match raw.nonequivariant_limit_line(0, &[gw_pn::algebra::Rational::one()]) {
            Ok(limit) => {
                let limit = RatFun::from_rational(limit);
                let ratio = if oracle.is_zero() {
                    "undefined".to_string()
                } else {
                    (limit.clone() / oracle.clone()).to_string()
                };
                println!("{degree},{},{limit},{oracle},{ratio}", raw == oracle);
            }
            Err(err) => {
                println!("{degree},{},limit-error:{err},{oracle},", raw == oracle);
            }
        },
        Ok(Err(err)) => {
            println!("{degree},error:{err},,{oracle},");
        }
        Err(_) => {
            println!("{degree},panic,,{oracle},");
        }
    }
    io::stdout().flush().unwrap();
}
