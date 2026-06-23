use gw_pn::algebra::Rational;
use gw_pn::geometry::CohomologyClass;
use gw_pn::givental::compute_semisimple_graph_value;
use gw_pn::tau;
use gw_pn::twisted::TwistedProjectiveSpaceProvider;

fn main() {
    let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
        2,
        vec![1],
        vec![Rational::from(1usize), Rational::from(2usize), Rational::from(4usize)],
        vec![Rational::from(0usize)],
    )
    .unwrap();
    let checks = [
        ("tau5(1)", tau(5, CohomologyClass::one(2)), "0"),
        ("tau4(H)", tau(4, CohomologyClass::h_power(2, 1)), "-1/480"),
        (
            "tau3(H^2)",
            tau(3, CohomologyClass::h_power(2, 2)),
            "-7/480",
        ),
    ];
    for (label, insertion, oracle) in checks {
        let value = compute_semisimple_graph_value(&provider, 2, 2, &[insertion], None).unwrap();
        println!("{label} value={value} oracle={oracle}");
    }
}
