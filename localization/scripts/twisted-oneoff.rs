use gw_pn::algebra::{RatFun, Rational};
use gw_pn::geometry::CohomologyClass;
use gw_pn::givental::compute_semisimple_graph_value;
use gw_pn::local_oracle::local_p2_gw;
use gw_pn::tau;
use gw_pn::twisted::{
    compute_negative_split_twisted, negative_split_inverse_mirror_map_coefficients,
    negative_split_mirror_map_coefficients, NegativeSplitBundleTwist, TwistedInvariantRequest,
    TwistedProjectiveSpaceProvider,
};

fn main() {
    let local_p2_default = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
    let local_p2_twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
    println!(
        "local-p2-mirror={:?}",
        negative_split_mirror_map_coefficients(2, &local_p2_twist, 4)
    );
    println!(
        "local-p2-inverse-mirror={:?}",
        negative_split_inverse_mirror_map_coefficients(2, &local_p2_twist, 4)
    );
    let local_p2_positive =
        TwistedProjectiveSpaceProvider::inverse_euler_with_positive_fiber_qrr(2, vec![3]).unwrap();
    for degree in 1..=3 {
        let oracle = local_p2_gw(2, degree).unwrap();
        let default_value =
            compute_semisimple_graph_value(&local_p2_default, 2, degree, &[], None).unwrap();
        let positive_value =
            compute_semisimple_graph_value(&local_p2_positive, 2, degree, &[], None).unwrap();
        println!(
            "local-p2-g2-d{degree} default={default_value} positive-fiber={positive_value} oracle={oracle}"
        );
    }
    let local_p2_weight_sets = [
        ([1usize, 2, 4], [0usize]),
        ([1usize, 2, 4], [7usize]),
        ([1usize, 3, 5], [11usize]),
        ([2usize, 5, 9], [13usize]),
        ([3usize, 7, 11], [17usize]),
    ];
    for (base, fiber) in local_p2_weight_sets {
        let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            2,
            vec![3],
            base.into_iter().map(Rational::from).collect(),
            fiber.into_iter().map(Rational::from).collect(),
        )
        .unwrap();
        match compute_semisimple_graph_value(&provider, 2, 2, &[], None) {
            Ok(value) => println!("local-p2-g2-d2 weights={base:?}/{fiber:?} value={value}"),
            Err(err) => println!("local-p2-g2-d2 weights={base:?}/{fiber:?} error={err}"),
        }
    }
    let hhh = vec![
        tau(0, CohomologyClass::h_power(2, 1)),
        tau(0, CohomologyClass::h_power(2, 1)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    for degree in 1..=3 {
        let value =
            compute_semisimple_graph_value(&local_p2_default, 0, degree, &hhh, None).unwrap();
        let oracle = RatFun::from_rational(local_p2_gw(0, degree).unwrap())
            * RatFun::from(degree).pow_usize(3);
        println!("local-p2-g0-d{degree}-HHH value={value} oracle={oracle}");
    }

    let genus_zero_insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let genus_zero_provider = TwistedProjectiveSpaceProvider::new(2, vec![1], false).unwrap();
    let o1_twist = NegativeSplitBundleTwist::new(vec![1]).unwrap();
    println!(
        "O(-1)-p2-mirror={:?}",
        negative_split_mirror_map_coefficients(2, &o1_twist, 4)
    );
    println!(
        "O(-1)-p2-inverse-mirror={:?}",
        negative_split_inverse_mirror_map_coefficients(2, &o1_twist, 4)
    );
    let genus_zero_value =
        compute_semisimple_graph_value(&genus_zero_provider, 0, 1, &genus_zero_insertions, None)
            .unwrap();
    println!("g0d1-pt-pt-H={genus_zero_value}");
    let genus_zero_plus_provider =
        TwistedProjectiveSpaceProvider::inverse_euler_with_positive_fiber_qrr(2, vec![1]).unwrap();
    let genus_zero_plus_value = compute_semisimple_graph_value(
        &genus_zero_plus_provider,
        0,
        1,
        &genus_zero_insertions,
        None,
    )
    .unwrap();
    println!("g0d1-pt-pt-H-positive-fiber={genus_zero_plus_value}");

    let req = TwistedInvariantRequest::new(
        2,
        vec![1],
        2,
        2,
        vec![tau(4, CohomologyClass::h_power(2, 1))],
    )
    .unwrap();
    match compute_negative_split_twisted(&req) {
        Ok(result) => {
            println!("early={}", result.value);
            println!("{}", result.engine);
            for note in result.notes {
                println!("note: {note}");
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }

    for scale in 1..=6 {
        let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_scale(
            2,
            vec![1],
            Rational::from(scale),
        )
        .unwrap();
        let value = compute_semisimple_graph_value(&provider, 2, 2, &req.insertions, None).unwrap();
        println!("scale={scale} value={value}");
    }
    let weight_sets = [
        ([1usize, 2, 4], [0usize]),
        ([1usize, 2, 4], [7usize]),
        ([1usize, 3, 5], [11usize]),
        ([2usize, 5, 9], [13usize]),
        ([3usize, 7, 11], [17usize]),
    ];
    for (base, fiber) in weight_sets {
        let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            2,
            vec![1],
            base.into_iter().map(Rational::from).collect(),
            fiber.into_iter().map(Rational::from).collect(),
        )
        .unwrap();
        let value = compute_semisimple_graph_value(&provider, 2, 2, &req.insertions, None).unwrap();
        println!("weights={base:?}/{fiber:?} value={value}");
    }
    let one_marking_checks = [
        ("tau5(1)", tau(5, CohomologyClass::one(2))),
        ("tau4(H)", tau(4, CohomologyClass::h_power(2, 1))),
        ("tau3(H^2)", tau(3, CohomologyClass::h_power(2, 2))),
    ];
    let o1_provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
        2,
        vec![1],
        vec![Rational::from(1usize), Rational::from(2usize), Rational::from(4usize)],
        vec![Rational::from(0usize)],
    )
    .unwrap();
    for (label, insertion) in one_marking_checks {
        let value = compute_semisimple_graph_value(&o1_provider, 2, 2, &[insertion], None).unwrap();
        println!("O(-1)-g2d2-{label}={value}");
    }
    let positive_fiber_provider =
        TwistedProjectiveSpaceProvider::inverse_euler_with_positive_fiber_qrr(2, vec![1]).unwrap();
    let positive_fiber_value =
        compute_semisimple_graph_value(&positive_fiber_provider, 2, 2, &req.insertions, None)
            .unwrap();
    println!("inverse-euler-positive-fiber-qrr={positive_fiber_value}");
    let euler_provider = TwistedProjectiveSpaceProvider::euler_twist(2, vec![1]).unwrap();
    match compute_semisimple_graph_value(&euler_provider, 2, 2, &req.insertions, None) {
        Ok(value) => println!("euler-mode={value}"),
        Err(err) => println!("euler-mode-error={err}"),
    }
}
