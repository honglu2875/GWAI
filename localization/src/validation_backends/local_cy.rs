//! Ground-truth values for local Calabi-Yau validation targets.
//!
//! These constants are validation oracles, not computation shortcuts.  The
//! local P2 values are from Coates-Iritani, Appendix C, Tables 2 and 3.  The
//! Gopakumar-Vafa table is intentionally kept separate from the Gromov-Witten
//! table because GV invariants must be transformed before comparison with GW
//! computations.

use crate::algebra::Rational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCurveClass {
    pub genus: usize,
    pub degree: usize,
}

pub fn resolved_conifold_gw(genus: usize, degree: usize) -> Option<Rational> {
    if degree == 0 {
        return None;
    }
    gv_to_gw(genus, degree, |h, d| (h == 0 && d == 1).then_some(1))
}

pub fn local_p2_gw(genus: usize, degree: usize) -> Option<Rational> {
    if degree == 0 || genus > 7 || degree > 15 {
        return None;
    }
    let row = LOCAL_P2_GW.get(genus)?;
    let &(num, den) = row.get(degree - 1)?;
    Some(Rational::new(num, den))
}

pub fn local_p2_gv(genus: usize, degree: usize) -> Option<i128> {
    if degree == 0 || genus > 7 || degree > 15 {
        return None;
    }
    let row = LOCAL_P2_GV.get(genus)?;
    row.get(degree - 1).copied()
}

pub fn local_p2_gw_from_gv(genus: usize, degree: usize) -> Option<Rational> {
    local_p2_gw(genus, degree)?;
    gv_to_gw(genus, degree, local_p2_gv)
}

pub fn gv_to_gw(
    genus: usize,
    degree: usize,
    gv: impl Fn(usize, usize) -> Option<i128>,
) -> Option<Rational> {
    if degree == 0 {
        return None;
    }
    let mut total = Rational::zero();
    for cover in divisors(degree) {
        let primitive_degree = degree / cover;
        for source_genus in 0..=genus {
            let gv_value = gv(source_genus, primitive_degree).unwrap_or(0);
            if gv_value == 0 {
                continue;
            }
            let coefficient = sine_factor_coefficient(source_genus, genus - source_genus);
            if coefficient.is_zero() {
                continue;
            }
            let cover_factor = Rational::from(cover).pow_usize(2 * genus).clone()
                / Rational::from(cover).pow_usize(3);
            total += Rational::from(gv_value) * cover_factor * coefficient;
        }
    }
    Some(total)
}

fn sine_factor_coefficient(source_genus: usize, extra_genus: usize) -> Rational {
    let exponent = 2isize * source_genus as isize - 2;
    let base = sine_ratio_series(extra_genus);
    pow_series_signed(&base, exponent, extra_genus)[extra_genus].clone()
}

fn sine_ratio_series(max_degree: usize) -> Vec<Rational> {
    (0..=max_degree)
        .map(|degree| {
            let sign = if degree % 2 == 0 { 1 } else { -1 };
            Rational::from(sign)
                / (Rational::from(2usize).pow_usize(2 * degree)
                    * factorial_rational(2 * degree + 1))
        })
        .collect()
}

fn pow_series_signed(series: &[Rational], exponent: isize, max_degree: usize) -> Vec<Rational> {
    if exponent == 0 {
        let mut out = vec![Rational::zero(); max_degree + 1];
        out[0] = Rational::one();
        return out;
    }
    if exponent > 0 {
        let mut out = vec![Rational::zero(); max_degree + 1];
        out[0] = Rational::one();
        for _ in 0..exponent {
            out = mul_series(&out, series, max_degree);
        }
        return out;
    }

    let positive = pow_series_signed(series, -exponent, max_degree);
    invert_unit_series(&positive, max_degree)
}

fn mul_series(left: &[Rational], right: &[Rational], max_degree: usize) -> Vec<Rational> {
    let mut out = vec![Rational::zero(); max_degree + 1];
    for left_degree in 0..=max_degree {
        if left[left_degree].is_zero() {
            continue;
        }
        for right_degree in 0..=max_degree - left_degree {
            if right[right_degree].is_zero() {
                continue;
            }
            out[left_degree + right_degree] +=
                left[left_degree].clone() * right[right_degree].clone();
        }
    }
    out
}

fn invert_unit_series(series: &[Rational], max_degree: usize) -> Vec<Rational> {
    assert_eq!(series.first(), Some(&Rational::one()));
    let mut out = vec![Rational::zero(); max_degree + 1];
    out[0] = Rational::one();
    for degree in 1..=max_degree {
        let mut sum = Rational::zero();
        for split in 1..=degree {
            sum += series[split].clone() * out[degree - split].clone();
        }
        out[degree] = -sum;
    }
    out
}

fn divisors(value: usize) -> Vec<usize> {
    (1..=value)
        .filter(|candidate| value.is_multiple_of(*candidate))
        .collect()
}

fn factorial_rational(value: usize) -> Rational {
    Rational::from((1..=value).product::<usize>().max(1))
}

const LOCAL_P2_GW: [[(i128, i128); 15]; 8] = [
    [
        (3, 1),
        (-45, 8),
        (244, 9),
        (-12333, 64),
        (211878, 125),
        (-102365, 6),
        (64639725, 343),
        (-1140830253, 512),
        (6742982701, 243),
        (-36001193817, 100),
        (6425982732150, 1331),
        (-9581431054999, 144),
        (2061386799232608, 2197),
        (-37021055156692659, 2744),
        (73982838271394248, 375),
    ],
    [
        (1, 4),
        (-3, 8),
        (-23, 3),
        (3437, 16),
        (-43107, 10),
        (79522, 1),
        (-39826681, 28),
        (803703117, 32),
        (-15878598203, 36),
        (154610243281, 20),
        (-2979940731399, 22),
        (7124283102275, 3),
        (-541814449674696, 13),
        (41013714834701487, 56),
        (-64436279290065616, 5),
    ],
    [
        (1, 80),
        (0, 1),
        (3, 20),
        (-514, 5),
        (43497, 8),
        (-1552743, 8),
        (92569957, 16),
        (-776658618, 5),
        (311565686229, 80),
        (-186103710373, 2),
        (17161329260151, 8),
        (-962191179023583, 20),
        (5278121482133523, 5),
        (-910206655959750921, 40),
        (966725384014894851, 2),
    ],
    [
        (1, 2016),
        (1, 336),
        (1, 56),
        (1480, 63),
        (-1385717, 336),
        (34386105, 112),
        (-4563656185, 288),
        (27816690931, 42),
        (-771022095237, 32),
        (400254094073885, 504),
        (-2722614157619637, 112),
        (9834759858880697, 14),
        (-1628439950424111871, 84),
        (4121486387127690091, 8),
        (-13266967197002009748, 1),
    ],
    [
        (1, 57600),
        (1, 1920),
        (7, 1600),
        (-2491, 900),
        (3865243, 1920),
        (-217225227, 640),
        (364416184789, 11520),
        (-316806697367, 150),
        (726200060335821, 6400),
        (-15051658211781731, 2880),
        (137299697068139103, 640),
        (-3220668414546452353, 400),
        (84382375637970689569, 300),
        (-14824230312581305514377, 1600),
        (46493722208852997775773, 160),
    ],
    [
        (1, 1774080),
        (1, 14080),
        (61, 49280),
        (4471, 22176),
        (-65308319, 98560),
        (5383395285, 19712),
        (-17012987874515, 354816),
        (64688948714407, 12320),
        (-11945278310269797, 28160),
        (350595910152610339, 12672),
        (-13785482612596271967, 8960),
        (465731911358273599411, 6160),
        (-248785036687799870780761, 73920),
        (6809369636793660022747587, 49280),
        (-65332009871525107439528907, 12320),
    ],
    [
        (691, 39626496000),
        (11747, 1320883200),
        (377977, 1100736000),
        (-4874687, 1238328000),
        (202790371913, 1320883200),
        (-24163714857019, 146764800),
        (64139775474690313, 1132185600),
        (-4310034999040379953, 412776000),
        (991900415691784747, 768000),
        (-239501070313053131971001, 1981324800),
        (1350252537724641260419439, 146764800),
        (-164221788876199036010533573, 275184000),
        (2164019137440273660185654977, 63504000),
        (-3029955595814315413860062951, 1728000),
        (433647145446345870048459770393, 5241600),
    ],
    [
        (1, 1916006400),
        (31, 29030400),
        (703, 7603200),
        (11705, 4790016),
        (-8293308997, 319334400),
        (540810103943, 7096320),
        (-20390495664732131, 383201280),
        (675146333220270311, 39916800),
        (-77105305044973449611, 23654400),
        (42491482875357032433349, 95800320),
        (-1655992931289521245824679, 35481600),
        (13400468324230111992071993, 3326400),
        (-23713389495101796065291526451, 79833600),
        (1025143548772512485242765294187, 53222400),
        (-2489343025368827360553366826757, 2217600),
    ],
];

const LOCAL_P2_GV: [[i128; 15]; 8] = [
    [
        3,
        -6,
        27,
        -192,
        1695,
        -17064,
        188454,
        -2228160,
        27748899,
        -360012150,
        4827935937,
        -66537713520,
        938273463465,
        -13491638200194,
        197287568723655,
    ],
    [
        0,
        0,
        -10,
        231,
        -4452,
        80948,
        -1438086,
        25301295,
        -443384578,
        7760515332,
        -135854179422,
        2380305803719,
        -41756224045650,
        733512068799924,
        -12903696488738656,
    ],
    [
        0,
        0,
        0,
        -102,
        5430,
        -194022,
        5784837,
        -155322234,
        3894455457,
        -93050366010,
        2145146041119,
        -48109281322212,
        1055620386953940,
        -22755110195405850,
        483361869975894765,
    ],
    [
        0,
        0,
        0,
        15,
        -3672,
        290853,
        -15363990,
        649358826,
        -23769907110,
        786400843911,
        -24130293606924,
        698473748830878,
        -19298221675559646,
        513289541565539286,
        // The arXiv source table repeats the GW value here.  The GV-to-GW
        // relation and the printed table value force this corrected integer.
        -13226687073790872894,
    ],
    [
        0,
        0,
        0,
        0,
        1386,
        -290400,
        29056614,
        -2003386626,
        109496290149,
        -5094944994204,
        210503102300868,
        -7935125096754762,
        278055282896359878,
        -9179532480730484952,
        288379973286696180135,
    ],
    [
        0,
        0,
        0,
        0,
        -270,
        196857,
        -40492272,
        4741754985,
        -396521732268,
        26383404443193,
        -1485630816648252,
        73613315148586317,
        -3295843339183602162,
        135875843241729533613,
        -5230662528295888702200,
    ],
    [
        0,
        0,
        0,
        0,
        21,
        -90390,
        42297741,
        -8802201084,
        1156156082181,
        -111935744536416,
        8698748079113310,
        -572001241783007370,
        32970159716836634586,
        -1707886552705077581628,
        80979854504456065293006,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        27538,
        -33388020,
        12991744968,
        -2756768768616,
        395499033672279,
        -42968546119317066,
        3786284014554551293,
        -283123099266200799858,
        18542695412600660315361,
        -1088520963699453440916068,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_conifold_gw_matches_closed_formula() {
        assert_eq!(resolved_conifold_gw(0, 3), Some(Rational::new(1, 27)));
        assert_eq!(resolved_conifold_gw(1, 3), Some(Rational::new(1, 36)));
        assert_eq!(resolved_conifold_gw(2, 3), Some(Rational::new(1, 80)));
        assert_eq!(resolved_conifold_gw(3, 3), Some(Rational::new(1, 224)));
    }

    #[test]
    fn local_p2_records_direct_gw_values() {
        assert_eq!(local_p2_gw(0, 2), Some(Rational::new(-45, 8)));
        assert_eq!(local_p2_gw(1, 3), Some(Rational::new(-23, 3)));
        assert_eq!(local_p2_gw(2, 4), Some(Rational::new(-514, 5)));
        assert_eq!(local_p2_gw(4, 1), Some(Rational::new(1, 57600)));
    }

    #[test]
    fn local_p2_gv_values_transform_to_gw_values() {
        for genus in 0..=7 {
            for degree in 1..=15 {
                assert_eq!(
                    local_p2_gw_from_gv(genus, degree),
                    local_p2_gw(genus, degree),
                    "g={genus}, d={degree}"
                );
            }
        }
    }
}
