//! Tightness: for the exact bounds, the constructed witness still differs by
//! at least tick/2 after n-1 shared bars and by less after n, when replayed
//! through the reference implementations (first-value and SMA-seeded runs;
//! adjusted-weight runs are only checked to agree at n).

use warmupbound::witness::extreme_witness;
use warmupbound::{bound, Indicator, Kind, Params, Seeding, Verdict};

const MARKETS: [(&str, &str, &str); 6] = [
    ("90", "110", "0.01"),
    ("90", "110", "0.5"),
    ("0", "1", "0.000001"),
    ("4.25", "4.75", "0.0025"),
    ("-3", "7", "0.001"),
    ("1000", "1300", "0.25"),
];

fn exact_indicators(period: usize) -> Vec<Indicator> {
    vec![
        Indicator::Ema { period },
        Indicator::Rma { period },
        Indicator::Atr { period },
        Indicator::Sma { period },
    ]
}

#[test]
fn witness_attains_the_bound_at_n_minus_one_and_not_at_n() {
    let mut checked = 0;
    for (lo, hi, tick) in MARKETS {
        let params = Params::from_decimals(lo, hi, tick, None).unwrap();
        let half = params.tick.value / 2.0;
        for period in [2, 3, 5, 9, 14, 20, 50, 200] {
            for ind in exact_indicators(period) {
                let report = bound(&ind, &params).unwrap();
                assert_eq!(report.kind, Kind::Exact, "{ind}");
                let Verdict::Proven(n) = report.verdict else {
                    panic!("{ind} should be finite")
                };
                assert!(
                    n >= 1,
                    "{ind} {lo}..{hi} tick {tick}: range exceeds tick/2 so n >= 1"
                );
                let base = extreme_witness(&ind, &params, n as usize + 5);
                assert!(base.is_admissible(&params));
                for sa in ind.conventions() {
                    for sb in ind.conventions() {
                        let w = warmupbound::witness::Witness {
                            seeding_a: sa,
                            seeding_b: sb,
                            ..base.clone()
                        };
                        let gaps = w.gaps(&ind);
                        // Adjusted weights give recent bars more weight than 1 - alpha while the
                        // history is finite, so they converge faster; tightness is claimed for the
                        // recursive conventions, soundness (below) for all of them.
                        let recursive = sa != Seeding::Adjusted && sb != Seeding::Adjusted;
                        if n >= 2 && recursive {
                            let g = gaps[n as usize - 2].expect("defined after seeding");
                            assert!(g >= half, "{ind} [{lo},{hi}] tick {tick} {sa}/{sb}: gap {g} at n-1={} < tick/2", n - 1);
                        }
                        let at_n = gaps[n as usize - 1].expect("defined");
                        assert!(at_n < half, "{ind} [{lo},{hi}] tick {tick} {sa}/{sb}: gap {at_n} at n={n} >= tick/2");
                        for g in &gaps[n as usize..] {
                            assert!(g.unwrap() < half);
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 500, "only {checked} witness replays");
}

#[test]
fn witness_gap_matches_closed_form_bound_value() {
    let params = Params::from_decimals("90", "110", "0.01", None).unwrap();
    for ind in [
        Indicator::Ema { period: 20 },
        Indicator::Rma { period: 14 },
        Indicator::Atr { period: 14 },
    ] {
        let report = bound(&ind, &params).unwrap();
        let b = report.gap_bound.unwrap();
        let w = extreme_witness(&ind, &params, 60);
        for (i, g) in w.gaps(&ind).iter().enumerate() {
            let expected = b.float(i as u64 + 1);
            let g = g.unwrap();
            assert!(
                (g - expected).abs() <= 1e-9 * expected.max(1e-12),
                "{ind} k={} gap {g} vs {expected}",
                i + 1
            );
        }
    }
}

#[test]
fn tiny_range_needs_no_warmup_and_sma_needs_the_full_window() {
    let params = Params::from_decimals("100", "100.004", "0.01", None).unwrap();
    for ind in exact_indicators(10) {
        assert_eq!(
            bound(&ind, &params).unwrap().verdict,
            Verdict::Proven(0),
            "{ind}"
        );
    }
    // D/N = 0.5 >= tick/2 = 0.005: every one of the 20 bars matters.
    let params = Params::from_decimals("90", "100", "0.01", None).unwrap();
    assert_eq!(
        bound(&Indicator::Sma { period: 20 }, &params)
            .unwrap()
            .verdict,
        Verdict::Proven(20)
    );
    // D (N - k)/N < 5 needs N - k < 5 -> k = 16.
    let coarse = Params::from_decimals("90", "110", "10", None).unwrap();
    assert_eq!(
        bound(&Indicator::Sma { period: 20 }, &coarse)
            .unwrap()
            .verdict,
        Verdict::Proven(16)
    );
}

#[test]
fn atr_needs_exactly_one_bar_more_than_rma() {
    for (lo, hi, tick) in MARKETS {
        let params = Params::from_decimals(lo, hi, tick, None).unwrap();
        for period in [2, 14, 100] {
            let rma = bound(&Indicator::Rma { period }, &params).unwrap().verdict;
            let atr = bound(&Indicator::Atr { period }, &params).unwrap().verdict;
            match (rma, atr) {
                (Verdict::Proven(r), Verdict::Proven(a)) => assert_eq!(a, r + 1),
                other => panic!("{other:?}"),
            }
        }
    }
}

#[test]
fn rejects_invalid_inputs() {
    assert!(Params::from_decimals("110", "90", "0.01", None).is_err());
    assert!(Params::from_decimals("90", "110", "0", None).is_err());
    assert!(Params::from_decimals("90", "110", "1e-2", None).is_err());
    let p = Params::from_decimals("90", "110", "0.01", None).unwrap();
    assert!(bound(&Indicator::Ema { period: 1 }, &p).is_err());
    // A move limit above the range is clamped to the range.
    let clamped = Params::from_decimals("90", "110", "0.01", Some("500")).unwrap();
    assert_eq!(clamped.max_move.value, 20.0);
}
