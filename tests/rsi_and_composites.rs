//! RSI's unbounded case, and the gap between composite bounds and the best
//! witness a randomised search can find.

use warmupbound::bounds::Decimal;
use warmupbound::witness::{condition_mask, rsi_flat_witness, search_witness};
use warmupbound::{bound, Indicator, Kind, MacdOutput, Params, Seeding, Verdict};

#[test]
fn rsi_has_no_unconditional_warmup_and_the_flat_witness_proves_it() {
    let params = Params::from_decimals("90", "110", "0.01", None).unwrap();
    for period in [2, 5, 9, 14, 15, 20, 50, 100] {
        let ind = Indicator::Rsi {
            period,
            min_move: None,
        };
        let report = bound(&ind, &params).unwrap();
        assert_eq!(report.verdict, Verdict::Unbounded);
        let w = rsi_flat_witness(period, &params, 10_000);
        assert!(w.is_admissible(&params));
        for sa in Seeding::ALL {
            for sb in Seeding::ALL {
                let w = warmupbound::witness::Witness {
                    seeding_a: sa,
                    seeding_b: sb,
                    ..w.clone()
                };
                let gaps = w.gaps(&ind);
                assert_eq!(gaps.len(), 10_000);
                // Run A's average loss is exactly 0 and B's average gain is exactly 0, so the
                // RSIs are pinned at 100 and 0 for as long as f64 can represent A's decaying
                // average gain. For periods >= 15 that is the full 10,000 bars; shorter periods
                // underflow first, which is an artefact of f64, not a loss of the witness.
                let closes: Vec<f64> = w.run_a().iter().map(|b| b.close).collect();
                let (gain_a, _) = warmupbound::indicators::rsi_averages(&closes, period, sa);
                let closes: Vec<f64> = w.run_b().iter().map(|b| b.close).collect();
                let (_, loss_b) = warmupbound::indicators::rsi_averages(&closes, period, sb);
                let hist = w.hist_a.len();
                let mut representable = 0;
                for (k, g) in gaps.iter().enumerate() {
                    if gain_a[hist + k].unwrap() == 0.0 || loss_b[hist + k].unwrap() == 0.0 {
                        break;
                    }
                    let g = g.expect("defined after seeding");
                    assert!(
                        g >= 99.999,
                        "RSI({period}) {sa}/{sb}: gap {g} after {} flat bars",
                        k + 1
                    );
                    representable += 1;
                }
                if period >= 15 {
                    assert_eq!(representable, 10_000, "RSI({period})");
                } else {
                    assert!(
                        representable >= 700,
                        "RSI({period}): only {representable} bars before underflow"
                    );
                }
            }
        }
    }
}

#[test]
fn stoch_rsi_without_assumptions_is_unbounded() {
    let params = Params::from_decimals("90", "110", "0.01", None).unwrap();
    let only_move = Indicator::StochRsi {
        period: 14,
        k: 14,
        min_move: Some(Decimal::parse("1").unwrap()),
        min_range: None,
    };
    assert_eq!(
        bound(&only_move, &params).unwrap().verdict,
        Verdict::Unbounded
    );
}

#[test]
fn flat_witness_violates_the_conditional_assumption() {
    let params = Params::from_decimals("90", "110", "0.01", None).unwrap();
    let ind = Indicator::Rsi {
        period: 14,
        min_move: Some(Decimal::parse("0.05").unwrap()),
    };
    let report = bound(&ind, &params).unwrap();
    assert_eq!(report.kind, Kind::Conditional);
    assert!(matches!(report.verdict, Verdict::Proven(_)));
    let w = rsi_flat_witness(14, &params, 2_000);
    let mask = condition_mask(&ind, &w);
    // Activity decays geometrically in a flat market, so the assumption fails for good.
    let first_bad = mask
        .iter()
        .position(|ok| !ok)
        .expect("assumption must fail eventually");
    assert!(mask[first_bad..].iter().all(|ok| !ok));
}

#[test]
fn conditional_rsi_bound_scales_with_the_activity_floor() {
    let params = Params::from_decimals("90", "110", "0.01", Some("2")).unwrap();
    let n = |m: &str| match bound(
        &Indicator::Rsi {
            period: 14,
            min_move: Some(Decimal::parse(m).unwrap()),
        },
        &params,
    )
    .unwrap()
    .verdict
    {
        Verdict::Proven(n) => n,
        v => panic!("{v:?}"),
    };
    // Halving the activity floor doubles the bound, costing ln 2 / ln(14/13) ~ 9.35 bars.
    let (a, b, c) = (n("1"), n("0.5"), n("0.25"));
    assert!(a < b && b < c);
    assert!(
        (9..=10).contains(&(b - a)) && (9..=10).contains(&(c - b)),
        "{a} {b} {c}"
    );
}

fn composite_cases() -> Vec<(Indicator, Params)> {
    let p = |lo: &str, hi: &str, tick: &str| Params::from_decimals(lo, hi, tick, None).unwrap();
    vec![
        (
            Indicator::Macd {
                fast: 12,
                slow: 26,
                signal: 9,
                output: MacdOutput::Line,
            },
            p("90", "110", "0.01"),
        ),
        (
            Indicator::Macd {
                fast: 12,
                slow: 26,
                signal: 9,
                output: MacdOutput::Signal,
            },
            p("90", "110", "0.01"),
        ),
        (
            Indicator::Macd {
                fast: 12,
                slow: 26,
                signal: 9,
                output: MacdOutput::Histogram,
            },
            p("90", "110", "0.01"),
        ),
        (
            Indicator::Macd {
                fast: 5,
                slow: 35,
                signal: 5,
                output: MacdOutput::Signal,
            },
            p("0", "1", "0.0001"),
        ),
        (
            Indicator::Kama {
                period: 10,
                fast: 2,
                slow: 30,
            },
            p("90", "110", "0.01"),
        ),
        (
            Indicator::Kama {
                period: 5,
                fast: 2,
                slow: 12,
            },
            p("0", "1", "0.001"),
        ),
        (
            Indicator::Rsi {
                period: 14,
                min_move: Some(Decimal::parse("0.3").unwrap()),
            },
            {
                let mut x = p("90", "110", "0.01");
                x.max_move = Decimal::parse("1").unwrap();
                x
            },
        ),
    ]
}

#[test]
fn composite_bounds_dominate_the_best_randomised_witness_and_the_gap_is_finite() {
    for (ind, params) in composite_cases() {
        let report = bound(&ind, &params).unwrap();
        assert_ne!(report.kind, Kind::Exact);
        let Verdict::Proven(n) = report.verdict else {
            panic!("{ind}: {:?}", report.verdict)
        };
        let (found, w) = search_witness(&ind, &params, n as usize + 10, 150, 7);
        let w = w.unwrap_or_else(|| panic!("{ind}: search found nothing"));
        assert!(w.is_admissible(&params));
        assert!(
            found <= n,
            "{ind}: witness forces {found} > proven {n}: the bound is unsound"
        );
        // Finite, and not vacuous: the witness reaches at least half the bound.
        let slack = n - found;
        assert!(
            found * 2 >= n,
            "{ind}: bound {n} vs witness {found}, slack {slack}"
        );
        // Replay confirms the reported forced warm-up.
        let gaps = w.gaps(&ind);
        let mask = condition_mask(&ind, &w);
        if found >= 2 {
            let i = found as usize - 2;
            assert!(
                mask[i] && gaps[i].unwrap() >= params.tick.value / 2.0,
                "{ind}"
            );
        }
    }
}

#[test]
fn witness_csv_contains_both_runs() {
    let params = Params::from_decimals("90", "110", "0.5", None).unwrap();
    let ind = Indicator::Ema { period: 5 };
    let w = warmupbound::witness::extreme_witness(&ind, &params, 12);
    let csv = w.to_csv(&ind);
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "bar,phase,shared_bars,high_a,low_a,close_a,value_a,high_b,low_b,close_b,value_b,gap"
    );
    assert_eq!(lines.len(), 1 + w.hist_a.len() + 12);
    let last: Vec<&str> = lines.last().unwrap().split(',').collect();
    assert_eq!(last[1], "shared");
    assert_eq!(last[2], "12");
    let gap: f64 = last[11].parse().unwrap();
    assert!((gap - 20.0 * (4.0f64 / 6.0).powi(12)).abs() < 1e-9);
    assert!(lines[1].starts_with("0,history,0,110,110,110,110,90,90,90,90,20"));
}
