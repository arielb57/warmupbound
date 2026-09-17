//! Soundness: random bounded price paths, random histories of different
//! lengths and random seeding conventions never differ by tick/2 or more at
//! or after the proven warm-up.

use proptest::prelude::*;
use proptest::test_runner::Config;
use warmupbound::witness::{condition_mask, Witness};
use warmupbound::{bound, Bar, Indicator, MacdOutput, Params, Seeding, Verdict};

fn config(cases: u32) -> Config {
    Config {
        cases,
        max_shrink_time: 30_000,
        max_shrink_iters: 2_000,
        ..Config::default()
    }
}

#[derive(Clone, Debug)]
struct Market {
    lo: String,
    hi: String,
    tick: String,
    /// Fraction of the range allowed per bar, in percent (100 = unconstrained).
    move_pct: u32,
}

impl Market {
    fn params(&self) -> Params {
        let range = self.hi.parse::<f64>().unwrap() - self.lo.parse::<f64>().unwrap();
        let max_move = if self.move_pct >= 100 {
            None
        } else {
            Some(format!("{:.4}", range * self.move_pct as f64 / 100.0))
        };
        Params::from_decimals(&self.lo, &self.hi, &self.tick, max_move.as_deref()).unwrap()
    }
}

/// tick/range between roughly 1e-4 and 5e-2.
fn market() -> impl Strategy<Value = Market> {
    (
        -5000i64..20000,
        50u64..5000,
        1u64..50,
        1u32..=4,
        prop_oneof![Just(100u32), 5u32..100],
    )
        .prop_map(|(lo_c, range_c, mant, extra_digits, move_pct)| {
            let lo = lo_c as f64 / 100.0;
            let hi = lo + range_c as f64 / 100.0;
            let tick_value =
                range_c as f64 / 100.0 * mant as f64 / 1000.0 / 10f64.powi(extra_digits as i32 - 1);
            Market {
                lo: format!("{lo:.2}"),
                hi: format!("{hi:.2}"),
                tick: format!("{tick_value:.8}"),
                move_pct,
            }
        })
}

/// A path of `len` closes walking backwards or forwards from `start`, respecting the move limit.
fn walk(params: &Params, start: f64, len: usize, style: u8, seed: &[f64]) -> Vec<f64> {
    let (lo, hi, m) = (params.lo, params.hi, params.max_move.value);
    let mut p = start;
    (0..len)
        .map(|i| {
            let u = seed[i % seed.len()];
            let step = match style % 4 {
                0 => (u * 2.0 - 1.0) * m,
                1 => m * if i % 2 == 0 { 1.0 } else { -1.0 },
                2 => m * u.signum() * if u > 0.5 { 1.0 } else { -1.0 },
                _ => 0.0,
            };
            p = (p + step).clamp(lo, hi);
            p
        })
        .collect()
}

fn bars(params: &Params, closes: &[f64], wick: &[f64]) -> Vec<Bar> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let w = wick[i % wick.len()] * params.range.value;
            Bar {
                high: (c + w).min(params.hi),
                low: (c - w * 0.7).max(params.lo),
                close: c,
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct Scenario {
    extra_a: usize,
    extra_b: usize,
    style_a: u8,
    style_b: u8,
    style_s: u8,
    start: f64,
    conv_a: usize,
    conv_b: usize,
    noise: Vec<f64>,
}

fn scenario() -> impl Strategy<Value = Scenario> {
    (
        0usize..30,
        0usize..30,
        any::<u8>(),
        any::<u8>(),
        any::<u8>(),
        0.0f64..1.0,
        0usize..3,
        0usize..3,
        prop::collection::vec(0.0f64..1.0, 7..40),
    )
        .prop_map(
            |(extra_a, extra_b, style_a, style_b, style_s, start, conv_a, conv_b, noise)| {
                Scenario {
                    extra_a,
                    extra_b,
                    style_a,
                    style_b,
                    style_s,
                    start,
                    conv_a,
                    conv_b,
                    noise,
                }
            },
        )
}

/// Builds two runs sharing `shared_len` bars, each with a fully seeded history of its own length.
fn build(ind: &Indicator, params: &Params, s: &Scenario, shared_len: usize) -> Witness {
    let convs = ind.conventions();
    let seeding_a = convs[s.conv_a % convs.len()];
    let seeding_b = convs[s.conv_b % convs.len()];
    let start = params.lo + s.start * params.range.value;
    let shared_closes = walk(params, start, shared_len, s.style_s, &s.noise);
    let first = shared_closes[0];
    let rev_noise: Vec<f64> = s.noise.iter().rev().cloned().collect();
    // Histories are generated backwards from the first shared close, so the move limit holds at the seam.
    let mut ha = walk(
        params,
        first,
        ind.seeded_history_len(seeding_a) + s.extra_a,
        s.style_a,
        &s.noise,
    );
    let mut hb = walk(
        params,
        first,
        ind.seeded_history_len(seeding_b) + s.extra_b,
        s.style_b,
        &rev_noise,
    );
    if s.style_a % 7 == 6 && params.max_move.value >= params.range.value {
        ha.iter_mut().for_each(|x| *x = params.hi);
    }
    if s.style_b % 7 == 6 && params.max_move.value >= params.range.value {
        hb.iter_mut().for_each(|x| *x = params.lo);
    }
    ha.reverse();
    hb.reverse();
    Witness {
        hist_a: bars(params, &ha, &s.noise),
        hist_b: bars(params, &hb, &rev_noise),
        shared: bars(params, &shared_closes, &rev_noise),
        seeding_a,
        seeding_b,
        description: "property test".into(),
    }
}

fn check(ind: &Indicator, m: &Market, s: &Scenario, max_n: u64) -> Result<(), TestCaseError> {
    let params = m.params();
    let report = bound(ind, &params).unwrap();
    let Verdict::Proven(n) = report.verdict else {
        return Err(TestCaseError::fail(format!(
            "{ind}: expected a finite bound, got {:?}",
            report.verdict
        )));
    };
    if n > max_n {
        return Ok(());
    }
    let w = build(ind, &params, s, n as usize + 40);
    prop_assert!(
        w.is_admissible(&params),
        "generator produced an inadmissible path"
    );
    let gaps = w.gaps(ind);
    let mask = condition_mask(ind, &w);
    let half = params.tick.value / 2.0;
    for k in n.max(1)..=gaps.len() as u64 {
        let i = k as usize - 1;
        if !mask[i] {
            continue;
        }
        if let Some(g) = gaps[i] {
            prop_assert!(
                g < half,
                "{ind} [{}, {}] tick {} move {}: gap {g} >= tick/2 after {k} shared bars (n = {n}), {} vs {}",
                m.lo, m.hi, m.tick, params.max_move.value, w.seeding_a, w.seeding_b
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(96))]

    #[test]
    fn ema_rma_sma_atr_never_exceed_half_tick_after_n(m in market(), s in scenario(), period in 2usize..40, which in 0u8..4) {
        let ind = match which {
            0 => Indicator::Ema { period },
            1 => Indicator::Rma { period },
            2 => Indicator::Sma { period },
            _ => Indicator::Atr { period },
        };
        check(&ind, &m, &s, 20_000)?;
    }

    #[test]
    fn macd_outputs_never_exceed_half_tick_after_n(m in market(), s in scenario(), fast in 2usize..15, extra in 1usize..20, signal in 2usize..12, out in 0u8..3) {
        let output = [MacdOutput::Line, MacdOutput::Signal, MacdOutput::Histogram][out as usize];
        check(&Indicator::Macd { fast, slow: fast + extra, signal, output }, &m, &s, 20_000)?;
    }

    #[test]
    fn rsi_conditional_bound_holds_where_activity_is_high_enough(m in market(), s in scenario(), period in 2usize..30, m_pct in 1u32..40) {
        let params = m.params();
        let min_move = format!("{:.6}", params.max_move.value * m_pct as f64 / 100.0);
        let ind = Indicator::Rsi { period, min_move: Some(warmupbound::bounds::Decimal::parse(&min_move).unwrap()) };
        // RSI tolerance is in RSI points, independent of the price tick.
        let rsi_market = Market { tick: "0.01".into(), ..m };
        check(&ind, &rsi_market, &s, 20_000)?;
    }

    #[test]
    fn stoch_rsi_conditional_bound_holds(m in market(), s in scenario(), period in 2usize..20, k in 2usize..10, m_pct in 5u32..40) {
        let params = m.params();
        let min_move = format!("{:.6}", params.max_move.value * m_pct as f64 / 100.0);
        let ind = Indicator::StochRsi {
            period,
            k,
            min_move: Some(warmupbound::bounds::Decimal::parse(&min_move).unwrap()),
            min_range: Some(warmupbound::bounds::Decimal::parse("5").unwrap()),
        };
        let stoch_market = Market { tick: "0.1".into(), ..m };
        check(&ind, &stoch_market, &s, 20_000)?;
    }
}

proptest! {
    #![proptest_config(config(32))]

    #[test]
    fn kama_never_exceeds_half_tick_after_n(m in market(), s in scenario(), period in 2usize..15, fast in 2usize..5, slow in 10usize..40) {
        check(&Indicator::Kama { period, fast, slow }, &m, &s, 12_000)?;
    }
}

/// Failure mode outside the stated precondition: if one run starts its SMA seed
/// inside the shared window, the flat seed weights let it lag more than a
/// recursive state would, and the gap at n can exceed tick/2.
#[test]
fn sma_seed_inside_the_shared_window_breaks_the_precondition() {
    let period = 10usize;
    let params = Params::from_decimals("0", "1", "0.001", None).unwrap();
    let ind = Indicator::Ema { period };
    let Verdict::Proven(n) = bound(&ind, &params).unwrap().verdict else {
        panic!()
    };
    let alpha = 2.0 / (period as f64 + 1.0);
    let q = 1.0 - alpha;
    let n = n as usize;
    // Bars where the recursive run weights the price more than the late-seeded run are set high.
    let shared: Vec<Bar> = (0..n)
        .map(|i| {
            let k = (n - 1 - i) as i32;
            let recursive_weight = alpha * q.powi(k);
            let seeded_weight = q.powi((n - period) as i32) / period as f64;
            Bar::flat(if i < period && recursive_weight > seeded_weight {
                1.0
            } else {
                0.0
            })
        })
        .collect();
    let late = Witness {
        hist_a: vec![Bar::flat(1.0); period],
        hist_b: vec![],
        shared: shared.clone(),
        seeding_a: Seeding::FirstValue,
        seeding_b: Seeding::SmaSeeded,
        description: "run B seeds inside the shared window".into(),
    };
    let gap = late.gaps(&ind)[n - 1].unwrap();
    assert!(
        gap >= params.tick.value / 2.0,
        "expected a violation, gap {gap}"
    );
    // With B's seed completed before the window, the same shared path agrees at n.
    let seeded = Witness {
        hist_b: vec![Bar::flat(0.0); period],
        ..late
    };
    assert!(seeded.gaps(&ind)[n - 1].unwrap() < params.tick.value / 2.0);
}
