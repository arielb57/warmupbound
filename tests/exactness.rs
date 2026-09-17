//! Exactness: hand-built boundary cases where (1-alpha)^k * D == tick/2 exactly.
//! "Agree" means strictly below tick/2, so the answer is k + 1; a float log formula
//! cannot see the equality and lands on k or k + 1 depending on rounding.

use warmupbound::bounds::gap_bound;
use warmupbound::exact::{ema_contraction, naive_log_warmup, rma_contraction};
use warmupbound::{bound, Indicator, Params, Verdict};

fn pow_str(base: u128, e: u32) -> String {
    base.pow(e).to_string()
}

/// For contraction p/q: D = q^k, tick = 2 p^k gives q^k (p/q)^k = p^k = tick/2.
/// Returns whether the f64 log formula got this boundary wrong.
fn boundary_case(ind: Indicator, p: u64, q: u64, k: u32) -> bool {
    let hi = pow_str(q as u128, k);
    let tick = (2 * (p as u128).pow(k)).to_string();
    let params = Params::from_decimals("0", &hi, &tick, None).unwrap();
    let b = gap_bound(&ind, &params).unwrap();
    let h = params.half_tick();
    assert!(
        b.exact_equals(k as u64, &h),
        "{ind} k={k}: bound(k) must equal tick/2 exactly"
    );
    assert!(!b.exact_below(k as u64, &h));
    assert!(b.exact_below(k as u64 + 1, &h));
    let exact = bound(&ind, &params).unwrap().verdict;
    assert_eq!(exact, Verdict::Proven(k as u64 + 1), "{ind} k={k}");
    let naive = naive_log_warmup(p as f64 / q as f64, params.range.value, params.tick.value);
    // Rounding in ln() lands on either side of the integer, so the float answer is k or k + 1 by luck.
    assert!(
        naive == k as u64 || naive == k as u64 + 1,
        "{ind} k={k}: naive {naive}"
    );
    Verdict::Proven(naive) != exact
}

#[test]
fn ema_period_3_halves_the_gap_each_bar() {
    let (p, q) = ema_contraction(3);
    assert_eq!((p, q), (2, 4));
    // ln(2^-k) / ln(1/2) evaluates to exactly k in f64, so the float formula
    // answers k at every one of these boundaries while the exact answer is k + 1.
    for k in [1, 5, 10, 20, 40, 60] {
        assert!(
            boundary_case(Indicator::Ema { period: 3 }, 1, 2, k),
            "k={k}"
        );
    }
}

#[test]
fn ema_period_7_three_quarters() {
    let (p, q) = ema_contraction(7);
    assert_eq!(p * 4, q * 3);
    let wrong = (1..=45)
        .filter(|&k| boundary_case(Indicator::Ema { period: 7 }, 3, 4, k))
        .count();
    assert!(
        wrong >= 10,
        "float formula wrong at only {wrong} of 45 boundaries"
    );
}

#[test]
fn rma_period_4_and_period_100() {
    let (p, q) = rma_contraction(4);
    let wrong4 = (1..=47)
        .filter(|&k| boundary_case(Indicator::Rma { period: 4 }, p, q, k))
        .count();
    let (p, q) = rma_contraction(100);
    let wrong100 = (1..=14)
        .filter(|&k| boundary_case(Indicator::Rma { period: 100 }, p, q, k))
        .count();
    assert!(
        wrong4 + wrong100 >= 10,
        "float formula wrong at {wrong4} + {wrong100} boundaries"
    );
}

#[test]
fn a_millionth_of_a_tick_moves_the_answer_by_one_bar() {
    // D = 4^30, tick/2 = 3^30 + 5e-7: now bound(30) is strictly below tick/2.
    let k = 30u32;
    let hi = pow_str(4, k);
    let tick = format!("{}.000001", 2 * 3u128.pow(k));
    let params = Params::from_decimals("0", &hi, &tick, None).unwrap();
    let ind = Indicator::Rma { period: 4 };
    assert_eq!(
        bound(&ind, &params).unwrap().verdict,
        Verdict::Proven(k as u64)
    );
    // ...and a millionth below the boundary needs k + 1.
    let tick = format!("{}.999999", 2 * 3u128.pow(k) - 1);
    let params = Params::from_decimals("0", &hi, &tick, None).unwrap();
    assert_eq!(
        bound(&ind, &params).unwrap().verdict,
        Verdict::Proven(k as u64 + 1)
    );
    // f64 cannot represent either tick distinctly from the exact boundary.
    assert_eq!(params.tick.value, 2.0 * 3f64.powi(k as i32));
}

#[test]
fn composite_bounds_are_decided_exactly_near_their_crossing() {
    let params = Params::from_decimals("90", "110", "0.01", None).unwrap();
    for ind in [
        Indicator::Macd {
            fast: 12,
            slow: 26,
            signal: 9,
            output: warmupbound::MacdOutput::Signal,
        },
        Indicator::Macd {
            fast: 5,
            slow: 35,
            signal: 5,
            output: warmupbound::MacdOutput::Histogram,
        },
        // signal contraction equals fast contraction: exercises the n * g^n term.
        Indicator::Macd {
            fast: 9,
            slow: 26,
            signal: 9,
            output: warmupbound::MacdOutput::Signal,
        },
    ] {
        let report = bound(&ind, &params).unwrap();
        let Verdict::Proven(n) = report.verdict else {
            panic!()
        };
        let b = report.gap_bound.unwrap();
        let h = params.half_tick();
        assert!(b.exact_below(n, &h), "{ind}");
        assert!(!b.exact_below(n - 1, &h), "{ind}");
        // The float evaluation of the closed form agrees with direct summation.
        let direct = direct_signal_bound(&ind, 20.0, n);
        assert!(
            (b.float(n) - direct).abs() < 1e-9 * direct,
            "{ind}: {} vs {direct}",
            b.float(n)
        );
    }
}

fn direct_signal_bound(ind: &Indicator, d: f64, n: u64) -> f64 {
    let Indicator::Macd {
        fast,
        slow,
        signal,
        output,
    } = *ind
    else {
        unreachable!()
    };
    let qf = 1.0 - 2.0 / (fast as f64 + 1.0);
    let qs = 1.0 - 2.0 / (slow as f64 + 1.0);
    let ag = 2.0 / (signal as f64 + 1.0);
    let qg = 1.0 - ag;
    let line = |j: u64| d * (qf.powi(j as i32) + qs.powi(j as i32));
    let sig = 2.0 * d * qg.powi(n as i32)
        + (1..=n)
            .map(|j| ag * qg.powi((n - j) as i32) * line(j))
            .sum::<f64>();
    match output {
        warmupbound::MacdOutput::Line => line(n),
        warmupbound::MacdOutput::Signal => sig,
        warmupbound::MacdOutput::Histogram => line(n) + sig,
    }
}
