//! Accuracy report: proven warm-ups against common rules of thumb over a grid.

use crate::bounds::{bound, talib_default_lookback, three_x_rule, Indicator, Params, Verdict};
use crate::exact::{ema_contraction, naive_log_warmup, rma_contraction};
use std::fmt::Write as _;

pub const PERIODS: [usize; 8] = [5, 10, 14, 20, 26, 50, 100, 200];
/// Tick / range ratios as decimal strings on a range of exactly 1.
pub const TICKS: [&str; 5] = ["0.01", "0.001", "0.0001", "0.00001", "0.000001"];
/// A fixed warm-up length used by many backtesting setups.
pub const FIXED_WARMUP: u64 = 100;

#[derive(Clone, Debug)]
pub struct Row {
    pub family: &'static str,
    pub period: usize,
    pub tick: &'static str,
    pub proven: Verdict,
    pub three_x: u64,
    pub talib: u64,
    /// `ceil(ln(tick/2D)/ln(1-alpha))` in f64, for the pure exponential averages.
    pub float_log: Option<u64>,
}

fn indicator(family: &str, period: usize) -> Indicator {
    match family {
        "EMA" => Indicator::Ema { period },
        "RMA" => Indicator::Rma { period },
        "ATR" => Indicator::Atr { period },
        "KAMA" => Indicator::Kama {
            period,
            fast: 2,
            slow: 30,
        },
        "RSI" => Indicator::Rsi {
            period,
            min_move: None,
        },
        _ => unreachable!("unknown family {family}"),
    }
}

pub const FAMILIES: [&str; 5] = ["EMA", "RMA", "ATR", "KAMA", "RSI"];

pub fn grid() -> Vec<Row> {
    let mut rows = Vec::new();
    for family in FAMILIES {
        for period in PERIODS {
            for tick in TICKS {
                let params = Params::from_decimals("0", "1", tick, None)
                    .expect("static grid parameters are valid");
                let ind = indicator(family, period);
                let report = bound(&ind, &params).expect("grid periods are valid");
                let tick_f = params.tick.value;
                let float_log = match family {
                    "EMA" => {
                        let (p, q) = ema_contraction(period as u64);
                        Some(naive_log_warmup(p as f64 / q as f64, 1.0, tick_f))
                    }
                    "RMA" => {
                        let (p, q) = rma_contraction(period as u64);
                        Some(naive_log_warmup(p as f64 / q as f64, 1.0, tick_f))
                    }
                    _ => None,
                };
                rows.push(Row {
                    family,
                    period,
                    tick,
                    proven: report.verdict,
                    three_x: three_x_rule(period),
                    talib: talib_default_lookback(&ind),
                    float_log,
                });
            }
        }
    }
    rows
}

fn too_short(rule: u64, proven: Verdict) -> bool {
    match proven {
        Verdict::Proven(n) => rule < n,
        Verdict::Unbounded | Verdict::ExceedsCap(_) => true,
    }
}

fn pct(k: usize, n: usize) -> String {
    format!("{k}/{n} ({:.0}%)", 100.0 * k as f64 / n as f64)
}

pub fn render(rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Grid: periods {:?}, tick/range ratios 1e-2 .. 1e-6 (range = 1), max move = range.\n",
        PERIODS
    );
    let _ = writeln!(
        out,
        "Share of configurations where the rule is shorter than the proven warm-up:\n"
    );
    let _ = writeln!(out, "| indicator | configs | 3 x period | TA-Lib default lookback | fixed {FIXED_WARMUP} bars |");
    let _ = writeln!(out, "|---|---|---|---|---|");
    let mut totals = [0usize; 4];
    for family in FAMILIES {
        let fam: Vec<&Row> = rows.iter().filter(|r| r.family == family).collect();
        let n = fam.len();
        let a = fam
            .iter()
            .filter(|r| too_short(r.three_x, r.proven))
            .count();
        let b = fam.iter().filter(|r| too_short(r.talib, r.proven)).count();
        let c = fam
            .iter()
            .filter(|r| too_short(FIXED_WARMUP, r.proven))
            .count();
        totals[0] += n;
        totals[1] += a;
        totals[2] += b;
        totals[3] += c;
        let _ = writeln!(
            out,
            "| {family} | {n} | {} | {} | {} |",
            pct(a, n),
            pct(b, n),
            pct(c, n)
        );
    }
    let _ = writeln!(
        out,
        "| all | {} | {} | {} | {} |\n",
        totals[0],
        pct(totals[1], totals[0]),
        pct(totals[2], totals[0]),
        pct(totals[3], totals[0])
    );
    for family in ["EMA", "RMA", "ATR", "KAMA"] {
        let _ = writeln!(
            out,
            "Proven warm-up in bars, {family} (3 x period in parentheses):\n"
        );
        let _ = writeln!(
            out,
            "| period | {} |",
            TICKS.map(|t| format!("tick {t}")).join(" | ")
        );
        let _ = writeln!(out, "|---|{}", "---|".repeat(TICKS.len()));
        for period in PERIODS {
            let cells: Vec<String> = rows
                .iter()
                .filter(|r| r.family == family && r.period == period)
                .map(|r| match r.proven {
                    Verdict::Proven(n) => format!("{n} ({})", r.three_x),
                    Verdict::Unbounded => "unbounded".into(),
                    Verdict::ExceedsCap(c) => format!("> {c}"),
                })
                .collect();
            let _ = writeln!(out, "| {period} | {} |", cells.join(" | "));
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "RSI: no finite unconditional warm-up in any configuration (flat-market witness).\n"
    );
    let float_rows: Vec<&Row> = rows.iter().filter(|r| r.float_log.is_some()).collect();
    let disagree = float_rows
        .iter()
        .filter(|r| match r.proven {
            Verdict::Proven(n) => r.float_log != Some(n),
            _ => true,
        })
        .count();
    let _ = writeln!(
        out,
        "f64 log formula ceil(ln(tick/2D)/ln(1-alpha)) disagrees with the exact answer in {} of the EMA/RMA grid.",
        pct(disagree, float_rows.len())
    );
    out
}
