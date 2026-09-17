//! Per-indicator warm-up bounds and their derivations.

use crate::bignum::{decimal_difference, decimal_to_ratio, BigUint, Ratio};
use crate::exact::{ema_contraction, half, rma_contraction, GeometricBound, Solved, Term};
use crate::indicators::{self, Bar, MacdOutput, Seeding};
use std::cmp::Ordering;
use std::fmt;

/// Largest warm-up the solver will search for.
pub const SEARCH_CAP: u64 = 10_000_000;

/// A decimal input kept both exactly and as `f64`.
#[derive(Clone, Debug)]
pub struct Decimal {
    pub exact: Ratio,
    pub value: f64,
}

impl Decimal {
    pub fn parse(s: &str) -> Result<Decimal, String> {
        let exact = decimal_to_ratio(s)?;
        let value = exact.to_f64();
        Ok(Decimal { exact, value })
    }
}

/// Market assumptions shared by every bound.
#[derive(Clone, Debug)]
pub struct Params {
    pub lo: f64,
    pub hi: f64,
    /// `hi - lo`, exact.
    pub range: Decimal,
    /// Tolerance on the indicator value; two runs "agree" when they differ by less than half of it.
    pub tick: Decimal,
    /// Largest close-to-close move per bar, at most `range`.
    pub max_move: Decimal,
}

impl Params {
    pub fn from_decimals(
        lo: &str,
        hi: &str,
        tick: &str,
        max_move: Option<&str>,
    ) -> Result<Params, String> {
        let range_exact = decimal_difference(hi, lo)?;
        let range = Decimal {
            value: range_exact.to_f64(),
            exact: range_exact,
        };
        let tick = Decimal::parse(tick)?;
        if tick.exact.num.is_zero() {
            return Err("--tick must be positive".into());
        }
        let max_move = match max_move {
            None => range.clone(),
            Some(s) => {
                let m = Decimal::parse(s)?;
                if m.exact.num.is_zero() {
                    return Err("--max-move must be positive".into());
                }
                if m.exact.cmp_ratio(&range.exact) == Ordering::Greater {
                    range.clone()
                } else {
                    m
                }
            }
        };
        let lo_f: f64 = lo.parse().map_err(|_| format!("bad --lo {lo}"))?;
        let hi_f: f64 = hi.parse().map_err(|_| format!("bad --hi {hi}"))?;
        Ok(Params {
            lo: lo_f,
            hi: hi_f,
            range,
            tick,
            max_move,
        })
    }

    pub fn half_tick(&self) -> Ratio {
        half(&self.tick.exact)
    }
}

#[derive(Clone, Debug)]
pub enum Indicator {
    Ema {
        period: usize,
    },
    Rma {
        period: usize,
    },
    Sma {
        period: usize,
    },
    Atr {
        period: usize,
    },
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
        output: MacdOutput,
    },
    Kama {
        period: usize,
        fast: usize,
        slow: usize,
    },
    /// `min_move`: lower bound on the Wilder average absolute move (avg gain + avg loss).
    Rsi {
        period: usize,
        min_move: Option<Decimal>,
    },
    /// `min_range`: lower bound on max - min of the RSI values in the stochastic window.
    StochRsi {
        period: usize,
        k: usize,
        min_move: Option<Decimal>,
        min_range: Option<Decimal>,
    },
}

impl fmt::Display for Indicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Indicator::Ema { period } => write!(f, "EMA({period})"),
            Indicator::Rma { period } => write!(f, "RMA({period})"),
            Indicator::Sma { period } => write!(f, "SMA({period})"),
            Indicator::Atr { period } => write!(f, "ATR({period})"),
            Indicator::Macd {
                fast,
                slow,
                signal,
                output,
            } => {
                write!(f, "MACD({fast},{slow},{signal}) {output:?}")
            }
            Indicator::Kama { period, fast, slow } => write!(f, "KAMA({period},{fast},{slow})"),
            Indicator::Rsi { period, .. } => write!(f, "RSI({period})"),
            Indicator::StochRsi { period, k, .. } => write!(f, "StochRSI({period},{k})"),
        }
    }
}

impl Indicator {
    /// Replays the indicator through its reference implementation.
    pub fn evaluate(&self, bars: &[Bar], seeding: Seeding) -> Vec<Option<f64>> {
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        match *self {
            Indicator::Ema { period } => indicators::ema(&closes, period, seeding),
            Indicator::Rma { period } => indicators::rma(&closes, period, seeding),
            Indicator::Sma { period } => indicators::sma(&closes, period),
            Indicator::Atr { period } => indicators::atr(bars, period, seeding),
            Indicator::Macd {
                fast,
                slow,
                signal,
                output,
            } => indicators::macd(&closes, fast, slow, signal, seeding, output),
            Indicator::Kama { period, fast, slow } => {
                indicators::kama(&closes, period, fast, slow, seeding)
            }
            Indicator::Rsi { period, .. } => indicators::rsi(&closes, period, seeding),
            Indicator::StochRsi { period, k, .. } => {
                indicators::stoch_rsi(&closes, period, k, seeding)
            }
        }
    }

    /// History length after which the convention's seeding is complete, so
    /// that every shared bar is a genuine update of an established state.
    pub fn seeded_history_len(&self, seeding: Seeding) -> usize {
        let s = |p: usize| seeding.seed_len(p);
        match *self {
            Indicator::Ema { period } | Indicator::Rma { period } | Indicator::Atr { period } => {
                s(period)
            }
            Indicator::Sma { period } => period,
            Indicator::Macd {
                fast, slow, signal, ..
            } => s(fast).max(s(slow)) + s(signal) - 1,
            Indicator::Kama { period, .. } => period,
            Indicator::Rsi { period, .. } => s(period) + 1,
            Indicator::StochRsi { period, k, .. } => s(period) + k,
        }
    }

    /// Conventions that are meaningful for this indicator.
    pub fn conventions(&self) -> Vec<Seeding> {
        match self {
            Indicator::Sma { .. } => vec![Seeding::FirstValue],
            Indicator::Kama { .. } => vec![Seeding::FirstValue, Seeding::SmaSeeded],
            _ => Seeding::ALL.to_vec(),
        }
    }

    fn periods(&self) -> Vec<usize> {
        match *self {
            Indicator::Ema { period }
            | Indicator::Rma { period }
            | Indicator::Sma { period }
            | Indicator::Atr { period }
            | Indicator::Rsi { period, .. } => vec![period],
            Indicator::Macd {
                fast, slow, signal, ..
            } => vec![fast, slow, signal],
            Indicator::Kama { period, fast, slow } => vec![period, fast, slow],
            Indicator::StochRsi { period, k, .. } => vec![period, k],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Any two runs sharing at least `n` bars agree to within tick/2.
    Proven(u64),
    /// No finite warm-up exists under the given assumptions.
    Unbounded,
    /// A finite warm-up exists but is larger than the search cap.
    ExceedsCap(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The bound is attained: a witness differs by at least tick/2 at n-1 shared bars.
    Exact,
    /// Several terms bounded separately; sound but possibly loose.
    Composite,
    /// Holds only under an extra market assumption (minimum activity).
    Conditional,
}

#[derive(Clone, Debug)]
pub struct BoundReport {
    pub indicator: String,
    pub verdict: Verdict,
    pub kind: Kind,
    pub derivation: Vec<String>,
    /// Closed form of the worst-case gap after `n` shared bars, when geometric.
    pub gap_bound: Option<GeometricBound>,
}

fn int(v: u64) -> Ratio {
    Ratio::from_u64s(v, 1)
}

fn show(r: &Ratio) -> String {
    format!("{:.6e}", r.to_f64())
}

fn frac(p: u64, q: u64) -> String {
    format!("{p}/{q}")
}

/// Worst-case gap after `n` shared bars as a geometric bound (all indicators except SMA).
pub fn gap_bound(ind: &Indicator, params: &Params) -> Option<GeometricBound> {
    let d = &params.range.exact;
    match *ind {
        Indicator::Ema { period } => {
            let (p, q) = ema_contraction(period as u64);
            Some(GeometricBound {
                start: 0,
                pre: d.clone(),
                terms: vec![Term::pos(d.clone(), p, q, 0)],
            })
        }
        Indicator::Rma { period } => {
            let (p, q) = rma_contraction(period as u64);
            Some(GeometricBound {
                start: 0,
                pre: d.clone(),
                terms: vec![Term::pos(d.clone(), p, q, 0)],
            })
        }
        Indicator::Atr { period } => {
            let (p, q) = rma_contraction(period as u64);
            Some(GeometricBound {
                start: 1,
                pre: d.clone(),
                terms: vec![Term::pos(d.clone(), p, q, 1)],
            })
        }
        Indicator::Sma { .. } => None,
        Indicator::Kama { period, slow, .. } => {
            let s1 = slow as u64 + 1;
            Some(GeometricBound {
                start: period as u64,
                pre: d.clone(),
                terms: vec![Term::pos(d.clone(), s1 * s1 - 4, s1 * s1, period as u64)],
            })
        }
        Indicator::Macd {
            fast,
            slow,
            signal,
            output,
        } => {
            let (fp, fq) = ema_contraction(fast as u64);
            let (sp, sq) = ema_contraction(slow as u64);
            let line = vec![
                Term::pos(d.clone(), fp, fq, 0),
                Term::pos(d.clone(), sp, sq, 0),
            ];
            let sig = macd_signal_terms(d, signal as u64, [(fp, fq), (sp, sq)]);
            let terms = match output {
                MacdOutput::Line => line,
                MacdOutput::Signal => sig,
                MacdOutput::Histogram => line.into_iter().chain(sig).collect(),
            };
            let pre = match output {
                MacdOutput::Histogram => d.mul(&int(4)),
                _ => d.mul(&int(2)),
            };
            Some(GeometricBound {
                start: 0,
                pre,
                terms,
            })
        }
        Indicator::Rsi {
            period,
            ref min_move,
        } => {
            let m = min_move.as_ref()?;
            let (p, q) = rma_contraction(period as u64);
            let coef = params
                .max_move
                .exact
                .mul(&int(100))
                .mul(&Ratio::new(m.exact.den.clone(), m.exact.num.clone()));
            Some(GeometricBound {
                start: 1,
                pre: int(100),
                terms: vec![Term::pos(coef, p, q, 1)],
            })
        }
        Indicator::StochRsi {
            period,
            k,
            ref min_move,
            ref min_range,
        } => {
            let m = min_move.as_ref()?;
            let w = min_range.as_ref()?;
            let (p, q) = rma_contraction(period as u64);
            let coef = params
                .max_move
                .exact
                .mul(&int(20_000))
                .mul(&Ratio::new(m.exact.den.clone(), m.exact.num.clone()))
                .mul(&Ratio::new(w.exact.den.clone(), w.exact.num.clone()));
            Some(GeometricBound {
                start: k as u64,
                pre: int(100),
                terms: vec![Term::pos(coef, p, q, k as u64)],
            })
        }
    }
}

/// Signal-line gap: `2D g^n + a_g D sum_{j=1..n} g^(n-j) (f^j + s^j)`, with each
/// convolution summed in closed form `r (r^n - g^n) / (r - g)` (or `n g^n` when `r = g`).
fn macd_signal_terms(d: &Ratio, signal: u64, lines: [(u64, u64); 2]) -> Vec<Term> {
    let (gp, gq) = ema_contraction(signal);
    let mut terms = vec![Term::pos(d.mul(&int(2)), gp, gq, 0)];
    for (rp, rq) in lines {
        let a = rp as u128 * gq as u128;
        let b = gp as u128 * rq as u128;
        if a == b {
            terms.push(Term {
                negative: false,
                coef: d.mul(&Ratio::from_u64s(2, signal + 1)),
                p: gp,
                q: gq,
                offset: 0,
                times_n: true,
            });
            continue;
        }
        let diff = a.abs_diff(b);
        // a_g * r / |r - g| = 2 rp gq / ((G+1) |rp gq - gp rq|)
        let coef = d.mul(&Ratio::new(
            BigUint::from_u128(2 * rp as u128 * gq as u128),
            BigUint::from_u128((signal as u128 + 1) * diff),
        ));
        let r_positive = a > b;
        terms.push(Term {
            negative: !r_positive,
            coef: coef.clone(),
            p: rp,
            q: rq,
            offset: 0,
            times_n: false,
        });
        terms.push(Term {
            negative: r_positive,
            coef,
            p: gp,
            q: gq,
            offset: 0,
            times_n: false,
        });
    }
    terms
}

fn solve_line(
    b: &GeometricBound,
    params: &Params,
    verdict: &mut Verdict,
    derivation: &mut Vec<String>,
) {
    let h = params.half_tick();
    match b.solve(&h, SEARCH_CAP) {
        Solved::Found(n) => {
            *verdict = Verdict::Proven(n);
            derivation.push(format!(
                "smallest n with bound(n) < tick/2 = {}, decided in exact integers: n = {n}",
                show(&h)
            ));
            if n > 0 {
                let (pp, pn) = b.exact(n - 1);
                let (qp, qn) = b.exact(n);
                derivation.push(format!(
                    "check: bound({}) = {} >= tick/2 > bound({n}) = {}",
                    n - 1,
                    format_signed(&pp, &pn),
                    format_signed(&qp, &qn)
                ));
            }
        }
        Solved::ExceedsCap(c) => {
            *verdict = Verdict::ExceedsCap(c);
            derivation.push(format!("bound(n) stays >= tick/2 for every n <= {c}"));
        }
    }
}

fn format_signed(pos: &Ratio, neg: &Ratio) -> String {
    format!("{:.6e}", pos.to_f64() - neg.to_f64())
}

/// Computes the warm-up bound and its derivation.
pub fn bound(ind: &Indicator, params: &Params) -> Result<BoundReport, String> {
    if ind.periods().iter().any(|&p| p < 2) {
        return Err("all periods must be at least 2".into());
    }
    let d = &params.range;
    let mut derivation = Vec::new();
    let mut verdict = Verdict::Unbounded;
    let kind;
    derivation.push(format!(
        "prices in [{}, {}], D = hi - lo = {}; tick = {}; max move per bar = {}",
        params.lo, params.hi, d.value, params.tick.value, params.max_move.value
    ));
    match *ind {
        Indicator::Ema { period } | Indicator::Rma { period } => {
            let is_ema = matches!(ind, Indicator::Ema { .. });
            let (p, q) = if is_ema {
                ema_contraction(period as u64)
            } else {
                rma_contraction(period as u64)
            };
            kind = Kind::Exact;
            derivation.push(format!(
                "alpha = {}, so each shared bar multiplies the state gap by exactly 1 - alpha = {}",
                if is_ema {
                    frac(2, period as u64 + 1)
                } else {
                    frac(1, period as u64)
                },
                frac(p, q)
            ));
            derivation.push(
                "every convention (first-value, SMA seed, adjusted weights) keeps the state inside [lo, hi], so once both runs have finished seeding the gap is at most D".into(),
            );
            derivation.push(format!(
                "bound(n) = ({})^n * D, attained by histories pinned at hi and at lo",
                frac(p, q)
            ));
        }
        Indicator::Atr { period } => {
            let (p, q) = rma_contraction(period as u64);
            kind = Kind::Exact;
            derivation.push(format!(
                "ATR = Wilder average (1 - alpha = {}) of true range, TR in [0, D]",
                frac(p, q)
            ));
            derivation.push("the first shared bar's TR still depends on the previous close, so it can differ by D too".into());
            derivation.push(format!(
                "bound(0) = D, bound(n) = ({})^(n-1) * ((1-alpha) D + alpha D) = ({})^(n-1) * D, attained",
                frac(p, q),
                frac(p, q)
            ));
        }
        Indicator::Sma { period } => {
            kind = Kind::Exact;
            derivation.push(format!(
                "SMA depends only on the last {period} bars; with n < {period} shared bars, {period} - n bars can differ by D each"
            ));
            derivation.push(format!(
                "bound(n) = D * ({period} - n) / {period} for n < {period}, 0 afterwards"
            ));
            let h = params.half_tick();
            let n = (0..=period as u64)
                .find(|&k| {
                    let g = d
                        .exact
                        .mul(&Ratio::from_u64s(period as u64 - k, period as u64));
                    g.cmp_ratio(&h) == Ordering::Less
                })
                .expect("bound reaches zero at n = period");
            verdict = Verdict::Proven(n);
            derivation.push(format!(
                "smallest n with bound(n) < tick/2 (exact): n = {n}"
            ));
            return Ok(BoundReport {
                indicator: ind.to_string(),
                verdict,
                kind,
                derivation,
                gap_bound: None,
            });
        }
        Indicator::Kama { period, fast, slow } => {
            let s1 = slow as u64 + 1;
            kind = Kind::Composite;
            derivation.push(format!(
                "smoothing constant sc = (ER * (2/{} - 2/{s1}) + 2/{s1})^2 lies in [4/{}, 4/{}]",
                fast + 1,
                s1 * s1,
                (fast + 1) * (fast + 1)
            ));
            derivation.push(format!(
                "the first {period} shared bars see efficiency-ratio windows reaching into the history, so they only keep the gap <= D"
            ));
            derivation.push(format!(
                "afterwards sc is common to both runs and the gap contracts by at most 1 - 4/{} = {}",
                s1 * s1,
                frac(s1 * s1 - 4, s1 * s1)
            ));
            derivation.push(format!(
                "bound(n) = ({})^(n - {period}) * D for n >= {period}",
                frac(s1 * s1 - 4, s1 * s1)
            ));
        }
        Indicator::Macd {
            fast,
            slow,
            signal,
            output,
        } => {
            kind = Kind::Composite;
            let (fp, fq) = ema_contraction(fast as u64);
            let (sp, sq) = ema_contraction(slow as u64);
            let (gp, gq) = ema_contraction(signal as u64);
            derivation.push(format!(
                "line gap after j shared bars <= D (({})^j + ({})^j) (fast and slow seed gaps bounded separately)",
                frac(fp, fq),
                frac(sp, sq)
            ));
            if output != MacdOutput::Line {
                derivation.push(format!(
                    "signal gap <= 2D ({})^n + (2/{}) sum_(j=1..n) ({})^(n-j) * line gap(j), summed in closed form",
                    frac(gp, gq),
                    signal + 1,
                    frac(gp, gq)
                ));
            }
            if output == MacdOutput::Histogram {
                derivation.push("histogram gap <= line gap + signal gap".into());
            }
        }
        Indicator::Rsi {
            period,
            ref min_move,
        } => match min_move {
            None => {
                derivation.push("RSI = 100 G / (G + L); its sensitivity to the seed is 1/(G + L), unbounded as the market goes flat".into());
                derivation.push(
                    "witness: one history rising into p, one falling into p, then a flat market at p; G_a > 0 = L_a and L_b > 0 = G_b forever, so RSI stays 100 vs 0"
                        .into(),
                );
                derivation.push(
                    "no finite warm-up exists; pass --min-move to get a conditional bound".into(),
                );
                return Ok(BoundReport {
                    indicator: ind.to_string(),
                    verdict: Verdict::Unbounded,
                    kind: Kind::Conditional,
                    derivation,
                    gap_bound: None,
                });
            }
            Some(m) => {
                kind = Kind::Conditional;
                let (p, q) = rma_contraction(period as u64);
                derivation.push(format!(
                    "assume avg gain + avg loss >= m = {} in both runs at the bar compared",
                    m.value
                ));
                derivation.push(format!(
                    "avg gain and avg loss each differ by <= max_move * ({})^(n-1) (first shared change still depends on history)",
                    frac(p, q)
                ));
                derivation.push("|grad of G/(G+L)| . (dG, dL) <= max(|dG|, |dL|) / (G+L) along the segment, where G+L >= m".into());
                derivation.push(format!(
                    "bound(n) = 100 * max_move * ({})^(n-1) / m",
                    frac(p, q)
                ));
            }
        },
        Indicator::StochRsi {
            period,
            k,
            ref min_move,
            ref min_range,
        } => {
            if min_move.is_none() || min_range.is_none() {
                derivation.push("StochRSI inherits RSI's flat-market witness and adds a second vanishing denominator (max - min of RSI)".into());
                derivation.push("no finite warm-up exists; pass both --min-move and --min-range for a conditional bound".into());
                return Ok(BoundReport {
                    indicator: ind.to_string(),
                    verdict: Verdict::Unbounded,
                    kind: Kind::Conditional,
                    derivation,
                    gap_bound: None,
                });
            }
            kind = Kind::Conditional;
            let (p, q) = rma_contraction(period as u64);
            derivation.push(format!(
                "assume avg gain + avg loss >= m = {} and RSI max - min >= w = {} over the {k}-bar window, in both runs",
                min_move.as_ref().map(|x| x.value).unwrap_or_default(),
                min_range.as_ref().map(|x| x.value).unwrap_or_default()
            ));
            derivation.push(
                "(R - lo)/(hi - lo) moves by <= 2 eta / w when R, lo, hi each move by <= eta"
                    .into(),
            );
            derivation.push(format!(
                "eta = RSI bound at the oldest window bar, so bound(n) = 200 * 100 * max_move * ({})^(n-{k}) / (w m)",
                frac(p, q)
            ));
        }
    }
    let b = gap_bound(ind, params).expect("geometric bound exists for this indicator");
    solve_line(&b, params, &mut verdict, &mut derivation);
    Ok(BoundReport {
        indicator: ind.to_string(),
        verdict,
        kind,
        derivation,
        gap_bound: Some(b),
    })
}

/// `3 * period`, the common rule of thumb.
pub fn three_x_rule(period: usize) -> u64 {
    3 * period as u64
}

/// TA-Lib's default lookback with unstable period 0: bars before the first output.
pub fn talib_default_lookback(ind: &Indicator) -> u64 {
    match *ind {
        Indicator::Ema { period } | Indicator::Sma { period } => period as u64 - 1,
        Indicator::Rma { period } | Indicator::Atr { period } | Indicator::Rsi { period, .. } => {
            period as u64
        }
        Indicator::Kama { period, .. } => period as u64,
        Indicator::Macd { slow, signal, .. } => (slow + signal - 2) as u64,
        Indicator::StochRsi { period, k, .. } => (period + k - 1) as u64,
    }
}

/// A seed-agnostic check used by tests and reports: the exact value of the
/// closed-form bound at `n` compared with `tick/2`.
pub fn bound_below_half_tick(ind: &Indicator, params: &Params, n: u64) -> Option<bool> {
    gap_bound(ind, params).map(|b| b.exact_below(n, &params.half_tick()))
}
