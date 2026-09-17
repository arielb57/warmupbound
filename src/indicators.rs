//! Reference implementations of the indicators under every seeding convention.
//!
//! These are deliberately plain `f64` batch computations: they are the ground
//! truth that witnesses are replayed through and that property tests attack.
//! Every function returns one `Option<f64>` per input bar (`None` = not yet defined).

use std::fmt;

/// How a recursive average is started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seeding {
    /// `y_0 = x_0`, then the recursion from bar 1 (pandas `adjust=False`).
    FirstValue,
    /// Undefined for the first `N-1` bars, `y_{N-1} = mean(x_0..x_{N-1})`
    /// (TA-Lib, Wilder's original RSI/ATR).
    SmaSeeded,
    /// `y_t = sum (1-a)^k x_{t-k} / sum (1-a)^k` over all bars seen so far
    /// (pandas `ewm(adjust=True)`).
    Adjusted,
}

impl Seeding {
    pub const ALL: [Seeding; 3] = [Seeding::FirstValue, Seeding::SmaSeeded, Seeding::Adjusted];

    pub fn parse(s: &str) -> Result<Seeding, String> {
        match s {
            "first" | "first-value" => Ok(Seeding::FirstValue),
            "sma" | "sma-seeded" => Ok(Seeding::SmaSeeded),
            "adjusted" | "adjust" => Ok(Seeding::Adjusted),
            other => Err(format!(
                "unknown convention {other:?} (first-value, sma-seeded, adjusted)"
            )),
        }
    }

    /// Number of input bars the convention consumes before the first defined value.
    pub fn seed_len(self, period: usize) -> usize {
        match self {
            Seeding::SmaSeeded => period,
            Seeding::FirstValue | Seeding::Adjusted => 1,
        }
    }
}

impl fmt::Display for Seeding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Seeding::FirstValue => "first-value",
            Seeding::SmaSeeded => "sma-seeded",
            Seeding::Adjusted => "adjusted",
        })
    }
}

/// One OHLC bar without the open (no indicator here uses it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl Bar {
    pub fn flat(price: f64) -> Bar {
        Bar {
            high: price,
            low: price,
            close: price,
        }
    }
}

/// Exponential smoothing with factor `alpha` over the defined entries of `xs`.
/// `period` is only used by `SmaSeeded`. Leading `None`s are skipped.
pub fn exp_smooth(
    xs: &[Option<f64>],
    alpha: f64,
    period: usize,
    seeding: Seeding,
) -> Vec<Option<f64>> {
    let mut out = vec![None; xs.len()];
    let q = 1.0 - alpha;
    let mut seen = 0usize;
    let mut sum = 0.0;
    let mut state: Option<f64> = None;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, x) in xs.iter().enumerate() {
        let Some(x) = *x else { continue };
        seen += 1;
        match seeding {
            Seeding::FirstValue => {
                state = Some(match state {
                    None => x,
                    Some(s) => alpha * x + q * s,
                });
            }
            Seeding::SmaSeeded => match state {
                None => {
                    sum += x;
                    if seen == period {
                        state = Some(sum / period as f64);
                    }
                }
                Some(s) => state = Some(alpha * x + q * s),
            },
            Seeding::Adjusted => {
                num = x + q * num;
                den = 1.0 + q * den;
                state = Some(num / den);
            }
        }
        out[i] = state;
    }
    out
}

pub fn ema_alpha(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

pub fn rma_alpha(period: usize) -> f64 {
    1.0 / period as f64
}

fn some(xs: &[f64]) -> Vec<Option<f64>> {
    xs.iter().map(|&x| Some(x)).collect()
}

pub fn ema(closes: &[f64], period: usize, seeding: Seeding) -> Vec<Option<f64>> {
    exp_smooth(&some(closes), ema_alpha(period), period, seeding)
}

/// Wilder's moving average (`alpha = 1/N`).
pub fn rma(closes: &[f64], period: usize, seeding: Seeding) -> Vec<Option<f64>> {
    exp_smooth(&some(closes), rma_alpha(period), period, seeding)
}

pub fn sma(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    (0..closes.len())
        .map(|i| {
            if i + 1 < period {
                None
            } else {
                Some(closes[i + 1 - period..=i].iter().sum::<f64>() / period as f64)
            }
        })
        .collect()
}

pub fn true_range(bars: &[Bar]) -> Vec<f64> {
    bars.iter()
        .enumerate()
        .map(|(i, b)| {
            let hl = b.high - b.low;
            if i == 0 {
                hl
            } else {
                let pc = bars[i - 1].close;
                hl.max((b.high - pc).abs()).max((b.low - pc).abs())
            }
        })
        .collect()
}

/// Average true range: Wilder smoothing of the true range.
pub fn atr(bars: &[Bar], period: usize, seeding: Seeding) -> Vec<Option<f64>> {
    exp_smooth(&some(&true_range(bars)), rma_alpha(period), period, seeding)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacdOutput {
    Line,
    Signal,
    Histogram,
}

impl MacdOutput {
    pub fn parse(s: &str) -> Result<MacdOutput, String> {
        match s {
            "line" => Ok(MacdOutput::Line),
            "signal" => Ok(MacdOutput::Signal),
            "hist" | "histogram" => Ok(MacdOutput::Histogram),
            o => Err(format!("unknown MACD output {o:?} (line, signal, hist)")),
        }
    }
}

pub fn macd(
    closes: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    seeding: Seeding,
    output: MacdOutput,
) -> Vec<Option<f64>> {
    let f = ema(closes, fast, seeding);
    let s = ema(closes, slow, seeding);
    let line: Vec<Option<f64>> = f.iter().zip(&s).map(|(a, b)| Some((*a)? - (*b)?)).collect();
    let sig = exp_smooth(&line, ema_alpha(signal), signal, seeding);
    match output {
        MacdOutput::Line => line,
        MacdOutput::Signal => sig,
        MacdOutput::Histogram => line
            .iter()
            .zip(&sig)
            .map(|(l, g)| Some((*l)? - (*g)?))
            .collect(),
    }
}

/// Kaufman adaptive moving average with efficiency-ratio window `period` and
/// fast/slow EMA periods. `Adjusted` has no meaning here and behaves as `FirstValue`.
pub fn kama(
    closes: &[f64],
    period: usize,
    fast: usize,
    slow: usize,
    seeding: Seeding,
) -> Vec<Option<f64>> {
    let fa = ema_alpha(fast);
    let sa = ema_alpha(slow);
    let mut out = vec![None; closes.len()];
    if closes.len() < period {
        return out;
    }
    let mut k = match seeding {
        Seeding::SmaSeeded => closes[..period].iter().sum::<f64>() / period as f64,
        _ => closes[period - 1],
    };
    out[period - 1] = Some(k);
    for t in period..closes.len() {
        let change = (closes[t] - closes[t - period]).abs();
        let vol: f64 = (t - period + 1..=t)
            .map(|i| (closes[i] - closes[i - 1]).abs())
            .sum();
        let er = if vol > 0.0 { change / vol } else { 0.0 };
        let sc = (er * (fa - sa) + sa).powi(2);
        k += sc * (closes[t] - k);
        out[t] = Some(k);
    }
    out
}

/// Wilder RSI on a 0..100 scale. When average gain and loss are both zero the
/// value is 50 (the common convention for a perfectly flat series).
pub fn rsi(closes: &[f64], period: usize, seeding: Seeding) -> Vec<Option<f64>> {
    let (g, l) = rsi_averages(closes, period, seeding);
    g.iter()
        .zip(&l)
        .map(|(g, l)| {
            let (g, l) = ((*g)?, (*l)?);
            Some(if g + l == 0.0 {
                50.0
            } else {
                100.0 * g / (g + l)
            })
        })
        .collect()
}

/// Smoothed average gain and loss per bar (bar 0 has no change and is `None`).
pub fn rsi_averages(
    closes: &[f64],
    period: usize,
    seeding: Seeding,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let mut gains = vec![None; closes.len()];
    let mut losses = vec![None; closes.len()];
    for t in 1..closes.len() {
        let d = closes[t] - closes[t - 1];
        gains[t] = Some(d.max(0.0));
        losses[t] = Some((-d).max(0.0));
    }
    let a = rma_alpha(period);
    (
        exp_smooth(&gains, a, period, seeding),
        exp_smooth(&losses, a, period, seeding),
    )
}

/// Stochastic RSI on a 0..100 scale over a window of `k` RSI values;
/// `None` while the window is incomplete or the RSI range is zero.
pub fn stoch_rsi(closes: &[f64], period: usize, k: usize, seeding: Seeding) -> Vec<Option<f64>> {
    let r = rsi(closes, period, seeding);
    (0..r.len())
        .map(|t| {
            if t + 1 < k {
                return None;
            }
            let window: Option<Vec<f64>> = r[t + 1 - k..=t].iter().copied().collect();
            let window = window?;
            let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if hi > lo {
                Some(100.0 * (r[t]? - lo) / (hi - lo))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conventions_agree_with_hand_computation() {
        let xs = [10.0, 12.0, 11.0, 13.0];
        let fv = ema(&xs, 3, Seeding::FirstValue);
        assert_eq!(fv[0], Some(10.0));
        assert_eq!(fv[1], Some(11.0));
        assert_eq!(fv[2], Some(11.0));
        assert_eq!(fv[3], Some(12.0));
        let sm = ema(&xs, 3, Seeding::SmaSeeded);
        assert_eq!(sm[1], None);
        assert_eq!(sm[2], Some(11.0));
        assert_eq!(sm[3], Some(12.0));
        // adjusted with a = 1/2: (12 + 0.5*10) / 1.5
        let ad = ema(&xs, 3, Seeding::Adjusted);
        assert!((ad[1].unwrap() - 17.0 / 1.5).abs() < 1e-12);
    }

    #[test]
    fn true_range_uses_previous_close_gaps() {
        let bars = [
            Bar {
                high: 10.0,
                low: 9.0,
                close: 9.5,
            },
            Bar {
                high: 12.0,
                low: 11.0,
                close: 11.5,
            },
            Bar {
                high: 11.0,
                low: 8.0,
                close: 8.5,
            },
        ];
        assert_eq!(true_range(&bars), vec![1.0, 2.5, 3.5]);
    }

    #[test]
    fn rsi_extremes_and_flat_convention() {
        let up: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        assert_eq!(rsi(&up, 14, Seeding::SmaSeeded)[29], Some(100.0));
        let flat = vec![5.0; 30];
        assert_eq!(rsi(&flat, 14, Seeding::SmaSeeded)[29], Some(50.0));
        assert_eq!(rsi(&flat, 14, Seeding::SmaSeeded)[13], None);
    }

    #[test]
    fn kama_on_a_straight_line_uses_the_fast_constant() {
        let up: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let k = kama(&up, 10, 2, 30, Seeding::FirstValue);
        let fa = 2.0 / 3.0f64;
        let expected = 9.0 + fa * fa * (10.0 - 9.0);
        assert!((k[10].unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn stoch_rsi_is_undefined_on_constant_rsi() {
        let up: Vec<f64> = (0..40).map(|i| i as f64).collect();
        assert!(stoch_rsi(&up, 5, 5, Seeding::SmaSeeded)
            .iter()
            .all(|v| v.is_none()));
    }
}
