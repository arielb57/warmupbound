//! Adversarial witnesses: two histories and a common price path whose indicator
//! values still differ by at least tick/2 after many shared bars.

use crate::bounds::{Indicator, Params};
use crate::indicators::{rsi_averages, Bar, Seeding};
use std::fmt::Write as _;

/// Small deterministic PRNG (xorshift64*), so witnesses are reproducible without dependencies.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

#[derive(Clone, Debug)]
pub struct Witness {
    pub hist_a: Vec<Bar>,
    pub hist_b: Vec<Bar>,
    pub shared: Vec<Bar>,
    pub seeding_a: Seeding,
    pub seeding_b: Seeding,
    pub description: String,
}

impl Witness {
    pub fn run_a(&self) -> Vec<Bar> {
        self.hist_a.iter().chain(&self.shared).copied().collect()
    }

    pub fn run_b(&self) -> Vec<Bar> {
        self.hist_b.iter().chain(&self.shared).copied().collect()
    }

    /// `gaps[k - 1]` = |A - B| after `k` shared bars (`None` if either is undefined).
    pub fn gaps(&self, ind: &Indicator) -> Vec<Option<f64>> {
        let va = ind.evaluate(&self.run_a(), self.seeding_a);
        let vb = ind.evaluate(&self.run_b(), self.seeding_b);
        let (ha, hb) = (self.hist_a.len(), self.hist_b.len());
        (0..self.shared.len())
            .map(|i| Some((va[ha + i]? - vb[hb + i]?).abs()))
            .collect()
    }

    /// Largest close-to-close move in either run.
    pub fn max_move(&self) -> f64 {
        [self.run_a(), self.run_b()]
            .iter()
            .flat_map(|r| {
                r.windows(2)
                    .map(|w| (w[1].close - w[0].close).abs())
                    .collect::<Vec<_>>()
            })
            .fold(0.0, f64::max)
    }

    /// Whether every bar stays inside the price bounds and respects the move limit.
    pub fn is_admissible(&self, params: &Params) -> bool {
        let slack = 1e-9 * params.range.value;
        let in_range = |b: &Bar| {
            b.low >= params.lo - slack
                && b.high <= params.hi + slack
                && b.low <= b.close
                && b.close <= b.high
        };
        self.run_a().iter().all(in_range)
            && self.run_b().iter().all(in_range)
            && self.max_move() <= params.max_move.value + slack
    }

    /// CSV with one row per bar of both runs aligned on the shared suffix.
    pub fn to_csv(&self, ind: &Indicator) -> String {
        let a = self.run_a();
        let b = self.run_b();
        let va = ind.evaluate(&a, self.seeding_a);
        let vb = ind.evaluate(&b, self.seeding_b);
        let len = a.len().max(b.len());
        let (oa, ob) = (len - a.len(), len - b.len());
        let hist = self.hist_a.len().max(self.hist_b.len());
        let mut out = String::from(
            "bar,phase,shared_bars,high_a,low_a,close_a,value_a,high_b,low_b,close_b,value_b,gap\n",
        );
        let opt = |v: Option<f64>| v.map(|x| format!("{x}")).unwrap_or_default();
        for i in 0..len {
            let ra = i.checked_sub(oa).map(|j| (a[j], va[j]));
            let rb = i.checked_sub(ob).map(|j| (b[j], vb[j]));
            let shared = i + 1 > hist;
            let k = if shared { i + 1 - hist } else { 0 };
            let gap = match (ra.and_then(|r| r.1), rb.and_then(|r| r.1)) {
                (Some(x), Some(y)) => Some((x - y).abs()),
                _ => None,
            };
            let cols = |r: Option<(Bar, Option<f64>)>| match r {
                Some((bar, v)) => format!("{},{},{},{}", bar.high, bar.low, bar.close, opt(v)),
                None => ",,,".to_string(),
            };
            let _ = writeln!(
                out,
                "{i},{},{k},{},{},{}",
                if shared { "shared" } else { "history" },
                cols(ra),
                cols(rb),
                opt(gap)
            );
        }
        out
    }
}

/// Number of shared bars after which the witness last differs by >= tick/2, plus one:
/// the warm-up this witness forces. Bars where `ok` is false are ignored.
pub fn forced_warmup(gaps: &[Option<f64>], half_tick: f64, ok: impl Fn(usize) -> bool) -> u64 {
    gaps.iter()
        .enumerate()
        .filter(|(i, g)| ok(*i) && g.is_some_and(|g| g >= half_tick))
        .map(|(i, _)| i as u64 + 2)
        .max()
        .unwrap_or(0)
}

fn zigzag(params: &Params, len: usize, start: f64) -> Vec<Bar> {
    let step = (params.range.value / 7.0).min(params.max_move.value);
    let mut p = start;
    let mut dir = 1.0;
    (0..len)
        .map(|_| {
            let b = Bar::flat(p);
            if p + dir * step > params.hi || p + dir * step < params.lo {
                dir = -dir;
            }
            p += dir * step;
            b
        })
        .collect()
}

/// The closed-form worst case for EMA, RMA, SMA and ATR: one history pinned at
/// `hi`, the other at `lo`. `shared_len` shared bars follow.
pub fn extreme_witness(ind: &Indicator, params: &Params, shared_len: usize) -> Witness {
    let hist_len = Seeding::ALL
        .iter()
        .map(|&s| ind.seeded_history_len(s))
        .max()
        .unwrap_or(1)
        .max(2);
    let (lo, hi) = (params.lo, params.hi);
    match ind {
        Indicator::Atr { .. } => {
            // Run A: full-range bars closing at lo (TR = D); run B: flat at hi (TR = 0).
            // First shared bar is flat at hi, so its TR is D in A and 0 in B.
            let hist_a = vec![
                Bar {
                    high: hi,
                    low: lo,
                    close: lo
                };
                hist_len
            ];
            let hist_b = vec![Bar::flat(hi); hist_len];
            let mut shared = vec![Bar::flat(hi)];
            shared.extend(zigzag(params, shared_len.saturating_sub(1), hi));
            Witness {
                hist_a,
                hist_b,
                shared,
                seeding_a: Seeding::FirstValue,
                seeding_b: Seeding::FirstValue,
                description:
                    "A: full-range bars closing at lo; B: flat at hi; shared path opens flat at hi"
                        .into(),
            }
        }
        _ => {
            let mid = lo + (hi - lo) / 2.0;
            Witness {
                hist_a: vec![Bar::flat(hi); hist_len],
                hist_b: vec![Bar::flat(lo); hist_len],
                shared: zigzag(params, shared_len, mid),
                seeding_a: Seeding::FirstValue,
                seeding_b: Seeding::FirstValue,
                description: "A: history pinned at hi; B: history pinned at lo; shared path zigzags from mid-range".into(),
            }
        }
    }
}

/// RSI's unbounded case: A rises into `p`, B falls into `p`, then the market is flat.
pub fn rsi_flat_witness(period: usize, params: &Params, flat_bars: usize) -> Witness {
    let mid = params.lo + params.range.value / 2.0;
    let hist_len = period + 2;
    let step = (params.range.value / 2.0 / hist_len as f64).min(params.max_move.value);
    let hist_a = (0..hist_len)
        .map(|i| Bar::flat(mid - step * (hist_len - i) as f64))
        .collect();
    let hist_b = (0..hist_len)
        .map(|i| Bar::flat(mid + step * (hist_len - i) as f64))
        .collect();
    Witness {
        hist_a,
        hist_b,
        shared: vec![Bar::flat(mid); flat_bars],
        seeding_a: Seeding::SmaSeeded,
        seeding_b: Seeding::SmaSeeded,
        description:
            "A: steady rise into mid-range; B: steady fall into mid-range; then a flat market"
                .into(),
    }
}

/// For each shared bar, whether the conditional assumptions of RSI / StochRSI
/// hold in both runs (always true for other indicators).
pub fn condition_mask(ind: &Indicator, w: &Witness) -> Vec<bool> {
    let (period, window, min_move, min_range) = match ind {
        Indicator::Rsi {
            period, min_move, ..
        } => (*period, 1, min_move.as_ref().map(|m| m.value), None),
        Indicator::StochRsi {
            period,
            k,
            min_move,
            min_range,
        } => (
            *period,
            *k,
            min_move.as_ref().map(|m| m.value),
            min_range.as_ref().map(|m| m.value),
        ),
        _ => return vec![true; w.shared.len()],
    };
    let rsi_ind = Indicator::Rsi {
        period,
        min_move: None,
    };
    let mut mask = vec![true; w.shared.len()];
    for (run, hist, seeding) in [
        (w.run_a(), w.hist_a.len(), w.seeding_a),
        (w.run_b(), w.hist_b.len(), w.seeding_b),
    ] {
        let closes: Vec<f64> = run.iter().map(|b| b.close).collect();
        let (g, l) = rsi_averages(&closes, period, seeding);
        let r = rsi_ind.evaluate(&run, seeding);
        for (i, ok) in mask.iter_mut().enumerate() {
            let idx = hist + i;
            if !*ok {
                continue;
            }
            if idx + 1 < window {
                *ok = false;
                continue;
            }
            let mut rlo = f64::INFINITY;
            let mut rhi = f64::NEG_INFINITY;
            for j in idx + 1 - window..=idx {
                match (g[j], l[j], r[j]) {
                    (Some(g), Some(l), Some(v)) if min_move.is_none_or(|m| g + l >= m) => {
                        rlo = rlo.min(v);
                        rhi = rhi.max(v);
                    }
                    _ => {
                        *ok = false;
                        break;
                    }
                }
            }
            if *ok && min_range.is_some_and(|wr| rhi - rlo < wr) {
                *ok = false;
            }
        }
    }
    mask
}

fn random_walk(rng: &mut Rng, params: &Params, len: usize, start: f64, scale: f64) -> Vec<Bar> {
    let mut p = start;
    (0..len)
        .map(|_| {
            p += (rng.unit() * 2.0 - 1.0) * params.max_move.value * scale;
            p = p.clamp(params.lo, params.hi);
            Bar::flat(p)
        })
        .collect()
}

fn history_candidate(rng: &mut Rng, params: &Params, len: usize) -> Vec<Bar> {
    let (lo, hi) = (params.lo, params.hi);
    match rng.below(5) {
        0 => vec![Bar::flat(if rng.below(2) == 0 { hi } else { lo }); len],
        1 => {
            let switch = len - 1 - rng.below(len.max(2) - 1);
            let (a, b) = if rng.below(2) == 0 {
                (hi, lo)
            } else {
                (lo, hi)
            };
            (0..len)
                .map(|i| Bar::flat(if i < switch { a } else { b }))
                .collect()
        }
        2 => {
            let start = lo + rng.unit() * (hi - lo);
            random_walk(rng, params, len, start, 1.0)
        }
        3 => {
            let up = rng.below(2) == 0;
            let step = params.max_move.value.min((hi - lo) / len as f64);
            (0..len)
                .map(|i| {
                    Bar::flat(if up {
                        lo + step * i as f64
                    } else {
                        hi - step * i as f64
                    })
                })
                .collect()
        }
        _ => {
            let a = lo + rng.unit() * (hi - lo);
            (0..len)
                .map(|i| Bar::flat(if i % 2 == 0 { a } else { hi + lo - a }))
                .collect()
        }
    }
}

fn shared_candidate(rng: &mut Rng, params: &Params, len: usize, from: f64) -> Vec<Bar> {
    match rng.below(4) {
        0 => vec![Bar::flat(from); len],
        1 => random_walk(rng, params, len, from, 1.0),
        2 => {
            let scale = 0.01 + rng.unit() * 0.2;
            random_walk(rng, params, len, from, scale)
        }
        _ => {
            let amp = params.max_move.value * (0.05 + rng.unit() * 0.5);
            (0..len)
                .map(|i| {
                    Bar::flat(
                        (from + if i % 2 == 0 { 0.0 } else { amp }).clamp(params.lo, params.hi),
                    )
                })
                .collect()
        }
    }
}

/// Best witness found by randomised search, as `(forced warm-up, witness)`.
/// Candidates violating the price bounds, move limit or conditional assumptions are ignored.
pub fn search_witness(
    ind: &Indicator,
    params: &Params,
    shared_len: usize,
    trials: usize,
    seed: u64,
) -> (u64, Option<Witness>) {
    let mut rng = Rng::new(seed);
    let half = params.tick.value / 2.0;
    let conventions = ind.conventions();
    let hist_len = conventions
        .iter()
        .map(|&s| ind.seeded_history_len(s))
        .max()
        .unwrap_or(1)
        + 20;
    let mut best: (u64, Option<Witness>) = (0, None);
    for t in 0..trials {
        let hist_a = history_candidate(&mut rng, params, hist_len);
        let hist_b = history_candidate(&mut rng, params, hist_len);
        let from = if t % 2 == 0 {
            hist_a[hist_len - 1].close
        } else {
            hist_b[hist_len - 1].close
        };
        let shared = shared_candidate(&mut rng, params, shared_len, from);
        let w = Witness {
            hist_a,
            hist_b,
            shared,
            seeding_a: conventions[rng.below(conventions.len())],
            seeding_b: conventions[rng.below(conventions.len())],
            description: format!("randomised search, trial {t}"),
        };
        if !w.is_admissible(params) {
            continue;
        }
        let gaps = w.gaps(ind);
        let mask = condition_mask(ind, &w);
        let forced = forced_warmup(&gaps, half, |i| mask[i]);
        if forced > best.0 {
            best = (forced, Some(w));
        }
    }
    best
}
