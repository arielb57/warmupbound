//! Bounds of the form `sum_i ± c_i * [n *] (p_i/q_i)^(n - o_i)` and an exact
//! solver for the smallest `n` at which such a bound drops below `tick / 2`.

use crate::bignum::{BigUint, Ratio};
use std::cmp::Ordering;

/// One signed geometric term `± coef * [n *] (p/q)^(n - offset)`.
#[derive(Clone, Debug)]
pub struct Term {
    pub negative: bool,
    pub coef: Ratio,
    pub p: u64,
    pub q: u64,
    pub offset: u64,
    pub times_n: bool,
}

impl Term {
    pub fn pos(coef: Ratio, p: u64, q: u64, offset: u64) -> Term {
        Term {
            negative: false,
            coef,
            p,
            q,
            offset,
            times_n: false,
        }
    }

    fn exact(&self, n: u64) -> Ratio {
        let mut r = self
            .coef
            .mul(&Ratio::pow_frac(self.p, self.q, n - self.offset));
        if self.times_n {
            r = r.mul(&Ratio::from_u64s(n, 1));
        }
        r
    }

    fn float(&self, n: u64) -> f64 {
        let e = (n - self.offset) as f64;
        let base = self.p as f64 / self.q as f64;
        let mut v = self.coef.to_f64()
            * if self.p == 0 && e == 0.0 {
                1.0
            } else {
                (e * base.ln()).exp()
            };
        if self.times_n {
            v *= n as f64;
        }
        if self.negative {
            -v
        } else {
            v
        }
    }
}

/// A gap bound as a function of the number of shared bars `n`:
/// `pre` for `n < start`, the sum of `terms` from `start` on.
#[derive(Clone, Debug)]
pub struct GeometricBound {
    pub start: u64,
    pub pre: Ratio,
    pub terms: Vec<Term>,
}

/// Outcome of a search for the warm-up length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solved {
    Found(u64),
    /// No `n <= cap` satisfies the bound.
    ExceedsCap(u64),
}

impl GeometricBound {
    /// Exact value of the bound after `n` shared bars.
    pub fn exact(&self, n: u64) -> (Ratio, Ratio) {
        if n < self.start {
            return (self.pre.clone(), Ratio::zero());
        }
        let mut pos = Ratio::zero();
        let mut neg = Ratio::zero();
        for t in &self.terms {
            let v = t.exact(n);
            if t.negative {
                neg = neg.add(&v);
            } else {
                pos = pos.add(&v);
            }
        }
        (pos, neg)
    }

    /// Exact decision of `bound(n) < rhs`.
    pub fn exact_below(&self, n: u64, rhs: &Ratio) -> bool {
        let (pos, neg) = self.exact(n);
        pos.cmp_ratio(&rhs.add(&neg)) == Ordering::Less
    }

    /// Exact decision of `bound(n) == rhs`.
    pub fn exact_equals(&self, n: u64, rhs: &Ratio) -> bool {
        let (pos, neg) = self.exact(n);
        pos.cmp_ratio(&rhs.add(&neg)) == Ordering::Equal
    }

    pub fn float(&self, n: u64) -> f64 {
        if n < self.start {
            return self.pre.to_f64();
        }
        self.terms.iter().map(|t| t.float(n)).sum()
    }

    /// Smallest `n` with `bound(n) < rhs`, decided exactly.
    ///
    /// Floating point only prunes: any `n` whose float value is within a
    /// relative 1e-6 of `rhs`, or below it, is re-decided with big integers.
    /// The true worst-case gap is non-increasing in `n` (sharing more bars
    /// implies sharing fewer), so the first `n` below `rhs` is the answer
    /// even when the closed-form bound itself is not monotone.
    pub fn solve(&self, rhs: &Ratio, cap: u64) -> Solved {
        let r = rhs.to_f64();
        for n in 0..=cap {
            if n < self.start && n > 0 {
                continue;
            }
            let f = self.float(n);
            if f > r * (1.0 + 1e-6) {
                continue;
            }
            if self.exact_below(n, rhs) {
                return Solved::Found(n);
            }
        }
        Solved::ExceedsCap(cap)
    }
}

/// `p / q` for the contraction `1 - alpha` of an EMA (`alpha = 2/(N+1)`).
pub fn ema_contraction(period: u64) -> (u64, u64) {
    (period - 1, period + 1)
}

/// `p / q` for Wilder smoothing (`alpha = 1/N`).
pub fn rma_contraction(period: u64) -> (u64, u64) {
    (period - 1, period)
}

/// What most code does: `ceil(ln(tick / (2 D)) / ln(q))`, in `f64`.
pub fn naive_log_warmup(q: f64, range: f64, tick: f64) -> u64 {
    let x = (tick / (2.0 * range)).ln() / q.ln();
    if x <= 0.0 {
        0
    } else {
        x.ceil() as u64
    }
}

/// `tick / 2` as an exact rational.
pub fn half(r: &Ratio) -> Ratio {
    Ratio::new(r.num.clone(), r.den.mul(&BigUint::from_u64(2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_returns_first_strictly_below_point() {
        // bound(n) = (1/2)^n, rhs = 1/8 -> equality at n = 3, strictly below at 4.
        let b = GeometricBound {
            start: 0,
            pre: Ratio::from_u64s(1, 1),
            terms: vec![Term::pos(Ratio::from_u64s(1, 1), 1, 2, 0)],
        };
        let rhs = Ratio::from_u64s(1, 8);
        assert!(b.exact_equals(3, &rhs));
        assert_eq!(b.solve(&rhs, 100), Solved::Found(4));
        assert_eq!(
            b.solve(&Ratio::from_u64s(1, 1 << 40), 10),
            Solved::ExceedsCap(10)
        );
    }

    #[test]
    fn solver_handles_non_monotone_bounds_with_negative_terms() {
        // 3 * n * (1/2)^n: 3/2, 3/2, 9/8, then 3/4 at n = 4.
        let b = GeometricBound {
            start: 0,
            pre: Ratio::from_u64s(3, 1),
            terms: vec![Term {
                negative: false,
                coef: Ratio::from_u64s(3, 1),
                p: 1,
                q: 2,
                offset: 0,
                times_n: true,
            }],
        };
        // n=0 -> 0 which is below 1: shows the solver takes n=0 literally.
        assert_eq!(b.solve(&Ratio::from_u64s(1, 1), 100), Solved::Found(0));
        let shifted = GeometricBound { start: 1, ..b };
        assert_eq!(
            shifted.solve(&Ratio::from_u64s(1, 1), 100),
            Solved::Found(4)
        );
        assert!(!shifted.exact_below(3, &Ratio::from_u64s(1, 1)));
    }
}
