//! Minimal arbitrary-precision arithmetic: non-negative integers and
//! non-negative rationals, enough to compare sums of rational powers exactly.

use std::cmp::Ordering;

/// Non-negative integer stored as little-endian base-2^32 limbs, no trailing zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    pub fn zero() -> Self {
        BigUint { limbs: Vec::new() }
    }

    pub fn from_u64(v: u64) -> Self {
        let mut b = BigUint {
            limbs: vec![v as u32, (v >> 32) as u32],
        };
        b.trim();
        b
    }

    pub fn from_u128(v: u128) -> Self {
        let mut b = BigUint {
            limbs: vec![
                v as u32,
                (v >> 32) as u32,
                (v >> 64) as u32,
                (v >> 96) as u32,
            ],
        };
        b.trim();
        b
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    pub fn add(&self, other: &BigUint) -> BigUint {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u64;
            let b = *other.limbs.get(i).unwrap_or(&0) as u64;
            let s = a + b + carry;
            out.push(s as u32);
            carry = s >> 32;
        }
        out.push(carry as u32);
        let mut r = BigUint { limbs: out };
        r.trim();
        r
    }

    /// `self - other`; panics if `other > self`.
    pub fn sub(&self, other: &BigUint) -> BigUint {
        assert!(
            self.cmp_big(other) != Ordering::Less,
            "BigUint subtraction underflow"
        );
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i64;
            let b = *other.limbs.get(i).unwrap_or(&0) as i64;
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(d as u32);
        }
        let mut r = BigUint { limbs: out };
        r.trim();
        r
    }

    pub fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() {
            return BigUint::zero();
        }
        let mut out = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let cur = out[i + j] as u64 + a as u64 * b as u64 + carry;
                out[i + j] = cur as u32;
                carry = cur >> 32;
            }
            let mut k = i + other.limbs.len();
            while carry > 0 {
                let cur = out[k] as u64 + carry;
                out[k] = cur as u32;
                carry = cur >> 32;
                k += 1;
            }
        }
        let mut r = BigUint { limbs: out };
        r.trim();
        r
    }

    pub fn pow(base: u64, exp: u64) -> BigUint {
        let mut result = BigUint::from_u64(1);
        let mut b = BigUint::from_u64(base);
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&b);
            }
            e >>= 1;
            if e > 0 {
                b = b.mul(&b);
            }
        }
        result
    }

    pub fn cmp_big(&self, other: &BigUint) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {
                for i in (0..self.limbs.len()).rev() {
                    match self.limbs[i].cmp(&other.limbs[i]) {
                        Ordering::Equal => continue,
                        o => return o,
                    }
                }
                Ordering::Equal
            }
            o => o,
        }
    }

    /// Nearest f64 (loses precision for large values; used only for display).
    pub fn to_f64(&self) -> f64 {
        self.limbs
            .iter()
            .rev()
            .fold(0.0, |acc, &l| acc * 4294967296.0 + l as f64)
    }

    /// Number of significant bits.
    pub fn bits(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() as u64 - 1) * 32 + (32 - top.leading_zeros() as u64),
        }
    }
}

/// Non-negative rational `num / den`, never reduced (cross-multiplication only).
#[derive(Clone, Debug)]
pub struct Ratio {
    pub num: BigUint,
    pub den: BigUint,
}

impl Ratio {
    pub fn new(num: BigUint, den: BigUint) -> Self {
        assert!(!den.is_zero(), "zero denominator");
        Ratio { num, den }
    }

    pub fn from_u64s(num: u64, den: u64) -> Self {
        Ratio::new(BigUint::from_u64(num), BigUint::from_u64(den))
    }

    pub fn zero() -> Self {
        Ratio::from_u64s(0, 1)
    }

    pub fn add(&self, o: &Ratio) -> Ratio {
        Ratio::new(
            self.num.mul(&o.den).add(&o.num.mul(&self.den)),
            self.den.mul(&o.den),
        )
    }

    pub fn mul(&self, o: &Ratio) -> Ratio {
        Ratio::new(self.num.mul(&o.num), self.den.mul(&o.den))
    }

    /// `(p/q)^e`
    pub fn pow_frac(p: u64, q: u64, e: u64) -> Ratio {
        Ratio::new(BigUint::pow(p, e), BigUint::pow(q, e))
    }

    pub fn cmp_ratio(&self, o: &Ratio) -> Ordering {
        self.num.mul(&o.den).cmp_big(&o.num.mul(&self.den))
    }

    pub fn to_f64(&self) -> f64 {
        // Keep only the top three limbs of each side so huge operands stay finite.
        fn top(b: &BigUint) -> (f64, i64) {
            let n = b.limbs.len();
            if n <= 3 {
                return (b.to_f64(), 0);
            }
            let v = b.limbs[n - 3..]
                .iter()
                .rev()
                .fold(0.0, |acc, &l| acc * 4294967296.0 + l as f64);
            (v, (n as i64 - 3) * 32)
        }
        let (a, ea) = top(&self.num);
        let (b, eb) = top(&self.den);
        if a == 0.0 {
            return 0.0;
        }
        (a / b) * 2f64.powi((ea - eb).clamp(-4000, 4000) as i32)
    }
}

/// Parses a plain decimal literal like `-12.345` into
/// `(numerator, scale)` meaning `numerator / 10^scale`.
pub fn parse_decimal(s: &str) -> Result<(i128, u32), String> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    if body.is_empty() {
        return Err(format!("not a decimal number: {s:?}"));
    }
    let (int_part, frac_part) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(format!("not a decimal number: {s:?}"));
    }
    if !int_part.chars().all(|c| c.is_ascii_digit())
        || !frac_part.chars().all(|c| c.is_ascii_digit())
    {
        return Err(format!("not a plain decimal number (no exponents): {s:?}"));
    }
    if frac_part.len() > 30 || int_part.len() > 30 {
        return Err(format!("too many digits: {s:?}"));
    }
    let digits = format!("{int_part}{frac_part}");
    let mut v: i128 = digits.parse().map_err(|_| format!("bad number {s:?}"))?;
    if neg {
        v = -v;
    }
    Ok((v, frac_part.len() as u32))
}

/// A non-negative decimal literal as an exact rational.
pub fn decimal_to_ratio(s: &str) -> Result<Ratio, String> {
    let (v, scale) = parse_decimal(s)?;
    if v < 0 {
        return Err(format!("expected a non-negative number, got {s}"));
    }
    Ok(Ratio::new(
        BigUint::from_u128(v as u128),
        BigUint::from_u128(10u128.pow(scale)),
    ))
}

/// Exact `hi - lo` of two decimal literals; errors unless it is positive.
pub fn decimal_difference(hi: &str, lo: &str) -> Result<Ratio, String> {
    let (h, sh) = parse_decimal(hi)?;
    let (l, sl) = parse_decimal(lo)?;
    let s = sh.max(sl);
    let hh = h.checked_mul(10i128.pow(s - sh)).ok_or("overflow")?;
    let ll = l.checked_mul(10i128.pow(s - sl)).ok_or("overflow")?;
    let d = hh - ll;
    if d <= 0 {
        return Err(format!("--hi ({hi}) must be greater than --lo ({lo})"));
    }
    Ok(Ratio::new(
        BigUint::from_u128(d as u128),
        BigUint::from_u128(10u128.pow(s)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_matches_u128() {
        let a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let b = 0x1234_5678_9ABC_DEF0u64;
        let p = BigUint::from_u64(a).mul(&BigUint::from_u64(b));
        assert_eq!(p, BigUint::from_u128(a as u128 * b as u128));
    }

    #[test]
    fn pow_and_sub_roundtrip() {
        let p = BigUint::pow(3, 200);
        let q = BigUint::pow(3, 199);
        let two_q = q.add(&q);
        assert_eq!(p.sub(&two_q), q);
        assert_eq!(BigUint::pow(2, 64), BigUint::from_u128(1u128 << 64));
    }

    #[test]
    fn comparison_distinguishes_one_unit_at_large_size() {
        let a = BigUint::pow(7, 500);
        let b = a.add(&BigUint::from_u64(1));
        assert_eq!(a.cmp_big(&b), Ordering::Less);
        assert_eq!(b.cmp_big(&a), Ordering::Greater);
        assert_eq!(a.cmp_big(&a.clone()), Ordering::Equal);
    }

    #[test]
    fn decimals_parse_exactly() {
        assert_eq!(parse_decimal("0.01").unwrap(), (1, 2));
        assert_eq!(parse_decimal("-12.50").unwrap(), (-1250, 2));
        assert!(parse_decimal("1e-3").is_err());
        assert!(parse_decimal("").is_err());
        let d = decimal_difference("110", "90.5").unwrap();
        assert_eq!(d.cmp_ratio(&Ratio::from_u64s(39, 2)), Ordering::Equal);
        assert!(decimal_difference("1", "2").is_err());
    }

    #[test]
    fn ratio_to_f64_handles_huge_operands() {
        let r = Ratio::pow_frac(199, 201, 3000);
        let expected = (3000.0 * (199f64 / 201.0).ln()).exp();
        assert!(
            (r.to_f64() / expected - 1.0).abs() < 1e-9,
            "{} vs {}",
            r.to_f64(),
            expected
        );
    }
}
