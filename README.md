# warmupbound

How many bars an EMA, RSI or ATR needs before its seed can't change a tick, proven.

## The problem

Recursive indicators (EMA, Wilder's RMA, ATR, MACD, KAMA, RSI) carry their starting point forever. The value depends on the seeding convention (first value, SMA of the first N bars, or pandas-style adjusted weights) and on where the history starts. So the same backtest run through a different library, or started a year later, can produce different signals. People handle this with rules of thumb like "discard 3x the period" or TA-Lib's unstable period. Nobody checks those against a price range and a tick size. For some indicators (RSI) no finite warm-up exists at all, and no tool warns you.

`warmupbound` computes the smallest number of bars `n` such that **any two runs sharing their last `n` bars agree to within tick/2**, whatever came before and whichever seeding convention each run used. It shows the derivation and gives an adversarial witness: two concrete runs that still disagree just before `n`.

## How it works

**Model.** Closes lie in `[lo, hi]`, `D = hi - lo`, and a close moves at most `max_move` per bar. Two runs have arbitrary, different histories, then share a path. Each run's seeding must be finished before the shared window starts (see Limitations). The tick is the tolerance on the indicator's value: price units for EMA/ATR/MACD/KAMA, RSI points for RSI.

**Contraction bounds.** An EMA update `y' = a x + (1-a) y` with a shared `x` multiplies the gap between two states by exactly `1-a`. Every convention keeps the state inside `[lo, hi]`, so once both runs are seeded the gap is at most `D`, and after `n` shared bars it is at most `(1-a)^n D`. Histories pinned at `hi` and at `lo` reach that value, so the bound is exact. Adjusted weights (pandas `adjust=True`) put more weight on the shared bars, so they only converge faster. Per indicator:

| indicator | worst-case gap after n shared bars | kind |
|---|---|---|
| EMA(N) | `((N-1)/(N+1))^n D` | exact |
| RMA(N) | `((N-1)/N)^n D` | exact |
| ATR(N) | `((N-1)/N)^(n-1) D`: the first shared true range still sees the previous close | exact |
| SMA(N) | `D (N-n)/N` for `n < N`, then 0 | exact |
| MACD line | `D (qf^n + qs^n)` | composite |
| MACD signal | `2D qg^n + ag D sum_j qg^(n-j) (qf^j + qs^j)`, summed in closed form | composite |
| MACD histogram | line + signal | composite |
| KAMA(N, fast, slow) | `D (1 - 4/(slow+1)^2)^(n-N)`: slowest smoothing constant once the efficiency-ratio window is shared | composite |
| RSI(N) | **unbounded**: flat-market witness | none |
| RSI(N) given avg gain + avg loss >= m | `100 max_move ((N-1)/N)^(n-1) / m` | conditional |
| StochRSI(N, K) given m and RSI range >= w | `20000 max_move ((N-1)/N)^(n-K) / (w m)` | conditional |

**Exact decision.** The answer is the first `n` with `bound(n) < tick/2`. Each bound is a sum of signed terms `c (p/q)^(n-o)`, possibly times `n`, with rational `c`, `p`, `q`. Inputs are parsed as exact decimals. The comparison is done in integers with a small built-in bignum: `p^n * D * 2 * tick_den < q^n * D_den * tick_num`. A floating-point pass only prunes. Any `n` whose float value is below `tick/2`, or within a relative 1e-6 of it, is re-decided exactly. The true worst-case gap cannot grow with `n`, since sharing more bars implies sharing fewer. So the first `n` below the threshold is the answer even when a closed form such as MACD's signal bound is not monotone.

Worked example: EMA(20) on `[90, 110]` with tick 0.01. `1-a = 19/21` and `D = 20`, so we need the smallest `n` with `19^n * 20 * 200 < 21^n * 1`. `(19/21)^82 * 20 = 0.005456` is still at least 0.005, and `(19/21)^83 * 20 = 0.004936` is below it, so `n = 83`. The rule of thumb says 60.

**RSI has no unconditional warm-up.** `RSI = 100 G/(G+L)`, and its sensitivity to the seed is `1/(G+L)`. Take a history rising into a price `p` and another falling into `p`, followed by a flat market at `p`. In run A the average loss is exactly 0 forever, and in run B the average gain is exactly 0 forever. RSI stays at 100 in one run and 0 in the other, for any number of bars. With a floor `m` on `G+L`, the gradient bound `|d(G/(G+L))| <= max(|dG|, |dL|)/(G+L)` along the segment between the two states gives the conditional bound above.

**Witnesses.** EMA, RMA, SMA and ATR get the closed-form extreme witness. It is replayed through the reference implementations in `src/indicators.rs` and differs by at least tick/2 after `n-1` shared bars and by less after `n`. The composite and conditional bounds get a randomised search over structured histories: pinned, step, ramp, random walk and alternating. The shared paths are flat, random walks and zigzags. Candidates that break the price bounds, the move limit or the conditional assumptions are discarded. The CLI reports the difference between the bound and the best witness it found.

## Install and usage

Needs a Rust toolchain (stable). No runtime dependencies. `proptest` is used only by the test suite.

```
git clone <this repository> warmupbound && cd warmupbound
cargo build --release
cargo test --release
```

Proven warm-up for EMA(20):

```
$ cargo run --release -q -- ema --period 20 --lo 90 --hi 110 --tick 0.01
EMA(20)
n = 83  (Exact bound)

derivation:
  - prices in [90, 110], D = hi - lo = 20; tick = 0.01; max move per bar = 20
  - alpha = 2/21, so each shared bar multiplies the state gap by exactly 1 - alpha = 19/21
  - every convention (first-value, SMA seed, adjusted weights) keeps the state inside [lo, hi], so once both runs have finished seeding the gap is at most D
  - bound(n) = (19/21)^n * D, attained by histories pinned at hi and at lo
  - smallest n with bound(n) < tick/2 = 5.000000e-3, decided in exact integers: n = 83
  - check: bound(82) = 5.455607e-3 >= tick/2 > bound(83) = 4.936026e-3

witness (A: history pinned at hi; B: history pinned at lo; shared path zigzags from mid-range):
  conventions: A = first-value, B = first-value; history 20 bars each; max move used 10; admissible: true
  forces a warm-up of 83 bars
  gap after      1 shared bars: 1.809524e1   (tick/2 = 5.000000e-3)
  gap after      2 shared bars: 1.637188e1   (tick/2 = 5.000000e-3)
  gap after      3 shared bars: 1.481266e1   (tick/2 = 5.000000e-3)
  gap after     82 shared bars: 5.455607e-3   (tick/2 = 5.000000e-3)
  gap after     83 shared bars: 4.936026e-3   (tick/2 = 5.000000e-3)
  gap after     84 shared bars: 4.465928e-3   (tick/2 = 5.000000e-3)
  shared closes: 100, 102.85714285714286, 105.71428571428572, 108.57142857142858, 105.71428571428572, 102.85714285714286, 100, 97.14285714285714 ...
```

Write both diverging runs to CSV (`bar, phase, shared_bars, high/low/close/value` for each run, `gap`):

```
$ cargo run --release -q -- ema --period 20 --lo 90 --hi 110 --tick 0.01 --csv witness.csv
...
wrote both runs to witness.csv
```

RSI, unconditionally and with an activity floor:

```
$ cargo run --release -q -- rsi --lo 90 --hi 110 --tick 0.01
RSI(14)
n = none: no finite warm-up exists
...
witness (A: steady rise into mid-range; B: steady fall into mid-range; then a flat market):
  RSI gap over 10000 flat bars never drops below 99.99999999999999

$ cargo run --release -q -- rsi --lo 90 --hi 110 --tick 0.01 --max-move 2 --min-move 0.5
RSI(14)
n = 154  (Conditional bound)
...
  forces a warm-up of 126 bars
  bound - best witness = 28 bars
```

Composites:

```
$ cargo run --release -q -- macd --output signal --lo 90 --hi 110 --tick 0.01
MACD(12,26,9) Signal
n = 113  (Composite bound)
...
  forces a warm-up of 113 bars
  bound - best witness = 0 bars

$ cargo run --release -q -- kama --lo 90 --hi 110 --tick 0.01
KAMA(10,2,30)
n = 1999  (Composite bound)
...
  forces a warm-up of 1918 bars
  bound - best witness = 81 bars
```

Other commands: `rma`, `sma`, `atr --period N`; `macd --fast --slow --signal --output line|signal|hist`; `kama --period --fast --slow`; `stochrsi --period --k [--min-move m --min-range w]`. Common options: `--max-move`, `--csv`, `--trials`, `--seed`. Run `cargo run --release -- --help` for the full list.

## Results

`cargo run --release -- report` reproduces the table below. Every number is a deterministic count of bars, not a timing, so hardware does not matter; the report runs in under a second. Grid: periods 5, 10, 14, 20, 26, 50, 100, 200 and tick/range ratios 1e-2 to 1e-6. Range = 1 and the move limit is unconstrained (max move = range). KAMA uses fast 2 and slow 30.

The table shows how often each rule of thumb is shorter than the proven warm-up:

| indicator | configs | 3 x period | TA-Lib default lookback (unstable period 0) | fixed 100 bars |
|---|---|---|---|---|
| EMA | 40 | 32/40 (80%) | 40/40 (100%) | 21/40 (52%) |
| RMA | 40 | 40/40 (100%) | 40/40 (100%) | 31/40 (78%) |
| ATR | 40 | 40/40 (100%) | 40/40 (100%) | 31/40 (78%) |
| KAMA | 40 | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) |
| RSI | 40 | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) |
| all | 200 | 192/200 (96%) | 200/200 (100%) | 163/200 (82%) |

Proven EMA warm-up in bars, with 3 x period in parentheses:

| period | tick/range 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|
| 5 | 14 (15) | 19 (15) | 25 (15) | 31 (15) | 36 (15) |
| 14 | 38 (42) | 54 (42) | 70 (42) | 86 (42) | 102 (42) |
| 20 | 53 (60) | 76 (60) | 99 (60) | 122 (60) | 145 (60) |
| 50 | 133 (150) | 190 (150) | 248 (150) | 306 (150) | 363 (150) |
| 200 | 530 (600) | 761 (600) | 991 (600) | 1221 (600) | 1451 (600) |

What the grid shows:

- For EMA, "3x the period" is only safe at coarse ticks, around 1e-2 of the range.
- Wilder smoothing (RMA, ATR, and RSI's internals) decays about twice as slowly as an EMA of the same period. The 3x rule is short everywhere on the grid. RMA(14) at tick/range 1e-4 needs 134 bars.
- KAMA with the standard slow constant needs 1,300 to 3,700 bars. A flat market pins it at its slowest smoothing constant. For KAMA(10,2,30) at tick 0.01 on a range of 20, the random search finds a witness forcing 1,918 bars against a bound of 1,999.
- The f64 formula `ceil(ln(tick/2D)/ln(1-a))` agrees with the exact answer at every point of this grid. It fails at exact boundaries, where `(1-a)^k D` equals tick/2: it returns `k` or `k+1` depending on rounding, while the correct answer is always `k+1`. `tests/exactness.rs` builds those cases by hand.

### What the tests check

- **Tightness** (`tests/tightness.rs`): EMA, RMA, ATR and SMA over 6 markets × 8 periods × convention pairs. The witness differs by at least tick/2 after `n-1` shared bars and by less after `n`. The replayed gap matches the closed form to 1e-9.
- **Soundness** (`tests/soundness.rs`, proptest): random markets, move limits, histories of different lengths, random walks, pinned extremes, wicks for ATR, and random conventions per run. No EMA, RMA, SMA, ATR, MACD (all outputs), KAMA, conditional RSI or conditional StochRSI gap reaches tick/2 at or after `n`. Shrinking is capped at 30 s.
- **Exactness** (`tests/exactness.rs`): hand-built equality boundaries for EMA(3), EMA(7), RMA(4) and RMA(100), with exponents up to 60. The answer is `k+1`, and the f64 log formula is shown to disagree. Moving the tick by 1e-6 moves the answer by one bar even though f64 cannot tell the two ticks apart.
- **RSI** (`tests/rsi_and_composites.rs`): the flat witness keeps the gap at 100 for 10,000 bars under every convention pair for periods of 15 and above. Shorter periods are checked until f64 underflows the decaying average. The conditional bound scales as ln 2 / ln(N/(N-1)) bars per halving of `m`.
- **Composites**: for MACD line, signal and histogram, KAMA and conditional RSI, the bound is at least the best randomised witness, and the gap between them is finite and at most half the bound.
- **Failure mode**: if one run starts its SMA seed inside the shared window, a hand-built path breaks the bound at `n`.

## Design notes

The main decision was to separate *finding* `n` from *deciding* it. A float log formula is fine almost everywhere, as the report shows. But "exactly at the boundary" is where a warm-up claim is either true or false, and there floats are right only by luck. Deciding every candidate with exact rationals costs little, because float pruning means only one or two bignum comparisons happen per query, with operands of a few thousand bits. It also lets one solver handle every bound, including MACD's signal line, where the geometric terms have mixed signs and the closed form is not monotone. The bignum is about 250 lines of schoolbook arithmetic rather than a dependency, which keeps the runtime dependency count at zero.

The second decision was what to promise. A single number of bars is useful only if it holds for every convention a user might meet, so the bound is the worst case over conventions and histories, and the witness shows it cannot be lowered. This makes some results strict. KAMA's bound is driven by a flat market, and RSI gets no unconditional number at all. Faking a finite RSI warm-up would have been easy and wrong. Instead the tool asks for an explicit assumption (`--min-move`) and says so in the derivation.

## Limitations

- **Seeding must finish before the shared window.** If a run starts inside the window, for example a later backtest whose SMA seed averages bars the other run also sees, the flat seed weights let that run lag more than a recursive state would. `tests/soundness.rs` builds a path where the gap at `n` is at least tick/2. In that situation, count `n` from the end of the later run's seeding window.
- **`--max-move` only tightens the conditional RSI and StochRSI bounds.** EMA, RMA, SMA, ATR, MACD and KAMA use the range `D` alone. They stay sound under a move limit, but the extreme witness jumps from `hi` or `lo` to mid-range, which a tight limit forbids, so they are no longer exact. ATR also does not use the move limit to bound the true range.
- **Composite and conditional bounds are not claimed tight.** MACD bounds the fast and slow seed gaps separately, although a single history drives both. KAMA assumes the slowest smoothing constant throughout. The randomised search gives a lower bound on the true warm-up, not the true value.
- **StochRSI** has only the conditional bound, which needs both an activity floor and a minimum RSI range. Its unconditional witness reuses the RSI one, where StochRSI itself is undefined.
- **Not covered:** Bollinger Bands, MACD variants with SMA signal lines, and volume-weighted indicators. The reference implementations are clear `f64` code for replaying witnesses, not a fast indicator library.
- In `f64`, the RSI flat witness for periods below 15 underflows a decaying average after roughly a thousand bars (period 2) to several thousand. The test requires at least 700 representable bars. After that the reference implementation reports the flat-market value 50. This is an artefact of floating point: the exact argument has no horizon.

## License

MIT. See [LICENSE](LICENSE).
