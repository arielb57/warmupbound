use std::collections::HashMap;
use std::process::ExitCode;
use warmupbound::bounds::{bound, Decimal, Indicator, Kind, Params, Verdict};
use warmupbound::indicators::MacdOutput;
use warmupbound::report;
use warmupbound::witness::{
    extreme_witness, forced_warmup, rsi_flat_witness, search_witness, Witness,
};

const USAGE: &str = "\
warmupbound: proven warm-up lengths for recursive indicators

USAGE:
  warmupbound <indicator> --lo <price> --hi <price> --tick <tick> [options]
  warmupbound report

INDICATORS:
  ema, rma, sma, atr     --period N
  macd                   --fast 12 --slow 26 --signal 9 --output line|signal|hist
  kama                   --period 10 --fast 2 --slow 30
  rsi                    --period 14 [--min-move m]
  stochrsi               --period 14 --k 14 [--min-move m --min-range w]

OPTIONS:
  --max-move <x>   largest close-to-close move per bar (default: hi - lo)
  --csv <path>     write both diverging witness runs to a CSV file
  --trials <n>     randomised witness search trials for composite bounds (default 300)
  --seed <n>       seed for the witness search (default 1)

Tick is the tolerance on the indicator's value (price units for EMA/ATR/MACD,
RSI points for RSI and StochRSI). Runs agree when they differ by less than tick/2.
";

fn parse_flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {:?}", args[i]))?;
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("--{key} needs a value"))?;
        map.insert(key.to_string(), val.clone());
        i += 2;
    }
    Ok(map)
}

fn usize_flag(
    flags: &HashMap<String, String>,
    key: &str,
    default: Option<usize>,
) -> Result<usize, String> {
    match flags.get(key) {
        Some(v) => v
            .parse()
            .map_err(|_| format!("--{key} must be a positive integer")),
        None => default.ok_or_else(|| format!("--{key} is required")),
    }
}

fn opt_decimal(flags: &HashMap<String, String>, key: &str) -> Result<Option<Decimal>, String> {
    flags.get(key).map(|v| Decimal::parse(v)).transpose()
}

fn build_indicator(name: &str, flags: &HashMap<String, String>) -> Result<Indicator, String> {
    Ok(match name {
        "ema" => Indicator::Ema {
            period: usize_flag(flags, "period", None)?,
        },
        "rma" => Indicator::Rma {
            period: usize_flag(flags, "period", None)?,
        },
        "sma" => Indicator::Sma {
            period: usize_flag(flags, "period", None)?,
        },
        "atr" => Indicator::Atr {
            period: usize_flag(flags, "period", None)?,
        },
        "macd" => Indicator::Macd {
            fast: usize_flag(flags, "fast", Some(12))?,
            slow: usize_flag(flags, "slow", Some(26))?,
            signal: usize_flag(flags, "signal", Some(9))?,
            output: MacdOutput::parse(flags.get("output").map(String::as_str).unwrap_or("line"))?,
        },
        "kama" => Indicator::Kama {
            period: usize_flag(flags, "period", Some(10))?,
            fast: usize_flag(flags, "fast", Some(2))?,
            slow: usize_flag(flags, "slow", Some(30))?,
        },
        "rsi" => Indicator::Rsi {
            period: usize_flag(flags, "period", Some(14))?,
            min_move: opt_decimal(flags, "min-move")?,
        },
        "stochrsi" => Indicator::StochRsi {
            period: usize_flag(flags, "period", Some(14))?,
            k: usize_flag(flags, "k", Some(14))?,
            min_move: opt_decimal(flags, "min-move")?,
            min_range: opt_decimal(flags, "min-range")?,
        },
        other => return Err(format!("unknown indicator {other:?}")),
    })
}

fn print_witness(ind: &Indicator, params: &Params, w: &Witness, forced: u64) {
    let gaps = w.gaps(ind);
    println!("\nwitness ({}):", w.description);
    println!(
        "  conventions: A = {}, B = {}; history {} bars each; max move used {}; admissible: {}",
        w.seeding_a,
        w.seeding_b,
        w.hist_a.len(),
        w.max_move(),
        w.is_admissible(params)
    );
    println!("  forces a warm-up of {forced} bars");
    let show = |k: usize| {
        if k >= 1 && k <= gaps.len() {
            let g = gaps[k - 1]
                .map(|g| format!("{g:.6e}"))
                .unwrap_or_else(|| "undefined".into());
            println!(
                "  gap after {k:>6} shared bars: {g}   (tick/2 = {:.6e})",
                params.tick.value / 2.0
            );
        }
    };
    let last = w.shared.len();
    let mut ks: Vec<usize> = vec![1, 2, 3];
    if forced >= 2 {
        ks.extend([forced as usize - 1, forced as usize]);
    }
    ks.push(last);
    ks.sort_unstable();
    ks.dedup();
    for k in ks {
        show(k);
    }
    let path: Vec<String> = w
        .shared
        .iter()
        .take(8)
        .map(|b| format!("{}", b.close))
        .collect();
    println!("  shared closes: {} ...", path.join(", "));
}

fn run_indicator(name: &str, args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args)?;
    let get = |k: &str| {
        flags
            .get(k)
            .map(String::as_str)
            .ok_or_else(|| format!("--{k} is required"))
    };
    let params = Params::from_decimals(
        get("lo")?,
        get("hi")?,
        get("tick")?,
        flags.get("max-move").map(String::as_str),
    )?;
    let ind = build_indicator(name, &flags)?;
    let report = bound(&ind, &params)?;
    println!("{}", report.indicator);
    match report.verdict {
        Verdict::Proven(n) => println!("n = {n}  ({:?} bound)", report.kind),
        Verdict::Unbounded => println!("n = none: no finite warm-up exists"),
        Verdict::ExceedsCap(c) => println!("n > {c} (search cap)"),
    }
    println!("\nderivation:");
    for line in &report.derivation {
        println!("  - {line}");
    }
    let trials = usize_flag(&flags, "trials", Some(300))?;
    let seed = usize_flag(&flags, "seed", Some(1))? as u64;
    let half = params.tick.value / 2.0;
    let witness = match (&ind, report.verdict) {
        (Indicator::Rsi { period, .. }, Verdict::Unbounded)
        | (Indicator::StochRsi { period, .. }, Verdict::Unbounded) => {
            let w = rsi_flat_witness(*period, &params, 10_000);
            let rsi = Indicator::Rsi {
                period: *period,
                min_move: None,
            };
            let gaps = w.gaps(&rsi);
            let min_gap = gaps.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
            println!("\nwitness ({}):", w.description);
            println!("  RSI gap over 10000 flat bars never drops below {min_gap}");
            Some((rsi, w))
        }
        (_, Verdict::Proven(n)) if report.kind == Kind::Exact => {
            let w = extreme_witness(&ind, &params, n as usize + 1);
            let forced = forced_warmup(&w.gaps(&ind), half, |_| true);
            print_witness(&ind, &params, &w, forced);
            Some((ind.clone(), w))
        }
        (_, Verdict::Proven(n)) => {
            let len = (n as usize + 2).min(20_000);
            let (forced, w) = search_witness(&ind, &params, len, trials, seed);
            match w {
                Some(w) => {
                    print_witness(&ind, &params, &w, forced);
                    println!("  bound - best witness = {} bars", n.saturating_sub(forced));
                    Some((ind.clone(), w))
                }
                None => {
                    println!(
                        "\nwitness search found no pair differing by tick/2 after any shared bar"
                    );
                    None
                }
            }
        }
        _ => None,
    };
    if let Some(path) = flags.get("csv") {
        let (ind, w) = witness.ok_or("no witness to write")?;
        std::fs::write(path, w.to_csv(&ind)).map_err(|e| format!("writing {path}: {e}"))?;
        println!("\nwrote both runs to {path}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("-h") | Some("--help") | Some("help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("report") => {
            print!("{}", report::render(&report::grid()));
            Ok(())
        }
        Some(name) => run_indicator(name, &args[1..]),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
