//! Proven warm-up lengths for recursive technical indicators.
//!
//! Given price bounds `[lo, hi]`, a tick size and a maximum move per bar,
//! [`bounds::bound`] returns the smallest number of shared bars after which
//! any two runs of an indicator agree to within tick/2, whatever their history
//! or seeding convention, together with the derivation. [`witness`] builds the
//! adversarial runs that show where the bound is attained.

pub mod bignum;
pub mod bounds;
pub mod exact;
pub mod indicators;
pub mod report;
pub mod witness;

pub use bounds::{bound, BoundReport, Indicator, Kind, Params, Verdict};
pub use indicators::{Bar, MacdOutput, Seeding};
