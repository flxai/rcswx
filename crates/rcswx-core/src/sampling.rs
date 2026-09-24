//! Native implementation of the pinned discrete skew-normal selector.
//!
//! The selector intentionally mirrors the reference's *discrete* grid law:
//! it uses the final enumerated mask cost as the grid resolution and keeps its
//! Python integer conversion and negative-index behavior. It does not sample a
//! continuous edit fraction and never substitutes a uniform distribution.
use crate::architecture::Budget;
use crate::error::{Error, Result};
use crate::selection::{self, Operation};
use rand_chacha::ChaCha12Rng;
use rand_core::{RngCore, SeedableRng};

pub const RNG_ALGORITHM: &str = "ChaCha12";
pub const RNG_STATE_VERSION: u32 = 1;
const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const INV_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;
const ADAPTIVE_TOLERANCE: f64 = 1.0e-14;
const ADAPTIVE_DEPTH: u32 = 48;

/// Serializable state for the named, pinned native generator.
///
/// Integer-valued fields that may exceed JavaScript/Python float precision are
/// decimal strings. `word_position` remains a `u128` count even though the
/// ChaCha implementation itself cycles its counter after 68 significant bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRngState {
    pub version: u32,
    pub algorithm: String,
    pub seed: String,
    pub stream: String,
    pub word_position: String,
}

/// A stateful ChaCha12 stream with no ambient entropy source.
#[derive(Clone)]
pub struct NativeRng {
    seed: [u8; 32],
    stream: u64,
    word_position: u128,
    rng: ChaCha12Rng,
}

impl NativeRng {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self {
            seed,
            stream: 0,
            word_position: 0,
            rng: ChaCha12Rng::from_seed(seed),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let value = self.rng.next_u64();
        // `next_u64` consumes two 32-bit ChaCha words. Retaining the complete
        // u128 counter makes serialized state lossless even after the generator
        // has wrapped its native 68-bit counter.
        self.word_position = self.word_position.wrapping_add(2);
        value
    }

    /// Return the high 53 bits of the next word as a uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * INV_TWO_POW_53
    }

    pub fn state(&self) -> NativeRngState {
        NativeRngState {
            version: RNG_STATE_VERSION,
            algorithm: RNG_ALGORITHM.to_owned(),
            seed: encode_hex(&self.seed),
            stream: self.stream.to_string(),
            word_position: self.word_position.to_string(),
        }
    }

    pub fn from_state(state: NativeRngState) -> Result<Self> {
        if state.version != RNG_STATE_VERSION {
            return Err(Error::InvalidInput(
                "unsupported native RNG state version".into(),
            ));
        }
        if state.algorithm != RNG_ALGORITHM {
            return Err(Error::InvalidInput(
                "unsupported native RNG algorithm".into(),
            ));
        }
        let seed = decode_seed(&state.seed)?;
        let stream = state.stream.parse::<u64>().map_err(|_| {
            Error::InvalidInput("native RNG stream must be an unsigned decimal string".into())
        })?;
        let word_position = state.word_position.parse::<u128>().map_err(|_| {
            Error::InvalidInput(
                "native RNG word position must be an unsigned decimal string".into(),
            )
        })?;
        let mut rng = ChaCha12Rng::from_seed(seed);
        rng.set_stream(stream);
        rng.set_word_pos(word_position);
        Ok(Self {
            seed,
            stream,
            word_position,
            rng,
        })
    }
}

fn encode_hex(seed: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for &byte in seed {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn decode_seed(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(Error::InvalidInput(
            "native RNG seed must contain 64 hexadecimal digits".into(),
        ));
    }
    let mut seed = [0u8; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        let high = hex_value(value.as_bytes()[index * 2]);
        let low = hex_value(value.as_bytes()[index * 2 + 1]);
        *byte = match (high, low) {
            (Some(high), Some(low)) => (high << 4) | low,
            _ => {
                return Err(Error::InvalidInput(
                    "native RNG seed is not hexadecimal".into(),
                ));
            }
        };
    }
    Ok(seed)
}

fn normal_cdf(value: f64) -> f64 {
    0.5 * libm::erfc(-value * INV_SQRT_2)
}

fn normal_pdf(value: f64) -> f64 {
    INV_SQRT_2PI * libm::exp(-0.5 * value * value)
}

fn simpson(left: f64, right: f64, f_left: f64, f_middle: f64, f_right: f64) -> f64 {
    (right - left) * (f_left + 4.0 * f_middle + f_right) / 6.0
}

fn adaptive_simpson<F>(
    function: &F,
    [left, right]: [f64; 2],
    [f_left, f_middle, f_right]: [f64; 3],
    whole: f64,
    tolerance: f64,
    depth: u32,
) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let middle = (left + right) * 0.5;
    let left_middle = (left + middle) * 0.5;
    let right_middle = (middle + right) * 0.5;
    let f_left_middle = function(left_middle);
    let f_right_middle = function(right_middle);
    if !f_left_middle.is_finite() || !f_right_middle.is_finite() {
        return Err(Error::Numerical("non-finite Owen T integrand".into()));
    }
    let first = simpson(left, middle, f_left, f_left_middle, f_middle);
    let second = simpson(middle, right, f_middle, f_right_middle, f_right);
    let difference = first + second - whole;
    if difference.abs() <= 15.0 * tolerance {
        return Ok(first + second + difference / 15.0);
    }
    if depth == 0 {
        return Err(Error::Numerical(
            "Owen T integration did not converge".into(),
        ));
    }
    Ok(adaptive_simpson(
        function,
        [left, middle],
        [f_left, f_left_middle, f_middle],
        first,
        tolerance * 0.5,
        depth - 1,
    )? + adaptive_simpson(
        function,
        [middle, right],
        [f_middle, f_right_middle, f_right],
        second,
        tolerance * 0.5,
        depth - 1,
    )?)
}

/// Owen's T, evaluated after the `t = tan(theta)` substitution so finite
/// skewness has a bounded integration interval even at extreme magnitudes.
fn owens_t(value: f64, skewness: f64) -> Result<f64> {
    let end = libm::atan(skewness);
    let integrand = |theta: f64| {
        let cosine = libm::cos(theta);
        libm::exp(-0.5 * value * value / (cosine * cosine))
    };
    let middle = end * 0.5;
    let f_left = integrand(0.0);
    let f_middle = integrand(middle);
    let f_right = integrand(end);
    if !f_left.is_finite() || !f_middle.is_finite() || !f_right.is_finite() {
        return Err(Error::Numerical("non-finite Owen T integrand".into()));
    }
    let integral = adaptive_simpson(
        &integrand,
        [0.0, end],
        [f_left, f_middle, f_right],
        simpson(0.0, end, f_left, f_middle, f_right),
        ADAPTIVE_TOLERANCE,
        ADAPTIVE_DEPTH,
    )?;
    Ok(integral / (2.0 * core::f64::consts::PI))
}

fn skew_normal_cdf(value: f64, skewness: f64) -> Result<f64> {
    let cdf = normal_cdf(value) - 2.0 * owens_t(value, skewness)?;
    if !cdf.is_finite() || !(-1.0e-12..=1.0 + 1.0e-12).contains(&cdf) {
        return Err(Error::Numerical("skew-normal CDF is outside [0, 1]".into()));
    }
    Ok(cdf.clamp(0.0, 1.0))
}

fn skew_normal_ppf(probability: f64, skewness: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&probability) {
        return Err(Error::InvalidInput(
            "skew-normal probability must be in (0, 1)".into(),
        ));
    }
    let mut lower = -1.0;
    while skew_normal_cdf(lower, skewness)? > probability {
        lower *= 2.0;
        if !lower.is_finite() {
            return Err(Error::Numerical(
                "skew-normal quantile could not be bracketed".into(),
            ));
        }
    }
    let mut upper = 1.0;
    while skew_normal_cdf(upper, skewness)? < probability {
        upper *= 2.0;
        if !upper.is_finite() {
            return Err(Error::Numerical(
                "skew-normal quantile could not be bracketed".into(),
            ));
        }
    }
    for _ in 0..80 {
        let middle = (lower + upper) * 0.5;
        if skew_normal_cdf(middle, skewness)? < probability {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Ok((lower + upper) * 0.5)
}

fn truncation(value: f64, field: &'static str) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::Numerical(format!(
            "cannot convert non-finite {field} to a reference index"
        )));
    }
    Ok(value.trunc())
}

fn grid_length(last_value: f64) -> Result<usize> {
    let value = truncation(last_value * 4.0, "last selection cost")?;
    if value < 0.0 {
        return Err(Error::InvalidInput(
            "reference linspace length must be non-negative".into(),
        ));
    }
    if value > usize::MAX as f64 {
        return Err(Error::Numerical(
            "reference linspace length exceeds native address space".into(),
        ));
    }
    Ok(value as usize)
}

fn grid_index(value: f64, length: usize) -> Result<usize> {
    let integer = truncation(value * 4.0, "selection cost")?;
    if integer < (isize::MIN + 1) as f64 || integer > isize::MAX as f64 {
        return Err(Error::Index);
    }
    let index = (integer as isize).checked_sub(1).ok_or(Error::Index)?;
    let index = if index < 0 {
        length.checked_add_signed(index).ok_or(Error::Index)?
    } else {
        index as usize
    };
    if index >= length {
        return Err(Error::Index);
    }
    Ok(index)
}

fn linspace_at(lower: f64, upper: f64, length: usize, index: usize) -> Result<f64> {
    if index >= length {
        return Err(Error::Index);
    }
    if length == 1 {
        return Ok(lower);
    }
    if index + 1 == length {
        return Ok(upper);
    }
    Ok(lower + (upper - lower) * index as f64 / (length - 1) as f64)
}

/// Preserve probability mass across large enumerations without an extra buffer.
#[derive(Default)]
struct CompensatedSum {
    value: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.value + corrected;
        self.correction = (next - self.value) - corrected;
        self.value = next;
    }
}

/// Construct the reference-compatible probabilities for ordered mask costs.
pub fn probabilities(values: &[f64], skewness: f64) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(Error::Index);
    }
    if !skewness.is_finite() {
        return Err(Error::Numerical("skewness must be finite".into()));
    }
    // The final enumerated value is deliberately not replaced by a maximum.
    let length = grid_length(values[values.len() - 1])?;
    let lower = skew_normal_ppf(0.01, skewness)?;
    let upper = skew_normal_ppf(0.99, skewness)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(values.len())
        .map_err(|_| Error::Memory)?;
    let mut total = CompensatedSum::default();
    for &value in values {
        let index = grid_index(value, length)?;
        let sample = linspace_at(lower, upper, length, index)?;
        let weight = 2.0 * normal_pdf(sample) * normal_cdf(skewness * sample);
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::Numerical(
                "skew-normal density produced an invalid weight".into(),
            ));
        }
        weights.push(weight);
        total.add(weight);
    }
    let total = total.value;
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Numerical(
            "skew-normal weights cannot be normalized".into(),
        ));
    }
    for weight in &mut weights {
        *weight /= total;
        if !weight.is_finite() || *weight < 0.0 {
            return Err(Error::Numerical(
                "skew-normal probability is invalid".into(),
            ));
        }
    }
    Ok(weights)
}

/// Construct probabilities while charging their live storage and per-value work.
pub fn probabilities_with_budget(
    values: &[f64],
    skewness: f64,
    budget: &mut Budget<'_>,
) -> Result<Vec<f64>> {
    let allocation = values
        .len()
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(Error::Limit("allocation"))?;
    budget.checkpoint(values.len(), 0, allocation)?;
    probabilities(values, skewness)
}

/// Draw a categorical index. Invalid input is rejected before the stream moves.
pub fn draw(probabilities: &[f64], rng: &mut NativeRng) -> Result<usize> {
    if probabilities.is_empty() {
        return Err(Error::Index);
    }
    let mut total = CompensatedSum::default();
    for &probability in probabilities {
        if !probability.is_finite() || probability < 0.0 {
            return Err(Error::InvalidInput(
                "categorical probabilities must be finite and non-negative".into(),
            ));
        }
        total.add(probability);
    }
    let total = total.value;
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InvalidInput(
            "categorical probabilities must have a positive finite sum".into(),
        ));
    }
    let target = rng.uniform() * total;
    let mut cumulative = CompensatedSum::default();
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative.add(*probability);
        if target < cumulative.value {
            return Ok(index);
        }
    }
    // Floating-point summation can leave a tiny tail after an otherwise valid
    // normalized distribution. The reference categorical path chooses the last
    // category rather than introducing another random draw.
    Ok(probabilities.len() - 1)
}

fn emit_stage(
    stage: &mut Option<&mut dyn FnMut(&'static str, bool)>,
    name: &'static str,
    starting: bool,
) {
    if let Some(callback) = stage.as_deref_mut() {
        callback(name, starting);
    }
}

fn sample_inner(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
    mut stage: Option<&mut dyn FnMut(&'static str, bool)>,
) -> Result<(String, f64)> {
    emit_stage(&mut stage, "enumeration", true);
    let enumerated = selection::combinations(operations).map_err(|error| match error {
        selection::Failure::Index => Error::Index,
        selection::Failure::Memory => Error::Memory,
    });
    emit_stage(&mut stage, "enumeration", false);
    let (masks, values) = enumerated?;

    emit_stage(&mut stage, "weights", true);
    let weighted = probabilities(&values, skewness);
    emit_stage(&mut stage, "weights", false);
    let probabilities = weighted?;

    emit_stage(&mut stage, "draw", true);
    let drawn = draw(&probabilities, rng);
    emit_stage(&mut stage, "draw", false);
    let selected = drawn?;
    Ok((masks[selected].clone(), values[selected]))
}

/// Enumerate, weight, and draw while keeping masks and weights in Rust.
///
/// This default path has neither clocks nor profiling callbacks.
pub fn sample(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
) -> Result<(String, f64)> {
    sample_inner(operations, skewness, rng, None)
}

/// Opt-in sampler phase boundaries for host-owned diagnostics.
///
/// `starting` is true immediately before a phase and false immediately after it,
/// including when that phase returns an error. The core neither reads a clock nor
/// stores profiling state.
pub fn sample_profiled(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
    stage: &mut dyn FnMut(&'static str, bool),
) -> Result<(String, f64)> {
    sample_inner(operations, skewness, rng, Some(stage))
}

fn sample_with_budget_inner(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
    budget: &mut Budget<'_>,
    mut stage: Option<&mut dyn FnMut(&'static str, bool)>,
) -> Result<(String, f64)> {
    emit_stage(&mut stage, "enumeration", true);
    let enumerated = selection::combinations_with_budget(operations, budget);
    emit_stage(&mut stage, "enumeration", false);
    let (masks, values) = enumerated?;

    emit_stage(&mut stage, "weights", true);
    let weighted = probabilities_with_budget(&values, skewness, budget);
    emit_stage(&mut stage, "weights", false);
    let probabilities = weighted?;

    emit_stage(&mut stage, "draw", true);
    let drawn = draw(&probabilities, rng);
    emit_stage(&mut stage, "draw", false);
    let selected = drawn?;
    Ok((masks[selected].clone(), values[selected]))
}

/// Budget-aware native selection for callers that already carry core limits.
pub fn sample_with_budget(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
    budget: &mut Budget<'_>,
) -> Result<(String, f64)> {
    sample_with_budget_inner(operations, skewness, rng, budget, None)
}

/// Budget-aware selection with host-owned phase diagnostics and no core clocks.
pub fn sample_with_budget_profiled(
    operations: &[Operation],
    skewness: f64,
    rng: &mut NativeRng,
    budget: &mut Budget<'_>,
    stage: &mut dyn FnMut(&'static str, bool),
) -> Result<(String, f64)> {
    sample_with_budget_inner(operations, skewness, rng, budget, Some(stage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_restoration_replays_the_named_stream() {
        let mut original = NativeRng::from_seed([0; 32]);
        let state = original.state();
        let first = original.next_u64();
        let second = original.next_u64();
        let mut restored = NativeRng::from_state(state).unwrap();
        assert_eq!([first, second], [restored.next_u64(), restored.next_u64()]);
    }

    #[test]
    fn uniform_uses_the_high_53_bits() {
        let mut words = NativeRng::from_seed([7; 32]);
        let word = words.next_u64();
        let mut uniform = NativeRng::from_seed([7; 32]);
        assert_eq!(uniform.uniform(), ((word >> 11) as f64) * INV_TWO_POW_53);
    }

    #[test]
    fn equal_costs_keep_the_discrete_grid_law() {
        let probabilities = probabilities(&[0.25, 0.25, 0.25, 0.25], 0.0).unwrap();
        assert_eq!(probabilities, vec![0.25; 4]);
    }

    #[test]
    fn long_repeated_cost_distribution_retains_unit_mass() {
        let count = 1_000_000;
        let probabilities = probabilities(&vec![0.25; count], 0.0).unwrap();
        // Equal costs have exactly uniform mass. Multiplication avoids repeating
        // the accumulation error under test in the assertion itself.
        assert!(
            probabilities
                .iter()
                .all(|probability| (probability * count as f64 - 1.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn final_value_and_fractional_indices_keep_reference_failures() {
        assert!(matches!(
            probabilities(&[0.5, 0.25], 0.0),
            Err(Error::Index)
        ));
        assert!(matches!(probabilities(&[0.125], 0.0), Err(Error::Index)));
    }

    #[test]
    fn profiled_sampling_emits_complete_phase_boundaries() {
        let operations = [Operation {
            id: 0,
            value: 0.25,
            enablers: vec![],
            disablers: vec![],
        }];
        let mut rng = NativeRng::from_seed([3; 32]);
        let mut stages = Vec::new();
        sample_profiled(&operations, 0.0, &mut rng, &mut |name, starting| {
            stages.push((name, starting));
        })
        .unwrap();
        assert_eq!(
            stages,
            vec![
                ("enumeration", true),
                ("enumeration", false),
                ("weights", true),
                ("weights", false),
                ("draw", true),
                ("draw", false),
            ]
        );
    }

    #[test]
    fn budgeted_profile_reports_the_failing_enumeration_boundary() {
        let operations = [Operation {
            id: 0,
            value: 0.25,
            enablers: vec![],
            disablers: vec![],
        }];
        let mut check = || -> crate::error::Result<()> { Ok(()) };
        let mut budget = Budget {
            limits: crate::architecture::Limits {
                max_output: Some(0),
                ..Default::default()
            },
            work: 0,
            output: 0,
            allocation_bytes: 0,
            check: &mut check,
            wait_check: None,
        };
        let mut rng = NativeRng::from_seed([3; 32]);
        let mut stages = Vec::new();
        let result = sample_with_budget_profiled(
            &operations,
            0.0,
            &mut rng,
            &mut budget,
            &mut |name, starting| stages.push((name, starting)),
        );
        assert!(matches!(result, Err(Error::Limit("output"))));
        assert_eq!(stages, vec![("enumeration", true), ("enumeration", false)]);
    }

    #[test]
    fn finite_extreme_skewness_is_a_numerical_case() {
        for skewness in [-1.0e300, 1.0e300] {
            let probabilities = probabilities(&[0.25, 0.5, 0.75], skewness).unwrap();
            assert!(
                probabilities
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            );
            assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        }
    }
}
