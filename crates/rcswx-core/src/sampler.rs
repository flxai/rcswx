use rand_chacha::ChaCha12Rng;
use rand_core::{RngCore, SeedableRng};
use std::mem::size_of;
use std::sync::Arc;

use crate::metadata::{Guard, allocation};
use crate::{CancellationToken, Error, Limits, Path, Result, check_limit, derive_seed};

#[derive(Clone, Copy, Debug)]
pub struct SamplingStats {
    pub visited_masks: usize,
    pub valid_masks: usize,
    pub charged_bytes: usize,
    pub total_weight: f64,
}
struct Entry {
    mask: u64,
    cumulative: f64,
}
#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) -> f64 {
        let updated = self.value + value;
        self.correction += if self.value.abs() >= value.abs() {
            (self.value - updated) + value
        } else {
            (value - updated) + self.value
        };
        self.value = updated;
        self.value + self.correction
    }
}

/// The CDF includes every structurally valid mask, never only shape-valid masks.
/// No request-global generator, acceptance memo, descriptor deduplication or cache.
pub struct Sampler {
    path: Arc<Path>,
    entries: Vec<Entry>,
    rng: ChaCha12Rng,
    guard: Guard,
    external_bytes: usize,
    pub stats: SamplingStats,
}
impl Sampler {
    pub fn new(
        path: Arc<Path>,
        seed: u64,
        limits: &Limits,
        cancel: &CancellationToken,
        external_bytes: usize,
    ) -> Result<Self> {
        let mut guard = Guard::new(limits, cancel)?;
        check_limit("max_core_bytes", limits.max_core_bytes, external_bytes)?;
        check_limit("max_edits", limits.max_edits, path.edit_count())?;
        check_limit(
            "max_predecessors",
            limits.max_predecessors,
            path.steps().len(),
        )?;
        let masks = 1_usize
            .checked_shl(path.edit_count() as u32)
            .ok_or(Error::BudgetExceeded {
                resource: "max_masks",
                limit: limits.max_masks as u64,
                observed: u64::MAX,
            })?;
        check_limit("max_masks", limits.max_masks, masks)?;
        let base = path.owned_bytes() + size_of::<Self>() + external_bytes;
        let mut entries = allocation::<Entry>(masks, base, &guard)?;
        let mut total = Sum::default();
        let mut cost = 0_u64;
        let mut length = path.source_len() as i64;
        for mask in 0..masks as u64 {
            if mask != 0 {
                // Adjacent unsigned masks differ in two bits on average. Cost and
                // emitted length are integer additive effects, so this is exact.
                let mut changed = (mask - 1) ^ mask;
                while changed != 0 {
                    let bit = changed.trailing_zeros() as usize;
                    changed &= changed - 1;
                    let effect = path.effects[bit];
                    if mask & (1 << bit) != 0 {
                        cost = cost.checked_add(effect.cost).ok_or_else(|| {
                            Error::InternalInvariant("selected cost overflow".into())
                        })?;
                        length += effect.length_delta as i64;
                    } else {
                        cost = cost.checked_sub(effect.cost).ok_or_else(|| {
                            Error::InternalInvariant("selected cost underflow".into())
                        })?;
                        length -= effect.length_delta as i64;
                    }
                }
            }
            if length > 0 {
                let weight = if path.cost_ticks() == 0 {
                    1.0
                } else {
                    let deviation = (cost as f64 / path.cost_ticks() as f64 - 0.5) / 0.2;
                    (-0.5 * (deviation * deviation)).exp()
                };
                entries.push(Entry {
                    mask,
                    cumulative: total.add(weight),
                });
            }
            guard.tick()?;
        }
        guard.check()?;
        let sum = entries
            .last()
            .ok_or_else(|| Error::InternalInvariant("no structurally valid endpoint mask".into()))?
            .cumulative;
        let stats = SamplingStats {
            visited_masks: masks,
            valid_masks: entries.len(),
            charged_bytes: base + entries.capacity() * size_of::<Entry>(),
            total_weight: sum,
        };
        Ok(Self {
            path,
            entries,
            rng: ChaCha12Rng::from_seed(derive_seed(seed, b"selection")),
            guard,
            external_bytes,
            stats,
        })
    }
    pub fn draw(&mut self) -> Result<u64> {
        self.guard.check()?;
        let unit = (self.rng.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        let threshold = unit * self.stats.total_weight;
        let index = self
            .entries
            .partition_point(|entry| entry.cumulative <= threshold);
        self.entries
            .get(index)
            .map(|entry| entry.mask)
            .ok_or_else(|| Error::InternalInvariant("CDF has no strict upper boundary".into()))
    }
    pub fn propose(
        &mut self,
        limits: &Limits,
        cancel: &CancellationToken,
    ) -> Result<(u64, Vec<(u8, usize)>)> {
        let mask = self.draw()?;
        let retained_bytes =
            size_of::<Self>() + self.entries.capacity() * size_of::<Entry>() + self.external_bytes;
        let origins = self
            .path
            .project_with_retained(mask, limits, cancel, retained_bytes)?;
        self.stats.charged_bytes = self.stats.charged_bytes.max(
            self.path.owned_bytes()
                + retained_bytes
                + origins.capacity() * size_of::<(u8, usize)>(),
        );
        Ok((mask, origins))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Architecture, Op, align};

    fn fixture() -> Arc<Path> {
        let identity = Op {
            kind: 0,
            width: 0,
            bias: false,
            param: 0.0,
            momentum: None,
            track: false,
        };
        let source = Arc::new(Architecture::new(vec![identity], &Limits::default()).unwrap());
        let target = Arc::new(
            Architecture::new(
                vec![
                    Op {
                        kind: 1,
                        width: 8,
                        bias: true,
                        ..identity
                    },
                    Op {
                        kind: 1,
                        width: 4,
                        bias: true,
                        ..identity
                    },
                ],
                &Limits::default(),
            )
            .unwrap(),
        );
        Arc::new(
            align(
                source,
                target,
                &Limits::default(),
                &CancellationToken::default(),
            )
            .unwrap(),
        )
    }
    #[test]
    fn chacha12_matches_independently_derived_words() {
        let mut rng = ChaCha12Rng::from_seed(derive_seed(42, b"selection"));
        assert_eq!(
            (0..8).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            vec![
                10266166836587437710,
                1822444519948767209,
                6656081839163481662,
                660504836312650541,
                8306220187888404226,
                2949079349722777010,
                1215706461372434336,
                1252025857116422685
            ]
        );
    }
    #[test]
    fn mixed_tick_cdf_preserves_seeded_invalid_shape_proposals() {
        let mut sampler = Sampler::new(
            fixture(),
            u64::MAX,
            &Limits::default(),
            &CancellationToken::default(),
            0,
        )
        .unwrap();
        assert_eq!(
            (0..8).map(|_| sampler.draw().unwrap()).collect::<Vec<_>>(),
            vec![1, 1, 1, 2, 0, 1, 0, 1]
        );
        assert_eq!(sampler.stats.valid_masks, 4);
        assert_eq!(
            sampler
                .entries
                .iter()
                .map(|entry| entry.cumulative)
                .collect::<Vec<_>>(),
            vec![
                0.04393693362340742,
                0.7505852114811237,
                1.4572334893388401,
                1.5011704229622476
            ]
        );
    }
    #[test]
    fn mask_and_byte_caps_stop_before_enumeration() {
        assert!(matches!(
            Sampler::new(
                fixture(),
                1,
                &Limits {
                    max_masks: 3,
                    ..Limits::default()
                },
                &CancellationToken::default(),
                0
            ),
            Err(Error::BudgetExceeded {
                resource: "max_masks",
                ..
            })
        ));
        assert!(matches!(
            Sampler::new(
                fixture(),
                1,
                &Limits {
                    max_core_bytes: 1,
                    ..Limits::default()
                },
                &CancellationToken::default(),
                0
            ),
            Err(Error::BudgetExceeded {
                resource: "max_core_bytes",
                ..
            })
        ));
    }
}
