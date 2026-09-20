#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};

mod alignment;
pub use alignment::{Distance, Path, Step, align, distance};

mod metadata;
pub use metadata::{
    Architecture, CancellationToken, Error, Limits, Op, Result, Stats, check_limit,
};

mod sampler;
pub use sampler::{Sampler, SamplingStats};

pub const PROFILE_VERSION: &str = "torch-grammar-v1";
pub const RNG_VERSION: &str = "rcswx-rng-v1";

/// Domain-separated request seed. Does not access an ambient generator.
pub fn derive_seed(seed: u64, domain: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rcswx-rng-v1\0");
    hash.update(seed.to_le_bytes());
    hash.update(domain);
    hash.finalize().into()
}
