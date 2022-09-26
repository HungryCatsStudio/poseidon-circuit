//! mpt demo circuits
//

#![allow(dead_code)]
#![allow(unused_macros)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod hash;
pub mod poseidon;
pub use hash::Hashable;
pub use halo2_proofs::halo2curves::bn256::Fr as Bn256Fr;
