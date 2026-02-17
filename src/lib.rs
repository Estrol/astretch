#![doc = include_str!("../README.md")]

extern crate alloc;

pub(crate) mod stretch;
pub mod dsp;

pub(crate) mod misc;

pub use stretch::Stretch;