
// Declare modules

pub mod array;

#[cfg(feature = "python")]
mod python;

// Re-export commonly used items for convenience
pub use array::NDArray;