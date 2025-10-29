//! Defines the interpolation algorithm variants 
//!
//! Provides the [`Algorithm`] enum, which enumerates all supported methods. 

/// Interpolation algorithm variants.
/// - [`Algorithm::Linear`]      linear interpolation 
#[derive(Debug, Copy, Clone)]
pub enum Algorithm {
    Linear,
    Newton, 
}

impl Algorithm {
    pub fn algorithm_name(self) -> &'static str {
        match self {
            Algorithm::Linear => "linear",
            Algorithm::Newton => "newton",
        }
    }
}
