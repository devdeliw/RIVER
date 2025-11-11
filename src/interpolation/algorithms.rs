//! Defines the interpolation algorithm variants 
//!
//! Provides the [`Algorithm`] enum, which enumerates all supported methods. 

/// Interpolation algorithm variants.
/// - [`Algorithm::Linear`]      linear interpolation 
#[derive(Debug, Copy, Clone)]
pub enum Algorithm {
    Linear,
    Newton, 
    SplineNatural, 
    SplineClamped,
    SplineMonotonic, 
}

impl Algorithm {
    pub fn algorithm_name(self) -> &'static str {
        match self {
            Algorithm::Linear => "linear",
            Algorithm::Newton => "newton",
            Algorithm::SplineNatural   => "natural cubic spline",
            Algorithm::SplineClamped   => "clamped cubic spline", 
            Algorithm::SplineMonotonic => "monotonic cubic spline"
        }
    }
}
