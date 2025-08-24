//! Root-finding error types.  
//! 
//! ┌ [`AlgorithmError`]   : algorithm misuse  
//! │  └ incompatible algorithm variant selection
//! │
//! ├ [`RootFindingError`] : common runtime errors  
//! │   ├ non-finite function evaluation  
//! │   └ invalid global parameters (e.g. max_iter) 
//! │
//! └ [`ToleranceError`]   : tolerance-related errors  
//!     ├ invalid input tolerances  
//!     ├ invalid or non-finite computed tolerances  
//!     └ mismatched tolerance type vs. algorithm ([`Algorithm`])  


use thiserror::Error; 
use super::algorithms::Algorithm; 


/// Algorithm selection errors.  
/// 
/// - Raised when an algorithm variant is requested that 
///   is not valid for the given runner.
#[derive(Debug, Error)]
pub enum AlgorithmError { 
    #[error("incompatible algorithm: got {algorithm}")]
    IncompatibleAlgorithm { algorithm: Algorithm }
}


/// Root-finding runtime errors.  
/// 
/// ┌ Non-finite function evaluation  
/// └ Invalid global configuration (e.g. max_iter < 1)
#[derive(Debug, Error)]
pub enum RootFindingError {
    #[error("function non-finite at x={x}, f(x)={fx}")]
    NonFiniteEvaluation { x: f64, fx: f64 },

    #[error("invalid max_iter: must be >= 1. got max_iter={got}")]
    InvalidMaxIter   { got: usize },
}


/// Tolerance configuration and evaluation errors.  
/// 
/// ┌ Invalid input tolerances (`abs_fx`, `abs_x`, `rel_x`)  
/// ├ Computed tolerance invalid (<= 0 or non-finite)  
/// └ Mismatched tolerance type vs. algorithm
#[derive(Debug, Error)]
pub enum ToleranceError { 
    #[error("invalid `abs_fx` tolerance: must be finite and > 0. got {got}")]
    InvalidAbsFx { got: f64 },

    #[error("invalid `abs_x` tolerance: must be finite and >= 0. got {got}")]
    InvalidAbsX  { got: f64 },

    #[error("invalid `rel_x` tolerance: must be finite and >= 0. got {got}")]
    InvalidRelX  { got: f64 },

    #[error("either `abs_x` or `rel_x` must be > 0. got {abs_x} and {rel_x}")]
    InvalidAbsRelX { abs_x: f64, rel_x: f64}, 

    #[error("width tolerance not applicable for algorithm {algorithm:?}")]
    WidthTolNotApplicable { algorithm: Algorithm },

    #[error("step tolerance not applicable for algorithm {algorithm:?}")]
    StepTolNotApplicable { algorithm: Algorithm },

    #[error("invalid computed tolerance: must be finite and > 0. got {got}")]
    InvalidTolerance { got: f64 },
}

