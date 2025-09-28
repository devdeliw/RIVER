//! Tolerance utilities for root-finding algorithms.
//!
//! Provides types and helpers for computing stopping tolerances
//! consistently across algorithm categories.
//!
//! `DynamicTolerance` : method-specific per-iteration tolerance  
//! - `WidthTol(a, b)` : bracketing methods  
//! - `StepTol(x)`     : open methods  
//!
//! Each [`Algorithm`] variant enforces that only the correct dynamic
//! tolerance type is used via `calculate_tolerance`.


use crate::root_finding::errors::ToleranceError;  
use crate::root_finding::algorithms::Algorithm;


/// Bracketing methods use DynamicTolerance::WidthTol 
/// Open methods       use DynamicTolerance::StepTol
/// Compound methods   use both 
#[derive(Debug, Copy, Clone)]
pub(crate) enum DynamicTolerance { 
    WidthTol { a: f64, b: f64 }, 
    StepTol  { x: [f64; 3] } 
}
impl DynamicTolerance { 
    pub fn step_two_scalars(x1: f64, x2: f64) -> Self { 
        DynamicTolerance::StepTol { x: [x1, x2, 0.0] }
    }
}


impl Algorithm {
    /// Compute the method-specific dynamic tolerance for an algorithm.
    /// - [`Algorithm::Bracket`] methods ([`DynamicTolerance::WidthTol`]):  
    ///   `abs_x + rel_x * max(|a|, |b|, 1.0)`
    /// - [`Algorithm::Open`] methods ([`DynamicTolerance::StepTol`]): 
    ///   `abs_x + rel_x * max(|x|, 1.0)` 
    /// - [`Algorithm::Compound`] methods can do both 
    ///
    /// # Notes 
    /// - For open methods the effective step tolerance is the *maximum* tolerance
    ///   across all iterates `x` that contribute to the next root estimate, not
    ///   just the most recent one.
    /// - For open methods that only use one iterate to calculate the next estimate 
    ///   (e.g. newton), the effective step tolerance is still the *maximum* tolerance 
    ///   between two consecutive estimates. 
    ///
    /// # Errors 
    /// - Returns a [`ToleranceError`] if the tolerance type does not 
    ///   match the algorithm type (e.g. width tolerance for an open method) 
    ///   or if the result is invalid (non-finite or <= 0).
    pub(crate) fn calculate_tolerance( 
        &self, 
        dynamic_tol : &DynamicTolerance, 
        abs_x   : f64, 
        rel_x   : f64 
    ) -> Result<f64, ToleranceError> {

        let calculated_tol = match (self, dynamic_tol) { 
            (
                Algorithm::Bracket(..) | Algorithm::Hybrid(..), 
                DynamicTolerance::WidthTol { a, b }
            ) 
            => abs_x + rel_x * a.abs().max(b.abs()).max(1.0), 

            (
                Algorithm::Open(..) | Algorithm::Hybrid(..), 
                DynamicTolerance::StepTol { x }
            )
            => { 
                // use max |x| over array 
                let mut max_abs = 0.0; 
                for &xi in x { 
                    let val = xi.abs(); 
                    if val > max_abs { 
                        max_abs = val; 
                    }
                } 

                let tol_val = abs_x + rel_x * max_abs.max(1.0);
                if !tol_val.is_finite() || tol_val <= 0.0 { 
                    return Err(ToleranceError::InvalidTolerance { got: tol_val });
                }   

                tol_val
            },

            (_, DynamicTolerance::WidthTol { .. }) 
            => return Err(ToleranceError::WidthTolNotApplicable { algorithm: *self }), 

            (_, DynamicTolerance::StepTol { .. })
            => return Err(ToleranceError::StepTolNotApplicable { algorithm: *self }),
        };

        if calculated_tol <= 0.0 || !calculated_tol.is_finite() {
            return Err(ToleranceError::InvalidTolerance { got: calculated_tol });
        }

        Ok(calculated_tol)
    }   
}
