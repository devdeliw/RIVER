//! Newton (Divided-Difference) Interpolation
//!
//! Implements global polynomial interpolation using the
//! [divided-difference method](https://en.wikipedia.org/wiki/Newton_polynomial).
//!
//! Coefficients are computed recursively by divided differences and
//! evaluated at query points using Horner’s scheme for numerical stability.


use crate::interpolation::algorithms::Algorithm;
use crate::interpolation::config::{impl_common_cfg, CommonCfg};
use crate::interpolation::errors::InterpolationError;
use crate::interpolation::report::InterpolationReport;


/// Newton interpolation configuration 
/// 
/// # Fields 
/// - `common` : [`CommonCfg`] 
///
/// # Construction 
/// - Use [`NewtonCfg::new`] then optional setters. 
///
/// # Defaults 
/// - Minimum allowed `x` spacing between consecutive requested eval points;
///   [`crate::interpolation::config::DEFAULT_X_TOL`] by default. 
#[derive(Debug, Clone, Copy)] 
pub struct NewtonCfg<'a> { 
    common: CommonCfg<'a>, 
}
impl<'a> NewtonCfg<'a> {
    pub fn new() -> Self {
        Self { common: CommonCfg::new() }
    }
}
impl_common_cfg!(NewtonCfg<'a>);


/// Computes Newton divided-difference coefficients.
///
/// Returns a coefficient vector `c` s.t. 
/// `P(x) = c[0] + c[1](x - x0) + ... + c[n-1](x - x0)...(x - x_{n-2})`.
#[inline]
fn divided_differences(x: &[f64], y: &[f64]) -> Vec<f64> { 
    let n = x.len(); 

    // doing this hurts, but it's the best way 
    let mut c = y.to_vec(); 

    for j in 1..n { 
        for i in (j..n).rev() { 
            c[i] = (c[i] - c[i - 1]) / (x[i] - x[i - j]); 
        }
    }

    c
}


/// Performs Newton divided-difference interpolation.
///
/// # Behavior
/// - Constructs the divided-difference table to obtain coefficients `c[i]`.
/// - For each evaluation point `xq` in `cfg.common.x_eval()`,
///   evaluates the polynomial using Horner’s nested form:
///
/// ```text
/// P(xq) = c[0] + (xq - x[0]) * [ c[1] + (xq - x[1]) * [ ... c[n-1] ... ] ]
/// ```
///
/// # Returns
/// [`InterpolationReport`] containing
/// - `algorithm_name` : `"newton"`
/// - `n_provided`     : number of (x, y) data points
/// - `n_evaluated`    : number of evaluation points
/// - `evaluated`      : interpolated values at each evaluation point
///
/// # Errors
/// - [`InterpolationError::OutOfBounds`] if any evaluation point lies
///   outside the provided x-range.
pub fn interpolate(cfg: NewtonCfg) -> Result<InterpolationReport, InterpolationError> { 
    let x     = cfg.common.x(); 
    let y     = cfg.common.y(); 
    let evals = cfg.common.x_eval(); 

    let n_provided  = x.len(); 
    let n_evaluated = evals.len(); 

    let mut report = InterpolationReport::new( 
        Algorithm::Newton, 
        n_provided, 
        n_evaluated, 
    ); 
    report.evaluated.reserve(n_evaluated); 

    let coeffs = divided_differences(x, y);

    let x_min = x[0]; 
    let x_max = x[n_provided - 1]; 

    // evaluate polynomials for query points 
    for &xq in evals { 
        if xq < x_min || xq > x_max { 
            return Err(InterpolationError::OutOfBounds { 
                got: xq, 
                x_min,
                x_max 
            }); 
        }

        let mut p = coeffs[n_provided - 1]; 
        for j in (0..n_provided - 1).rev() { 
            p = coeffs[j] + (xq - x[j]) * p; 
        }

        report.evaluated.push(p); 
    }

    Ok(report)
}
