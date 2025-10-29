//! Linear Interpolation 
//!
//! Implements piecewise-[linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation). 
//!
//! Each consecutive pair `(x[i], y[i])`, `(x[i+1], y[i+1])` defines 
//! a line segment. Evaluation points lying within `[x[i], x[i+1]]` 
//! are interpolated linearly between the two end points. 


use std::cmp::Ordering;
use crate::interpolation::algorithms::Algorithm; 
use crate::interpolation::config::{impl_common_cfg, CommonCfg}; 
use crate::interpolation::errors::InterpolationError; 
use crate::interpolation::report::InterpolationReport; 


/// Linear interpolation configuration 
/// 
/// # Fields 
/// - `common` : [`CommonCfg`]
///
/// # Construction 
/// - Use [`LinearCfg::new`] then optional setters. 
///
/// # Defaults 
/// - Minimum allowed `x` spacing between consecutive requested eval points;
///   [`crate::interpolation::config::DEFAULT_X_TOL`] by default. 
#[derive(Debug, Clone, Copy)] 
pub struct LinearCfg<'a> { 
    common: CommonCfg<'a>, 
}
impl<'a> LinearCfg<'a> {
    pub fn new() -> Self {
        Self { common: CommonCfg::new() }
    }
}
impl_common_cfg!(LinearCfg<'a>);


#[inline] 
fn lerp(x0: f64, x1: f64, y0: f64, y1: f64, xq: f64) -> f64 { 
    y0 + (y1 - y0) * (xq - x0) / (x1 - x0) 
}

/// Performs linear interpolation over the data in [`CommonCfg`].
///
/// # Behavior
/// For each evaluation point `xq` in `cfg.common.x_eval()`:
/// - If `xq` lies outside `[x[0], x[-1]]`, returns
///   [`InterpolationError::OutOfBounds { got: xq, x_min: x[0], x_max: x[-1] }`].
/// - Otherwise finds the enclosing interval `[x[i], x[i+1]]`
///   and computes
///
/// ```text
/// yq = y[i] + (y[i+1] - y[i]) * (xq - x[i]) / (x[i+1] - x[i])
/// ```
///
/// # Returns
/// [`InterpolationReport`] containing
/// - `algorithm_name` : `"linear"`
/// - `n_provided`     : number of (x, y) data points
/// - `n_evaluated`    : number of evaluation points
/// - `evaluated`      : interpolated y-values
///
/// # Errors
/// - [`InterpolationError::OutOfBounds`] if any evaluation point lies
///   outside the provided x-range.
pub fn interpolate(cfg: LinearCfg) -> Result<InterpolationReport, InterpolationError> { 
    let x     = cfg.common.x(); 
    let y     = cfg.common.y();
    let evals = cfg.common.x_eval(); 

    let n_provided  = x.len(); 
    let n_evaluated = evals.len(); 

    let mut report = InterpolationReport::new( 
        Algorithm::Linear, 
        n_provided, 
        n_evaluated,
    ); 
    report.evaluated.reserve(n_evaluated);

    let x_min = x[0]; 
    let x_max = x[n_provided - 1];  
    for &xq in evals { 
        // domain check 
        if xq < x_min || xq > x_max { 
            return Err(InterpolationError::OutOfBounds { 
                got: xq, 
                x_min,
                x_max 
            }); 
        }

        // binary search for interval 
        match x.binary_search_by(
            |xi| { 
                if xi < &xq      { Ordering::Less    } 
                else if xi > &xq { Ordering::Greater } 
                else             { Ordering::Equal   } 
            }
        ) { 
            Ok(idx)  => { report.evaluated.push(y[idx]); }
            Err(idx) => { 
                // x[idx - 1] < xq < x[idx] 
                let i = idx - 1;

                let (x0, x1) = (x[i], x[i + 1]); 
                let (y0, y1) = (y[i], y[i + 1]); 
                
                let yq = lerp(x0, x1, y0, y1, xq);

                report.evaluated.push(yq); 
            }
        }
    }

    Ok(report) 
}
