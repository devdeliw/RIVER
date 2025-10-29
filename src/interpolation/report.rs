//! Defines the struct returned by all interpolation algorithms.
//!
//! Defines the [`InterpolationReport`] struct returned by all
//! interpolation algorithms.
//!
//! This report summarizes key metadata about the interpolation process,
//! including the algorithm used, number of data and evaluation points,
//! and results of evaluating the interpolant.

use crate::interpolation::algorithms::Algorithm;

/// Summary of an interpolation run.
///
/// [`InterpolationReport`]
/// - `algorithm_name` : name of the interpolation method (e.g. `"linear"`)
/// - `n_provided`     : number of input data points `(x, y)`
/// - `n_evaluated`    : number of points at which interpolation was performed
/// - `evaluated`      : interpolated values at each evaluation point
#[derive(Debug, Clone)]
pub struct InterpolationReport {
    pub algorithm_name: &'static str,
    pub n_provided: usize,
    pub n_evaluated: usize,
    pub evaluated: Vec<f64>,
}

impl InterpolationReport {
    pub fn new(algorithm: Algorithm, n_provided: usize, n_evaluated: usize) -> Self {
        Self {
            algorithm_name: algorithm.algorithm_name(),
            n_provided,
            n_evaluated,
            evaluated: Vec::new(),
        }
    }
}

