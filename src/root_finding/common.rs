use thiserror::Error;


/// Hard cap on automatically computed iteration counts for two-bracket methods.
///
/// This limit is only applied when `max_iter` is `None`/not specified
/// to prevent pathological cases (e.g., extremely small tolerances or huge 
/// initial brackets) from causing unreasonably long runtimes.
///
/// This is a safety limit, not a mathematical bound. Theoretical bounds may be
/// much larger, but in practical floating-point ops, going beyond this
/// rarely improves accuracy.
pub(crate) const DEFAULT_MAX_ITER_LIMIT: usize = 500;


/// Summarizes the outcome of a root-finding run.
///
/// Final report returned by all root-finding algorithms. 
/// It collects the computed root, residual, iteration count, 
/// function evaluation count, final bracket bounds, termination condition, 
/// tolerance reason, and algorithm name into a single structured result.
///
/// # Fields 
/// ├ `root`        -  The computed root approximation.
/// ├ `f_root`      -  The value of `f(root)` at the computed root.
/// ├ `iterations`  -  Total number of iterations performed.
/// ├ `evals`       -  Total number of function evaluations.
/// ├ `left`        -  Final left bound of the bracketing interval.
/// ├ `right`       -  Final right bound of the bracketing interval.
/// ├ `termination` - [`Termination`] variant indicating why the algorithm stopped.
/// ├ `tolerance`   - [`ToleranceReason`] indicating which tolerance criterion was met.
/// └ `algorithm`   -  Name of the algorithm (e.g., `"Bisection"`, `"Regula Falsi"`).
#[derive(Debug, Copy, Clone)]
pub struct RootReport { 
    pub root:        f64,
    pub f_root:      f64,
    pub iterations:  usize,
    pub evals:       usize,
    pub left:        f64,
    pub right:       f64,
    pub termination: Termination,
    pub tolerance:   ToleranceReason,
    pub algorithm:   &'static str, 
}


/// Common error type for root-finding algorithms.
///
/// Encapsulates invalid input parameters, tolerance issues, and
/// evaluation failures that can occur during execution.
#[derive(Debug, Error)]
pub enum RootFindingError {
    #[error("function non-finite at x={x}, f(x)={fx}")]
    NonFiniteEvaluation { x: f64, fx: f64 },

    #[error("invalid `abs_fx` tolerance: must be finite and > 0. got {got}")]
    InvalidAbsFx { got: f64 },

    #[error("invalid `abs_x` tolerance: must be finite and >= 0. got {got}")]
    InvalidAbsX  { got: f64 },

    #[error("invalid `rel_x` tolerance: must be finite and >= 0. got {got}")]
    InvalidRelX  { got: f64 },
    
    #[error("invalid calculated width tolerance (abs_x + rel_x*scale) must be finite and > 0. got {got}")]
    InvalidTolerance { got: f64 },

    #[error("invalid max_iter: must be >= 1. got max_iter={got}")]
    InvalidMaxIter   { got: usize },
}


/// Termination variants for root-finding algorithms after completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    ToleranceReached,
    IterationLimit,
    Stagnation  
}


/// Tolerance variants for root-finding algorithms after completion.  
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceReason { 
    AbsFxReached, 
    WidthTolReached, 
    ToleranceNotReached 
} 


/// Compute the bisection-based iteration upper bound given a target width tolerance.  
///
/// This gets adjusted in specific two-bracket implementations if there are edge cases 
/// where more iterations should be required by default. (e.g. 2x for regula falsi to 
/// account for stagnant endpoints)
///
/// # Arguments 
/// ├ a - left endpoint 
/// ├ b - right endpoint 
/// └ `width_tol`, the convergence tolerance on x-values

/// Returns:
/// ├ `Ok(theoretical_iters)` - theoretical # bisections to satisfy given width tol 
/// └ `Err(RootFindingError::InvalidTolerance)` - width_tol <= 0 or non-finite. 
///
/// # Visibility
/// This function is crate-visible (`pub(crate)`) and is intended to be used
/// by all two-bracket algorithms to produce a relatively consistent default 
/// number of iterations to use. 
pub(crate) fn bisection_theoretical_iter( 
    a: f64, 
    b: f64, 
    width_tol: f64
) -> Result<usize, RootFindingError> { 
    if !(width_tol.is_finite() && width_tol > 0.0) {
        return Err(RootFindingError::InvalidTolerance { got: width_tol });
    }
    let w0 = b - a; 
    let theoretical_iters = if w0 <= width_tol { 0 } else {
        (w0 / width_tol).log2().ceil() as usize
    };

    Ok(theoretical_iters)
}


/// Common method for configuration structs of two-point bracketing methods.
///
/// Provides default implementation of [`cfg.validate`] that 
/// ├ checks tolerances
/// ├ fills defaults  
/// └ ensures `max_iter` sanity 
pub(crate) fn validate_tolerances(
    abs_fx: Option<f64>,
    abs_x: Option<f64>,
    rel_x: Option<f64>,
    max_iter: Option<usize>,
    default_abs_fx: f64,
    default_abs_x: f64,
    default_rel_x: f64,
) -> Result<(f64, f64, f64, Option<usize>), RootFindingError> {
    let abs_fx_val = abs_fx.unwrap_or(default_abs_fx);
    if !(abs_fx_val.is_finite() && abs_fx_val > 0.0) {
        return Err(RootFindingError::InvalidAbsFx { got: abs_fx_val });
    }

    let abs_x_val = abs_x.unwrap_or(default_abs_x);
    if !(abs_x_val.is_finite() && abs_x_val >= 0.0) {
        return Err(RootFindingError::InvalidAbsX { got: abs_x_val });
    }

    let rel_x_val = rel_x.unwrap_or(default_rel_x);
    if !(rel_x_val.is_finite() && rel_x_val >= 0.0) {
        return Err(RootFindingError::InvalidRelX { got: rel_x_val });
    }

    if rel_x_val.max(abs_x_val) == 0.0 {
        return Err(RootFindingError::InvalidTolerance { got: 0.0 });
    }

    let max_iter_val = match max_iter {
        None => None,
        Some(0) => return Err(RootFindingError::InvalidMaxIter { got: 0 }),
        Some(n) => Some(n),
    };

    Ok((abs_fx_val, abs_x_val, rel_x_val, max_iter_val))
} 


/// Combined absolute + relative width tolerance for the current bracket [a, b].
///
/// Formula: `abs_x + rel_x * max(max(|a|, |b|), 1.0)`
///
/// Ensures the relative scale is never below 1.0 to avoid tiny tolerances
/// near zero. Used by all bracketing algorithms for consistent checks.
pub(crate) fn width_tol_current(a: f64, b: f64, abs_x: f64, rel_x: f64) -> f64 { 
    abs_x + rel_x * a.abs().max(b.abs()).max(1.0)
}


/// Determines whether `u` and `v` are of opposite sign. 
///
/// Used for two-bracketing algorithms to determine which direction 
/// to shrink the interval/bracket by, or if there is an initial 
/// error if f(a) is same sign as f(b), not guaranteeing a root exists. 
pub(crate) fn opposite_signs(u: f64, v: f64) -> bool {
    (u > 0.0 && v < 0.0) || (u < 0.0 && v > 0.0)
}
