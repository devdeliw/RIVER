use super::algorithms::{Algorithm, OpenFamily, GLOBAL_MAX_ITER_FALLBACK}; 
use super::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil}; 
use super::tolerances::DynamicTolerance; 
use super::errors::{RootFindingError, ToleranceError}; 
use super::config::{CommonCfg, impl_common_cfg}; 
use thiserror::Error; 


#[derive(Debug, Error)] 
pub enum SecantError{ 
    #[error(transparent)] 
    RootFinding(#[from] RootFindingError),

    #[error(transparent)]
    Tolerance(#[from] ToleranceError),

    #[error("invalid initial guesses: x0 and x1 must be finite and distinct")]
    InvalidGuess { x0: f64, x1: f64 }, 

    #[error("degenerate secant: |fx2 - fx1| near 0")]
    DegenerateSecantStep 
}


/// Secant configuration 
///
/// # Fields 
/// - `common` : [`CommonCfg`] with tolerances and optional `max_iter`.
///
/// # Construction 
/// - Use [`SecantCfg::new`] then optional setters. 
///
/// # Defaults 
/// - If `common.max_iter` is `None`, [`secant`] resolves it using 
///   [`Algorithm::default_max_iter`] for [`OpenFamily::Secant`], or 
///   [`GLOBAL_MAX_ITER_FALLBACK`] if unavailable. 
#[derive(Debug, Copy, Clone)]
pub struct SecantCfg { 
    common: CommonCfg, 
}
impl SecantCfg { 
    pub fn new() -> Self { 
        Self { 
            common: CommonCfg::new()
        }
    }
}
impl_common_cfg!(SecantCfg);


/// Calculates the secant x-intercept for the line 
/// connecting `(x1, fx1)` and `(x2, fx2)` 
///
/// # Arguments 
/// - `(x1, fx1)` : secant endpoint 1 and function value
/// - `(x2, fx2)` : secant endpoint 2 and function value 
///
/// # Returns 
/// - `Ok(x_secant)` if denominator `fx2 - fx1` is well-scaled 
/// - `Err(DegenerateSecantStep)` if denominator is too small. 
#[inline]
pub(crate) fn calculate_secant_x_intercept(
    (x1, fx1): (f64, f64), 
    (x2, fx2): (f64, f64), 
)-> Result<f64, SecantError> {
    let denom = fx2 - fx1;
    let scale  = fx1.abs().max(fx2.abs()).max(1.0);
    let thresh = f64::EPSILON * scale + f64::MIN_POSITIVE;

    if denom.abs() <= thresh {
        return Err(SecantError::DegenerateSecantStep );
    } 

    Ok(( x1 * fx2 - x2 * fx1 ) / denom)
}


/// Calculates the secant x-intersection point for the line 
/// connecting `(x1, fx1)` and `(x2, fx2)` and its function eval 
///
/// # Arguments 
/// - `(x1, fx1)` : secant endpoint 1 and function value
/// - `(x2, fx2)` : secant endpoint 2 and function value 
/// - `eval`      : function closure defined in [`secant`]
///
/// # Returns 
/// - `Ok(x_next, f(x_next))`     : if denominator `fx2 - fx1` is well-scaled 
/// - `Err(DegenerateSecantStep)` : if denominator is too small 
///     - *Handled internally. Replaces with a bisection step.*   
#[inline] 
fn next_sol_estimate<F> (
    (x1, fx1): (f64, f64), 
    (x2, fx2): (f64, f64), 
    eval: &mut F 
) -> Result<(f64, f64), SecantError> 
where F: FnMut(f64) -> Result<f64, SecantError> { 
    let x_next = match calculate_secant_x_intercept((x1, fx1), (x2, fx2)) { 
        Ok(x)                                  => x, 
        // default to bisection  
        Err(SecantError::DegenerateSecantStep) => x1 - (x1 - x2) * 0.5,
        Err(e)                                 => return Err(e),
    };
    let f_next  = eval(x_next)?;

    Ok((x_next, f_next))
}


/// Calculates the step tolerance for the [`secant`] algorithm 
/// using the two points (stencil) used in the update formula.  
///
/// # Arguments 
/// - `x1`/`x2`         : x-values used in update formula 
/// - `abs_x`/`rel_x`   : absolute and relative tolerance 
/// - `algorithm`       : [`Algorithm::Open`] with [`OpenFamily::Secant`]
///
/// # Returns 
/// - `Ok(step_tol)` : if tolerance finite and > 0 
/// - `Err(ToleranceError::InvalidTolerance)` if tolerance non-finite or <= 0
#[inline]
fn step_tolerance(
    x1: f64, 
    x2: f64,
    abs_x: f64, 
    rel_x: f64, 
    algorithm: Algorithm,
) -> Result<f64, ToleranceError> {
    let tol1 = algorithm.calculate_tolerance(
        &DynamicTolerance::StepTol { x: [x1, 0.0, 0.0] }, 
        abs_x, 
        rel_x
    )?;
    let tol2 = algorithm.calculate_tolerance(
        &DynamicTolerance::StepTol { x: [x2, 0.0, 0.0] }, 
        abs_x, 
        rel_x
    )?;

    Ok(tol1.max(tol2))
}


/// Finds a root of a function using the 
/// [secant method](https://en.wikipedia.org/wiki/Secant_method).
///
/// # Arguments
/// - `func` : The function whose root is to be found
/// - `x0`   : First initial guess.  Must be finite and not equal to `x1`
/// - `x1`   : Second initial guess. Must be finite and not equal to `x0`
/// - `cfg`  : [`SecantCfg`] (tolerances, optional `max_iter`)
///
/// # Returns
/// [`RootFindingReport`] with 
/// - `root`                : approximate root
/// - `f_root`              : function value at `root`
/// - `iterations`          : number of iterations performed
/// - `evaluations`         : total function evaluations 
/// - `termination_reason`  : why it stopped
/// - `tolerance_satisfied` : which tolerance triggered
/// - `stencil`             : previous iterates used to form the step
/// - `algorithm_name`      : "secant"
///
/// # Errors
/// - [`SecantError::InvalidGuess`]             : `x0` or `x1` is NaN/inf or equal
/// 
/// * Propagated via [`SecantError::RootFinding`]
/// - [`RootFindingError::NonFiniteEvaluation`] : `f(x)` produced NaN/inf
/// - [`RootFindingError::InvalidMaxIter`]      : `max_iter` = 0
/// 
/// * Propagated via [`SecantError::Tolerance`] 
/// - [`ToleranceError::InvalidAbsFx`]          : `abs_fx` <= 0.0 or inf
/// - [`ToleranceError::InvalidAbsX`]           : `abs_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidRelX`]           : `rel_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidAbsRelX`]        : both `abs_x` and `rel_x` not > 0.
///
/// # Behavior
/// - Update:
///     - secant step: 
///       x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
///     - denominator collapse (f(x_k) ~ f(x_{k-1})) triggers a safeguard:
///       fall back to a half-step between x_k and x_{k-1}
/// - Tolerances: 
///     - if |x_{k+1} - x_k| <= tolerance, return with [`ToleranceSatisfied::StepSizeReached`]
///     - if |f(x_k)| <= abs_fx at any stage, [`ToleranceSatisfied::AbsFxReached`]
/// - Stencil:
///     - stores the pair of previous iterates {x_{k-1}, x_k}
///       used to form the last secant step; on immediate success
///       both entries are the initial guesses.
///
/// # Notes
/// - Convergence is superlinear (~1.618) near simple roots but requires two
///   distinct starting guesses.
///
/// # Warning 
/// - Poor initial guesses may lead to divergence or extremely slow convergence.
///   For guaranteed convergence, use a **bracketed method** (e.g. bisection/Brent)
#[must_use]
pub fn secant<F> ( 
    mut func: F, 
    x0: f64, 
    x1: f64,
    cfg: SecantCfg
) -> Result<RootFindingReport, SecantError> 
where F: FnMut(f64) -> f64 { 
    
    if !(x0.is_finite() && x1.is_finite()) || x0 == x1 { 
        return Err(SecantError::InvalidGuess { x0, x1 });
    }

    let abs_x      = cfg.common.abs_x(); 
    let rel_x      = cfg.common.rel_x(); 
    let abs_fx     = cfg.common.abs_fx(); 
    let max_iter   = cfg.common.max_iter(); 
    let algorithm  = Algorithm::Open(OpenFamily::Secant);
    let algo_name  = algorithm.algorithm_name(); 

    let num_iter = match max_iter {
        // already validated via building config; redundant guard
        Some(0) => return Err(RootFindingError::InvalidMaxIter { got: 0 }.into()), 

        Some(v) => v, 
        None    => algorithm.default_max_iter().unwrap_or(GLOBAL_MAX_ITER_FALLBACK),
    };

    // track function evaluations 
    let mut evals = 0; 

    // wraps func, increments evals, enforces finiteness
    let mut eval = |x: f64| -> Result<f64, SecantError> { 
        let fx = { evals += 1; func(x) }; 
        if !fx.is_finite() { 
            return Err(RootFindingError::NonFiniteEvaluation { x, fx }.into()) 
        } 

        Ok(fx) 
    };

    // early exit: x0 is root
    let fx0 = eval(x0)?; 
    if fx0.abs() <= abs_fx { 
        return Ok(RootFindingReport {
            root                : x0, 
            f_root              : fx0, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::singleton(x0),
            algorithm_name      : algo_name
        }); 
    }
    // early exit: x1 is root 
    let fx1 = eval(x1)?; 
    if fx1.abs() <= abs_fx {
        return Ok(RootFindingReport {
            root                : x1, 
            f_root              : fx1, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::singleton(x1),
            algorithm_name      : algo_name
        }); 
    }


    // step tolerance already satisfied 
    let mut step_tol = step_tolerance(x0, x1, abs_x, rel_x, algorithm)?; 
    if (x1 - x0).abs() <= step_tol { 
        return Ok(RootFindingReport {
            root                : x1, 
            f_root              : fx1, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::StepSizeReached,
            stencil             : Stencil::doubleton(x0, x1), 
            algorithm_name      : algo_name
        }); 
    }

    // main loop 
    let mut x_parent1 = x1; 
    let mut x_parent2 = x0; 
    let mut f_parent1 = fx1; 
    let mut f_parent2 = fx0; 
    for iter in 1..=num_iter { 
        let (x_next, f_next) = next_sol_estimate(
            (x_parent1, f_parent1), (x_parent2, f_parent2), 
            &mut eval
        )?;

        // check |f(x)| tolerance 
        if f_next.abs() <= abs_fx { 
            return Ok(RootFindingReport {
                root                : x_next, 
                f_root              : f_next, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
                stencil             : Stencil::doubleton(x_parent1, x_parent2),
                algorithm_name      : algo_name
            }); 
        }

        // check step tolerance  
        step_tol = step_tolerance(x_next, x_parent1, abs_x, rel_x, algorithm)?;
        if (x_next - x_parent1).abs() <= step_tol { 
            return Ok(RootFindingReport {
                root                : x_next, 
                f_root              : f_next, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::StepSizeReached,
                stencil             : Stencil::doubleton(x_parent1, x_parent2),
                algorithm_name      : algo_name
            }); 
        }

        x_parent2 = x_parent1; 
        f_parent2 = f_parent1; 
        x_parent1 = x_next; 
        f_parent1 = f_next; 
    }

    Ok(RootFindingReport {
        root                : x_parent1, 
        f_root              : f_parent1, 
        iterations          : num_iter, 
        evaluations         : evals,
        termination_reason  : TerminationReason::IterationLimit, 
        tolerance_satisfied : ToleranceSatisfied::ToleranceNotReached,
        stencil             : Stencil::doubleton(x_parent1, x_parent2),
        algorithm_name      : algo_name
    }) 
}

