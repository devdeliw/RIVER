use super::algorithms::{Algorithm, BracketFamily, GLOBAL_MAX_ITER_FALLBACK}; 
use super::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil}; 
use super::tolerances::DynamicTolerance; 
use super::signs::{opposite_sign, same_sign};
use super::errors::{RootFindingError, ToleranceError}; 
use super::config::{CommonCfg, impl_common_cfg};
use thiserror::Error;


#[derive(Debug, Error)]
pub enum BisectionError {
    #[error(transparent)]
    RootFinding(#[from] RootFindingError),

    #[error(transparent)]
    Tolerance(#[from] ToleranceError),

    #[error("no sign change on [{a}, {b}]: sign(f(a)) = sign(f(b))")]
    NoSignChange  { a: f64, b: f64 },

    #[error("invalid bounds: a and b must be finite with a < b. got [{a}, {b}]")] 
    InvalidBounds { a: f64, b: f64 },
}


/// Bisection Configuration 
/// 
/// # Fields 
/// └ `common` : [`CommonCfg`] with tolerances and optional `max_iter`. 
///
/// # Construction 
/// └ Use [`BisectionCfg::new`] then optional setters from [`impl_common_cfg`]. 
///
/// # Defaults 
/// └ If `common.max_iter` is `None`, [`bisection`] resolves it using 
///   the theoretical number of iterations for guaranteed convergence.
#[derive(Debug, Copy, Clone)]
pub struct BisectionCfg { 
    common: CommonCfg, 
}
impl BisectionCfg { 
    pub fn new() -> Self { 
        Self { 
            common: CommonCfg::new()
        }
    }
}
impl_common_cfg!(BisectionCfg);


/// Compute the max iteration count for guaranteed width tolerance convergence.
/// This is used as a default if `max_iter` is not provided. 
///
/// # Arguments 
/// ├ `a`       ` : left endpoint 
/// ├ `b`         : right endpoint 
/// └ `width_tol` : the convergence tolerance on x-values
///
/// # Returns:
/// ├ `Ok(theoretical_iters)` : # bisections to guarantee `width_tol` convergence.
/// └ `Err(ToleranceError::InvalidTolerance)` : if width_tol <= 0 or inf. 
#[inline] 
pub(crate) fn theoretical_iter( 
    a: f64, 
    b: f64, 
    width_tol: f64
) -> Result<usize, ToleranceError> { 
    if !(width_tol.is_finite() && width_tol > 0.0) {
        return Err(ToleranceError::InvalidTolerance { got: width_tol });
    }
    let w0 = b - a; 
    let theoretical_iters = if w0 <= width_tol { 0 } else {
        (w0 / width_tol).log2().ceil() as usize
    };

    Ok(theoretical_iters)
}


#[inline] 
pub(crate) fn midpoint(a: f64, b: f64) -> f64 { 
    a + (b - a) * 0.5
}


/// Calculates the function evaluation for the midpoint of [a, b].
///
/// # Arguments 
/// ├ `a`    : left endpoint 
/// ├ `b`    : right endpoint 
/// └ `eval` : function that also updates `evals` count; made in [`bisection`] 
/// 
/// # Returns 
/// ├ `Ok((midpoint, f(midpoint)))` if evaluation is finite. 
/// └ `Err(BisectionError::RootFinding(RootFindingError::NonFiniteEvaluation))`
///   if the function evaluation is non-finite. 
#[inline] 
fn next_sol_estimate<F>(
    a: f64, b: f64, eval: &mut F 
) -> Result<(f64, f64), BisectionError> 
where F: FnMut(f64) -> Result<f64, BisectionError> { 
    let m  = midpoint(a, b);
    let fm = eval(m)?;

    Ok((m, fm))
}


/// Finds a root of a function using the 
/// [bisection method](https://en.wikipedia.org/wiki/Bisection_method).
///
/// This method assumes that the function is continuous on the interval `[a, b]`
/// and that `f(a)` and `f(b)` have opposite signs, guaranteeing a root exists
/// within the interval.
///
/// # Arguments
/// ├ `func` : The function whose root is to be found.
/// ├ `a`    : Lower bound of the search interval. Must be finite and less than `b`.
/// ├ `b`    : Upper bound of the search interval. Must be finite and greater than `a`.
/// └ `cfg`  : [`BisectionCfg`] (tolerances, optional `max_iter`).  
///
/// # Returns
/// [`RootFindingReport`] with 
/// ├ `root`                : approximate root
/// ├ `f_root`              : function value at `root`
/// ├ `iterations`          : number of iterations performed
/// ├ `evaluations`         : total function evaluations 
/// ├ `termination_reason`  : why it stopped
/// ├ `tolerance_satisfied` : which tolerance triggered
/// ├ `stencil`             : previous bracket used to form the step
/// └ `algorithm_name`      : "bisection"
///
/// # Errors
/// ├ [`BisectionError::InvalidBounds`]         : `a` or `b` is NaN/inf or if `a >= b`.
/// ├ [`BisectionError::NoSignChange`]          : `func(a)` and `func(b)` do not have opposite signs.
/// │
/// * Propagated via [`BisectionError::RootFinding`]
/// ├ [`RootFindingError::NonFiniteEvaluation`] : `func(x)` produces NaN or inf during evaluation.
/// ├ [`RootFindingError::InvalidMaxIter`]      : `cfg.max_iter` = 0
/// │
/// * Propagated via [`BisectionError::Tolerance`] 
/// ├ [`ToleranceError::InvalidAbsFx`]          : `abs_fx` <= 0.0 or inf
/// ├ [`ToleranceError::InvalidAbsX`]           : `abs_x`  <  0.0 or inf
/// ├ [`ToleranceError::InvalidRelX`]           : `rel_x`  <  0.0 or inf
/// └ [`ToleranceError::InvalidAbsRelX`]        : one of `abs_x` and `rel_x` not > 0.
///
/// # Behavior
/// ├ Update:
/// │   x_mid = (x_lo + x_hi) / 2.0, maintaining a bracket
/// │   where f(x_lo)*f(x_hi) < 0.
/// │
/// ├ Tolerances:
/// │   |f(x_mid)| <= abs_fx returns with [`ToleranceSatisfied::AbsFxReached`] 
/// │   |x_hi - x_lo|/2 <= width_tol returns with [`ToleranceSatisfied::WidthTolReached`]
/// │
/// └ Stencil:
///     stores the current bracket [a, b] that produced the calculated root.
///
/// # Notes
/// ├ Bisection is globally convergent and robust but only linearly convergent.
/// └ Always requires an initial bracket with f(a) * f(b) < 0 (opposite sign).
///
/// # Warning 
/// └ Even if `(b - a)` already meets interval width tolerance, a sign change is required. 
pub fn bisection<F> (
    mut func: F,                
    mut a: f64, 
    mut b: f64, 
    cfg: BisectionCfg
) -> Result<RootFindingReport, BisectionError> 
where F: FnMut(f64) -> f64 {

    if !a.is_finite() || !b.is_finite() || a >= b { 
        return Err(BisectionError::InvalidBounds { a, b }); 
    }

    let abs_x    = cfg.common.abs_x(); 
    let rel_x    = cfg.common.rel_x(); 
    let abs_fx   = cfg.common.abs_fx();
    let max_iter = cfg.common.max_iter(); 
    let algorithm = Algorithm::Bracket(BracketFamily::Bisection);
    let algo_name = algorithm.algorithm_name();

    // already validated via building config; redundant guard 
    if let Some(0) = max_iter {
        return Err(RootFindingError::InvalidMaxIter { got: 0 }.into());
    }

    let mut dynamic_tol = DynamicTolerance::WidthTol { a, b };
    let mut width_tol   = algorithm.calculate_tolerance(&dynamic_tol, abs_x, rel_x)?;
    let theoretical_iters = theoretical_iter(a, b, width_tol)?;

    let num_iter = match max_iter { 
        Some(v) => v, 
        None    => theoretical_iters.max(GLOBAL_MAX_ITER_FALLBACK)
    };

    // track function evaluations
    let mut evals = 0; 

    // wraps func, increments evals, enforces finiteness
    let mut eval = |x: f64| -> Result<f64, BisectionError> {
        let fx = { evals += 1; func(x) }; 
        if !fx.is_finite() { 
            return Err(RootFindingError::NonFiniteEvaluation { x, fx }.into());
        } 
        
        Ok(fx) 
    };

    // early exit: a is root
    let mut fa = eval(a)?;
    if fa.abs() <= abs_fx { 
        return Ok(RootFindingReport {
            root                : a, 
            f_root              : fa, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::Bracket { bounds: [a, b] }, 
            algorithm_name      : algo_name 
        });
    } 
    // early exit: b is root
    let fb = eval(b)?;  
    if fb.abs() <= abs_fx { 
        return Ok(RootFindingReport {
            root                : b, 
            f_root              : fb, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::Bracket { bounds: [a, b] }, 
            algorithm_name      : algo_name 
        });
    }

    // error: no sign change across [a, b]
    if same_sign(fa, fb) { 
        return Err(BisectionError::NoSignChange { a, b }); 
    } 

    // early exit: width tolerance satisfied
    if b - a <= width_tol {
        let(midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;
        return Ok(RootFindingReport {
            root                : midpoint, 
            f_root              : fm, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
            stencil             : Stencil::Bracket { bounds: [a, b] }, 
            algorithm_name      : algo_name 
        });
    }

    // main loop 
    let mut midpoint = a;       // gets overwritten 
    let mut fm       = fa;      // gets overwritten
    for iter in 1..=num_iter {
        (midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;  
        
        // check |f(x)| tolerance 
        if fm.abs() <= abs_fx { 
            return Ok(RootFindingReport {
                root                : midpoint, 
                f_root              : fm, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
                stencil             : Stencil::Bracket { bounds: [a, b] }, 
                algorithm_name      : algo_name 
            });
        }

        // updated bracket 
        if opposite_sign(fa, fm) { 
            b = midpoint; 
        } else { 
            a = midpoint; 
            fa = fm; 
        }

        // check width tolerance 
        dynamic_tol = DynamicTolerance::WidthTol { a, b }; 
        width_tol   = algorithm.calculate_tolerance(&dynamic_tol, abs_x, rel_x)?;
        if b - a <= width_tol { 
            let (midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;
            return Ok(RootFindingReport {
                root                : midpoint, 
                f_root              : fm, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
                stencil             : Stencil::Bracket { bounds: [a, b] }, 
                algorithm_name      : algo_name 
            });
        }        
    }

    Ok(RootFindingReport {
        root                : midpoint, 
        f_root              : fm, 
        iterations          : num_iter, 
        evaluations         : evals,
        termination_reason  : TerminationReason::IterationLimit, 
        tolerance_satisfied : ToleranceSatisfied::ToleranceNotReached,
        stencil             : Stencil::Bracket { bounds: [a, b] }, 
        algorithm_name      : algo_name 
    })
}
