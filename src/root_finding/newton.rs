//! Newton-Rapshon method

use super::algorithms::{Algorithm, OpenFamily, GLOBAL_MAX_ITER_FALLBACK}; 
use super::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil}; 
use super::tolerances::DynamicTolerance; 
use super::errors::{RootFindingError, ToleranceError}; 
use super::config::{CommonCfg, impl_common_cfg}; 
use thiserror::Error;


#[derive(Debug, Error)] 
pub enum NewtonError {
    #[error(transparent)]
    RootFinding(#[from] RootFindingError),

    #[error(transparent)]
    Tolerance(#[from] ToleranceError), 

    #[error("invalid initial guess: x0={x0} must be finite")]
    InvalidGuess { x0: f64 }, 

    #[error("invalid max step, must be > 0 or f64::INFINITY")] 
    InvalidMaxStep { step: f64 },

    #[error("step non-finite at x={x}, step={step}; x + step undefined")] 
    StepNotFinite { x: f64, step: f64 }, 

    #[error("step non-finite from vanishing derivative at x={x}, f'(x)={dfx}")]
    DerivativeTooSmall { x: f64, dfx: f64 },

    #[error("derivative non-finite at x={x}, f'(x)={dfx}")]
    DerivativeNotFinite { x: f64, dfx: f64 },

    #[error("finite-difference step not representable at x={x}, h={h};\
             try smaller |x| scaling or analytic derivative"
    )]
    FiniteDifferenceStepUnrepresentable { x: f64, h: f64 }
}


/// Newton configuration.
/// 
/// # Fields
/// - `common`   : [`CommonCfg`] with tolerances and optional `max_iter`.
/// - `max_step` : optional limit on the absolute Newton step (default: ∞).
///
/// # Construction
/// - Use [`NewtonCfg::new`] then optional setters.
/// - Set an explicit step cap via [`NewtonCfg::set_max_step`] (must be > 0).
///
/// # Defaults
/// - If `common.max_iter` is `None`, [`newton`] resolves it using
///   [`Algorithm::default_max_iter`] for [`OpenFamily::Newton`], or
///   [`GLOBAL_MAX_ITER_FALLBACK`] if unavailable.
#[derive(Debug, Copy, Clone)] 
pub struct NewtonCfg {
    common: CommonCfg,
    max_step: f64
}
impl NewtonCfg {
    #[must_use]
    pub fn new() -> Self { 
        Self { 
            common: CommonCfg::new(),
            max_step: f64::INFINITY 
        }
    }
    #[must_use]
    pub fn set_max_step(mut self, v: f64) -> Result<Self, NewtonError> { 
        if  v <= 0.0 || v.is_nan() { 
            return Err(NewtonError::InvalidMaxStep { step: v });
        }
        self.max_step = v; 
        Ok(self) 
    }
}
impl_common_cfg!(NewtonCfg);


/// ULP helpers for finite-difference fallback near representability edges 
#[inline] 
fn next_up(x: f64) -> f64 { 
    if x.is_nan() || x == f64::INFINITY { return x; }
    // smallest positive subnormal 
    if x == 0.0 { return f64::from_bits(1); } 
    
    let bits   = x.to_bits(); 
    let bumped = if x > 0.0 { bits + 1 } else { bits - 1 }; 
    f64::from_bits(bumped)
}
#[inline] 
fn next_down(x: f64) -> f64 { 
    if x.is_nan() || x == f64::NEG_INFINITY { return x; } 
    // largest negative subnormal 
    if x == 0.0 { return -f64::from_bits(1); } 

    let bits = x.to_bits(); 
    let bumped = if x > 0.0 { bits - 1 } else { bits + 1 };
    f64::from_bits(bumped)
}


/// Helpers 
/// - `eval_fx_checked`   : evaluates `f(x)` with finite-check
/// - `eval_dfx_analytic` : evaluates user-supplied derivative `df(x)` 
/// - `eval_dfx_fd`       : central finite-difference with ULP rescue  
#[inline] 
fn eval_fx_checked<F>(
    f: &mut F, 
    x: f64,
    evals: &mut usize
) -> Result<f64, NewtonError>  where F: FnMut(f64) -> f64 { 
    let fx = { *evals += 1; f(x) }; 
    if !fx.is_finite() { 
        return Err(RootFindingError::NonFiniteEvaluation { x, fx }.into()); 
    }

    Ok(fx)
}
#[inline] 
fn eval_dfx_analytic<G>(
    df: &mut G, 
    x: f64, 
    evals: &mut usize
) -> Result<f64, NewtonError> where G: FnMut(f64) -> f64 { 
    let dfx = { *evals += 1; df(x) };  
    if !dfx.is_finite() { 
        return Err(NewtonError::DerivativeNotFinite { x, dfx }); 
    }

    Ok(dfx)
}
#[inline] 
fn eval_dfx_fd<F>(
    f: &mut F, 
    x: f64, 
    evals: &mut usize
) -> Result<f64, NewtonError>  where F: FnMut(f64) -> f64 { 
    // central finite-difference 
    let mut h  = f64::EPSILON.cbrt() * x.abs().max(1.0); 
    let mut xp = x + h; 
    let mut xm = x - h; 

    // try rescue if representability collapses 
    if !xp.is_finite() || !xm.is_finite() || xp == x || xm == x { 
        xp = next_up(x); 
        xm = next_down(x); 
        h = 0.5 * (xp - xm); 

        if !xp.is_finite() || !xm.is_finite() || xp == x || xm == x { 
            return Err(NewtonError::FiniteDifferenceStepUnrepresentable { x, h });
        }
    }

    let fxp = eval_fx_checked(f, xp, evals)?; 
    let fxm = eval_fx_checked(f, xm, evals)?; 
    let dfx = (fxp - fxm) / (2.0 * h); 
    if !dfx.is_finite() { 
        return Err(NewtonError::DerivativeNotFinite { x, dfx }); 
    }

    Ok(dfx)
}


fn newton_loop<F, G>(
    mut f: F, 
    mut df: Option<G>, 
    x0: f64, 
    cfg: NewtonCfg, 
) -> Result<RootFindingReport, NewtonError> 
where 
    F: FnMut(f64) -> f64, 
    G: FnMut(f64) -> f64 
{   
    let algorithm = Algorithm::Open(OpenFamily::Newton);
    let algo_name = algorithm.algorithm_name(); 

    let abs_fx    = cfg.common.abs_fx(); 
    let abs_x     = cfg.common.abs_x(); 
    let rel_x     = cfg.common.rel_x(); 
    let max_iter  = cfg.common.max_iter(); 
    let max_step  = cfg.max_step; 
    
    let num_iter  = match max_iter { 
        Some(0) => { 
            return Err(RootFindingError::InvalidMaxIter { got: 0 }.into());
        },

        Some(v) => v, 
        None    => algorithm
            .default_max_iter()
            .unwrap_or(GLOBAL_MAX_ITER_FALLBACK)
    };

    let mut evals: usize = 0; 

    // early exit: x0 is root
    let mut x  = x0; 
    let mut fx = eval_fx_checked(&mut f, x, &mut evals)?; 
    if fx.abs() <= abs_fx { 
        return Ok(RootFindingReport {
            root                : x0,
            f_root              : fx,
            iterations          : 0,
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached,
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::singleton(x0), // equal to root; see docs 
            algorithm_name      : algo_name,
        });
    }

    // main loop 
    let mut prev_x = x; 
    for iter in 1..=num_iter {
        // compute derivative
        let dfx = match df.as_mut() { 
            Some(v) => eval_dfx_analytic(v, x, &mut evals)?, 
            None    => eval_dfx_fd(&mut f, x, &mut evals)? 
        }; 

        // raw step 
        let mut step = -fx / dfx; 
        if !step.is_finite() { 
            return Err(NewtonError::DerivativeTooSmall { x, dfx }); 
        }

        // clip to max_step
        if step.abs() > max_step {
            step = step.signum() * max_step; 
        }

        let x_next = x + step; 
        if !x_next.is_finite() { 
            return Err(NewtonError::StepNotFinite { x, step }); 
        }

        // machine stagnation 
        if x_next == x { 
            return Ok(RootFindingReport { 
                root                : x, 
                f_root              : fx, 
                iterations          : iter, 
                evaluations         : evals, 
                termination_reason  : TerminationReason::MachinePrecisionReached, 
                tolerance_satisfied : ToleranceSatisfied::StepSizeReached, 
                stencil             : Stencil::singleton(x), // same as root; see docs
                algorithm_name      : algo_name 
            });
        }

        // check |f(x)| tolerance 
        let fx_next = eval_fx_checked(&mut f, x_next, &mut evals)?; 
        if fx_next.abs() <= abs_fx { 
            return Ok(RootFindingReport {
                root                : x_next,
                f_root              : fx_next,
                iterations          : iter,
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached,
                tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
                stencil             : Stencil::singleton(x), // previous iterate; see docs
                algorithm_name      : algo_name,
            });
        }

        // check step tolerance 
        let step_tol = algorithm.calculate_tolerance(
            &DynamicTolerance::step_two_scalars(x, x_next), 
            abs_x, 
            rel_x
        )?; 
        if (x_next - x).abs() <= step_tol { 
            return Ok(RootFindingReport {
                root                : x_next,
                f_root              : fx_next,
                iterations          : iter,
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached,
                tolerance_satisfied : ToleranceSatisfied::StepSizeReached,
                stencil             : Stencil::singleton(x), // previous iterate; see docs 
                algorithm_name      : algo_name,
            });
        }

        prev_x = x; 
        x  = x_next; 
        fx = fx_next; 
    }

    Ok(RootFindingReport {
        root                : x,
        f_root              : fx,
        iterations          : num_iter,
        evaluations         : evals,
        termination_reason  : TerminationReason::IterationLimit,
        tolerance_satisfied : ToleranceSatisfied::ToleranceNotReached,
        stencil             : Stencil::singleton(prev_x), // previous iterate; see docs
        algorithm_name      : algo_name,
    })
}


/// Finds a root of `func` using the
/// [Newton–Raphson method](https://en.wikipedia.org/wiki/Newton_method).
/// Supports analytic derivatives or a central finite-difference fallback.
///
/// # Arguments
/// - `func`  : function whose root is sought
/// - `dfunc` : optional analytic derivative; if `None`, use finite-difference
/// - `x0`    : finite initial guess
/// - `cfg`   : [`NewtonCfg`] (tolerances, optional `max_iter`, optional `max_step`)
///
/// # Returns
/// [`RootFindingReport`] with:
/// - `root`                : approximate root
/// - `f_root`              : function value at `root`
/// - `iterations`          : number of iterations performed
/// - `evaluations`         : total evaluations (f and f')
/// - `termination_reason`  : why it stopped
/// - `tolerance_satisfied` : which tolerance triggered 
/// - `stencil`             : previous iterate used to form the step
/// - `algorithm_name`      : "newton"
///
/// # Errors
/// - [`NewtonError::InvalidGuess`]                 : `x0` non-finite
/// - [`NewtonError::InvalidMaxStep`]               : `max_step <= 0` or NaN
/// - [`NewtonError::StepNotFinite`]                : `x + step` not representable
/// - [`NewtonError::DerivativeTooSmall`]           : derivative too small for a reliable step
/// - [`NewtonError::DerivativeNotFinite`]          : derivative non-finite
/// - [`NewtonError::FiniteDifferenceStepUnrepresentable`]  : FD step unrepresentable near `x`
///  
/// * Propagated via [`NewtonError::RootFinding`]:
/// - [`RootFindingError::NonFiniteEvaluation`]     : `f(x)` produced NaN/inf
/// - [`RootFindingError::InvalidMaxIter`]          : `max_iter = 0`
/// 
/// * Propagated via [`NewtonError::Tolerance`]: 
/// - [`ToleranceError::InvalidAbsFx`]              : `abs_fx` <= 0.0 or inf 
/// - [`ToleranceError::InvalidAbsX`]               : `abs_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidRelX`]               : `rel_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidAbsRelX`]            : both `abs_x` and `rel_x` not > 0.

/// # Behavior
/// - Derivative:
///     - analytic path uses `df(x)`
///     - FD path uses central finite-difference with `h = eps^{1/3} * max(|x|, 1)`,
///       rescued by ULP nudges (`next_up/down`) if `x +/- h` collapses.
/// - Step:
///     - raw step `-f/df` is computed; if non-finite, errors
///     - step is clipped to `max_step` if provided
/// - Stagnation: if `x + step == x`, returns [`TerminationReason::MachinePrecisionReached`]
/// - Report:
///     `stencil` is the previous iterate used to form the last step; on
///     immediate success at `x0`, it equals `x0`; on iteration limit, it is the
///     last iterate; on machine stagnation it equals the root. 
///
/// # Notes
/// - Quadratic convergence requires a good initial guess and smooth `f`
/// - For guaranteed convergence, prefer a bracketed method (e.g., bisection)
/// - Convergence is *local only* and depends on a good initial guess `x0` and
///   smoothness of `f`. Poor guesses or ill-behaved functions can diverge or cycle.
///   For guaranteed convergence, use a **bracketed method** (e.g. bisection/Brent)
pub fn newton<F, G>( 
    func: F, 
    dfunc: Option<G>, 
    x0: f64, 
    cfg: NewtonCfg, 
) -> Result<RootFindingReport, NewtonError> 
where 
    F: FnMut(f64) -> f64, 
    G: FnMut(f64) -> f64 { 
    
    if !x0.is_finite() { 
        return Err(NewtonError::InvalidGuess { x0 }); 
    }

    newton_loop(func, dfunc, x0, cfg) 
}
