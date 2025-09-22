use super::algorithms::{Algorithm, CompoundFamily, GLOBAL_MAX_ITER_FALLBACK}; 
use super::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil}; 
use super::tolerances::DynamicTolerance; 
use super::signs::{opposite_sign, same_sign};
use super::errors::{RootFindingError, ToleranceError}; 
use super::config::{CommonCfg, impl_common_cfg};
use super::bisection::{theoretical_iter, midpoint}; 
use thiserror::Error;


#[derive(Debug, Error)] 
pub enum BrentError { 
    #[error(transparent)]
    RootFinding(#[from] RootFindingError), 

    #[error(transparent)] 
    Tolerance(#[from] ToleranceError), 

    #[error("invalid bounds: a and b must be finite with a < b got [{a}, {b}]")]
    InvalidBounds { a: f64, b: f64 },

    #[error("no sign change on [{a}, {b}]: sign(f(a)) = sign(f(b))")]
    NoSignChange { a: f64, b: f64 },
}


/// Brent's Configuration 
/// 
/// # Fields 
/// - `common` : [`CommonCfg`] with tolerances and optional `max_iter`. 
///
/// # Construction 
/// - Use [`BrentCfg::new`] then optional setters. 
///
/// # Defaults 
/// - If `common.max_iter` is `None`, [`brent`] resolves it using the 
///   worst-case theoretical number of iterations using pure bisection
///   from using the initial window.
#[derive(Debug, Copy, Clone)] 
pub struct BrentCfg { 
    common: CommonCfg
}
impl BrentCfg {
    #[must_use] 
    pub fn new() -> Self { 
        Self { 
            common: CommonCfg::new()
        }
    }
}
impl_common_cfg!(BrentCfg); 


/// Checks degeneracies in iqi and secant calculations.
#[inline] 
fn near_equal(x: f64, y: f64) -> bool { 
    (x - y).abs() <= 8.0 * f64::EPSILON * (x.abs() + y.abs()).max(1.0)
}


/// Inverse quadratic interpolation (Brent-style ratios).
///
/// Uses three distinct abscissae `(a,b,c)` with values `(fa,fb,fc)` to compute
/// a candidate `s = b + p/qd`. Rejects non-finite/degenerate inputs.
///
/// # Arguments
/// - `(a, fa)` : `a` and function value `f(a)`
/// - `(b, fb)` : `b` (current best) and value `f(b)`
/// - `(c, fc)` : `c` (previous) and value `f(c)`
///
/// # Returns
/// - `Some(s)` : finite IQI estimate
/// - `None`    : invalid inputs (non-finite, duplicate points, fa~fb~fc degeneracy,
///               zero/inf/NaN denominator, or non-finite result)
#[inline] 
fn iqi(
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
    (c, fc): (f64, f64), 
) -> Option<f64> {
    if !(a.is_finite()  && b.is_finite()  && c.is_finite() && 
        fa.is_finite()  && fb.is_finite() && fc.is_finite()) {
        return None;
    } 

    if near_equal(a, b)   
    || near_equal(a, c)   
    || near_equal(b, c) { 
        return None; 
    }   

    if near_equal(fa, fb) 
    || near_equal(fa, fc)
    || near_equal(fb, fc) { 
        return None; 
    }

    // brent's ratios 
    let q = fa / fc;
    let r = fb / fc;
    let t = fb / fa;

    let p = t * ( (c - b) * q * (q - r) - (b - a) * (r - 1.0) );
    let qd = (q - 1.0) * (r - 1.0) * (t - 1.0);

    if !p.is_finite() || !qd.is_finite() || qd == 0.0 { return None; }

    let d = p / qd;
    let sx = b + d;
    if !sx.is_finite() { return None; }
    Some(sx)
}


/// One secant step: line through `(a, fa)` and `(b, fb)` intersecting the x-axis.
///
/// Rejects non-finite inputs and (near-)parallel secant.
///
/// # Arguments
/// - `(a, fa)`  : `a` and function value `f(a)`
/// - `(b, fb)` : `b` and function value `f(b)`
///
/// # Returns
/// - `Some(s)`  : finite secant intersection
/// - `None`     : invalid inputs or degenerate denominator / non-finite result
#[inline]
fn secant( 
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
) -> Option<f64> { 

    if !(a.is_finite() && b.is_finite() && 
        fa.is_finite() && fb.is_finite()) { 
        return None; 
    }

    if near_equal(fa, fb) { return None; }     
    let denom = fb - fa; 

    let sx = ( a * fb - b * fa ) / denom; 
    if !sx.is_finite() { return None; }
    Some(sx)
}


/// Brent's "interior window" test for candidate `s`.
///
/// Checks that `s` lies strictly inside the open interval
/// `((3a + b)/4, b)` when `a < b` (mirrored when `a > b`).
/// This guards against overly aggressive extrapolation.
///
/// # Arguments
/// - `a`     : bracket endpoint A
/// - `b`     : bracket endpoint B
/// - `s`     : proposed root
///
/// # Returns
/// - `true`  : `s` is strictly inside the interior window
/// - `false` : otherwise
#[inline] 
fn interior_window_ok(a: f64, b: f64, s: f64) -> bool { 
    let lower = (3.0 * a + b) / 4.0; 
    if a < b { 
        s > lower && s < b 
    } else { 
        s < lower && s > b 
    }
}


/// Finds a root using Brent's method (bisection + secant + inverse quadratic interpolation).
///
/// This method assumes that the function is continuous on the interval `[a, b]`
/// and that `f(a)` and `f(b)` have opposite signs, guaranteeing a root exists
/// within the interval.
///
/// # Arguments
/// - `func` : function to evaluate
/// - `a`    : lower bound of the initial bracket (finite)
/// - `b`    : upper bound of the initial bracket (finite, strictly greater than `a`)
/// - `cfg`  : [`BrentCfg`] (tolerances; optional max_iter)
///
/// # Returns
/// [`RootFindingReport`] with
/// - `root`                : final root
/// - `f_root`              : function value at `root`
/// - `iterations`          : iterations performed
/// - `evaluations`         : total function evaluations
/// - `termination_reason`  : why it stopped
/// - `tolerance_satisfied` : which tolerance triggered or not reached
/// - `stencil`             : bracket snapshot at termination
/// - `algorithm_name`      : "brent"
///
/// # Errors
/// - [`BrentError::InvalidBounds`]        : `a`/`b` non-finite or `a >= b`
/// - [`BrentError::NoSignChange`]         : f(a) and f(b) share sign on [a, b]
///  
/// * Propagated via [`BrentError::RootFinding`]
/// - [`RootFindingError::NonFiniteEvaluation`] : f(x) produced NaN/inf
/// - [`RootFindingError::InvalidMaxIter`]      : cfg.common.max_iter = Some(0)
/// 
/// * Propagated via [`BrentError::Tolerance`]
/// - [`ToleranceError::InvalidAbsFx`]      : `abs_fx` <= 0.0 or inf 
/// - [`ToleranceError::InvalidAbsX`]       : `abs_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidRelX`]       : `rel_x`  <  0.0 or inf 
/// - [`ToleranceError::InvalidAbsRelX`]    : one of `abs_x` and `rel_x` not > 0
/// 
/// # Notes
/// - Globally convergent for continuous f with a valid sign-change bracket.
/// - Typically superlinear near simple roots and often as fast as Newton.
/// - May reduce to linear if interpolation is repeatedly rejected.
///
/// # Warning
/// - Even if `(b - a)` already meets interval tolerance, a sign change is required. 
/// - Very wide or poorly placed initial brackets, flat regions, or near-multiple roots can slow
///   progress; ensure a tight, genuine sign-change bracket when possible.
pub fn brent<F> ( 
    mut func: F, 
    mut a: f64, 
    mut b: f64, 
    cfg: BrentCfg 
) -> Result<RootFindingReport, BrentError> 
where F: FnMut(f64) -> f64 { 
    
    if !a.is_finite() || !b.is_finite() || a >= b { 
        return Err(BrentError::InvalidBounds { a, b });
    }   

    let abs_x     = cfg.common.abs_x(); 
    let rel_x     = cfg.common.rel_x(); 
    let abs_fx    = cfg.common.abs_fx(); 
    let max_iter  = cfg.common.max_iter(); 
    let algorithm = Algorithm::Compound(CompoundFamily::Brent);
    let algo_name = algorithm.algorithm_name();

    // already validated via building config; redundant guard
    if let Some(0) = max_iter { 
        return Err(RootFindingError::InvalidMaxIter { got: 0 }.into());
    }

    // track function evaluations 
    let mut evals: usize = 0; 

    // wraps func, increments evals, enforces finiteness
    let mut eval  = |x: f64| -> Result<f64, BrentError> { 
        evals += 1; 
        let fx = func(x); 
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
            stencil             : Stencil::doubleton(a, b), 
            algorithm_name      : algo_name 
        });
    }
    // early exit: b is root 
    let mut fb = eval(b)?; 
    if fb.abs() <= abs_fx { 
        return Ok(RootFindingReport {
            root                : b, 
            f_root              : fb, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
            stencil             : Stencil::doubleton(a, b), 
            algorithm_name      : algo_name 
        });
    }

    // error: no sign change across [a, b] 
    if same_sign(fa, fb) { 
        return Err(BrentError::NoSignChange { a, b }); 
    }

    let mut width_tol = algorithm.calculate_tolerance(
        &DynamicTolerance::WidthTol { a, b }, 
        abs_x, 
        rel_x
    )?; 
    let theoretical_iters = theoretical_iter(a, b, width_tol)?; 
    let num_iter = match max_iter { 
        Some(v) => v, 
        None    => theoretical_iters.max(GLOBAL_MAX_ITER_FALLBACK)
    };

    // early exit: width tolerance satisfied 
    if (b - a).abs() <= width_tol { 
        let mid = midpoint(a, b); 
        let fm  = eval(mid)?; 
        return Ok(RootFindingReport {
            root                : mid, 
            f_root              : fm, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
            stencil             : Stencil::doubleton(a, b), 
            algorithm_name      : algo_name 
        });
    }

    // ensure |fb| < |fa| 
    if fa.abs() > fb.abs() { 
        std::mem::swap(&mut a, &mut b); 
        std::mem::swap(&mut fa, &mut fb); 
    } 

    let mut c  = a;
    let mut d  = c; 
    let mut fc = fa; 
    let mut mflag = true; 

    // main loop 
    for iter in 1..=num_iter {  

        // candidate via iqi or secant 
        let candidate_iqi = iqi(
            (a, fa), 
            (b, fb), 
            (c, fc), 
        ); 
        let candidate_secant = secant(
            (b, fb), 
            (c, fc), 
        ); 

        let mut s = candidate_iqi 
            .or(candidate_secant) 
            .unwrap_or_else(|| midpoint(a, b)); 
        
        let step_bc = (b - c).abs(); 
        let step_cd = (c - d).abs(); 

        let reject =
            !interior_window_ok(a, b, s)
            || (mflag && (s - b).abs() >= 0.5 * step_bc)
            || (!mflag && (s - b).abs() >= 0.5 * step_cd)
            || (mflag && step_bc < width_tol)
            || (!mflag
                && step_cd < algorithm.calculate_tolerance(
                    &DynamicTolerance::step_two_scalars(c, d),
                    abs_x,
                    rel_x,
                )?);

        if reject { 
            // use bisection 
            s = midpoint(a, b); 
            mflag = true; 
        } else { 
            mflag = false; 
        }

        // brent rotation 
        let fs = eval(s)?; 
        d  = c; 
        c  = b; 
        fc = fb; 

        if opposite_sign(fa, fs) { 
            // root inside [a, s] 
            b  = s; 
            fb = fs; 
        } else { 
            // root inside [s, b] 
            a  = s; 
            fa = fs; 
        }

        // maintain |fb| <= |fa| 
        if fa.abs() < fb.abs() { 
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb); 
        }

        width_tol   = algorithm.calculate_tolerance(
            &DynamicTolerance::WidthTol { a, b }, 
            abs_x, 
            rel_x
        )?; 

        if fb.abs() <= abs_fx { 
            return Ok(RootFindingReport {
                root                : b, 
                f_root              : fb, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
                stencil             : Stencil::doubleton(a, b),
                algorithm_name      : algo_name
            }); 
        }

        if (b - a).abs() <= width_tol { 
            return Ok(RootFindingReport {
                root                : b, 
                f_root              : fb, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
                stencil             : Stencil::doubleton(a, b),
                algorithm_name      : algo_name
            }); 
        }
    }

    Ok(RootFindingReport {
        root                : b, 
        f_root              : fb, 
        iterations          : num_iter, 
        evaluations         : evals,
        termination_reason  : TerminationReason::IterationLimit, 
        tolerance_satisfied : ToleranceSatisfied::ToleranceNotReached,
        stencil             : Stencil::doubleton(a, b),
        algorithm_name      : algo_name
    }) 
}
