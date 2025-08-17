use super::common::{GLOBAL_MAX_ITER_FALLBACK};
use super::common::{
    RootReport, RootMeta, RootFindingError, Termination, ToleranceReason, Algorithm
};
use super::common::{
    bisection_theoretical_iter, validate_tolerances, width_tol_current, 
    opposite_signs
}; 
use thiserror::Error; 

const ALGORITHM: &str = Algorithm::Bisection.algorithm_name();

#[derive(Debug, Error)]
pub enum BisectionError {
    #[error(transparent)]
    Common(#[from] RootFindingError),

    #[error("no sign change on [{a}, {b}]: f(a) * f(b) > 0")]
    NoSignChange  { a: f64, b: f64 },

    #[error("invalid bounds: a and b must be finite with a < b. got [{a}, {b}]")] 
    InvalidBounds { a: f64, b: f64 },
}

/// Bisection Configuration 
/// 
/// # Defaults
///
/// ┌ DEFAULT_ABS_FX - Default absolute tolerance for convergence 
/// ├ DEFAULT_ABS_X  - Default absolute tolerance for interval width 
/// └ DEFAULT_REL_X  - Default relative tolerance for convergence
///
/// # Notes: 
/// └ If `max_iter` is None, it will be set to [`bisection_theoretical_iter`] 
///     ├ where [`bisection_theoretical_iter`] is the theoretical 
///     └ number of iterations for convergence. 
///
/// # Validation: 
/// └ Configuration validation occurs in [`bisection`] via [`BisectionCfg::validate()`].
///
///    The following checks are performed: 
///    ├ `abs_fx` >  0 and finite 
///    ├ `abs_x`  >= 0 and finite 
///    ├ `rel_x`  >= 0 and finite 
///    ├ Either `abs_x` or `rel_x` must be > 0 
///    └ `max_iter` is `None` or >= 1
#[derive(Debug, Copy, Clone)]
pub struct BisectionCfg {
    abs_fx:     Option<f64>, 
    abs_x:      Option<f64>, 
    rel_x:      Option<f64>, 
    max_iter:   Option<usize>, 
}
impl BisectionCfg { 
    pub const DEFAULT_ABS_FX: f64 = 1e-12; 
    pub const DEFAULT_ABS_X:  f64 = 0.0; 
    pub const DEFAULT_REL_X:  f64 = 4.0 * f64::EPSILON; 

    #[must_use]
    pub fn new() -> Self { Self::default() } 

    pub fn with_abs_fx(mut self, v: f64) -> Self { self.abs_fx = Some(v); self }
    pub fn with_abs_x (mut self, v: f64) -> Self { self.abs_x  = Some(v); self }
    pub fn with_rel_x (mut self, v: f64) -> Self { self.rel_x  = Some(v); self }
    pub fn with_max_iter(mut self, v: usize) -> Self { self.max_iter = Some(v); self } 

    #[inline] #[must_use] pub fn abs_fx(&self) -> f64 { self.abs_fx.unwrap_or(Self::DEFAULT_ABS_FX) }
    #[inline] #[must_use] pub fn abs_x (&self) -> f64 { self.abs_x .unwrap_or(Self::DEFAULT_ABS_X)  }
    #[inline] #[must_use] pub fn rel_x (&self) -> f64 { self.rel_x .unwrap_or(Self::DEFAULT_REL_X)  }
    #[inline] #[must_use] pub fn max_iter(&self) -> Option<usize> { self.max_iter }
 
    pub fn validate(&self) -> Result<BisectionCfg, RootFindingError> {
        let (abs_fx, abs_x, rel_x, max_iter) = validate_tolerances(
            self.abs_fx,
            self.abs_x,
            self.rel_x,
            self.max_iter,
            Self::DEFAULT_ABS_FX,
            Self::DEFAULT_ABS_X,
            Self::DEFAULT_REL_X
        )?;

        Ok(Self {
            abs_fx: Some(abs_fx),
            abs_x:  Some(abs_x),
            rel_x:  Some(rel_x),
            max_iter,
        })
    }
}

impl Default for BisectionCfg { 
    fn default() -> Self { 
        Self { 
            abs_fx:     Some(Self::DEFAULT_ABS_FX), 
            abs_x:      Some(Self::DEFAULT_ABS_X), 
            rel_x:      Some(Self::DEFAULT_REL_X), 
            max_iter:   None
        }
    }
}

/// Calculates midpoint of [a, b]
#[inline] 
fn calculate_bisection(a: f64, b: f64) -> f64 { 
    a + (b - a) * 0.5
}

/// Calculates the function evaluation for the midpoint/bisection of [a, b]
///
/// # Arguments 
/// ├ `a` - left endpoint 
/// ├ `b` - right endpoint 
/// └ `eval` - function; made by default with finite checks 
/// 
/// # Returns 
/// ├ Ok((midpoint, f(midpoint)) if the function evaluation f(m) is finite. 
/// └ Err(BisectionError::NonFiniteEval) if the function evaluation f(m) is non-finite. 
#[inline] 
fn next_sol_estimate<F>(
    a: f64, b: f64, eval: &mut F 
) -> Result<(f64, f64), BisectionError> 
where F: FnMut(f64) -> Result<f64, BisectionError> { 
    let midpoint = calculate_bisection(a, b);
    let fm       = eval(midpoint)?; 
    Ok((midpoint, fm))
}

/// Finds a root of a function using the 
/// [bisection method](https://en.wikipedia.org/wiki/Bisection_method).
///
/// This method assumes that the function `func` is continuous on the interval `[a, b]`
/// and that `func(a)` and `func(b)` have opposite signs, guaranteeing a root exists
/// within the interval.
///
/// # Arguments
///
/// ┌ `func` - The function whose root is to be found.
/// ├ `a`    - Lower bound of the search interval. Must be finite and less than `b`.
/// ├ `b`    - Upper bound of the search interval. Must be finite and greater than `a`.
/// └ `cfg`  - Configuration containing optional absolute (`abs_fx`, `abs_x`) and relative 
///            (`rel_x`) tolerances for convergence. See [`BisectionCfg`]
///    Defaults: 
///    ├ cfg.abs_fx = 1e-12 
///    ├ cfg.abs_x  = 0.0                  
///    └ cfg.rel_x  = 4 * machine_epsilon 
///
/// # Returns
///
/// On success, returns a [`RootReport::Bracket`] enum containing a struct:
/// ├ `meta` : [`RootMeta`] struct containing 
/// │          ├ `root`       : Approximate root location
/// │          ├ `f_root`     : The function value at calculated root 
/// │          ├ `iterations` : Number of iterations performed, 0 if bounds are already roots
/// │          ├ `evals`      : Number of function evaluations performed
/// │          ├ `termination`: Reason for termination 
/// │          │  ├ [`Termination::ToleranceReached`]  
/// │          │  └ [`Termination::IterationLimit`]
/// │          ├ `tolerance`  : Which tolerance was reached
/// │          │  ├ [`ToleranceReason::AbsFxReached`] 
/// │          │  ├ [`ToleranceReason::WidthTolReached`] 
/// │          │  └ [`ToleranceReason::ToleranceNotReached`]
/// │          └ `algorithm`  : "bisection" 
/// ├ `left` : Final left interval bound after convergence.   
/// └ `right`: Final right interval bound after convergence. 
///
/// # Errors
///
/// ┌ [`BisectionError::InvalidBounds`]       - `a` or `b` is NaN/inf or if `a >= b`.
/// ├ [`BisectionError::NoSignChange`]        - `func(a)` and `func(b)` do not have opposite signs.
/// │
/// │ 
/// The following are propagated via [`BisectionError::Common`]
/// ├ [`BisectionError::NonFiniteEvaluation`] - `func(x)` produces NaN or inf during evaluation.
/// ├ [`BisectionError::InvalidAbsFx`]        - `cfg.abs_fx` <= 0 or not finite.
/// ├ [`BisectionError::InvalidAbsX`]         - `cfg.abs_x` < 0 or not finite.
/// ├ [`BisectionError::InvalidRelX`]         - `cfg.rel_x` < 0 or not finite.
/// ├ [`BisectionError::InvalidTolerance`]    - computed width tolerance (cfg.abs_x + cfg.rel_x*scale) <= 0 or not finite.
/// └ [`BisectionError::InvalidMaxIter`]      - `cfg.max_iter` == 0.
///
/// # Notes 
/// ├ Theoretical iteration limits are based on equivalent bisection steps. 
/// │   └ Used only if `max_iter` is `None`/not provided.  
/// └ On early width-tolerance success, `iterations = 0` but the midpoint and its function eval 
///   is computed for reporting. This incurs exactly one extra evaluation. 
///
/// # Warning 
/// └ Even if `(b - a)` already meets interval width tolerance, a sign change is required. 
pub fn bisection<F>(
    mut func: F,                
    mut a: f64, 
    mut b: f64, 
    cfg: BisectionCfg
) -> Result<RootReport, BisectionError> 
where F: FnMut(f64) -> f64 {
    
    if !(a.is_finite() && b.is_finite()) || a >= b { 
        return Err(BisectionError::InvalidBounds { a, b }); 
    }

    let cfg = cfg.validate()?; 

    let abs_x    = cfg.abs_x(); 
    let rel_x    = cfg.rel_x(); 
    let abs_fx   = cfg.abs_fx();
    let max_iter = cfg.max_iter;
    let width_tol0 = width_tol_current(a, b, abs_x, rel_x);
    let theoretical_iters = bisection_theoretical_iter(a, b, width_tol0)?;

    let num_iter = match max_iter { 
        Some(m) => m, 
        None    => theoretical_iters.min(GLOBAL_MAX_ITER_FALLBACK)
    };

    // number of function evaluations 
    let mut evals = 0; 

    // closure function, checks finiteness  
    let mut eval = |x: f64| -> Result<f64, BisectionError> {
        let fx = { evals += 1; func(x) }; 
        if !fx.is_finite() { 
            Err(RootFindingError::NonFiniteEvaluation { x, fx }.into()) 
        } else { 
            Ok(fx) 
        }
    };

    // immediate bounds are roots
    let mut fa = eval(a)?;
    if fa.abs() <= abs_fx { 
        return Ok(RootReport::Bracket {
                meta: RootMeta { 
                root        : a, 
                f_root      : fa, 
                iterations  : 0, 
                evals       : evals,
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::AbsFxReached, 
                algorithm   : ALGORITHM 
            }, 
            left  : a, 
            right : b, 
        });
    }
    let fb = eval(b)?;  
    if fb.abs() <= abs_fx { 
        return Ok(RootReport::Bracket {
            meta: RootMeta { 
                root        : b, 
                f_root      : fb, 
                iterations  : 0, 
                evals       : evals, 
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::AbsFxReached, 
                algorithm   : ALGORITHM 
            }, 
            left  : a, 
            right : b, 
        }); 
    }

    if !opposite_signs(fa, fb) { 
        return Err(BisectionError::NoSignChange { a, b }); 
    } 

    // immediate narrow width success 
    if b - a <= width_tol0 {
        let(midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;
        return Ok(RootReport::Bracket {
            meta: RootMeta { 
                root        : midpoint, 
                f_root      : fm, 
                iterations  : 0, 
                evals       : evals, 
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::WidthTolReached, 
                algorithm   : ALGORITHM 
            }, 
            left  : a, 
            right : b, 
        }); 
    }

    // algorithm
    let mut midpoint = a;       // gets overwritten 
    let mut fm       = fa;      // gets overwritten
    for iter in 1..=num_iter {
        (midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;  
        
        // check for abs fx tolerance 
        if fm.abs() <= abs_fx { 
            return Ok(RootReport::Bracket {
                meta: RootMeta { 
                    root        : midpoint, 
                    f_root      : fm, 
                    iterations  : iter, 
                    evals       : evals, 
                    termination : Termination::ToleranceReached, 
                    tolerance   : ToleranceReason::AbsFxReached, 
                    algorithm   : ALGORITHM 
                },
                left  : a, 
                right : b, 
            });
        }

        // shrink interval
        if opposite_signs(fa, fm) { 
            b = midpoint; 
        } else { 
            a = midpoint; 
            fa = fm; 
        }

        // check for width tolerance 
        if b - a <= width_tol_current(a, b, abs_x, rel_x) { 
            let (midpoint, fm) = next_sol_estimate(a, b, &mut eval)?;
            return Ok(RootReport::Bracket {
                meta: RootMeta { 
                    root        : midpoint, 
                    f_root      : fm, 
                    iterations  : iter, 
                    evals       : evals, 
                    termination : Termination::ToleranceReached, 
                    tolerance   : ToleranceReason::WidthTolReached,
                    algorithm   : ALGORITHM
                },
                left  : a, 
                right : b, 
            }); 
        }        
    }

    Ok(RootReport::Bracket {
        meta: RootMeta { 
            root        : midpoint, 
            f_root      : fm, 
            iterations  : num_iter, 
            evals       : evals, 
            termination : Termination::IterationLimit, 
            tolerance   : ToleranceReason::ToleranceNotReached,
            algorithm   : ALGORITHM 
        }, 
        left  : a, 
        right : b, 
    })
}
