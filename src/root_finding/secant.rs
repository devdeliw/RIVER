use super::common::{
    RootReport, RootMeta, RootFindingError, Termination, ToleranceReason, Algorithm
}; 
use super::common::{validate_tolerances, step_tol_current};
use thiserror::Error; 

const ALGORITHM: &str = Algorithm::Secant.algorithm_name(); 

#[derive(Debug, Error)] 
pub enum SecantError{ 
    #[error(transparent)] 
    Common(#[from] RootFindingError), 

    #[error("invalid initial guesses: x0 and x1 must be finite and distinct")]
    InvalidGuess { x0: f64, x1: f64 }, 

    #[error(
        "degenerate secant: |fx2 - fx1|={denom:.3e} vs scale≈max(|fx|, 1); \
        fx1={fx1:.3e}, fx2={fx2:.3e}"
    )]
    DegenerateSecantStep { fx1: f64, fx2: f64, denom: f64 },
}


#[derive(Debug, Copy, Clone)] 
pub struct SecantCfg { 
    abs_fx:     Option<f64>, 
    abs_x:      Option<f64>, 
    rel_x:      Option<f64>, 
    max_iter:   Option<usize>, 
}
impl SecantCfg { 
    pub const DEFAULT_ABS_FX: f64 = 1e-12; 
    pub const DEFAULT_ABS_X:  f64 = 0.0; 
    pub const DEFAULT_REL_X:  f64 = 4.0 * f64::EPSILON; 

    #[must_use] 
    pub fn new() -> Self { Self::default() } 

    pub fn with_abs_fx   (mut self, v: f64)   -> Self { self.abs_fx = Some(v); self }
    pub fn with_abs_x    (mut self, v: f64)   -> Self { self.abs_x  = Some(v); self }
    pub fn with_rel_x    (mut self, v: f64)   -> Self { self.rel_x  = Some(v); self }
    pub fn with_max_iter (mut self, v: usize) -> Self { self.max_iter = Some(v); self } 

    #[inline] #[must_use] pub fn abs_fx(&self)   -> f64 { self.abs_fx.unwrap_or(Self::DEFAULT_ABS_FX) }
    #[inline] #[must_use] pub fn abs_x (&self)   -> f64 { self.abs_x .unwrap_or(Self::DEFAULT_ABS_X)  }
    #[inline] #[must_use] pub fn rel_x (&self)   -> f64 { self.rel_x .unwrap_or(Self::DEFAULT_REL_X)  }
    #[inline] #[must_use] pub fn max_iter(&self) -> Option<usize> { self.max_iter }

    #[must_use] 
    pub fn validate(&self) -> Result<SecantCfg, RootFindingError> { 
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
            max_iter
        })
    }
}

impl Default for SecantCfg { 
    fn default() -> Self { 
        Self { 
            abs_fx:     Some(Self::DEFAULT_ABS_FX), 
            abs_x:      Some(Self::DEFAULT_ABS_X), 
            rel_x:      Some(Self::DEFAULT_REL_X), 
            max_iter:   None, 
        }
    }
}


/// Calculates the secant x-intercept for the line 
/// connecting `(x1, fx1)` and `(x2, fx2)` 
///
/// # Arguments 
/// ├ `(x1, fx1)` - secant endpoint 1 and function value
/// └ `(x2, fx2)` - secant endpoint 2 and function value 
///
/// # Returns 
/// ├ `Ok(x_secant)` if denominator `fx2 - fx1` is well-scaled 
/// └ `Err(DegenerateSecantStep)` if denominator is too small. 
#[inline]
fn calculate_secant_x_intercept(
    (x1, fx1): (f64, f64), 
    (x2, fx2): (f64, f64), 
)-> Result<f64, SecantError> {
    let denom = fx2 - fx1;
    let scale  = fx1.abs().max(fx2.abs()).max(1.0);
    let thresh = f64::EPSILON * scale + f64::MIN_POSITIVE;

    if denom.abs() <= thresh {
        return Err(SecantError::DegenerateSecantStep { fx1, fx2, denom });
    } 

    // Guard for if x1 ~ x2. 
    // However, typically this will trigger satisfied [`ToleranceReason::StepSizeReached`] 
    // unless user provides initial guesses that differ by ~ machine epsilon. 
    let dx = (x2 - x1).abs(); 
    if dx <= f64::MIN_POSITIVE {
        // default to bisection 
        return Ok((x1 + x2) * 0.5);
    }

    Ok(x2 - fx2 * (x2 - x1) / denom)
}
/// Calculates the secant x-intersection point for the line 
/// connecting `(x1, fx1)` and `(x2, fx2)` and its function eval 
///
/// # Arguments 
/// ├ `(x1, fx1)` - secant endpoint 1 and function value
/// └ `(x2, fx2)` - secant endpoint 2 and function value 
///
/// # Returns 
/// ├ `Ok(x_next, f(x_next))` if denominator `fx2 - fx1` is well-scaled 
/// └ `Err(DegenerateSecantStep)` if denominator is too small. 
#[inline] 
fn next_sol_estimate<F>(
    (x1, fx1): (f64, f64), 
    (x2, fx2): (f64, f64), 
    eval: &mut F 
) -> Result<(f64, f64), SecantError> 
where F: FnMut(f64) -> Result<f64, SecantError> { 
    let x_next = calculate_secant_x_intercept((x1, fx1), (x2, fx2))?;
    let f_next  = eval(x_next)?; 
    Ok((x_next, f_next))
}


/// Finds a root of a function using the 
/// [secant method](https://en.wikipedia.org/wiki/Secant_method).
///
/// This method assumes two distinct initial guesses `x0` and `x1` are provided.
/// It constructs successive secant lines through the last two iterates to estimate
/// the root of `func`. Convergence is not guaranteed unless the function behaves
/// sufficiently like a smooth monotone function near the root.
///
/// # Arguments
///
/// ┌ `func` - The function whose root is to be found.
/// ├ `x0`   - First initial guess.  Must be finite and not equal to `x1`.
/// ├ `x1`   - Second initial guess. Must be finite and not equal to `x0`.
/// └ `cfg`  - Configuration containing optional absolute (`abs_fx`, `abs_x`) and relative 
///            (`rel_x`) tolerances for convergence. See [`SecantCfg`].
///    Defaults: 
///    ├ cfg.abs_fx = 1e-12 
///    ├ cfg.abs_x  = 0.0                  
///    └ cfg.rel_x  = 4 * machine_epsilon 
///
/// # Returns
///
/// On success, returns a [`RootReport::TwoPoint`] enum containing a struct:
/// ├ `meta` : [`RootMeta`] struct containing 
/// │          ├ `root`       : Approximate root location
/// │          ├ `f_root`     : The function value at calculated root 
/// │          ├ `iterations` : Number of iterations performed, 0 if an initial guess was root
/// │          ├ `evals`      : Number of function evaluations performed
/// │          ├ `termination`: Reason for termination 
/// │          │  ├ [`Termination::ToleranceReached`]  
/// │          │  └ [`Termination::IterationLimit`]
/// │          ├ `tolerance`  : Which tolerance was reached
/// │          │  ├ [`ToleranceReason::AbsFxReached`] 
/// │          │  ├ [`ToleranceReason::StepSizeReached`] 
/// │          │  └ [`ToleranceReason::ToleranceNotReached`]
/// │          └ `algorithm`  : "secant"
/// ├ `curr` : Most recent iterate before final secant.  
/// └ `prev` : The one before `curr` 
///     
/// Together `parent1` and `parent2` are what calculated the final secant root.
/// `parent1` is *not* the root itself. 
///
/// # Errors
///
/// ┌ [`SecantError::InvalidGuess`]           - If `x0` or `x1` is NaN/inf or if `x0 == x1`.
/// ├ [`SecantError::DegenerateSecantStep`]   - If f(x1) close to f(x2), denom f(x2) - f(x1) ~ 0
/// │
/// * The following are propagated via [`SecantError::Common`]
/// ├ [`SecantError::NonFiniteEvaluation`]    - `func(x)` produces NaN or inf during evaluation.
/// ├ [`SecantError::InvalidAbsFx`]           - `cfg.abs_fx` <= 0 or not finite.
/// ├ [`SecantError::InvalidAbsX`]            - `cfg.abs_x` < 0 or not finite.
/// ├ [`SecantError::InvalidRelX`]            - `cfg.rel_x` < 0 or not finite.
/// └ [`SecantError::InvalidMaxIter`]         - `cfg.max_iter` == 0.
///
/// # Notes 
/// ├ If `max_iter` is `None`/not provided, will use default # iterations from 
/// │ [`Algorithm::default_max_iter()`]
/// ├ if the initial guesses are already close enough below the calculated step tolerance,  
/// │ `root` will be chosen as `x1`, `f_root` will be chosen as `func(x1)`. 
/// ├ if the initial guesses are already roots (f(guess) <= `abs_fx`): 
/// │   ├ if `x0` is root: `root=x0`, `curr=x0`, `prev=x1` (even though no previous)  
/// │   └ if `x1` is root: `root=x1`, `curr=x1`, `prev=x0` (even though no previous)
/// ├ Convergence is superlinear if close to the root but may diverge otherwise.  
/// └ On immediate tolerance success, `iterations = 0`. 
///
/// # Warning 
/// └ Poor initial guesses may lead to divergence or extremely slow convergence.
#[must_use]
pub fn secant<F> ( 
    mut func: F, 
    x0: f64, 
    x1: f64,
    cfg: SecantCfg
) -> Result<RootReport, SecantError> 
where F: FnMut(f64) -> f64 { 
    
    if !(x0.is_finite() && x1.is_finite()) || x0 == x1 { 
        return Err(SecantError::InvalidGuess { x0, x1 });
    }

    let cfg = cfg.validate()?; 
    let abs_x       = cfg.abs_x(); 
    let rel_x       = cfg.rel_x(); 
    let abs_fx      = cfg.abs_fx(); 
    let max_iter    = cfg.max_iter(); 

    // default to algorithm default # iterations 
    let num_iter = match max_iter { 
        Some(m) => m, 
        None    => Algorithm::Secant.default_max_iter(), 
    };

    // number of function evaluations 
    let mut evals = 0; 

    // closure function, checks finiteness 
    let mut eval = |x: f64| -> Result<f64, SecantError> { 
        let fx = { evals += 1; func(x) }; 
        if !fx.is_finite() { 
            return Err(RootFindingError::NonFiniteEvaluation { x, fx }.into()) 
        } else { 
            Ok(fx) 
        }
    };

    // initial guesses are roots 
    let fx0 = eval(x0)?; 
    if fx0.abs() <= abs_fx { 
        return Ok(RootReport::TwoPoint {
            meta: RootMeta {
                root        : x0, 
                f_root      : fx0, 
                iterations  : 0, 
                evals       : evals,
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::AbsFxReached, 
                algorithm   : ALGORITHM
            }, 
            parent1 : x0, 
            parent2 : x1,
        }); 
    }
    let fx1 = eval(x1)?; 
    if fx1.abs() <= abs_fx {
        return Ok(RootReport::TwoPoint { 
            meta: RootMeta {
                root        : x1, 
                f_root      : fx1, 
                iterations  : 0, 
                evals       : evals,
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::AbsFxReached, 
                algorithm   : ALGORITHM
            }, 
            parent1 : x1, 
            parent2 : x0,
        }); 
    }

    // immediate step-size success
    let step_tol0 = step_tol_current(x1, abs_x, rel_x); 
    if (x1 - x0).abs() <= step_tol0 { 
        return Ok(RootReport::TwoPoint {
            meta: RootMeta { 
                root        : x1, 
                f_root      : fx1, 
                iterations  : 0, 
                evals       : evals, 
                termination : Termination::ToleranceReached, 
                tolerance   : ToleranceReason::StepSizeReached, 
                algorithm   : ALGORITHM 
            }, 
            parent1  : x1, 
            parent2 :  x0, 
        }); 
    }

    let mut x_curr = x1; 
    let mut x_prev = x0; 
    let mut f_curr = fx1; 
    let mut f_prev = fx0; 
    
    // algorithm 
    for iter in 1..=num_iter { 
        let (x_next, f_next) = next_sol_estimate(
            (x_curr, f_curr), (x_prev, f_prev), 
            &mut eval
        )?;

        // check for abs_fx tolerance 
        if f_next.abs() <= abs_fx { 
            return Ok(RootReport::TwoPoint {
                meta: RootMeta { 
                    root        : x_next,
                    f_root      : f_next,
                    iterations  : iter,
                    evals       : evals,
                    termination : Termination::ToleranceReached,
                    tolerance   : ToleranceReason::AbsFxReached,
                    algorithm   : ALGORITHM,
                },
                parent1 : x_curr,
                parent2 : x_prev,
            }); 
        }

        // check for step tolerance 
        let step_tol = step_tol_current(x_next, abs_x, rel_x); 
        if (x_next - x_curr).abs() <= step_tol { 
            return Ok(RootReport::TwoPoint {
                meta: RootMeta { 
                    root        : x_next,
                    f_root      : f_next,
                    iterations  : iter,
                    evals       : evals,
                    termination : Termination::ToleranceReached,
                    tolerance   : ToleranceReason::StepSizeReached,
                    algorithm   : ALGORITHM,
                },
                parent1 : x_curr,
                parent2 : x_prev,
            });
        }

        x_prev = x_curr; 
        f_prev = f_curr; 
        x_curr = x_next; 
        f_curr = f_next; 
    }

    Ok(RootReport::TwoPoint {
        meta: RootMeta { 
            root        : x_curr,
            f_root      : f_curr,
            iterations  : num_iter,
            evals       : evals,
            termination : Termination::IterationLimit,
            tolerance   : ToleranceReason::ToleranceNotReached,
            algorithm   : ALGORITHM,
        }, 
        parent1 : x_curr,
        parent2 : x_prev,
    })
}

