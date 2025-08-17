use super::common::{
    RootReport, RootMeta, RootFindingError, Termination, ToleranceReason, Algorithm 
}; 
use super::common::{
    validate_tolerances, width_tol_current, opposite_signs, 
}; 
use thiserror::Error; 

#[derive(Debug, Error)]
pub enum RegulaFalsiError{ 
    #[error(transparent)] 
    Common(#[from] RootFindingError), 

    #[error("no sign change on [{a}, {b}]: f(a) * f(b) > 0")]
    NoSignChange  { a: f64, b: f64 },

    #[error("invalid bounds: a and b must be finite with a < b. got [{a}, {b}]")] 
    InvalidBounds { a: f64, b: f64 }, 

    #[error("denominator fb - fa too small in secant step: fa={fa}, fb={fb}, denom={denom}")]
    DegenerateSecantStep { fa: f64, fb: f64, denom: f64 },
}

/// RegulaFalsi Configuration 
/// 
/// # Defaults
///
/// ┌ DEFAULT_ABS_FX - Default absolute tolerance for convergence 
/// ├ DEFAULT_ABS_X  - Default absolute tolerance for interval width 
/// └ DEFAULT_REL_X  - Default relative tolerance for convergence
///
/// # Notes:
/// ├ If `max_iter` is None, it will be set to 2 * [`bisection_theoretical_iter()`]
/// └ Factor 2 is a heuristic in case of roughly stagnant endpoints 
///
/// # Validation: 
/// └ Configuration validation occurs in [`regula_falsi`] via [`RegulaFalsiCfg::validate()`].
///
///    The following checks are performed: 
///    ├ `abs_fx` >  0 and finite 
///    ├ `abs_x`  >= 0 and finite 
///    ├ `rel_x`  >= 0 and finite 
///    ├ Either `abs_x` or `rel_x` must be > 0 
///    └ `max_iter` is `None` or >= 1
#[derive(Debug, Copy, Clone)] 
pub struct RegulaFalsiCfg { 
    abs_fx:     Option<f64>, 
    abs_x:      Option<f64>, 
    rel_x:      Option<f64>, 
    max_iter:   Option<usize>, 
    algorithm:  Algorithm
}
impl RegulaFalsiCfg { 

    pub const DEFAULT_ABS_FX: f64 = 1e-12; 
    pub const DEFAULT_ABS_X:  f64 = 0.0; 
    pub const DEFAULT_REL_X:  f64 = 4.0 * f64::EPSILON; 

    #[must_use] 
    pub fn new() -> Self { Self::default() } 

    pub fn with_abs_fx   (mut self, v: f64)   -> Self { self.abs_fx = Some(v); self }
    pub fn with_abs_x    (mut self, v: f64)   -> Self { self.abs_x  = Some(v); self }
    pub fn with_rel_x    (mut self, v: f64)   -> Self { self.rel_x  = Some(v); self }
    pub fn with_max_iter (mut self, v: usize) -> Self { self.max_iter = Some(v); self } 
    pub fn with_algorithm (mut self, v: Algorithm) -> Self { self.algorithm = v; self }

    #[inline] #[must_use] pub fn abs_fx(&self)   -> f64 { self.abs_fx.unwrap_or(Self::DEFAULT_ABS_FX) }
    #[inline] #[must_use] pub fn abs_x (&self)   -> f64 { self.abs_x .unwrap_or(Self::DEFAULT_ABS_X)  }
    #[inline] #[must_use] pub fn rel_x (&self)   -> f64 { self.rel_x .unwrap_or(Self::DEFAULT_REL_X)  }
    #[inline] #[must_use] pub fn max_iter(&self) -> Option<usize> { self.max_iter }
    #[inline] #[must_use] pub fn algorithm(&self) -> Algorithm { self.algorithm } 
    
    pub fn validate(&self) -> Result<RegulaFalsiCfg, RootFindingError> {

        match self.algorithm { 
            Algorithm::RegulaFalsiPure 
            | Algorithm::RegulaFalsiIllinois 
            | Algorithm::RegulaFalsiPegasus 
            | Algorithm::RegulaFalsiAndersonBjorck => {}, 
            _ => return Err(RootFindingError::InvalidAlgorithm { algorithm: self.algorithm }.into()),
        }

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
            algorithm: self.algorithm
        })
    }

    #[must_use]
    pub const fn algorithm_in_use(&self) -> &'static str { 
        self.algorithm.algorithm_name()
    }
}

impl Default for RegulaFalsiCfg { 
    fn default() -> Self { 
        Self { 
            abs_fx:    Some(Self::DEFAULT_ABS_FX), 
            abs_x:     Some(Self::DEFAULT_ABS_X), 
            rel_x:     Some(Self::DEFAULT_REL_X), 
            max_iter:  None, 
            algorithm: Algorithm::RegulaFalsiIllinois
        }
    } 
}

/// Calculates the secant x-intersection point for the line 
/// connecting `(a, fa)` and `(b, fb)` 
///
/// # Arguments 
/// ├ `(a, fa)` - left endpoint and function value 
/// └ `(b, fb)` - right endpoint and function value
///
/// # Returns 
/// ├ `Ok(x_secant)` if denominator `fb - fa` is well-scaled 
/// └ `Err(DegenerateSecantStep)` if denominator is too small. 
///    └ Handled internally. Replaces with a bisection step. 
#[inline]
fn calculate_secant_x_intercept(
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
)-> Result<f64, RegulaFalsiError> {
    let denom = fb - fa;
    let scale = fa.abs().max(fb.abs()).max(1.0);

    if denom.abs() <= f64::EPSILON * scale {
        return Err(RegulaFalsiError::DegenerateSecantStep { fa, fb, denom });
    }

    Ok(b - fb * (b - a) / denom)
}

/// Calculates the secant intersection point for the line 
/// connecting `(a, fa)` and `(b, fb)` and its function eval
///
/// # Arguments 
/// ├ `(a, fa)` - left endpoint and function value 
/// ├ `(b, fb)` - right endpoint and function value
/// └ `eval`    - function; made by default with finite checks 
///
/// # Returns 
/// ├ `Ok(x_secant)` if denominator `fb - fa` is well-scaled 
/// └ `Err(DegenerateSecantStep)` if denominator is too small. 
///     └ Handled internally. Replaces with a bisection step. 
#[inline] 
fn next_sol_estimate<F>(
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
    eval: &mut F 
) -> Result<(f64, f64), RegulaFalsiError> 
where F: FnMut(f64) -> Result<f64, RegulaFalsiError> { 
    let x = match calculate_secant_x_intercept((a, fa), (b, fb)) { 
        Ok(estimate) if a < estimate && estimate < b => estimate, 
        
        // Replaces with bisection
        Ok(_) | Err(_) => 0.5 * (a + b),
    }; 

    let fx = eval(x)?; 
    Ok((x, fx))
}


/// Finds a root of a function using the ancient 
/// [regula falsi method](https://en.wikipedia.org/wiki/Regula_falsi).
///
/// This method assumes that the function `func` is continuous on the interval `[a, b]`
/// and that `func(a)` and `func(b)` have opposite signs, guaranteeing a root exists
/// within the interval.
///
/// There are four different versions of regula falsi root-finding, each having 
/// a different practical convergence speed:  
/// ├ [`Algorithm::RegulaFalsiPure`]      : converges linearly, but often endpoint stalls.
/// ├ [`Algorithm::RegulaFalsiIllinois`]  : converges linearly, but better than pure and rarely stalls. 
/// ├ [`Algorithm::RegulaFalsiPegasus`]   : converges linearly, but typically faster than illinois. 
/// └ [`Algorithm::RegulaFalsiAndersonBjorck`] : fastest among regula falsi family.
///
/// # Arguments
///
/// ┌ `func` - The function whose root is to be found.
/// ├ `a`    - Lower bound of the search interval. Must be finite and less than `b`.
/// ├ `b`    - Upper bound of the search interval. Must be finite and greater than `a`.
/// └ `cfg`  - Configuration containing optional absolute (`abs_fx`, `abs_x`) and relative 
///            (`rel_x`) tolerances for convergence. See [`RegulaFalsiCfg`] 
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
/// │          └ `algorithm`  : [`Algorithm::algorithm_name()`] 
///               └ "regula_falsi_pure/illinois/pegasus/anderson_bjorck"     
/// ├ `left` : Final left interval bound after convergence.   
/// └ `right`: Final right interval bound after convergence. 
///
/// # Errors
///
/// ┌ [`RegulaFalsiError::InvalidBounds`]        - `a` or `b` is NaN/inf or if `a >= b`.
/// ├ [`RegulaFalsiError::NoSignChange`]         - `func(a)` and `func(b)` do not have opposite signs.
/// │
/// │
/// The following are propagated via [`RegulaFalsiError::Common`]
/// ├ [`RegulaFalsiError::NonFiniteEvaluation`] - `func(x)` produces NaN or inf during evaluation.
/// ├ [`RegulaFalsiError::InvalidAbsFx`]        - `cfg.abs_fx` <= 0 or not finite.
/// ├ [`RegulaFalsiError::InvalidAbsX`]         - `cfg.abs_x` < 0 or not finite.
/// ├ [`RegulaFalsiError::InvalidRelX`]         - `cfg.rel_x` < 0 or not finite.
/// ├ [`RegulaFalsiError::InvalidTolerance`]    - computed interval width tolerance (cfg.abs_x + cfg.rel_x*scale) <= 0 or not finite.
/// └ [`RegulaFalsiError::InvalidMaxIter`]      - `cfg.max_iter` == 0.
/// 
///    Internally Handled 
///    └ [`RegulaFalsiError::DegenerateSecantStep`] - denominator `fb - fa` ~ 0.
///        ├ Replaces with a bisection step.  
///        └ Algorithm won't terminate with this error. 
///
/// # Notes
/// ├ If `max_iter` is `None`/not provided, will use default # iterations from 
/// │ [`Algorithm::default_max_iter()`]
/// └ On early width-tolerance success, `iterations = 0` but the secant point is computed for reporting.
///   This incurs exactly one extra evaluation.
///
/// # Warning 
/// └ Even if `(b - a)` already meets the interval width tolerance, a sign change is still required.
pub fn regula_falsi<F> (
    mut func: F, 
    mut a: f64, 
    mut b: f64, 
    cfg: RegulaFalsiCfg 
) -> Result<RootReport, RegulaFalsiError> 
where F: FnMut(f64) -> f64 { 

    if !(a.is_finite() && b.is_finite()) || a >= b { 
        return Err(RegulaFalsiError::InvalidBounds { a, b }); 
    }

    let cfg = cfg.validate()?;
    let algorithm = cfg.algorithm_in_use();    

    let abs_x       = cfg.abs_x(); 
    let rel_x       = cfg.rel_x(); 
    let abs_fx      = cfg.abs_fx(); 
    let max_iter    = cfg.max_iter(); 
    let width_tol0  = width_tol_current(a, b, abs_x, rel_x);

    let num_iter = match max_iter { 
        Some(m) => m,
        None    => cfg.algorithm.default_max_iter(), 
    };

    // number of function evaluations 
    let mut evals = 0; 

    // closure function, checks finiteness 
    let mut eval = |x: f64| -> Result<f64, RegulaFalsiError> { 
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
                algorithm   : algorithm,
            },
            left        : a,
            right       : b,
        }); 
    }
    let mut fb = eval(b)?; 
    if fb.abs() <= abs_fx { 
        return Ok(RootReport::Bracket {
            meta: RootMeta { 
                root        : b,
                f_root      : fb,
                iterations  : 0,
                evals       : evals,
                termination : Termination::ToleranceReached,
                tolerance   : ToleranceReason::AbsFxReached,
                algorithm   : algorithm,
            }, 
            left  : a,
            right : b,
        });
    }

    if !opposite_signs(fa, fb) { 
        return Err(RegulaFalsiError::NoSignChange { a, b }); 
    }

    // immediate narrow width success 
    if b - a <= width_tol0 { 
        let (sol_estimate, fsol) = next_sol_estimate((a, fa), (b, fb), &mut eval)?;
        return Ok(RootReport::Bracket { 
            meta: RootMeta { 
                root        : sol_estimate,
                f_root      : fsol,
                iterations  : 0,
                evals       : evals,
                termination : Termination::ToleranceReached,
                tolerance   : ToleranceReason::WidthTolReached,
                algorithm   : algorithm,
            }, 
            left  : a,
            right : b,
        }); 
    }

    // algorithm 
    let mut sol_estimate = a;     // gets overwritten
    let mut fsol         = fa;    // gets overwritten 
    
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Side { Left, Right } 
    let mut last_side: Option<Side> = None; 

    let variant = cfg.algorithm();
    
    for iter in 1..=num_iter { 
        (sol_estimate, fsol) = next_sol_estimate((a, fa), (b, fb), &mut eval)?; 
        
        // check for abs fx tolerance 
        if fsol.abs() <= abs_fx {
            return Ok(RootReport::Bracket {
                meta: RootMeta { 
                    root        : sol_estimate,
                    f_root      : fsol,
                    iterations  : iter,
                    evals       : evals,
                    termination : Termination::ToleranceReached,
                    tolerance   : ToleranceReason::AbsFxReached,
                    algorithm   : algorithm,
                }, 
                left  : a,
                right : b,
            }); 
        }

        // shrink interval 
        // perform endpoint updates based on RegulaFalsiVariant
        if opposite_signs(fa, fsol) {
            match (variant, last_side) { 
                (Algorithm::RegulaFalsiPure, Some(Side::Right))           => {
                    // do nothing, leave fa/fb unchanged
                }, 
                (Algorithm::RegulaFalsiIllinois, Some(Side::Right))       => { 
                    // half retained endpoint
                    fa *= 0.5; 
                }, 
                (Algorithm::RegulaFalsiPegasus,  Some(Side::Right))       => { 
                    // scale retained endpoint adaptively 
                    let denom = fsol + fb; 
                    let scale = if denom.abs() <= f64::EPSILON * (fsol.abs() + fb.abs()).max(1.0) {
                        // fallback to illinois algorithm to avoid blowup
                        0.5 
                    } else { 
                        fsol / denom 
                    };
                    fa *= scale 
                }, 
                (Algorithm::RegulaFalsiAndersonBjorck, Some(Side::Right)) => {
                    // more aggressive scaling than pegasus 
                    let denom = fb; 
                    let ratio = if denom.abs() <= f64::EPSILON * (fsol.abs() + fb.abs()).max(1.0) { 
                        0.5 
                    } else { 
                        1.0 - fsol / denom
                    }; 

                    if ratio <= 0.0 { 
                        fa *= 0.5; 
                    } else { 
                        fa *= ratio; 
                    }
                },
                _ => {}
            }

            b  = sol_estimate; 
            fb = fsol;

            last_side = Some(Side::Right);
        } else { 
            match (variant, last_side) {
                (Algorithm::RegulaFalsiPure, Some(Side::Left))           => {
                    // do nothing, leave fa/fb unchanged
                }, 
                (Algorithm::RegulaFalsiIllinois, Some(Side::Left))       => { 
                    // half retained endpoint
                    fb *= 0.5; 
                }, 
                (Algorithm::RegulaFalsiPegasus,  Some(Side::Left))       => { 
                    // scale retained endpoint adaptively 
                    let denom = fsol + fa; 
                    let scale = if denom.abs() <= f64::EPSILON * (fsol.abs() + fa.abs()).max(1.0) {
                        // fallback to illinois algorithm to avoid blowup
                        0.5 
                    } else { 
                        fsol / denom 
                    };
                    fb *= scale 
                }, 
                (Algorithm::RegulaFalsiAndersonBjorck, Some(Side::Left)) => {
                    // more aggressive scaling than pegasus 
                    let denom = fa; 
                    let ratio = if denom.abs() <= f64::EPSILON * (fsol.abs() + fa.abs()).max(1.0) { 
                        0.5 
                    } else { 
                        1.0 - fsol / denom
                    }; 

                    if ratio <= 0.0 { 
                        fb *= 0.5; 
                    } else { 
                        fb *= ratio; 
                    }
                },
                _ => {}
            }

            a  = sol_estimate; 
            fa = fsol;

            last_side = Some(Side::Left); 
        }

        // check for interval width tolerance 
        if b - a <= width_tol_current(a, b, abs_x, rel_x) {
            let (sol_estimate, fsol) = next_sol_estimate(
                (a, fa), 
                (b, fb),
                &mut eval
            )?;
            return Ok(RootReport::Bracket {
                meta: RootMeta { 
                    root        : sol_estimate,
                    f_root      : fsol,
                    iterations  : iter,
                    evals       : evals,
                    termination : Termination::ToleranceReached,
                    tolerance   : ToleranceReason::WidthTolReached,
                    algorithm   : algorithm,
                }, 
                left  : a,
                right : b,
            }); 
        }
    }

    Ok(RootReport::Bracket {
        meta: RootMeta { 
            root        : sol_estimate,
            f_root      : fsol, 
            iterations  : num_iter,
            evals       : evals,
            termination : Termination::IterationLimit,
            tolerance   : ToleranceReason::ToleranceNotReached,
            algorithm   : algorithm,
        }, 
        left  : a,
        right : b,
    })
}

