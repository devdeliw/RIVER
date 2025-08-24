use super::algorithms::{Algorithm, BracketFamily, GLOBAL_MAX_ITER_FALLBACK}; 
use super::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil}; 
use super::tolerances::DynamicTolerance; 
use super::signs::{opposite_sign, same_sign};
use super::errors::{RootFindingError, ToleranceError, AlgorithmError}; 
use super::config::{CommonCfg, impl_common_cfg};
use thiserror::Error;


#[derive(Debug, Error)]
pub enum RegulaFalsiError { 
    #[error(transparent)] 
    RootFinding(#[from] RootFindingError), 

    #[error(transparent)] 
    Tolerance(#[from] ToleranceError),

    #[error("no sign change on [{a}, {b}]: f(a) * f(b) > 0")]
    NoSignChange  { a: f64, b: f64 },

    #[error("invalid bounds: a and b must be finite with a < b. got [{a}, {b}]")] 
    InvalidBounds { a: f64, b: f64 }, 

    #[error("denominator |fb - fa| near 0 ")]
    DegenerateSecantStep
}


/// RegulaFalsi configuration 
///
/// # Fields 
/// ├ `common`  : [`CommonCfg`] with tolerances and optional `max_iter`. 
/// └ `variant` : [`BracketFamily`] specfying which regula falsi variant.  
///
/// # Construction 
/// ├ Use [`RegulaFalsiCfg::new`] then optional setters from [`impl_common_cfg`]. 
/// └ Set an explicit step cap via [`RegulaFalsiCfg::set_variant`].
///
/// # Defaults 
/// ├ `variant` is by default [`BracketFamily::RegulaFalsiIllinois`]
/// └ If `common.max_iter` is `None`, [`regula_falsi`] resolves it using 
///   [`Algorithm::default_max_iter`] for the associated `variant`, or 
///   [`GLOBAL_MAX_ITER_FALLBACK`] if unavailable. 
#[derive(Debug, Copy, Clone)] 
pub struct RegulaFalsiCfg { 
    common  : CommonCfg, 
    variant : BracketFamily
}
impl RegulaFalsiCfg { 
    pub fn new () -> Self { 
        Self { 
            common  : CommonCfg::new(),
            variant : BracketFamily::RegulaFalsiIllinois
        }
    }

    pub fn set_variant(
        mut self, 
        variant: BracketFamily
    ) -> Result<Self, AlgorithmError> { 
        let v = match variant { 
            BracketFamily::RegulaFalsiPure 
            | BracketFamily::RegulaFalsiIllinois 
            | BracketFamily::RegulaFalsiPegasus 
            | BracketFamily::RegulaFalsiAndersonBjorck => variant, 
            _ => { 
                return Err(AlgorithmError::IncompatibleAlgorithm { 
                    algorithm: Algorithm::Bracket(variant) 
                }) 
            }
        }; 

        self.variant = v;
        Ok(self)
    }
}
impl_common_cfg!(RegulaFalsiCfg); 


/// Calculates the secant x-intersection point for the line 
/// connecting `(a, fa)` and `(b, fb)` 
///
/// # Arguments 
/// ├ `(a, fa)` - left endpoint and function value 
/// └ `(b, fb)` - right endpoint and function value
///
/// # Returns 
/// ├ `Ok(x_secant)` if denominator `fb - fa` is well-scaled 
/// └ `Err(RegulaFalsiError::DegenerateSecantStep)` if denominator is too small. 
///    this is *handled internally* in [`next_sol_estimate`]; replaces with bisection. 
#[inline]
fn secant_x_intercept(
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
)-> Result<f64, RegulaFalsiError> {
    let denom = fb - fa;
    let scale = fa.abs().max(fb.abs()).max(1.0);

    if denom.abs() <= f64::EPSILON * scale {
        return Err(RegulaFalsiError::DegenerateSecantStep);
    }

    Ok(b - fb * (b - a) / denom)
}


/// Calculates the secant intersection point for the line 
/// connecting `(a, fa)` and `(b, fb)` and its function value.
///
/// # Arguments 
/// ├ `(a, fa)` - left endpoint and function value 
/// ├ `(b, fb)` - right endpoint and function value
/// └ `eval`    - function; made by default with finite checks 
///
/// # Returns 
/// ├ `Ok(x_secant)` if denominator `fb - fa` is well-scaled otherwise 
/// │ `Ok(midpoint)` having defaulted to a bisection.
/// └ `Err(RegulaFalsiError::RootFinding(RootFindingError::NonFiniteEvaluation))` 
///    if the function evaluation is non-finite. 
#[inline] 
fn next_sol_estimate<F> (
    (a, fa): (f64, f64), 
    (b, fb): (f64, f64), 
    eval: &mut F 
) -> Result<(f64, f64), RegulaFalsiError> 
where F: FnMut(f64) -> Result<f64, RegulaFalsiError> { 
    let x = match secant_x_intercept((a, fa), (b, fb)) { 
        Ok(estimate) if a < estimate && estimate < b => estimate,  
        // default to bisection
        Ok(_) | Err(_) => 0.5 * (a + b),
    }; 
    let fx = eval(x)?; 

    Ok((x, fx))
}


/// Finds a root of a function using the ancient 
/// [regula falsi method](https://en.wikipedia.org/wiki/Regula_falsi).
///
/// This method assumes that the function is continuous on the interval `[a, b]`
/// and that `f(a)` and `f(b)` have opposite signs, guaranteeing a root exists
/// within the interval.
///
/// Four variants of regula falsi root-finding are available, differing
/// in convergence behavior and how they rescale the retained endpoint:  
/// ├ [`BracketFamily::RegulaFalsiPure`]          
/// │   ├ Linear convergence; stalls often. 
/// │   └ `f_same` unchanged.  
/// │
/// ├ [`BracketFamily::RegulaFalsiIllinois`]      
/// │   ├ Linear convergence; rarely stalls.  
/// │   └ `f_same *= 0.5`  
/// │
/// ├ [`BracketFamily::RegulaFalsiPegasus`]       
/// │   ├ Linear convergence; typically faster than Illinois.  
/// │   └ `f_same *= f_other / (f_other + f_new)`  
/// │
/// └ [`BracketFamily::RegulaFalsiAndersonBjorck`] 
///     ├ Fastest among the family.  
///     └ `f_same *= max(0.5, 1 - f_new / f_other)`  
///
/// where `f_same` is the function value of the consecutively retained endpoint,  
///       `f_other` is the function value of the replaced endpoint,  
///       `f_new` is the function value of the new solution estimate.
///
/// # Arguments
/// ├ `func` : The function whose root is to be found.
/// ├ `a`    : Lower bound of the search interval. Must be finite and less than `b`.
/// ├ `b`    : Upper bound of the search interval. Must be finite and greater than `a`.
/// └ `cfg`  : [`RegulaFalsiCfg`] (tolerances, optional `max_iter`, optional `variant`).  
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
/// └ `algorithm_name`      : "regula_falsi_{pure/illinois/pegasus/anderson_bjorck}"
//
/// # Errors
/// ├ [`RegulaFalsiError::InvalidBounds`]       : `a` or `b` is NaN/inf or `a >= b`.
/// ├ [`RegulaFalsiError::NoSignChange`]        : `f(a)` and `f(b)` are same sign.
/// │
/// * Propagated via [`RegulaFalsiError::RootFinding`]
/// ├ [`RootFindingError::NonFiniteEvaluation`] : `f(x)` NaN/inf.
/// ├ [`RootFindingError::InvalidMaxIter`]      : `max_iter` = 0
/// │
/// * Propagated via [`RegulaFalsiError::Tolerance`] 
/// ├ [`ToleranceError::InvalidAbsFx`]          : `abs_fx` <= 0.0 or inf 
/// ├ [`ToleranceError::InvalidAbsX`]           : `abs_x`  <  0.0 or inf 
/// ├ [`ToleranceError::InvalidRelX`]           : `rel_x`  <  0.0 or inf 
/// └ [`ToleranceError::InvalidAbsRelX`]        : one of `abs_x` and `rel_x` not > 0.
///
/// # Behavior
/// ├ Update:
/// │   └ x_new = (x0*f(x1) - x1*f(x0)) / (f(x1) - f(x0)), maintaining a bracket
/// │     where f(x0)*f(x1) < 0.
/// │
/// ├ Tolerances:
/// │   ├ |f(x)| <= abs_fx return with [`ToleranceSatisfied::AbsFxReached`]
/// │   └ |x_new - x_old|  return with [`ToleranceSatisfied::StepSizeReached`] 
/// │
/// ├ Stencil: 
/// │   └ stores the bracket [a, b] that produced the calculated root. 
/// │
/// └ Variants:
///     ├ Pure            : standard false-position, can stagnate if one endpoint never moves
///     ├ Illinois        : halves stagnant endpoint’s function value
///     ├ Pegasus         : rescales stagnant endpoint by f(new)/[f(new)+f(old)]
///     └ Anderson-Bjorck : adaptive blend, avoids both stagnation and oscillation
///
/// # Notes
/// ├ Regula Falsi is globally convergent like bisection but faster in practice.
/// └ Pure method may stall; Illinois, Pegasus, and Anderson-Bjorck are fixes.
pub fn regula_falsi<F> (
    mut func: F, 
    mut a: f64, 
    mut b: f64, 
    cfg: RegulaFalsiCfg 
) -> Result<RootFindingReport, RegulaFalsiError> 
where F: FnMut(f64) -> f64 { 

    if !(a.is_finite() && b.is_finite()) || a >= b { 
        return Err(RegulaFalsiError::InvalidBounds { a, b }); 
    } 

    let abs_x     = cfg.common.abs_x();
    let rel_x     = cfg.common.rel_x(); 
    let abs_fx    = cfg.common.abs_fx(); 
    let max_iter  = cfg.common.max_iter(); 
    let variant   = cfg.variant;
    let algorithm = Algorithm::Bracket(variant); 

    let num_iter = match max_iter {
        // already validated via building config; redundant guard
        Some(0) => return Err(RootFindingError::InvalidMaxIter { got: 0 }.into()),

        Some(v) => v,
        None    => algorithm.default_max_iter().unwrap_or(GLOBAL_MAX_ITER_FALLBACK)
    };

    // track function evaluations 
    let mut evals = 0; 

    // wraps func, increments evals, enforces finiteness
    let mut eval = |x: f64| -> Result<f64, RegulaFalsiError> { 
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
            algorithm_name      : algorithm.algorithm_name() 
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
            stencil             : Stencil::Bracket { bounds: [a, b] }, 
            algorithm_name      : algorithm.algorithm_name() 
        });
    }

    // error: no sign change across [a, b]
    if same_sign(fa, fb) { 
        return Err(RegulaFalsiError::NoSignChange { a, b }); 
    }

    let mut dynamic_tol = DynamicTolerance::WidthTol { a, b }; 
    let mut width_tol   = algorithm.calculate_tolerance(&dynamic_tol, abs_x, rel_x)?;

    // width tolerance already satisfied 
    if b - a <= width_tol { 
        let (sol_estimate, fsol) = next_sol_estimate((a, fa), (b, fb), &mut eval)?;
        return Ok(RootFindingReport {
            root                : sol_estimate, 
            f_root              : fsol, 
            iterations          : 0, 
            evaluations         : evals,
            termination_reason  : TerminationReason::ToleranceReached, 
            tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
            stencil             : Stencil::Bracket { bounds: [a, b] }, 
            algorithm_name      : algorithm.algorithm_name() 
        });
    }

    // main loop  
    let mut sol_estimate = a;     
    let mut fsol         = fa;     

    enum Side { Left, Right } 
    let mut last_side: Option<Side> = None; 

    for iter in 1..=num_iter { 
        (sol_estimate, fsol) = next_sol_estimate((a, fa), (b, fb), &mut eval)?; 
        
        // check |f(x)| tolerance 
        if fsol.abs() <= abs_fx {
            return Ok(RootFindingReport {
                root                : sol_estimate, 
                f_root              : fsol, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::AbsFxReached,
                stencil             : Stencil::Bracket { bounds: [a, b] }, 
                algorithm_name      : algorithm.algorithm_name() 
            });
        }

        // updated bracket 
        // method-specific 
        if opposite_sign(fa, fsol) {
            match (variant, last_side) { 
                (BracketFamily::RegulaFalsiPure, Some(Side::Right))           => {
                    // do nothing, fa/fb unchanged
                }, 
                (BracketFamily::RegulaFalsiIllinois, Some(Side::Right))       => { 
                    // half retained endpoint
                    fa *= 0.5; 
                }, 
                (BracketFamily::RegulaFalsiPegasus,  Some(Side::Right))       => { 
                    // scale retained endpoint adaptively 
                    let denom = fsol + fb; 
                    let scale = if denom.abs() <= f64::EPSILON * (fsol.abs() + fb.abs()).max(1.0) {
                        // fallback to illinois algorithm to avoid blowup
                        0.5 
                    } else { 
                        fb / denom 
                    };
                    fa *= scale 
                }, 
                (BracketFamily::RegulaFalsiAndersonBjorck, Some(Side::Right)) => {
                    // more aggressive scaling than pegasus 
                    let denom = fb; 
                    let ratio = if denom.abs() <= f64::EPSILON * (fsol.abs() + fb.abs()).max(1.0) { 
                        0.5 
                    } else { 
                        (1.0 - fsol / denom).max(0.5)
                    }; 
                    fa *= ratio; 
                },
                _ => {}
            }

            b  = sol_estimate; 
            fb = fsol;
            last_side = Some(Side::Right);

        } else { 
            match (variant, last_side) {
                (BracketFamily::RegulaFalsiPure, Some(Side::Left))           => {
                    // do nothing, leave fa/fb unchanged
                }, 
                (BracketFamily::RegulaFalsiIllinois, Some(Side::Left))       => { 
                    // half retained endpoint
                    fb *= 0.5; 
                }, 
                (BracketFamily::RegulaFalsiPegasus,  Some(Side::Left))       => { 
                    // scale retained endpoint adaptively 
                    let denom = fsol + fa; 
                    let scale = if denom.abs() <= f64::EPSILON * (fsol.abs() + fa.abs()).max(1.0) {
                        // fallback to illinois algorithm to avoid blowup
                        0.5 
                    } else { 
                        fa / denom 
                    };
                    fb *= scale 
                }, 
                (BracketFamily::RegulaFalsiAndersonBjorck, Some(Side::Left)) => {
                    // more aggressive scaling than pegasus 
                    let denom = fa; 
                    let ratio = if denom.abs() <= f64::EPSILON * (fsol.abs() + fa.abs()).max(1.0) { 
                        0.5 
                    } else { 
                        (1.0 - fsol / denom).max(0.5)
                    }; 
                    fb *= ratio; 
                },
                _ => {}
            }

            a  = sol_estimate; 
            fa = fsol;
            last_side = Some(Side::Left); 
        }

        // check width tolerance 
        dynamic_tol = DynamicTolerance::WidthTol { a, b };
        width_tol   = Algorithm::Bracket(variant).calculate_tolerance(&dynamic_tol, abs_x, rel_x)?;
        if b - a <= width_tol {
            let (sol_estimate, fsol) = next_sol_estimate(
                (a, fa), 
                (b, fb),
                &mut eval
            )?;
            return Ok(RootFindingReport {
                root                : sol_estimate, 
                f_root              : fsol, 
                iterations          : iter, 
                evaluations         : evals,
                termination_reason  : TerminationReason::ToleranceReached, 
                tolerance_satisfied : ToleranceSatisfied::WidthTolReached,
                stencil             : Stencil::Bracket { bounds: [a, b] }, 
                algorithm_name      : algorithm.algorithm_name() 
            });

        }
    }

    Ok(RootFindingReport {
        root                : sol_estimate, 
        f_root              : fsol, 
        iterations          : num_iter, 
        evaluations         : evals,
        termination_reason  : TerminationReason::IterationLimit, 
        tolerance_satisfied : ToleranceSatisfied::ToleranceNotReached,
        stencil             : Stencil::Bracket { bounds: [a, b] }, 
        algorithm_name      : algorithm.algorithm_name() 
    })
}

