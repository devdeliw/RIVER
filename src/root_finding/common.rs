use thiserror::Error;

/// All algorithms for root-finding. 
#[derive(Debug, Copy, Clone)]
pub enum Algorithm {
    Bisection,
    RegulaFalsiPure,
    RegulaFalsiIllinois,
    RegulaFalsiPegasus,
    RegulaFalsiAndersonBjorck,
    Secant,
}
impl Algorithm {
    /// Hard cap on automatically computed iteration counts for methods.
    ///
    /// # Notes
    /// ├ Only applied when `max_iter` is `None`.
    /// ├ Values are method-specific heuristics for reasonable convergence.
    /// └ Methods with theoretical iteration limits (e.g. [`Algorithm::Bisection`])  
    ///   return `0` here, meaning "compute theoretical bound instead". 
    ///     ├ If the theoretical limit is too large practically, 
    ///     └ [`DEFAULT_MAX_ITER_FALLBACK`] fallback is used instead.
    pub const fn default_max_iter(self) -> usize {
        match self {
            Algorithm::Bisection                 => 0,   // theoretical limit
            Algorithm::RegulaFalsiPure           => 200,
            Algorithm::RegulaFalsiIllinois       => 100,
            Algorithm::RegulaFalsiPegasus        => 100,
            Algorithm::RegulaFalsiAndersonBjorck => 100,
            Algorithm::Secant                    => 100,
        }
    }

    /// Algorithm names for the [`RootMeta::algorithm`] field for each method.  
    pub const fn algorithm_name(self) -> &'static str {
        match self {
            Algorithm::Bisection                 => "bisection",
            Algorithm::RegulaFalsiPure           => "regula_falsi_pure",
            Algorithm::RegulaFalsiIllinois       => "regula_falsi_illinois",
            Algorithm::RegulaFalsiPegasus        => "regula_falsi_pegasus",
            Algorithm::RegulaFalsiAndersonBjorck => "regula_falsi_anderson_bjorck",
            Algorithm::Secant                    => "secant",
        }
    }
}
impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.algorithm_name())
    }
}

/// Global hard cap on iterations, applied if a method’s theoretical
/// or heuristic default would otherwise exceed this value.
///
pub const GLOBAL_MAX_ITER_FALLBACK: usize = 500;


/// Shared metadata returned by all root-finding algorithms.
/// Contains the universal outcome fields of a solver run.
#[derive(Debug, Clone, Copy)]
pub struct RootMeta {
    pub root: f64,                  // root approximation 
    pub f_root: f64,                // f(root)
    pub iterations: usize,          // # iterations performed 
    pub evals: usize,               // # function evaluations
    pub termination: Termination,   // reason why algorithm stopped 
    pub tolerance: ToleranceReason, // which tolerance criterion was satisifed (or not) 
    pub algorithm: &'static str,    // algorithm name (e.g. "bisection", "secant", ...)
}

/// Summarizes the outcome of a root-finding run.
///
/// Final report enum returned by all root-finding algorithms.
/// Each variant contains [`RootMeta`] with the common fields
/// (root, residual, iterations, evals, termination, tolerance, algorithm),
/// plus method-specific fields.
///
/// Variants:
/// ├ [`RootReport::Bracket`]   
/// │   Bracketing methods (e.g. bisection, regula falsi)  
/// │   ├ `meta`   : [`RootMeta`] (common fields)  
/// │   ├ `left`   : Final left bound of the bracketing interval.  
/// │   └ `right`  : Final right bound of the bracketing interval.  
/// │
/// ├ [`RootReport::OnePoint`]  
/// │   One-point methods (e.g. Newton)  
/// │   ├ `meta`   : [`RootMeta`] (common fields)  
/// │   └ `point`  : Final iterate xₙ that gave the result.  
/// │
/// ├ [`RootReport::TwoPoint`]  
/// │   Two-point methods (e.g. secant)  
/// │   ├ `meta`   : [`RootMeta`] (common fields)  
/// │   ├ `curr`   : Final iterate xₙ.  
/// │   └ `prev`   : Previous iterate xₙ₋₁.  
/// │
/// └ [`RootReport::MultiPoint`]  
///     Multi-point methods (e.g. inverse quadratic interpolation)  
///     ├ `meta`    : [`RootMeta`] (common fields)  
///     └ `points`  : Recent sequence of points contributing to result.  
///
/// # Common Fields in [`RootMeta`]
/// ├ `root`        : The computed root approximation.  
/// ├ `f_root`      : Value of f(root).  
/// ├ `iterations`  : Total number of iterations performed.  
/// ├ `evals`       : Total number of function evaluations.  
/// ├ `termination` : [`Termination`] reason for stopping.  
/// └ `tolerance`   : [`ToleranceReason`] satisfied (or not).  
#[derive(Debug)]
pub enum RootReport {
    Bracket {
        meta: RootMeta,
        /// Final left bound of bracketing interval.
        left: f64,
        /// Final right bound of bracketing interval.
        right: f64,
    },

    OnePoint {
        meta: RootMeta,
        /// Final iterate x_n that yielded the result.
        point: f64,
    },

    TwoPoint {
        meta: RootMeta,
        /// Parent iterate that produced root x_n (x_{n-1})
        parent1: f64,
        /// Parent iterate that produced root x_n (x_{n-2})
        parent2: f64,
    },

    MultiPoint {
        meta: RootMeta,
        /// Recent sequence of points before calculated root.
        points: Vec<f64>,
    },
}

/// # Common accessors
///
/// These methods are available for all solver variants.
/// ├ [`RootReport::termination()`] : [`Termination`]
/// ├ [`RootReport::tolerance()`]   : [`ToleranceReason`]
/// ├ [`RootReport::root()`]        : `f64`
/// ├ [`RootReport::f_root()`]      : `f64`
/// ├ [`RootReport::iterations()`]  : `usize`
/// ├ [`RootReport::evals()`]       : `usize`
/// └ [`RootReport::algorithm()`]   : `&'static str`
/// # Variant-specific accessors
///
/// These return `Option<T>` because the field only exists
/// for some variants.  
///
/// [`RootReport::Bracket`]
/// └ [`RootReport::left()`], [`RootReport::right()`] : Option<f64>  
/// [`RootReport::OnePoint`]
/// └ [`RootReport::point()`]                         : Option<f64> 
/// [`RootReport::TwoPoint`]
/// └ [`RootReport::curr()`], [`RootReport::prev()`]  : Option<f64> 
/// [`RootReport::MultiPoint`] 
/// └ [`RootReport::points()`]                        : Option<Vec<f64>>  
///
/// ## Warning: For non-matching variants they return `None`.
impl RootReport {
    /// Access shared [`RootMeta`] directly.
    pub fn meta(&self) -> &RootMeta {
        match self {
            RootReport::Bracket { meta, .. } => meta,
            RootReport::OnePoint { meta, .. } => meta,
            RootReport::TwoPoint { meta, .. } => meta,
            RootReport::MultiPoint { meta, .. } => meta,
        }
    }

    /// Computed root approximation.
    pub fn root(&self) -> f64 {
        self.meta().root
    }

    /// Function value at the computed root.
    pub fn f_root(&self) -> f64 {
        self.meta().f_root
    }

    /// Number of iterations performed.
    pub fn iterations(&self) -> usize {
        self.meta().iterations
    }

    /// Number of function evaluations.
    pub fn evals(&self) -> usize {
        self.meta().evals
    }

    /// Why the algorithm stopped.
    pub fn termination(&self) -> Termination {
        self.meta().termination
    }

    /// Tolerance criteria that was satisfied (or not).
    pub fn tolerance(&self) -> ToleranceReason {
        self.meta().tolerance
    }

    /// Algorithm name (e.g., "bisection", "secant").
    pub fn algorithm(&self) -> &'static str {
        self.meta().algorithm
    }

    pub fn left(&self) -> Option<f64> {
        match self {
            RootReport::Bracket { left, .. } => Some(*left),
            _ => None,
        }
    }

    pub fn right(&self) -> Option<f64> {
        match self {
            RootReport::Bracket { right, .. } => Some(*right),
            _ => None,
        }
    }

    pub fn parent1(&self) -> Option<f64> {
        match self {
            RootReport::TwoPoint { parent1, .. } => Some(*parent1),
            _ => None,
        }
    }

    pub fn parent2(&self) -> Option<f64> {
        match self {
            RootReport::TwoPoint { parent2, .. } => Some(*parent2),
            _ => None,
        }
    }

    pub fn points(&self) -> Option<&[f64]> {
        match self {
            RootReport::MultiPoint { points, .. } => Some(points),
            _ => None,
        }
    }

    pub fn point(&self) -> Option<f64> {
        match self {
            RootReport::OnePoint { point, .. } => Some(*point),
            _ => None,
        }
    }
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

    #[error("invalid algorithm: got {algorithm}")]
    InvalidAlgorithm { algorithm: Algorithm }
}


/// Termination variants for root-finding algorithms after completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    ToleranceReached,
    IterationLimit,
    Stagnation  
}


/// Tolerance variants for root-finding algorithms after completion.
/// ├ [`ToleranceReason::AbsFxReached`]    
/// │   ├ All methods 
/// │   └ |f(x)| <= tol
/// │
/// ├ [`ToleranceReason::WidthTolReached`]
/// │   ├ [`RootReport::Bracket`] 
/// │   └ [a, b] -> (b - a).abs() <= tol 
/// │
/// ├ [`ToleranceReason::StepSizeReached`]
/// │   ├ [`RootReport::TwoPoint`], [`RootReport::MultiPoint`]
/// │   └ x_n - x_{n - 1} <= tol 
/// │
/// └ [`ToleranceReason::ToleranceNotReached`] 
///     ├ All methods 
///     └ Tolerance was not reached, usually alongside [`Termination::IterationLimit`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceReason { 
    AbsFxReached, 
    WidthTolReached,
    StepSizeReached,
    ToleranceNotReached, 
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
///
/// Returns:
/// ├ `Ok(theoretical_iters)` - theoretical # bisections to satisfy given width tol 
/// └ `Err(RootFindingError::InvalidTolerance)` - width_tol <= 0 or non-finite. 
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
/// `abs_x + rel_x * max(max(|a|, |b|), 1.0)`
///
/// Ensures the relative scale is never below 1.0 to avoid tiny tolerances
/// near zero. Used by all bracketing algorithms for consistent checks.
///
/// Only used for Bracketing methods for root-finding.
pub(crate) fn width_tol_current(a: f64, b: f64, abs_x: f64, rel_x: f64) -> f64 { 
    abs_x + rel_x * a.abs().max(b.abs()).max(1.0)
}

/// Combined absolute + relative tolerance.
///
/// `abs_x + rel_x * |x|` 
///
/// Only used for Open methods for root-finding. 
pub(crate) fn step_tol_current(x: f64, abs_x: f64, rel_x: f64) -> f64 { 
    abs_x + rel_x * x.abs()
}


/// Determines whether `u` and `v` are of opposite sign. 
///
/// Used for two-bracketing algorithms to determine which direction 
/// to shrink the interval/bracket by, or if there is an initial 
/// error when sign(f(a)) == sign(f(b))
pub(crate) fn opposite_signs(u: f64, v: f64) -> bool {
    (u.is_sign_positive()) != (v.is_sign_positive())
}
