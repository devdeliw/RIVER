//! Defines the [`RootFindingReport`] struct returned by all 
//! root-finding algorithms. 

/// Reasons a root-finding algorithm may terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] 
pub enum TerminationReason { 
    ToleranceReached, 
    IterationLimit,
    MachinePrecisionReached, 
}


/// Which tolerance condition was satisfied (or not).
/// - [`ToleranceSatisfied::AbsFxReached`]    
///     - All methods 
///     - |f(x)| <= tol
/// - [`ToleranceSatisfied::WidthTolReached`]
///     - [`algorithms::Algorithm::Bracket`] 
///     - [a, b] -> (b - a).abs() <= tol 
/// - [`ToleranceSatisfied::StepSizeReached`]
///     - [`algorithms::Algorithm::Open`] 
///     - x_n - x_{n - 1} <= tol 
///   [`ToleranceSatisfied::ToleranceNotReached`] 
///     - All methods 
///     - Tolerance not reached, usually with [`TerminationReason::IterationLimit`] 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceSatisfied { 
    AbsFxReached, 
    WidthTolReached, 
    StepSizeReached, 
    ToleranceNotReached
}


/// Method-specific data returned by a solver. 
/// Contains the last set of points used in the update formula. 
/// - [`Stencil::Bracket`] : bracketing methods  
///     - `left`, `right` bounds of the final interval  
/// - [`Stencil::Open`]    : open methods  
///     - `x` = last iterate used to compute the root  
#[derive(Debug, Copy, Clone)]
pub enum Stencil { 
    Bracket { bounds: [f64; 2] },
    Open    { x: [f64; 3], len: usize }, 
}
impl Stencil { 
    pub fn stencil(&self) -> &[f64] { 
        match self { 
            Stencil::Bracket { bounds } => &bounds[..],
            Stencil::Open { x, len }    => &x[..*len],
        }
    }
    pub fn singleton(x: f64) -> Self { 
        Stencil::Open { x: [x, 0.0, 0.0], len: 1 }
    }
    pub fn doubleton(x1: f64, x2: f64) -> Self { 
        Stencil::Open { x: [x1, x2, 0.0], len: 2}
    }
}


/// Final report returned by all root-finding algorithms.  
/// 
/// [`RootFindingReport`]
/// - `root`                : best root estimate  
/// - `f_root`              : function value at `root`  
/// - `iterations`          : total iterations  
/// - `evaluations`         : total function evaluations  
/// - `termination_reason`  : why the solver stopped  ([`TerminationReason`])  
/// - `tolerance_satisfied` : which tolerance was met ([`ToleranceSatisfied`]) 
/// - `stencil`             : last set of points used in update formula    
/// - `algorithm_name`      : algorithm name (e.g. `"bisection"`)  
#[derive(Debug, Copy, Clone)] 
pub struct RootFindingReport {
    pub root                : f64, 
    pub f_root              : f64, 
    pub iterations          : usize, 
    pub evaluations         : usize, 
    pub termination_reason  : TerminationReason, 
    pub tolerance_satisfied : ToleranceSatisfied, 
    pub stencil             : Stencil, 
    pub algorithm_name      : &'static str, 
}
