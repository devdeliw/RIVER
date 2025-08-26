//! Root-finding algorithm definitions.  
//!
//! Provides the [`Algorithm`] enum, which enumerates all supported methods, 
//! along with shared [`GLOBAL_MAX_ITER_FALLBACK`] hard cap.  


/// Most methods use heuristic defaults from [`Algorithm::default_max_iter`]. 
/// This cap is only applied when a bracket algorithm’s theoretical iteration bound 
/// would otherwise exceed it (e.g. [`BracketFamily::Bisection`]).  
///
/// Serves as a practical safeguard against iteration counts that are 
/// mathematically valid but computationally excessive.
pub const GLOBAL_MAX_ITER_FALLBACK: usize = 500; 


/// Root-finding algorithm variants. 
/// - [`Algorithm::Bracket`] contains bracket methods for root-finding 
/// - [`Algorithm::Open`]    contains open methods for root-finding 
#[derive(Debug, Copy, Clone)]
pub enum Algorithm { 
    Bracket(BracketFamily), 
    Open(OpenFamily),
    Compound(CompoundFamily)
}

#[derive(Debug, Copy, Clone)]
pub enum BracketFamily { 
    Bisection,
    RegulaFalsiPure, 
    RegulaFalsiIllinois, 
    RegulaFalsiPegasus, 
    RegulaFalsiAndersonBjorck,
}

#[derive(Debug, Copy, Clone)]
pub enum OpenFamily { 
    Secant, 
    Newton
}

#[derive(Debug, Copy, Clone)] 
pub enum CompoundFamily { 
    Brent 
}

impl Algorithm { 
    /// Default iteration count if `max_iter` is unset in config. 
    ///  
    /// # Notes 
    /// - Applied only when `max_iter` is unset.  
    /// - Values are heuristic and method-specific.  
    /// - Methods with theoretical bounds (e.g. [`BracketFamily::Bisection`]) 
    ///   return `None`, meaning “compute theoretical bound instead”.  
    ///   - If that bound exceeds practical limits, 
    ///     [`GLOBAL_MAX_ITER_FALLBACK`] is used.  
    pub const fn default_max_iter(self) -> Option<usize> { 
        match self { 
            Algorithm::Bracket(BracketFamily::Bisection)                   => None, 
            Algorithm::Bracket(BracketFamily::RegulaFalsiPure)             => Some(200), 
            Algorithm::Bracket(BracketFamily::RegulaFalsiIllinois) 
            | Algorithm::Bracket(BracketFamily::RegulaFalsiPegasus) 
            | Algorithm::Bracket(BracketFamily::RegulaFalsiAndersonBjorck) => Some(100), 
            Algorithm::Open(OpenFamily::Secant)                            => Some(100), 
            Algorithm::Open(OpenFamily::Newton)                            => Some(50), 
            Algorithm::Compound(CompoundFamily::Brent)                      => None, 
        }
    }

    pub const fn algorithm_name(self) -> &'static str { 
        match self { 
            Algorithm::Bracket(BracketFamily::Bisection)                 => "bisection", 
            Algorithm::Bracket(BracketFamily::RegulaFalsiPure)           => "regula_falsi_pure", 
            Algorithm::Bracket(BracketFamily::RegulaFalsiIllinois)       => "regula_falsi_illinois",
            Algorithm::Bracket(BracketFamily::RegulaFalsiPegasus)        => "regula_falsi_pegasus",
            Algorithm::Bracket(BracketFamily::RegulaFalsiAndersonBjorck) => "regula_falsi_anderson_bjorck", 
            Algorithm::Open(OpenFamily::Secant)                          => "secant", 
            Algorithm::Open(OpenFamily::Newton)                          => "newton",
            Algorithm::Compound(CompoundFamily::Brent)                   => "brent",   
        }
    }
}
impl std::fmt::Display for Algorithm { 
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { 
        write!(f, "{}", self.algorithm_name())
    }
}

