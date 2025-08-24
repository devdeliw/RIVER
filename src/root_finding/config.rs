//! Shared configuration for root-finding algorithms.  
//! 
//! Provides [`CommonCfg`] with default tolerances and iteration limits, 
//! used by all root-finding configs.
//!
//! [`CommonCfg`] — universal fields  
//! ├ `abs_fx`   : function-value tolerance  
//! ├ `abs_x`    : absolute step/width tolerance  
//! ├ `rel_x`    : relative step/width tolerance  
//! └ `max_iter` : iteration cap (optional) 
//!
//! [`CommonCfg::new`] initializes configuration with default values. 
//!
//! Some algorithms (e.g. regula_falsi) have additional config arguments 
//! to specify which variant of the algortihm to use (e.g. pegasus)


pub const DEFAULT_ABS_FX : f64 = 1e-12;
pub const DEFAULT_ABS_X  : f64 = 0.0;
pub const DEFAULT_REL_X  : f64 = 4.0 * f64::EPSILON;


#[derive(Debug, Copy, Clone)]
pub struct CommonCfg {
    abs_fx: f64,
    abs_x:  f64,
    rel_x:  f64,
    max_iter: Option<usize>,
}

impl CommonCfg {
    pub fn new() -> Self {
        Self { 
            abs_fx   : DEFAULT_ABS_FX, 
            abs_x    : DEFAULT_ABS_X, 
            rel_x    : DEFAULT_REL_X, 
            max_iter : None 
        }
    }

    // getters  
    pub fn abs_fx(&self)   -> f64 { self.abs_fx }
    pub fn abs_x(&self)    -> f64 { self.abs_x }
    pub fn rel_x(&self)    -> f64 { self.rel_x }
    pub fn max_iter(&self) -> Option<usize> { self.max_iter }

    // setters (internal) 
    pub(crate) fn with_abs_fx   (&mut self, v: f64)   { self.abs_fx   = v; }
    pub(crate) fn with_abs_x    (&mut self, v: f64)   { self.abs_x    = v; }
    pub(crate) fn with_rel_x    (&mut self, v: f64)   { self.rel_x    = v; }
    pub(crate) fn with_max_iter (&mut self, v: usize) { self.max_iter = Some(v); }
}

macro_rules! impl_common_cfg {
    ($cfg:ty) => {
        impl $cfg {
            pub fn set_abs_fx(
                mut self, v: f64
            ) -> Result<Self, $crate::root_finding::errors::ToleranceError> {
                if !v.is_finite() || v <= 0.0 {
                    return Err(
                        $crate::root_finding::errors::ToleranceError::InvalidAbsFx { got: v }
                    );
                }
                self.common.with_abs_fx(v);
                Ok(self)
            }
            pub fn set_abs_x(
                mut self, v: f64
            ) -> Result<Self, $crate::root_finding::errors::ToleranceError> {
                if !v.is_finite() || v < 0.0 {
                    return Err(
                        $crate::root_finding::errors::ToleranceError::InvalidAbsX { got: v }
                    );
                }

                let rel_x = self.common.rel_x(); 
                if v == 0.0 && rel_x <= 0.0 {
                    return Err(
                        $crate::root_finding::errors::ToleranceError::InvalidAbsRelX { abs_x: v, rel_x }  
                    );
                }
                self.common.with_abs_x(v);
                Ok(self)
            }
            pub fn set_rel_x(
                mut self, v: f64
            ) -> Result<Self, $crate::root_finding::errors::ToleranceError> {
                if !v.is_finite() || v < 0.0 {
                    return Err(
                        $crate::root_finding::errors::ToleranceError::InvalidRelX { got: v }
                    );
                }

                let abs_x = self.common.abs_x(); 
                if v <= 0.0 && abs_x == 0.0 {
                    return Err(
                        $crate::root_finding::errors::ToleranceError::InvalidAbsRelX { abs_x, rel_x: v }
                    );
                }
                self.common.with_rel_x(v);
                Ok(self)
            }
            pub fn set_max_iter(
                mut self, v: usize
            ) -> Result<Self, $crate::root_finding::errors::RootFindingError> {
                if v == 0 {
                    return Err(
                        $crate::root_finding::errors::RootFindingError::InvalidMaxIter { got: v }
                    );
                }
                self.common.with_max_iter(v);
                Ok(self)
            }
        }
    };
}
pub(crate) use impl_common_cfg; 
