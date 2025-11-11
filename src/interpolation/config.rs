//! Shared configuration for interpolation algorithms.  
//! 
//! Provides [`CommonCfg`] with default minimum allowed spacing between adjacent 
//! `x` data; [`DEFAULT_X_TOL`]. Shared by all interpolation algorithms. 
//!
//! [`CommonCfg`] — universal fields  
//! - `x`      : x values provided 
//! - `y`      : y values provided 
//! - `x_eval` : x values to evaluate 
//! - `x_tol`  : tolerance between values in eval_points
//!
//! [`CommonCfg::new`] initializes configuration with empty `Vec<f64>`s. 


use crate::interpolation::errors::InterpolationError;

pub const DEFAULT_X_TOL: f64 = 1e-12; 


#[derive(Debug, Copy, Clone)]
pub struct CommonCfg<'a> {
    pub(crate) x      : &'a [f64],
    pub(crate) y      : &'a [f64],
    pub(crate) x_eval : &'a [f64],       
    pub(crate) x_min_spacing: f64,
}

impl<'a> CommonCfg<'a> {
    pub fn new() -> Self {
        Self {
            x      : &[],
            y      : &[],
            x_eval : &[],
            x_min_spacing: DEFAULT_X_TOL, 
        }
    }
    pub fn validate(&self) -> Result<(), InterpolationError> {
        let x = self.x;
        let y = self.y;

        if x.is_empty() || y.is_empty() {
            return Err(InterpolationError::EmptyInput);
        }
        if x.len() != y.len() {
            return Err(InterpolationError::UnequalLength { x_len: x.len(), y_len: y.len() });
        }
        if x.len() < 2 {
            return Err(InterpolationError::InsufficientPoints { got: x.len() });
        }
        Ok(())
    }

    // getters
    pub fn x(&self) -> &'a [f64] { &self.x }
    pub fn y(&self) -> &'a [f64] { &self.y }
    pub fn x_eval(&self) -> &'a [f64] { &self.x_eval }
    pub fn x_min_spacing(&self)  -> f64 { self.x_min_spacing }

    // setters
    pub(crate) fn with_x(&mut self, v: &'a[f64]) { self.x = v; }
    pub(crate) fn with_y(&mut self, v: &'a[f64]) { self.y = v; }
    pub(crate) fn with_x_eval(&mut self, v: &'a[f64]) { self.x_eval = v; }
    pub(crate) fn with_x_min_spacing(&mut self, v: f64) { self.x_min_spacing = v; }
}


pub(crate) fn non_finite_idx(xs: &[f64]) -> Option<usize> {
    xs.iter().position(|x| !x.is_finite())
}

macro_rules! impl_common_cfg {
    ($cfg:ty) => {
        impl<'a> $cfg {
            pub fn set_x(
                mut self,
                v: &'a [f64],
            ) -> Result<Self, $crate::interpolation::errors::InterpolationError> {
                use $crate::interpolation::errors::InterpolationError;

                if v.is_empty() {
                    return Err(InterpolationError::EmptyInput);
                }
                if let Some(idx) = $crate::interpolation::config::non_finite_idx(v) {
                    return Err(InterpolationError::NonFiniteVec { idx });
                }
                if v.len() < 2 {
                    return Err(InterpolationError::InsufficientPoints { got: v.len() });
                }
                for i in 1..v.len() {
                    if (v[i] - v[i - 1]).abs() < self.common.x_min_spacing {
                        return Err(InterpolationError::DuplicateX {
                            x1: v[i - 1],
                            x2: v[i],
                        });
                    }
                    if v[i] <= v[i - 1] {
                        return Err(InterpolationError::NonIncreasingX);
                    }
                }

                self.common.with_x(v);

                // length agreement check 
                // symmetric with set_y
                let y_len = self.common.y.len();
                if y_len != 0 && y_len != v.len() {
                    return Err(InterpolationError::UnequalLength { x_len: v.len(), y_len });
                }

                Ok(self)
            }

            pub fn set_y(
                mut self,
                v: &'a [f64],
            ) -> Result<Self, $crate::interpolation::errors::InterpolationError> {
                use $crate::interpolation::errors::InterpolationError;

                if v.is_empty() {
                    return Err(InterpolationError::EmptyInput);
                }
                if let Some(idx) = $crate::interpolation::config::non_finite_idx(v) {
                    return Err(InterpolationError::NonFiniteVec { idx });
                }

                let x_len = self.common.x.len();
                let y_len = v.len();
                if x_len != 0 && y_len != x_len {
                    return Err(InterpolationError::UnequalLength { x_len, y_len });
                }

                self.common.with_y(v);
                Ok(self)
            }

            pub fn set_x_eval(
                mut self,
                v: &'a [f64],
            ) -> Result<Self, $crate::interpolation::errors::InterpolationError> {
                use $crate::interpolation::errors::InterpolationError;

                if let Some(idx) = $crate::interpolation::config::non_finite_idx(v) {
                    return Err(InterpolationError::NonFiniteVec { idx });
                }

                self.common.with_x_eval(v);
                Ok(self)
            }

            pub fn set_x_tol(
                mut self,
                v: f64,
            ) -> Result<Self, $crate::interpolation::errors::InterpolationError> {
                use $crate::interpolation::errors::InterpolationError;

                if !v.is_finite() || v <= 0.0 {
                    return Err(InterpolationError::InvalidXTol { got: v });
                }

                self.common.with_x_min_spacing(v);
                Ok(self)
            }
        }
    };
}
pub(crate) use impl_common_cfg;
