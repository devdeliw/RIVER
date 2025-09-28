//! Sign utilities for root-finding algorithms.
//! - `opposite_sign` : `true` if values have opposite sign  
//! - `same_sign`     : `true` if values share the same sign

/// Returns `true` if `x` and `y` have opposite signs.
#[inline]
pub(crate) fn opposite_sign(x: f64, y: f64) -> bool {
    x.is_sign_positive() != y.is_sign_positive()
}


/// Returns `true` if `x` and `y` have the same sign.
#[inline]
pub(crate) fn same_sign(x: f64, y: f64) -> bool {
    x.is_sign_positive() == y.is_sign_positive()
}
