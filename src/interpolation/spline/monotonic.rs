use crate::interpolation::algorithms::Algorithm;
use crate::interpolation::config::{impl_common_cfg, CommonCfg};
use crate::interpolation::errors::InterpolationError;
use crate::interpolation::report::InterpolationReport;
use crate::interpolation::spline::helpers::{spacings, deltas, find_interval};


#[derive(Debug, Copy, Clone)]
pub struct MonotonicSplineCfg<'a> {
    common: CommonCfg<'a>,
}
impl<'a> MonotonicSplineCfg<'a> {
    pub fn new() -> Self {
        Self { common: CommonCfg::new() }
    }
}
impl_common_cfg!(MonotonicSplineCfg<'a>);


#[inline]
fn endpoint_slope_left(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    if d0.signum() * d1.signum() <= 0.0 { return 0.0; }

    let m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if m0.signum() * d0.signum() <= 0.0 { 0.0 }
    else if m0.abs() > 3.0 * d0.abs() { 3.0 * d0 }
    else { m0 }
}


#[inline]
fn endpoint_slope_right(hm2: f64, hm1: f64, dm2: f64, dm1: f64) -> f64 {
    if dm1.signum() * dm2.signum() <= 0.0 { return 0.0; }

    let mn = ((2.0 * hm1 + hm2) * dm1 - hm1 * dm2) / (hm2 + hm1);
    if mn.signum() * dm1.signum() <= 0.0 { 0.0 }
    else if mn.abs() > 3.0 * dm1.abs() { 3.0 * dm1 }
    else { mn }
}


#[inline]
fn interior_slopes(h: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len() + 1;
    let mut m = vec![0.0; n];

    for i in 1..n-1 {
        let d0 = d[i - 1];
        let d1 = d[i];

        if d0.signum() * d1.signum() <= 0.0 {
            m[i] = 0.0;
        } else {
            let w0 = 2.0 * h[i] + h[i - 1];
            let w1 = h[i] + 2.0 * h[i - 1];
            m[i]   = (w0 + w1) / (w0 / d0 + w1 / d1);
        }
    }
    m
}


#[inline]
fn slopes(x: &[f64], h: &[f64], d: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 2 { 
        return vec![d[0], d[0]]; 
    }

    let mut m = interior_slopes(h, d);
    m[0]      = endpoint_slope_left(h[0], h[1], d[0], d[1]);
    m[n-1]    = endpoint_slope_right(h[n - 3], h[n - 2], d[n - 3], d[n - 2]);
    m
}


/// Evaluate a monotonic cubic spline; PCHIP / Fritsch–Carlson
pub fn interpolate(cfg: MonotonicSplineCfg) -> Result<InterpolationReport, InterpolationError> {
    let x     = cfg.common.x();
    let y     = cfg.common.y();
    let evals = cfg.common.x_eval();

    let n = x.len();
    if n < 2 {
        return Err(InterpolationError::InsufficientPoints { got: n });
    }

    let x_min = x[0];
    let x_max = x[n-1];

    let h = spacings(x);
    let d = deltas(y, &h);
    let m = slopes(x, &h, &d);

    let mut report = InterpolationReport::new(Algorithm::SplineMonotonic, n, evals.len());
    report.evaluated.reserve(evals.len());

    for &xq in evals {
        if xq < x_min || xq > x_max {
            return Err(InterpolationError::OutOfBounds { got: xq, x_min, x_max });
        }
        let i  = find_interval(x, xq);
        let hi = h[i];
        let t  = (xq - x[i]) / hi;

        let h00 = (2.0 * t - 3.0) * t * t + 1.0;
        let h10 = (t - 2.0) * t * t + t;
        let h01 = -((2.0 * t - 3.0) * t * t);
        let h11 = (t * t) * (t - 1.0);

        let s = h00 * y[i]
            + h10 * hi * m[i]
            + h01 * y[i + 1]
            + h11 * hi * m[i + 1];

        report.evaluated.push(s);
    }

    Ok(report)
}

