use crate::interpolation::algorithms::Algorithm;
use crate::interpolation::config::{impl_common_cfg, CommonCfg};
use crate::interpolation::errors::InterpolationError;
use crate::interpolation::report::InterpolationReport;
use crate::interpolation::spline::helpers::{
    spacings, 
    lu_to_dense_for_trsv, 
    coeffs, 
    find_interval
};
use coral::enums::{CoralDiagonal, CoralTranspose, CoralTriangular};
use coral::level2::dtrsv;


#[derive(Debug, Copy, Clone)]
pub struct ClampedSplineCfg<'a> {
    common: CommonCfg<'a>,
    slope_start: f64,
    slope_final: f64,
}
impl<'a> ClampedSplineCfg<'a> {
    pub fn new(slope_start: f64, slope_final: f64) -> Self {
        Self { common: CommonCfg::new(), slope_start, slope_final }
    }

    pub fn with_slope_start(mut self, v: f64) -> Self { self.slope_start = v; self }
    pub fn with_slope_final(mut self, v: f64) -> Self { self.slope_final = v; self }
}
impl_common_cfg!(ClampedSplineCfg<'a>);


fn solve_c_clamped(
    n: usize, 
    h: &[f64],
    y: &[f64], 
    fp0: f64, 
    fpn: f64
) -> Vec<f64> {
    // nxn tridiagonal for c[0..n-1]
    let mut a_sub  = vec![0.0; n];  
    let mut b_diag = vec![0.0; n];
    let mut c_sup  = vec![0.0; n];  
    let mut rhs    = vec![0.0; n];

    // i = 0:
    // 2 h0 c0 + h0 c1 = 3[(y1 - y0)/h0 - fp0]
    {
        let h0 = h[0];
        b_diag[0] = 2.0 * h0;
        c_sup[0]  = h0;
        rhs[0]    = 3.0 * ((y[1] - y[0]) / h0 - fp0);
    }

    // interior
    for i in 1..n - 1 {
        let him1 = h[i-1];
        let hi   = h[i];
        a_sub[i]  = him1;
        b_diag[i] = 2.0 * (him1 + hi);
        c_sup[i]  = hi;
        rhs[i]    = 3.0 * ((y[i+1] - y[i]) / hi - (y[i] - y[i-1]) / him1);
    }

    // i = n-1: 
    // h_{n-2} c_{n-2} + 2 h_{n-2} c_{n-1} = 
    // 3[slope_final - (y_n - y_{n-1})/h_{n-2}]
    {
        let i = n - 1;
        let hnm1 = h[n - 2];

        a_sub[i]  = hnm1;
        b_diag[i] = 2.0 * hnm1;
        c_sup[i]  = 0.0;
        rhs[i]    = 3.0 * (fpn - (y[n-1] - y[n-2]) / hnm1);
    }

    // thomas LU 
    let mut l_sub  = vec![0.0; n];
    let mut u_diag = vec![0.0; n];
    u_diag[0] = b_diag[0];
    for k in 1..n {
        l_sub[k]  = a_sub[k] / u_diag[k-1];
        u_diag[k] = b_diag[k] - l_sub[k] * c_sup[k-1];
    }

    let (l, u, lda) = lu_to_dense_for_trsv(&l_sub, &u_diag, &c_sup);

    // L y = rhs
    dtrsv(
        CoralTriangular::LowerTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::UnitDiagonal,
        n,
        &l,
        lda,
        &mut rhs,
        1,
    );
    // U c = y
    dtrsv(
        CoralTriangular::UpperTriangular,
        CoralTranspose::NoTranspose,
        CoralDiagonal::NonUnitDiagonal,
        n,
        &u,
        lda,
        &mut rhs,
        1,
    );

    rhs
}


/// Evaluate a clamped cubic spline
pub fn interpolate(cfg: ClampedSplineCfg) -> Result<InterpolationReport, InterpolationError> {
    let x     = cfg.common.x();
    let y     = cfg.common.y();
    let evals = cfg.common.x_eval();

    let n = x.len();
    if n < 2 {
        return Err(InterpolationError::InsufficientPoints { got: n });
    }

    let x_min = x[0];
    let x_max = x[n - 1];

    let h = spacings(x);

    // hermite fast path
    if n == 2 {
        let hi    = h[0];
        let delta = (y[1] - y[0]) / hi;

        let c0 = (-2.0 * cfg.slope_start - cfg.slope_final + 3.0 * delta) / hi;
        let c1 = ( cfg.slope_start + 2.0 * cfg.slope_final - 3.0 * delta) / hi;
        let b0 = delta - hi * (2.0 * c0 + c1) / 3.0;
        let d0 = (c1 - c0) / (3.0 * hi);

        let n_provided  = n;
        let n_evaluated = evals.len();
        let mut report = InterpolationReport::new(
            Algorithm::SplineClamped,
            n_provided, 
            n_evaluated
        );
        report.evaluated.reserve(n_evaluated);

        for &xq in evals {
            let dx = xq - x[0];
            let s  = y[0] + b0 * dx + c0 * dx * dx + d0 * dx * dx * dx;
            report.evaluated.push(s);
        }
        return Ok(report);
    }

    let c_full = solve_c_clamped(
        n, 
        &h,
        y,
        cfg.slope_start,
        cfg.slope_final
    );

    let (bcoef, dcoef) = coeffs(n, &h, y, &c_full);

    let n_provided  = n;
    let n_evaluated = evals.len();
    let mut report = InterpolationReport::new(
        Algorithm::SplineClamped,
        n_provided, 
        n_evaluated
    );
    report.evaluated.reserve(n_evaluated);

    for &xq in evals {
        if xq < x_min || xq > x_max {
            return Err(InterpolationError::OutOfBounds { got: xq, x_min, x_max });
        }
        let lo = find_interval(x, xq);
        let dx = xq - x[lo];
        let s  = y[lo]
            + bcoef[lo]  * dx
            + c_full[lo] * dx * dx
            + dcoef[lo]  * dx * dx * dx;
        report.evaluated.push(s);
    }

    Ok(report)
}

