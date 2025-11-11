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
pub struct NaturalSplineCfg<'a> {
    common: CommonCfg<'a>
}
impl<'a> NaturalSplineCfg<'a> {
    pub fn new() -> Self {
        Self { common: CommonCfg::new() }
    }
}
impl_common_cfg!(NaturalSplineCfg<'a>);


fn solve_c_natural(
    n: usize,
    h: &[f64],
    y: &[f64]
) -> Vec<f64> {
    // number of interior unknowns
    let m = n.saturating_sub(2);

    // tridiagonal ssystem for interior c = rhs
    // subdiag   a[k] = h[i-1]
    // diag      b[k] = 2(h[i-1]+h[i])
    // superdiag c[k] = h[i]
    // rhs[k] = 3[(y[i+1]-y[i])/h[i] - (y[i] - y[i-1])/h[i-1]]
    let mut c_full = vec![0.0; n];
    if m > 0 {
        let mut a_sub  = vec![0.0; m];
        let mut b_diag = vec![0.0; m];
        let mut c_sup  = vec![0.0; m];
        let mut rhs    = vec![0.0; m];

        for k in 0..m {
            let i = k + 1;
            a_sub[k]  = h[i - 1];
            b_diag[k] = 2.0 * (h[i-1] + h[i]);
            c_sup[k]  = h[i];
            rhs[k]    = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]);
        }

        let mut l_sub  = vec![0.0; m];
        let mut u_diag = vec![0.0; m];
        u_diag[0] = b_diag[0];
        for k in 1..m {
            l_sub[k]  = a_sub[k] / u_diag[k - 1];
            u_diag[k] = b_diag[k] - l_sub[k] * c_sup[k - 1];
        }

        let (l, u, lda) = lu_to_dense_for_trsv(&l_sub, &u_diag, &c_sup);

        // L y = rhs
        dtrsv(
            CoralTriangular::LowerTriangular,
            CoralTranspose::NoTranspose,
            CoralDiagonal::UnitDiagonal,
            m,
            &l,
            lda,
            &mut rhs,
            1,
        );

        // U c_interior = y
        dtrsv(
            CoralTriangular::UpperTriangular,
            CoralTranspose::NoTranspose,
            CoralDiagonal::NonUnitDiagonal,
            m,
            &u,
            lda,
            &mut rhs,
            1,
        );

        for k in 0..m {
            c_full[k + 1] = rhs[k];
        }
    }

    c_full
}


/// Evaluate a natural cubic spline
pub fn interpolate(cfg: NaturalSplineCfg) -> Result<InterpolationReport, InterpolationError> {
    let x = cfg.common.x();
    let y = cfg.common.y();
    let evals = cfg.common.x_eval();

    let n = x.len();
    if n < 2 {
        return Err(InterpolationError::InsufficientPoints { got: n });
    }

    let x_min = x[0];
    let x_max = x[n - 1];

    // compute spacings
    let h = spacings(x);

    let c_full = solve_c_natural(n, &h, y);

    let (bcoef, dcoef) = coeffs(n, &h, y, &c_full);

    // evaluate
    let n_provided  = n;
    let n_evaluated = evals.len();
    let mut report  = InterpolationReport::new(
        Algorithm::SplineNatural, 
        n_provided,
        n_evaluated
    );
    report.evaluated.reserve(n_evaluated);

    for &xq in evals {
        if xq < x_min || xq > x_max {
            return Err(InterpolationError::OutOfBounds { got: xq, x_min, x_max });
        }

        // interval search
        let lo = find_interval(x, xq);

        let dx = xq - x[lo];
        let s = y[lo]
            + bcoef[lo]  * dx
            + c_full[lo] * dx * dx
            + dcoef[lo]  * dx * dx * dx;

        report.evaluated.push(s);
    }

    Ok(report)
}

