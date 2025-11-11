use river::interpolation::spline::clamped::{interpolate, ClampedSplineCfg};
use river::interpolation::errors::InterpolationError;

type RiverResult = Result<(), InterpolationError>;

const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;

#[inline]
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * b.abs()
}

#[inline]
fn assert_vec_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            approx_eq(*ai, *bi),
            "mismatch at {}: left={}, right={}, ATOL={}, RTOL={}",
            i, ai, bi, ATOL, RTOL
        );
    }
}

#[test]
fn report_metadata() -> RiverResult {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 4.0];
    let x_eval = [0.5, 1.5];
    let fp0 = 0.0;     // dy/dx at x=0 for y=x^2
    let fpn = 4.0;     // dy/dx at x=2 for y=x^2

    let cfg = ClampedSplineCfg::new(fp0, fpn)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_eq!(rep.algorithm_name, "clamped cubic spline");
    assert_eq!(rep.n_provided, 3);
    assert_eq!(rep.n_evaluated, 2);
    Ok(())
}

#[test]
fn exact_hits() -> RiverResult {
    // y  = x^3 - 2x + 1
    // y' = 3x^2 - 2
    let x = [0.0, 1.0, 2.0, 3.0];
    let y: Vec<f64> = x.iter().map(|&t| t*t*t - 2.0*t + 1.0).collect();
    let fp0 = 3.0*0.0*0.0 - 2.0;
    let fpn = 3.0*3.0*3.0 - 2.0;
    let cfg = ClampedSplineCfg::new(fp0, fpn)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x)?;
    let rep = interpolate(cfg)?;
    assert_vec_close(&rep.evaluated, &y);
    Ok(())
}

#[test]
fn constant_function() -> RiverResult {
    let x = [0.0, 0.2, 1.1, 3.7, 5.0];
    let y = [2.5; 5];
    let x_eval = [-0.0, 0.2, 1.0, 2.5, 3.7, 5.0];
    let y_expected = [2.5; 6];

    let cfg = ClampedSplineCfg::new(0.0, 0.0)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn linear_function() -> RiverResult {
    // y  = 3x - 1
    // y' = 3
    let x = [-2.0, 0.0, 0.3, 1.7, 4.2];
    let y: Vec<f64> = x.iter().map(|&xi| 3.0*xi - 1.0).collect();
    let x_eval = [-2.0, -1.0, 0.0, 0.3, 1.0, 1.7, 3.0, 4.2];
    let y_expected: Vec<f64> = x_eval.iter().map(|&t| 3.0*t - 1.0).collect();

    let cfg = ClampedSplineCfg::new(3.0, 3.0)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn two_points_fast_path() -> RiverResult {
    let x = [2.0, 5.0];
    let y = [7.0, 1.0];
    let h = x[1] - x[0];
    let m = (y[1] - y[0]) / h;

    let x_eval = [2.0, 3.0, 4.0, 5.0];
    let y_expected: Vec<f64> = x_eval.iter().map(|&t| y[0] + m*(t - x[0])).collect();

    let cfg = ClampedSplineCfg::new(m, m)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn bounds_ok_at_endpoints() -> RiverResult {
    // y  = x^2
    // y' = 2x
    let x = [-1.0, 2.0, 6.0];
    let y: Vec<f64> = x.iter().map(|&t| t*t).collect();

    let fp0 = 2.0 * x[0];
    let fpn = 2.0 * x[2];
    let x_eval = [x[0], x[2]];

    let cfg = ClampedSplineCfg::new(fp0, fpn)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert!(approx_eq(rep.evaluated[0], y[0]));
    assert!(approx_eq(rep.evaluated[1], y[2]));
    Ok(())
}

#[test]
fn out_of_bounds_low() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 2.0];
    let x_eval = [-0.1];

    let cfg = ClampedSplineCfg::new(0.0, 0.0)
        .set_x(&x).unwrap()
        .set_y(&y).unwrap()
        .set_x_eval(&x_eval).unwrap();

    let err = interpolate(cfg).unwrap_err();
    assert!(matches!(err, InterpolationError::OutOfBounds { got, x_min, x_max }
        if got == -0.1 && (x_min, x_max) == (0.0, 2.0)));
}

#[test]
fn out_of_bounds_high() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 2.0];
    let x_eval = [2.0000001];

    let cfg = ClampedSplineCfg::new(0.0, 0.0)
        .set_x(&x).unwrap()
        .set_y(&y).unwrap()
        .set_x_eval(&x_eval).unwrap();

    let err = interpolate(cfg).unwrap_err();
    assert!(matches!(err, InterpolationError::OutOfBounds { got, x_min, x_max }
        if got == 2.0000001 && (x_min, x_max) == (0.0, 2.0)));
}

#[test]
fn empty_x_eval_ok() -> RiverResult {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 4.0];
    let cfg = ClampedSplineCfg::new(0.0, 4.0).set_x(&x)?.set_y(&y)?.set_x_eval(&[])?;
    let rep = interpolate(cfg)?;
    assert_eq!(rep.n_provided, 3);
    assert_eq!(rep.n_evaluated, 0);
    assert!(rep.evaluated.is_empty());
    Ok(())
}

#[test]
fn nonuniform_spacing() -> RiverResult {
    // y = x^2
    let x = [0.0, 0.1, 0.1000001, 2.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&t| t*t).collect();
    let x_eval = [0.0, 0.05, 0.1, 0.1000001, 1.0, 2.0, 5.0, 10.0];

    let cfg = ClampedSplineCfg::new(2.0*x[0], 2.0*x[x.len()-1])
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;
    
    for (i, v) in rep.evaluated.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at {}", i);
    }
    Ok(())
}

fn clamped_reference_eval(
    x: &[f64], 
    y: &[f64], 
    x_eval: &[f64], 
    fp0: f64, 
    fpn: f64
) -> Vec<f64> {
    let n = x.len();
    assert!(n >= 2);
    let mut h = Vec::with_capacity(n - 1);
    for i in 0..n - 1 { h.push(x[i + 1] - x[i]); }

    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    b[0] = 2.0 * h[0];
    c[0] = h[0];
    d[0] = 3.0 * ((y[1] - y[0]) / h[0] - fp0);

    for i in 1..n - 1 {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    a[n - 1] = h[n - 2];
    b[n - 1] = 2.0 * h[n - 2];
    d[n - 1] = 3.0 * (fpn - (y[n - 1] - y[n - 2]) / h[n - 2]);

    // nxn thomas
    c[0] /= b[0];
    d[0] /= b[0];
    for i in 1..n {
        let denom = b[i] - a[i] * c[i - 1];
        if i < n - 1 { c[i] /= denom; }
        d[i] = (d[i] - a[i] * d[i - 1]) / denom;
    }
    for i in (0..n - 1).rev() {
        d[i] -= c[i] * d[i + 1];
    }
    let c_full = d;

    let mut bcoef = vec![0.0; n - 1];
    let mut dcoef = vec![0.0; n - 1];
    for i in 0..n - 1 {
        bcoef[i] = (y[i + 1] - y[i]) / h[i] - (h[i] * (2.0 * c_full[i] + c_full[i + 1])) / 3.0;
        dcoef[i] = (c_full[i + 1] - c_full[i]) / (3.0 * h[i]);
    }

    let (xmin, xmax) = (x[0], x[n - 1]);
    let mut out = Vec::with_capacity(x_eval.len());
    for &xq in x_eval {
        assert!(xq >= xmin && xq <= xmax);
        let mut lo = 0;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if x[mid] <= xq { lo = mid; } else { hi = mid; }
        }
        let dx = xq - x[lo];
        out.push(y[lo] + bcoef[lo]*dx + c_full[lo]*dx*dx + dcoef[lo]*dx*dx*dx);
    }
    out
}

#[test]
fn cross_checks_solver() -> RiverResult {
    let x: Vec<f64> = (0..21).map(|k| (k as f64).powf(1.3)).collect();
    let y: Vec<f64> = x.iter().map(|&t| (t + 1.0).ln() + 0.1 * (0.5*t).sin()).collect();
    let x_eval: Vec<f64> = (0..51).map(|k| (k as f64) * x.last().unwrap() / 50.0).collect();

    let fp0 = 1.0/(x[0] + 1.0) + 0.1 * 0.5 * (0.5*x[0]).cos();
    let xn  = *x.last().unwrap();
    let fpn = 1.0/(xn + 1.0) + 0.1 * 0.5 * (0.5*xn).cos();

    let cfg = ClampedSplineCfg::new(fp0, fpn)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    let ref_vals = clamped_reference_eval(&x, &y, &x_eval, fp0, fpn);
    assert_vec_close(&rep.evaluated, &ref_vals);
    Ok(())
}

#[test]
fn large_n_reasonable() -> RiverResult {
    let n = 2000;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&t| (t + 1.0).ln()).collect();
    let x_eval: Vec<f64> = (0..1000).map(|i| i as f64 * (x[n-1]) / 999.0).collect();

    let fp0 = 1.0 / (x[0] + 1.0);
    let xn  = *x.last().unwrap();
    let fpn = 1.0 / (xn + 1.0);

    let cfg = ClampedSplineCfg::new(fp0, fpn)
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_eq!(rep.n_provided, n);
    assert_eq!(rep.n_evaluated, x_eval.len());
    for v in &rep.evaluated { assert!(v.is_finite()); }

    Ok(())
}

