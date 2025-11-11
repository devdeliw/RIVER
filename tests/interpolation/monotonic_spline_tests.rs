use river::interpolation::monotonic_spline::{interpolate, MonotonicSplineCfg};
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

#[inline]
fn find_interval(x: &[f64], xq: f64) -> usize {
    let n = x.len();
    let mut lo = 0;
    let mut hi = n - 1;

    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if x[mid] <= xq { lo = mid; } else { hi = mid; }
    }

    lo
}

#[test]
fn report_metadata() -> RiverResult {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 4.0];
    let x_eval = [0.5, 1.5];

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_eq!(rep.algorithm_name, "monotonic cubic spline");
    assert_eq!(rep.n_provided, 3);
    assert_eq!(rep.n_evaluated, 2);
    Ok(())
}

#[test]
fn exact_hits() -> RiverResult {
    let x = [0.0, 1.0, 2.0, 4.0];
    let y = [0.0, 1.0, 1.5, 3.0];

    let cfg = MonotonicSplineCfg::new()
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

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn linear_function() -> RiverResult {
    let x = [-2.0, 0.0, 0.3, 1.7, 4.2];
    let y: Vec<f64> = x.iter().map(|&xi| 3.0*xi - 1.0).collect();
    let x_eval = [-2.0, -1.0, 0.0, 0.3, 1.0, 1.7, 3.0, 4.2];
    let y_expected: Vec<f64> = x_eval.iter().map(|&t| 3.0*t - 1.0).collect();

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn two_points_degenerate() -> RiverResult {
    let x = [2.0, 5.0];
    let y = [7.0, 1.0];
    let x_eval = [2.0, 3.0, 4.0, 5.0];
    let y_expected: Vec<f64> = x_eval.iter().map(|&t| {
        let h = x[1] - x[0];
        let m = (y[1] - y[0]) / h;
        y[0] + m * (t - x[0])
    }).collect();

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn bounds_ok_at_endpoints() -> RiverResult {
    let x = [-1.0, 2.0, 6.0];
    let y = [10.0, 40.0, 55.0];
    let x_eval = [-1.0, 6.0];

    let cfg = MonotonicSplineCfg::new()
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

    let cfg = MonotonicSplineCfg::new()
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

    let cfg = MonotonicSplineCfg::new()
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
    let cfg = MonotonicSplineCfg::new().set_x(&x)?.set_y(&y)?.set_x_eval(&[])?;
    let rep = interpolate(cfg)?;
    assert_eq!(rep.n_provided, 3);
    assert_eq!(rep.n_evaluated, 0);
    assert!(rep.evaluated.is_empty());
    Ok(())
}

#[test]
fn nonuniform_spacing() -> RiverResult {
    let x = [0.0, 0.1, 0.1000001, 2.0, 10.0];
    let y = [0.0, 0.01, 0.01000001, 4.0, 12.0];
    let x_eval = [0.0, 0.05, 0.1, 0.1000001, 1.0, 2.0, 5.0, 10.0];

    let cfg = MonotonicSplineCfg::new().set_x(&x)?.set_y(&y)?.set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    for (i, v) in rep.evaluated.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at {}", i);
    }
    Ok(())
}

#[test]
fn monotone_no_overshoot() -> RiverResult {
    let x = [0.0, 1.0, 2.0, 4.0];
    let y = [0.0, 1.0, 1.5, 3.0];

    let x_grid: Vec<f64> = (0..=80).map(|k| 4.0 * k as f64 / 80.0).collect();

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_grid)?;
    let rep = interpolate(cfg)?;

    for w in rep.evaluated.windows(2) {
        assert!(w[1] >= w[0] - 1e-12, "not monotone: {} -> {}", w[0], w[1]);
    }

    for (&xq, &yq) in x_grid.iter().zip(rep.evaluated.iter()) {
        let i = find_interval(&x, xq);
        let lo = y[i].min(y[i + 1]);
        let hi = y[i].max(y[i + 1]);
        assert!(yq >= lo - 1e-12 && yq <= hi + 1e-12, "overshoot at x={}", xq);
    }

    Ok(())
}


#[test]
fn flat_segment_preserved() -> RiverResult {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 1.0, 2.0];
    let midpoints = [1.25, 1.5, 1.75];

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&midpoints)?;
    let rep = interpolate(cfg)?;

    for &v in &rep.evaluated {
        assert!(approx_eq(v, 1.0));
    }

    Ok(())
}

#[test]
fn large_n_reasonable() -> RiverResult {
    let n = 2000;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&t| (t + 1.0).ln()).collect();
    let x_eval: Vec<f64> = (0..1000).map(|i| i as f64 * (x[n-1]) / 999.0).collect();

    let cfg = MonotonicSplineCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;
    let rep = interpolate(cfg)?;

    assert_eq!(rep.n_provided, n);
    assert_eq!(rep.n_evaluated, x_eval.len());
    for v in &rep.evaluated { assert!(v.is_finite()); }

    Ok(())
}

