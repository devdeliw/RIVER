use river::interpolation::newton::{interpolate, NewtonCfg};
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
            "mismatch at index {}: left={}, right={}, ATOL={}, RTOL={}",
            i, ai, bi, ATOL, RTOL
        );
    }
}

#[test]
fn quadratic_global_match() -> RiverResult {
    let x      = [0.0, 1.0, 2.0];
    let y      = [0.0, 1.0, 4.0];
    let x_eval = [0.5, 1.5];

    let cfg = NewtonCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;

    let rep = interpolate(cfg)?;
    assert_eq!(rep.n_provided, 3);
    assert_eq!(rep.n_evaluated, 2);
    assert!(approx_eq(rep.evaluated[0], 0.25));
    assert!(approx_eq(rep.evaluated[1], 2.25));
    Ok(())
}

#[test]
fn exact_hits() -> RiverResult {
    let x_for_cfg  = [0.0, 1.0, 2.0, 3.0];
    let y_for_cfg  = [0.0, 1.0, 4.0, 9.0];
    let x_eval     = [0.0, 1.0, 2.0, 3.0];
    let y_expected = [0.0, 1.0, 4.0, 9.0];

    let cfg = NewtonCfg::new()
        .set_x(&x_for_cfg)?
        .set_y(&y_for_cfg)?
        .set_x_eval(&x_eval)?;

    let rep = interpolate(cfg)?;
    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn bounds_ok_at_endpoints() -> RiverResult {
    let x          = [-1.0, 2.0];
    let y          = [10.0, 40.0];
    let x_eval     = [-1.0, 2.0];
    let y_expected = [10.0, 40.0];

    let cfg = NewtonCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;

    let rep = interpolate(cfg)?;
    assert_vec_close(&rep.evaluated, &y_expected);
    Ok(())
}

#[test]
fn out_of_bounds_low() {
    let x      = [0.0, 1.0, 2.0];
    let y      = [0.0, 1.0, 2.0];
    let x_eval = [-0.1];

    let cfg = NewtonCfg::new()
        .set_x(&x).unwrap()
        .set_y(&y).unwrap()
        .set_x_eval(&x_eval).unwrap();

    let err = interpolate(cfg).unwrap_err();
    assert!(matches!(err, InterpolationError::OutOfBounds { got, x_min, x_max }
        if got == -0.1 && x_min == 0.0 && x_max == 2.0));
}

#[test]
fn out_of_bounds_high() {
    let x      = [0.0, 1.0, 2.0];
    let y      = [0.0, 1.0, 2.0];
    let x_eval = [2.1];

    let cfg = NewtonCfg::new()
        .set_x(&x).unwrap()
        .set_y(&y).unwrap()
        .set_x_eval(&x_eval).unwrap();

    let err = interpolate(cfg).unwrap_err();
    assert!(matches!(err, InterpolationError::OutOfBounds { got, x_min, x_max }
        if got == 2.1 && x_min == 0.0 && x_max == 2.0));
}

#[test]
fn unequal_length_error() {
    let x  = [0.0, 1.0, 2.0];
    let y  = [0.0, 1.0];
    let cfg = NewtonCfg::new().set_x(&x).unwrap();
    let err = cfg.set_y(&y).unwrap_err();
    assert!(matches!(err, InterpolationError::UnequalLength { x_len: 3, y_len: 2 }));
}

#[test]
fn non_increasing_x_error() {
    let x = [0.0, 0.0, 2.0];
    let err = NewtonCfg::new().set_x(&x).unwrap_err();
    assert!(matches!(err, InterpolationError::NonIncreasingX));
}

#[test]
fn near_duplicate_x_error() {
    let x = [0.0, 1e-13, 1.0];
    let err = NewtonCfg::new().set_x(&x).unwrap_err();
    assert!(matches!(err, InterpolationError::DuplicateX { .. }));
}

#[test]
fn empty_x_eval_ok() -> RiverResult {
    let x = [0.0, 1.0];
    let y = [0.0, 1.0];

    let cfg = NewtonCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&[])?; 

    let rep = interpolate(cfg)?;
    assert_eq!(rep.n_provided, 2);
    assert_eq!(rep.n_evaluated, 0);
    assert!(rep.evaluated.is_empty());
    Ok(())
}

#[test]
fn two_points() -> RiverResult {
    let x      = [2.0, 4.0];
    let y      = [5.0, 9.0];
    let x_eval = [3.0];

    let cfg = NewtonCfg::new()
        .set_x(&x)?
        .set_y(&y)?
        .set_x_eval(&x_eval)?;

    let rep = interpolate(cfg)?;
    assert!(approx_eq(rep.evaluated[0], 7.0));
    Ok(())
}

#[test]
fn many_points() -> RiverResult {
    let x_for_cfg      = [0.0, 1.0, 3.0, 6.0, 10.0];
    let y_for_cfg      = [0.0, 2.0, 3.0, 3.0, 8.0];
    let x_eval_for_cfg = [0.0, 1.0, 3.0, 6.0, 10.0];
    let y_at_nodes     = [0.0, 2.0, 3.0, 3.0, 8.0];

    let cfg = NewtonCfg::new()
        .set_x(&x_for_cfg)?
        .set_y(&y_for_cfg)?
        .set_x_eval(&x_eval_for_cfg)?;

    let rep = interpolate(cfg)?;
    assert_vec_close(&rep.evaluated, &y_at_nodes);
    Ok(())
}

