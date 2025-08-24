use numena::root_finding::newton::{newton, NewtonCfg, NewtonError};
use numena::root_finding::errors::{RootFindingError, ToleranceError};
use numena::root_finding::report::{TerminationReason, ToleranceSatisfied};

type TestResult = Result<(), NewtonError>;

#[test]
fn finds_sqrt_2_with_analytic_derivative() -> TestResult {
    let f  = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;
    let tol = 1e-12;

    let cfg = NewtonCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(50)?;

    let res = newton(f, Some(df), 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - 2.0_f64.sqrt()).abs() <= tol);
    assert!(res.iterations > 0);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    Ok(())
}

#[test]
fn finds_sqrt_2_with_fd_derivative() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let tol = 1e-12;

    let cfg = NewtonCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(60)?;

    let res = newton(f, None::<fn(f64)->f64>, 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - 2.0_f64.sqrt()).abs() <= tol);
    Ok(())
}

#[test]
fn finds_linear_root_analytic() -> TestResult {
    let f  = |x: f64| 2.0 * x - 6.0;
    let df = |_x: f64| 2.0;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-14)?
        .set_abs_x(1e-15)?
        .set_rel_x(0.0)?
        .set_max_iter(20)?;

    let res = newton(f, Some(df), 10.0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - 3.0).abs() <= 1e-12);
    Ok(())
}

#[test]
fn early_abs_fx_exit_at_x0() -> TestResult {
    let f   = |x: f64| x;
    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-20)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let res = newton(f, None::<fn(f64)->f64>, 0.0, cfg)?;
    assert_eq!(res.iterations, 0);
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], 0.0);
    Ok(())
}

#[test]
fn step_size_tolerance_on_first_iteration() -> TestResult {
    let f  = |x: f64| (x - 1.0) + 1e-6 * (x - 1.0).powi(2);
    let df = |x: f64| 1.0 + 2e-6 * (x - 1.0);

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(0.15)?
        .set_rel_x(0.0)?
        .set_max_iter(5)?;

    let res = newton(f, Some(df), 1.1, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert!(res.iterations >= 1);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert!((s[0] - 1.1).abs() <= 0.0_f64);
    Ok(())
}

#[test]
fn machine_precision_stagnation_triggers() -> TestResult {
    let x0 = 1.0e308;
    let f  = |_x: f64| -1.0;
    let df = |_x: f64|  1.0;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?
        .set_max_iter(10)?;

    let res = newton(f, Some(df), x0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::MachinePrecisionReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert_eq!(res.iterations, 1);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], x0);
    Ok(())
}

#[test]
fn uses_max_iter_hits_limit() -> TestResult {
    let f  = |x: f64| x * x + 1.0;
    let df = |x: f64| 2.0 * x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?
        .set_max_iter(1)?;

    let res = newton(f, Some(df), 1.0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    Ok(())
}

#[test]
fn invalid_abs_fx_rejected_by_setter() {
    let err = NewtonCfg::new().set_abs_fx(0.0).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidAbsFx { .. }));
}

#[test]
fn invalid_abs_x_rejected_by_setter() {
    let err = NewtonCfg::new().set_abs_x(f64::NAN).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidAbsX { .. }));
}

#[test]
fn invalid_rel_x_rejected_by_setter() {
    let err = NewtonCfg::new().set_rel_x(f64::NAN).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidRelX { .. }));
}

#[test]
fn invalid_max_step_rejected_by_setter() {
    let err = NewtonCfg::new().set_max_step(0.0).unwrap_err();
    assert!(matches!(err, NewtonError::InvalidMaxStep { step } if step == 0.0));
}

#[test]
fn non_finite_eval_on_initial() {
    let f   = |x: f64| 1.0 / x;
    let cfg = NewtonCfg::new().set_abs_fx(1e-12).unwrap();
    let err = newton(f, None::<fn(f64)->f64>, 0.0, cfg).unwrap_err();
    assert!(matches!(
        err,
        NewtonError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
}

#[test]
fn non_finite_eval_mid_iteration_via_analytic_step_to_singularity() -> TestResult {
    let f  = |x: f64| 1.0 / x;
    let df = |_x: f64| 1.0;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), 1.0, cfg).unwrap_err();
    assert!(matches!(
        err,
        NewtonError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
    Ok(())
}

#[test]
fn derivative_too_small_error_when_df_zero() -> TestResult {
    let f  = |_x: f64| 1.0;
    let df = |_x: f64| 0.0;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), 1.0, cfg).unwrap_err();
    assert!(matches!(err, NewtonError::DerivativeTooSmall { x, dfx } if x == 1.0 && dfx == 0.0));
    Ok(())
}

#[test]
fn derivative_not_finite_error_when_df_nan() -> TestResult {
    let f  = |_x: f64| 1.0;
    let df = |_x: f64| f64::NAN;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), 1.0, cfg).unwrap_err();
    assert!(matches!(err, NewtonError::DerivativeNotFinite { x, dfx } if x == 1.0 && dfx.is_nan()));
    Ok(())
}

#[test]
fn fd_step_unrepresentable_near_huge_x() -> TestResult {
    let x0 = f64::MAX;
    let f  = |x: f64| x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?
        .set_max_iter(2)?;

    let err = newton(f, None::<fn(f64)->f64>, x0, cfg).unwrap_err();
    assert!(matches!(err, NewtonError::FiniteDifferenceStepUnrepresentable { x, h: _ } if x == x0));
    Ok(())
}

#[test]
fn step_not_finite_when_x_plus_step_overflows() -> TestResult {
    let x0  = f64::MAX / 2.0;
    let step_target = f64::MAX;
    let df_val = -1.0e-308;
    let fx_val = -df_val * step_target;

    let f  = move |_x: f64| fx_val;
    let df = move |_x: f64| df_val;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), x0, cfg).unwrap_err();
    assert!(matches!(err, NewtonError::StepNotFinite { x, step } if x == x0 && step.is_finite()));
    Ok(())
}

#[test]
fn max_step_clip_effect_observable_via_iteration_limit() -> TestResult {
    let f  = |x: f64| x;
    let df = |_x: f64| 1.0;

    let cfg = NewtonCfg::new()
        .set_max_step(1.0)?
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?
        .set_max_iter(1)?;

    let x0 = 10.0;
    let res = newton(f, Some(df), x0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);
    assert_eq!(res.root, x0 - 1.0);
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], x0);
    Ok(())
}

#[test]
fn algorithm_field_is_newton() -> TestResult {
    let f  = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;

    let cfg = NewtonCfg::new().set_abs_fx(1e-10)?;
    let res = newton(f, Some(df), 1.0, cfg)?;

    assert!(res.algorithm_name.to_lowercase().contains("newton"));
    Ok(())
}

#[test]
fn parent_fields_reproduce_newton_update_on_final_step() -> TestResult {
    let f  = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-15)?
        .set_rel_x(0.0)?
        .set_max_iter(50)?;

    let res = newton(f, Some(df), 1.0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    let x_prev = s[0];
    let x_newton = x_prev - f(x_prev) / df(x_prev);
    assert!((x_newton - res.root).abs() <= 1e-12);
    Ok(())
}

#[test]
fn invalid_guess_nan_rejected() {
    let f   = |x: f64| x;
    let cfg = NewtonCfg::new();
    let err = newton(f, None::<fn(f64)->f64>, f64::NAN, cfg).unwrap_err();
    assert!(matches!(err, NewtonError::InvalidGuess { x0 } if x0.is_nan()));
}

