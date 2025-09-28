use river::root_finding::newton::{newton, NewtonCfg, NewtonError};
use river::root_finding::errors::RootFindingError;
use river::root_finding::report::{TerminationReason, ToleranceSatisfied};

type RiverResult = Result<(), NewtonError>; 

#[test]
fn sqrt2_analytic_derivative() -> RiverResult {
    let f  = |x: f64| x.powi(2) - 2.0;
    let df = |x: f64| 2.0 * x;

    let abs_fx   = 1e-20; 
    let abs_x    = 1e-10; 
    let rel_x    = 0.0; 
    let max_iter = 50;

    let cfg = NewtonCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?;

    let res = newton(f, Some(df), 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert!((res.root - (2.0 as f64).sqrt()).abs() <= abs_x);
    assert!(res.iterations > 0);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);

    Ok(())
}


#[test]
fn sqrt2_fd_derivative() -> RiverResult {
    let f   = |x: f64| x.powi(2) - 2.0;
    let tol = 1e-12;

    let cfg = NewtonCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(60)?;

    let res = newton(f, None::<fn(f64)->f64>, 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - (2.0 as f64).sqrt()).abs() <= tol);

    Ok(())
}

#[test]
fn three_analytic() -> RiverResult {
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
fn early_absfx_exit() -> RiverResult {
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
fn steptol_iter1() -> RiverResult {
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
    assert!((s[0] - 1.1).abs() <= 0.0);

    Ok(())
}

#[test]
fn epsilon_stag_triggers() -> RiverResult {
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
fn max_iter() -> RiverResult {
    let f  = |x: f64| x.powi(2) + 1.0;
    let df = |x: f64| 2.0 * x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?
        .set_max_iter(1)?;

    let res = newton(f, Some(df), 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);

    Ok(())
}

#[test]
fn infinite_eval_start() -> RiverResult {
    let f   = |x: f64| 1.0 / x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?;

    let err = newton(f, None::<fn(f64)->f64>, 0.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        NewtonError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));

    Ok(())
}

#[test]
fn infinite_eval_mid() -> RiverResult {
    let f  = |x: f64| 1.0 / x;
    let df = |_x: f64| 1.0; // wrong 

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
fn small_derivative_err() -> RiverResult {
    let f  = |_x: f64| 1.0;
    let df = |_x: f64| 0.0;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), 1.0, cfg).unwrap_err();

    assert!(matches!(
            err,
            NewtonError::DerivativeTooSmall { x, dfx } 
            if x == 1.0 && dfx == 0.0
    ));

    Ok(())
}

#[test]
fn nan_derivative_err() -> RiverResult {
    let f  = |_x: f64| 1.0;
    let df = |_x: f64| f64::NAN;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), 1.0, cfg).unwrap_err();
    assert!(matches!(
            err, 
            NewtonError::DerivativeNotFinite { x, dfx }
            if x == 1.0 && dfx.is_nan()
    ));

    Ok(())
}

#[test]
fn fd_unrepresentable() -> RiverResult {
    let x0 = f64::MAX; // x + h -> infinity
    let f  = |x: f64| x;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?
        .set_max_iter(2)?;

    let err = newton(f, None::<fn(f64)->f64>, x0, cfg).unwrap_err();

    assert!(matches!(
            err,
            NewtonError::FiniteDifferenceStepUnrepresentable { x, h: _ }
            if x == x0
    ));

    Ok(())
}

#[test]
fn infinite_step() -> RiverResult {
    let x0          = f64::MAX / 2.0;
    let step_target = f64::MAX;
    let df_val      = -1.0e-308;
    let fx_val      = -df_val * step_target;

    let f  = move |_x: f64| fx_val;
    let df = move |_x: f64| df_val;

    let cfg = NewtonCfg::new()
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?;

    let err = newton(f, Some(df), x0, cfg).unwrap_err();

    assert!(matches!(
            err, 
            NewtonError::StepNotFinite { x, step } 
            if x == x0 && step.is_finite()
    ));

    Ok(())
}

#[test]
fn max_step_clip() -> RiverResult {
    let f  = |x: f64| x;
    let df = |_x: f64| 1.0;

    let cfg = NewtonCfg::new()
        .set_max_step(1.0)?
        .set_abs_fx(1e-300)?
        .set_abs_x(1e-300)?
        .set_rel_x(0.0)?
        .set_max_iter(1)?;

    let x0  = 10.0;
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
fn parents_yield_root() -> RiverResult {
    let f  = |x: f64| x.powi(2) - 2.0;
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
fn invalid_guess_nan() {
    let f   = |x: f64| x;
    let cfg = NewtonCfg::new();
    let err = newton(f, None::<fn(f64)->f64>, f64::NAN, cfg).unwrap_err();

    assert!(matches!(err, NewtonError::InvalidGuess { x0 } if x0.is_nan()));
}
