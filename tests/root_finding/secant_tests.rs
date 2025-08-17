//! tests for the secant root-finding algorithm
use numena::root_finding::common::{RootFindingError, Termination, ToleranceReason};
use numena::root_finding::secant::{secant, SecantCfg, SecantError};

type TestResult = Result<(), SecantError>;

#[test]
fn finds_sqrt_2() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let tol = 1e-10;

    let cfg = SecantCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(1e-14)     
        .with_rel_x(0.0)
        .with_max_iter(60);

    let res = secant(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.termination(), Termination::ToleranceReached);
    assert_eq!(res.tolerance(), ToleranceReason::AbsFxReached);
    assert!((res.root() - 2.0_f64.sqrt()).abs() <= tol);
    assert!(res.iterations() > 0);
    Ok(())
}

#[test]
fn finds_3() -> TestResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    let cfg = SecantCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(1e-14)
        .with_rel_x(0.0)
        .with_max_iter(40);

    let res = secant(f, 0.0, 10.0, cfg)?;

    assert_eq!(res.termination(), Termination::ToleranceReached);
    assert_eq!(res.tolerance(), ToleranceReason::AbsFxReached);
    assert!((res.root() - 3.0).abs() <= tol);
    assert!(res.iterations() > 0);
    Ok(())
}

#[test]
fn invalid_equal_guesses() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new();
    let err = secant(f, 1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::InvalidGuess { x0, x1 } if x0 == 1.0 && x1 == 1.0));
}

#[test]
fn invalid_tolerance_both_zero() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().with_abs_x(0.0).with_rel_x(0.0).with_abs_fx(1e-12);
    let err = secant(f, 0.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::Common(RootFindingError::InvalidTolerance { got }) if got == 0.0));
}

#[test]
fn invalid_abs_fx() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().with_abs_fx(0.0); 
    let err = secant(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::Common(RootFindingError::InvalidAbsFx { .. })));
}

#[test]
fn invalid_abs_x() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().with_abs_x(f64::NAN);
    let err = secant(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::Common(RootFindingError::InvalidAbsX { .. })));
}

#[test]
fn invalid_rel_x() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().with_rel_x(f64::NAN);
    let err = secant(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::Common(RootFindingError::InvalidRelX { .. })));
}

#[test]
fn invalid_max_iter_zero() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().with_max_iter(0);
    let err = secant(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::Common(RootFindingError::InvalidMaxIter { got: 0 })));
}

#[test]
fn non_finite_eval_on_initial() {
    let f   = |x: f64| 1.0 / x; // f(0) = inf
    let cfg = SecantCfg::new().with_abs_fx(1e-12);
    let err = secant(f, 0.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        SecantError::Common(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
}

#[test]
fn non_finite_eval_mid_iteration() {
    let f   = |x: f64| 1.0 / x;
    let cfg = SecantCfg::new()
        .with_abs_fx(1e-12)
        .with_abs_x(1e-16) 
        .with_rel_x(0.0);
    let err = secant(f, 1.0, -1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        SecantError::Common(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
}

#[test]
fn step_size_tolerance_hits_immediately() -> TestResult {
    let f   = |x: f64| x + 1.0;
    let cfg = SecantCfg::new()
        .with_abs_x(1e-3)
        .with_rel_x(0.0)
        .with_abs_fx(1e-20);

    let res = secant(f, 0.0, 5e-4, cfg)?;

    assert_eq!(res.iterations(), 0);
    assert_eq!(res.termination(), Termination::ToleranceReached);
    assert_eq!(res.tolerance(), ToleranceReason::StepSizeReached);
    assert_eq!(res.root(), 5e-4);
    Ok(())
}

#[test]
fn uses_max_iter_hits_limit() -> TestResult {
    let f   = |x: f64| x * x + 1.0;
    let cfg = SecantCfg::new()
        .with_abs_fx(1e-30)
        .with_abs_x(1e-16)
        .with_rel_x(0.0)
        .with_max_iter(1);

    let res = secant(f, 0.0, 1.0, cfg)?;

    assert_eq!(res.termination(), Termination::IterationLimit);
    assert_eq!(res.iterations(), 1);
    Ok(())
}

#[test]
fn degenerate_secant_denominator_errors() {
    let f   = |_x: f64| 1.0;
    let cfg = SecantCfg::new()
        .with_abs_fx(1e-12)
        .with_abs_x(1e-16)
        .with_rel_x(0.0);
    let err = secant(f, 0.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, SecantError::DegenerateSecantStep { .. }));
}

#[test]
fn parent_fields_generate_returned_root() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let cfg = SecantCfg::new()
        .with_abs_fx(1e-12)
        .with_abs_x(1e-14)
        .with_rel_x(0.0)
        .with_max_iter(50);

    let res = secant(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.termination(), Termination::ToleranceReached);
    assert_eq!(res.tolerance(), ToleranceReason::AbsFxReached);

    let p1 = res.parent1().unwrap();
    let p2 = res.parent2().unwrap();
    let xp = p2 - f(p2) * (p2 - p1) / (f(p2) - f(p1));
    assert!((xp - res.root()).abs() <= 1e-10);
    Ok(())
}

#[test]
fn algorithm_field_is_secant() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let cfg = SecantCfg::new().with_abs_fx(1e-10);
    let res = secant(f, 1.0, 2.0, cfg)?;

    assert!(res.algorithm().contains("secant"));
    Ok(())
}

#[test]
fn parent_fields_on_step_tolerance() -> Result<(), SecantError> {

    // case 1: immediate step size 
    let f = |x: f64| 3.0*x + 1.0;

    let cfg0 = SecantCfg::new()
        .with_abs_x(1e-3).with_rel_x(0.0).with_abs_fx(1e-30);
    let res0 = secant(f, 0.0, 5e-4, cfg0)?;

    assert_eq!(res0.termination(), Termination::ToleranceReached);
    assert_eq!(res0.tolerance(), ToleranceReason::StepSizeReached);
    assert_eq!(res0.iterations(), 0);
    assert_eq!(res0.root(), res0.parent1().unwrap());

    // case 2: first iteration step size 
    let f = |x: f64| (x - 1.0) + 1e-6 * (x - 1.0).powi(2);
    let cfg = SecantCfg::new()
        .with_abs_fx(1e-30)   
        .with_abs_x(0.15)
        .with_rel_x(0.0)
        .with_max_iter(5);

    let res = secant(f, 0.9, 1.1, cfg)?;

    assert_eq!(res.termination(), Termination::ToleranceReached);
    assert_eq!(res.tolerance(), ToleranceReason::StepSizeReached);
    assert!(res.iterations() >= 1);

    let p1 = res.parent1().unwrap();
    let p2 = res.parent2().unwrap();
    let xp = p2 - f(p2) * (p2 - p1) / (f(p2) - f(p1));
    assert!((xp - res.root()).abs() <= 1e-10);

    Ok(())
}


