use numena::root_finding::secant::{secant, SecantCfg, SecantError};
use numena::root_finding::errors::{RootFindingError, ToleranceError};
use numena::root_finding::report::{TerminationReason, ToleranceSatisfied};

type TestResult = Result<(), SecantError>;

#[test]
fn finds_sqrt_2() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let tol = 1e-10;

    let cfg = SecantCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(60)?;

    let res = secant(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - 2.0_f64.sqrt()).abs() <= tol);
    assert!(res.iterations > 0);
    Ok(())
}

#[test]
fn finds_3() -> TestResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    let cfg = SecantCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(40)?;

    let res = secant(f, 0.0, 10.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!((res.root - 3.0).abs() <= tol);
    assert!(res.iterations > 0);
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
fn invalid_abs_fx_rejected_by_setter() {
    // Setters validate; invalid abs_fx should error before running the solver.
    let err = SecantCfg::new().set_abs_fx(0.0).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidAbsFx { .. }));
}

#[test]
fn invalid_abs_x_rejected_by_setter() {
    let err = SecantCfg::new().set_abs_x(f64::NAN).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidAbsX { .. }));
}

#[test]
fn invalid_rel_x_rejected_by_setter() {
    let err = SecantCfg::new().set_rel_x(f64::NAN).unwrap_err();
    assert!(matches!(err, ToleranceError::InvalidRelX { .. }));
}

#[test]
fn non_finite_eval_on_initial() {
    let f   = |x: f64| 1.0 / x; // f(0) = inf
    let cfg = SecantCfg::new().set_abs_fx(1e-12).unwrap();
    let err = secant(f, 0.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        SecantError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
}

#[test]
fn non_finite_eval_mid_iteration() {
    let f   = |x: f64| 1.0 / x;
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-12).unwrap()
        .set_abs_x(1e-16).unwrap()
        .set_rel_x(0.0).unwrap();

    let err = secant(f, 1.0, -1.0, cfg).unwrap_err();
    assert!(matches!(
        err,
        SecantError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
}

#[test]
fn step_size_tolerance_hits_immediately() -> TestResult {
    let f   = |x: f64| x + 1.0;
    let cfg = SecantCfg::new()
        .set_abs_x(1e-3)?
        .set_rel_x(0.0)?
        .set_abs_fx(1e-20)?;

    let res = secant(f, 0.0, 5e-4, cfg)?;
    assert_eq!(res.iterations, 0);
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert_eq!(res.root, 5e-4);
    // stencil should have both initial points for step-size early exit
    let s = res.stencil.stencil();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0], 0.0);
    assert_eq!(s[1], 5e-4);
    Ok(())
}

#[test]
fn uses_max_iter_hits_limit() -> TestResult {
    let f   = |x: f64| x * x + 1.0; // never hits |f| tol
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?
        .set_max_iter(1)?;

    let res = secant(f, 0.0, 1.0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);
    Ok(())
}

#[test]
fn degenerate_secant_denominator_is_safeguarded() {
    let f   = |_x: f64| 1.0; // fx2 - fx1 == 0 -> safeguard half-step
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-12).unwrap()
        .set_abs_x(1e-16).unwrap()
        .set_rel_x(0.0).unwrap();

    let res = secant(f, 0.0, 1.0, cfg).expect("should safeguard, not error");
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert!((res.root - (2.0/3.0)).abs() < 1e-12);
}

#[test]
fn parent_fields_generate_returned_root() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-14)?
        .set_rel_x(0.0)?
        .set_max_iter(50)?;

    let res = secant(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 2);
    let (p1, p2) = (s[0], s[1]);
    let xp = p2 - f(p2) * (p2 - p1) / (f(p2) - f(p1));
    assert!((xp - res.root).abs() <= 1e-10);
    Ok(())
}

#[test]
fn algorithm_field_is_secant() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let cfg = SecantCfg::new().set_abs_fx(1e-10)?;
    let res = secant(f, 1.0, 2.0, cfg)?;

    assert!(res.algorithm_name.contains("secant"));
    Ok(())
}

#[test]
fn early_abs_fx_exit_stencil_len_is_one_x0() -> TestResult {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().set_abs_fx(1e-12)?;
    let res = secant(f, 0.0, 5.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert_eq!(res.iterations, 0);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], 0.0);
    Ok(())
}

#[test]
fn early_abs_fx_exit_stencil_len_is_one_x1() -> TestResult {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new().set_abs_fx(1e-12)?;
    let res = secant(f, -5.0, 0.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert_eq!(res.iterations, 0);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], 0.0);
    Ok(())
}

#[test]
fn parent_fields_on_step_tolerance() -> TestResult {
    // case 1: immediate step-size
    let f = |x: f64| 3.0*x + 1.0;
    let cfg0 = SecantCfg::new()
        .set_abs_x(1e-3)? .set_rel_x(0.0)? .set_abs_fx(1e-30)?;
    let res0 = secant(f, 0.0, 5e-4, cfg0)?;
    assert_eq!(res0.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res0.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert_eq!(res0.iterations, 0);
    let s0 = res0.stencil.stencil();
    assert_eq!(s0.len(), 2);
    assert_eq!(res0.root, s0[1]); // root = x1 when immediate

    // case 2: first-iteration step-size
    let f = |x: f64| (x - 1.0) + 1e-6 * (x - 1.0).powi(2);
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-30)?   
        .set_abs_x(0.15)?
        .set_rel_x(0.0)?
        .set_max_iter(5)?;
    let res = secant(f, 0.9, 1.1, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert!(res.iterations >= 1);

    let s = res.stencil.stencil();
    assert_eq!(s.len(), 2);
    let (p1, p2) = (s[0], s[1]);
    let xp = p2 - f(p2) * (p2 - p1) / (f(p2) - f(p1));
    assert!((xp - res.root).abs() <= 1e-10);
    Ok(())
}

