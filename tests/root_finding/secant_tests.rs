use river::root_finding::secant::{secant, SecantCfg, SecantError};
use river::root_finding::errors::RootFindingError;
use river::root_finding::report::{TerminationReason, ToleranceSatisfied};

type RiverResult = Result<(), SecantError>; 

#[test]
fn sqrt2() -> RiverResult {
    let f   = |x: f64| x * x - 2.0;

    let a = 1.0; 
    let b = 2.0; 

    let abs_fx   = 1e-20; 
    let abs_x    = 1e-10; 
    let rel_x    = 0.0; 
    let max_iter = 60; 

    let cfg = SecantCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?;

    let res = secant(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert!(res.iterations > 0);

    assert!((res.root - (2.0 as f64).sqrt()).abs() <= abs_x);

    Ok(())
}

#[test]
fn three() -> RiverResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    let cfg = SecantCfg::new()
        .set_abs_fx(1e-20)?
        .set_abs_x(tol)?
        .set_rel_x(0.0)?
        .set_max_iter(40)?;

    let res = secant(f, 0.0, 10.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!(res.iterations > 0);

    assert!((res.root - 3.0).abs() <= tol);
    Ok(())
}

#[test]
fn identical_bounds() {
    let f   = |x: f64| x;
    let cfg = SecantCfg::new();
    let err = secant(f, 1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, 
            SecantError::InvalidGuess { x0, x1 }
            if x0 == 1.0 && x1 == 1.0
    ));
}


#[test]
fn infinite_eval_start() -> RiverResult {
    let f   = |x: f64| 1.0 / x; 
    let cfg = SecantCfg::new()
        .set_abs_fx(1e-12)?; 

    let err = secant(f, 0.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        SecantError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));

    Ok(())
}

#[test]
fn infinite_eval_mid() -> RiverResult{
    let f   = |x: f64| 1.0 / x;

    let cfg = SecantCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-16)?
        .set_rel_x(0.0)?;

    let err = secant(f, 1.0, -1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        SecantError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));

    Ok(())
}

#[test]
fn step_tol_immediate() -> RiverResult {
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
fn max_iter() -> RiverResult {
    let f   = |x: f64| x.powi(2) + 1.0;
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
fn degenerate_denominator() {
    let f   = |_x: f64| 1.0; 

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
fn parents_generate_root() -> RiverResult {
    let f   = |x: f64| x.powi(2) - 2.0;

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
fn early_absfx_stencil() -> RiverResult {
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
fn early_steptol_stencil() -> RiverResult {
    let f = |x: f64| 3.0*x + 1.0;

    let cfg0 = SecantCfg::new()
        .set_abs_x(1e-3)? 
        .set_rel_x(0.0)?
        .set_abs_fx(1e-30)?;

    let res0 = secant(f, 0.0, 5e-4, cfg0)?;

    assert_eq!(res0.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res0.tolerance_satisfied, ToleranceSatisfied::StepSizeReached);
    assert_eq!(res0.iterations, 0);

    let s0 = res0.stencil.stencil();
    assert_eq!(s0.len(), 2);

    // root in s0[1] when immediate 
    assert_eq!(res0.root, s0[1]);

    Ok(())
}

