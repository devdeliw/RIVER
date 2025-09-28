use river::root_finding::brent::{brent, BrentCfg, BrentError};
use river::root_finding::errors::RootFindingError;
use river::root_finding::report::{TerminationReason, ToleranceSatisfied};

type RiverResult = Result<(), BrentError>;

#[test]
fn sqrt2() -> RiverResult {
    let f   = |x: f64| x.powi(2) - 2.0;

    let a = 0.0; 
    let b = 2.0; 

    let abs_fx   = 1e-30; 
    let abs_x    = 1e-15; 
    let rel_x    = 0.0; 
    let max_iter = 60; 

    let cfg = BrentCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?;

    let res = brent(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);

    assert!((res.root - (2.0 as f64).sqrt()).abs() <= abs_x);
    assert!(res.iterations > 0);

    Ok(())
}

#[test]
fn three() -> RiverResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-12;

    let cfg = BrentCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(tol)?
        .set_rel_x(0.0)?
        .set_max_iter(30)?;

    let res = brent(f, 0.0, 10.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert!((res.root - 3.0).abs() <= tol);
    
    Ok(())
}

#[test]
fn negative5() -> RiverResult {
    let f   = |x: f64| x + 5.0;
    let tol = 1e-12;

    let cfg = BrentCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(tol)?
        .set_rel_x(0.0)?
        .set_max_iter(30)?;

    let res = brent(f, -10.0, 0.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert!((res.root + 5.0).abs() <= tol);

    Ok(())
}

#[test]
fn no_sign_change() -> RiverResult {
    let f   = |x: f64| x * x + 1.0;
    let cfg = BrentCfg::new()
        .set_abs_fx(1e-10)?;

    let err = brent(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
            err, 
            BrentError::NoSignChange { a: -1.0, b: 1.0 }
    ));

    Ok(())
}

#[test]
fn infinite_eval_start() -> RiverResult {
    let f   = |x: f64| x.sqrt() - 2.0;
    let cfg = BrentCfg::new().set_abs_fx(1e-10)?;

    let err = brent(f, -1.0, 5.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        BrentError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == -1.0 && fx.is_nan()
    ));

    Ok(())
}

#[test]
fn infinite_eval_mid() -> RiverResult {
    let f   = |x: f64| 1.0 / x;
    let cfg = BrentCfg::new().set_abs_fx(1e-12)?;

    let err = brent(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        BrentError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));

    Ok(())
}

#[test]
fn invalid_bounds() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new();

    let err = brent(f, 2.0, 0.0, cfg).unwrap_err();

    assert!(matches!(
            err, 
            BrentError::InvalidBounds { .. }
    ));

    Ok(())
}

#[test]
fn identical_bounds() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new();

    let err = brent(f, 1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
            err, 
            BrentError::InvalidBounds { a, b } 
            if a == 1.0 && b == 1.0
    ));

    Ok(())
}

#[test]
fn endpoint_a_root() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new().set_abs_fx(1e-10)?;

    let res = brent(f, 0.0, 5.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert_eq!(res.root, 0.0);
    assert_eq!(res.iterations, 0);

    Ok(())
}

#[test]
fn endpoint_b_root() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new().set_abs_fx(1e-10)?;

    let res = brent(f, -5.0, 0.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert_eq!(res.root, 0.0);
    assert_eq!(res.iterations, 0);

    Ok(())
}

#[test]
fn small_interval() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new()
        .set_abs_x(1e-12)?
        .set_abs_fx(1e-20)?;

    let res = brent(f, -3e-16, 1e-16, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);

    Ok(())
}

#[test]
fn high_tol_exits() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BrentCfg::new()
        .set_abs_fx(1.0)?
        .set_max_iter(5)?;

    let res = brent(f, -5.0, 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert!(res.iterations == 0);

    Ok(())
}

#[test]
fn max_iter() -> RiverResult {
    let f   = |x: f64| (x - 1.0).powi(3);

    let cfg = BrentCfg::new()
        .set_abs_fx(1e-30)?
        .set_abs_x(0.0)?
        .set_rel_x(1e-16)?
        .set_max_iter(1)?;

    let res = brent(f, -2.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);

    Ok(())
}

#[test]
fn pathological_flat() -> RiverResult {
    let f   = |x: f64| (x - 1.0).powi(3);
    let tol = 1e-10;

    let cfg = BrentCfg::new()
        .set_abs_fx(tol)?
        .set_abs_x(tol)?
        .set_max_iter(100)?;

    let res = brent(f, -2.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert!(matches!(
        res.tolerance_satisfied,
        ToleranceSatisfied::AbsFxReached | ToleranceSatisfied::WidthTolReached
    ));
    assert!((res.root - 1.0).abs() <= 1e-6); 

    Ok(())
}

#[test]
fn both_endpoints_root() -> RiverResult {
    let f   = |_x: f64| 0.0;
    let cfg = BrentCfg::new().set_abs_fx(1e-12)?;

    let res = brent(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.root, 1.0);
    assert_eq!(res.iterations, 0);
    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    Ok(())
}

#[test]
fn high_rel_tol() -> RiverResult {
    let f   = |x: f64| x - 10.0;

    let cfg = BrentCfg::new()
        .set_abs_x(1e-12)?
        .set_rel_x(0.5)? 
        .set_max_iter(10)?;

    let res = brent(f, -2.0, 21.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);
    assert!(res.iterations <= 2);

    Ok(())
}

#[test]
fn nearly_equal_fafb() -> RiverResult {
    let f   = |x: f64| x - 1.0;

    let a = 1.0 - 1.0e-14;
    let b = 1.0 + 1.0e-14;

    let cfg = BrentCfg::new()
        .set_abs_fx(1e-20)?
        .set_abs_x(0.0)?     
        .set_rel_x(1e-16)?   
        .set_max_iter(10)?;

    let res = brent(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!(res.iterations <= 2);

    Ok(())
}

#[test]
fn very_large_scale() -> RiverResult {
    let f   = |x: f64| x - 1.0e12;

    let cfg = BrentCfg::new()
        .set_abs_fx(1e-8)?     
        .set_abs_x(1e-6)?
        .set_rel_x(1e-12)?     
        .set_max_iter(100)?;

    let res = brent(f, 1.0e11, 2.0e12, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert!( (res.root - 1.0e12).abs() <= 1e6 ); 

    Ok(())
}

#[test]
fn asymmetric_bracket() -> RiverResult {
    let f   = |x: f64| x - 2.0;

    let cfg = BrentCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-12)?
        .set_rel_x(1e-12)?
        .set_max_iter(100)?;

    let res = brent(f, -1.0e9, 3.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert!((res.root - 2.0).abs() <= 1e-6);

    Ok(())
}

#[test]
fn curved_monotone() -> RiverResult {
    let f   = |x: f64| x.exp() - 3.0;

    let cfg = BrentCfg::new()
        .set_abs_fx(1e-12)?
        .set_abs_x(1e-12)?
        .set_rel_x(1e-12)?
        .set_max_iter(50)?;

    let res = brent(f, 0.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert!((res.root - 3.0_f64.ln()).abs() <= 1e-8);

    Ok(())
}

