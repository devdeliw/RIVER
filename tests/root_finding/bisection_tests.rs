//! tests for the bisection root finding algorithm 
use numena::root_finding::common::{RootFindingError, Termination, ToleranceReason}; 
use numena::root_finding::bisection::{bisection, BisectionCfg, BisectionError};

type TestResult = Result<(), BisectionError>;

#[test]
fn finds_sqrt_2() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let tol = 1e-10;

    let cfg = BisectionCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(tol)
        .with_max_iter(60)
        .with_rel_x(0.0); 

    let res = bisection(f, 0.0, 2.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert!((res.root - 2.0_f64.sqrt()).abs() <= tol);
    assert!(res.iterations > 0);
    Ok(())
}

#[test]
fn finds_3() -> TestResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    let cfg = BisectionCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(tol)
        .with_max_iter(60)
        .with_rel_x(0.0);

    let res = bisection(f, 0.0, 10.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert!((res.root - 3.0_f64).abs() <= tol);
    assert!(res.iterations > 0);
    Ok(())
}

#[test]
fn no_sign_change() -> TestResult {
    let f   = |x: f64| x * x + 1.0;
    let cfg = BisectionCfg::new().with_abs_fx(1e-10);
    let err = bisection(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, BisectionError::NoSignChange { a: -1.0, b: 1.0 }));
    Ok(())
}

#[test]
fn non_finite_eval() -> TestResult {
    let f   = |x: f64| x.sqrt() - 2.0; 
    let cfg = BisectionCfg::new().with_abs_fx(1e-10);
    let err = bisection(f, -1.0, 5.0, cfg).unwrap_err(); 

    assert!(matches!(
        err, 
        BisectionError::Common(RootFindingError::NonFiniteEvaluation { x, fx }) 
        if x == -1.0 && fx.is_nan()));
    Ok(())
}

#[test]
fn finds_negative_5() -> TestResult {
    let f   = |x: f64| x + 5.0;
    let tol = 1e-10;

    let cfg = BisectionCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(tol)
        .with_max_iter(60);

    let res = bisection(f, -10.0, 0.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert!((res.root - (-5.0_f64)).abs() <= tol);
    assert!(res.iterations > 0);
    Ok(())
}

#[test]
fn uses_max_iter() -> TestResult {
    let f     = |x: f64| x;
    let niter = 10;

    let cfg = BisectionCfg::new()
        .with_abs_fx(1e-30)    
        .with_rel_x(1e-12) 
        .with_abs_x(0.0)      
        .with_max_iter(niter);

    let res = bisection(f, -3.0, 2.0, cfg)?;

    assert_eq!(res.termination, Termination::IterationLimit);
    assert_eq!(res.iterations, niter);
    Ok(())
}

#[test]
fn detects_invalid_bounds() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new(); 
    let err = bisection(f, 2.0, 0.0, cfg).unwrap_err();
    assert!(matches!(err, BisectionError::InvalidBounds { a: _, b: _ }));
    Ok(())
}

#[test]
fn endpoint_a_is_root_iterations_0() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new().with_abs_fx(1e-10);
    let res = bisection(f, 0.0, 5.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert_eq!(res.iterations, 0); 
    Ok(())
}

#[test]
fn endpoint_b_is_root_iterations_0() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new().with_abs_fx(1e-10);
    let res = bisection(f, -5.0, 0.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert_eq!(res.iterations, 0); 
    Ok(())
}

#[test]
fn pathological_flat() -> TestResult {
    let f   = |x: f64| (x - 1.0).powi(3);
    let tol = 1e-10;

    let cfg = BisectionCfg::new()
        .with_abs_fx(tol)
        .with_abs_x(tol)
        .with_max_iter(80);

    let res = bisection(f, -2.0, 2.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert!((res.root - 1.0).abs() <= tol);
    assert!(res.iterations > 0);
    Ok(())
}

#[test]
fn narrow_interval_stops_on_width() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new()
        .with_abs_x(1e-12)
        .with_abs_fx(1e-20);
    let res = bisection(f, -3e-16, 1e-16, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::WidthTolReached);
    Ok(())
}

#[test]
fn high_function_tol_stops_quickly() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new()
        .with_abs_fx(1.0)      
        .with_max_iter(5);     

    let res = bisection(f, -5.0, 1.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    assert!(res.iterations < 3); 
    Ok(())
}

#[test]
fn max_iter_1_hits_limit() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new().with_max_iter(1);
    let res = bisection(f, -5.0, 1.0, cfg)?;

    assert_eq!(res.termination, Termination::IterationLimit);
    assert_eq!(res.iterations, 1);
    Ok(())
}

#[test]
fn identical_bounds_are_invalid() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new();
    let err = bisection(f, 1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, BisectionError::InvalidBounds { a, b } if a == 1.0 && b == 1.0));
    Ok(())
}

#[test]
fn one_endpoint_exact_root_other_not() -> TestResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new().with_abs_fx(1e-12);
    let res = bisection(f, 0.0, 2.0, cfg)?;

    assert_eq!(res.root, 0.0);
    assert_eq!(res.iterations, 0);
    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    Ok(())
}

#[test]
fn infinite_function_value() -> TestResult {
    let f   = |x: f64| 1.0 / x;
    let cfg = BisectionCfg::new().with_abs_fx(1e-12);
    let err = bisection(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err, 
        BisectionError::Common(RootFindingError::NonFiniteEvaluation{ x, fx })
        if x == 0.0 && fx.is_infinite()));
    Ok(())
}

#[test]
fn both_endpoints_are_roots_picks_first() -> TestResult {
    let f   = |_x: f64| 0.0;
    let cfg = BisectionCfg::new().with_abs_fx(1e-12);
    let res = bisection(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.root, 1.0);
    assert_eq!(res.iterations, 0);
    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
    Ok(())
}

#[test]
fn high_rel_tol_ignores_abs_x() -> TestResult {
    let f   = |x: f64| x - 10.0;
    let cfg = BisectionCfg::new()
        .with_abs_x(1e-12)
        .with_rel_x(0.5) // huge relative tolerance
        .with_max_iter(100);
    let res = bisection(f, 0.0, 21.0, cfg)?;

    assert_eq!(res.termination, Termination::ToleranceReached);
    assert_eq!(res.tolerance, ToleranceReason::WidthTolReached);
    assert!(res.iterations < 5);
    Ok(())
}
