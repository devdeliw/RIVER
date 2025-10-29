use river::root_finding::bisection::{bisection, BisectionCfg, BisectionError}; 
use river::root_finding::errors::RootFindingError; 
use river::root_finding::report::{TerminationReason, ToleranceSatisfied};

type RiverResult = Result<(), BisectionError>;
    
#[test] 
fn sqrt2() -> RiverResult { 
    let f = |x: f64| x.powi(2) - 2.0; 

    // bounds 
    let a = 0.0; 
    let b = 2.0; 

    // config 
    let abs_fx   = 1e-15; 
    let abs_x    = 1e-10; 
    let rel_x    = 0.0; 
    let max_iter = 60;  

    let cfg = BisectionCfg::new() 
        .set_abs_fx(abs_fx)? 
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?; 

    // render
    let res = bisection(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached); 
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached); 
    assert!(res.iterations > 0);     

    assert!((res.root - (2.0 as f64).sqrt()).abs() <= abs_x);

    Ok(())
}

#[test]
fn three() -> RiverResult {
    let f = |x: f64| 2.0 * x - 6.0;

    let a = 0.0; 
    let b = 10.0; 

    let abs_fx   = 1e-15;
    let abs_x    = 1e-10; 
    let rel_x    = 0.0; 
    let max_iter = 60;

    let cfg = BisectionCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?; 

    let res = bisection(f, a, b, cfg)?; 

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);
    assert!(res.iterations > 0);

    assert!((res.root - 3.0).abs() <= abs_x);

    Ok(())
}

#[test]
fn no_sign_change() -> RiverResult {
    let f = |x: f64| x.powi(2) + 1.0;

    let cfg = BisectionCfg::new()
        .set_abs_fx(1e-10)?;

    let err = bisection(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(err, BisectionError::NoSignChange { a: -1.0, b: 1.0 }));

    Ok(())
}

#[test]
fn non_finite_eval() -> RiverResult {
    let f = |x: f64| x.sqrt() - 2.0;

    let cfg = BisectionCfg::new()
        .set_abs_fx(1e-10)?;

    let err = bisection(f, -1.0, 5.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        BisectionError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == -1.0 && fx.is_nan()
    ));

    Ok(())
}

#[test]
fn negative5() -> RiverResult {
    let f = |x: f64| x + 5.0;

    let a = -10.0; 
    let b = 0.0; 
    
    let abs_fx   = 1e-20; 
    let abs_x    = 1e-10; 
    let rel_x    = 0.0; 
    let max_iter = 60;

    let cfg = BisectionCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?;

    let res = bisection(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!(res.iterations > 0);

    assert!((res.root - -5.0).abs() <= abs_x);

    Ok(())
}

#[test]
fn max_iter() -> RiverResult {
    let f = |x: f64| x;

    let a = -3.0; 
    let b = 2.0; 

    let abs_fx   = 1e-20; 
    let abs_x    = 1e-20; 
    let rel_x    = 0.0; 
    let max_iter = 30; 

    let cfg = BisectionCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_rel_x(rel_x)?
        .set_max_iter(max_iter)?;

    let res = bisection(f, a, b, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, max_iter); 

    Ok(())
}

#[test]
fn invalid_bounds() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new();
    let err = bisection(f, 2.0, 0.0, cfg).unwrap_err();

    assert!(matches!(err, BisectionError::InvalidBounds { .. }));

    Ok(())
}

#[test]
fn endpoint_a_root() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new(); 

    let res = bisection(f, 0.0, 5.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert_eq!(res.iterations, 0);
    Ok(())
}

#[test]
fn endpoint_b_root() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new();

    let res = bisection(f, -5.0, 0.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    assert_eq!(res.iterations, 0);
    Ok(())
}

#[test]
fn pathological_flat() -> RiverResult {
    let f = |x: f64| (x - 1.0).powi(3);

    let abs_fx   = 1e-10;  
    let abs_x    = 1e-10;
    let max_iter = 80; 

    let cfg = BisectionCfg::new()
        .set_abs_fx(abs_fx)?
        .set_abs_x(abs_x)?
        .set_max_iter(max_iter)?;

    let res = bisection(f, -2.0, 2.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!(res.iterations > 0);

    assert!((res.root - 1.0).abs() <= abs_x);

    Ok(())
}

#[test]
fn small_interval_exits() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new()
        .set_abs_x(1e-12)?
        .set_abs_fx(1e-20)?;

    let res = bisection(f, -3e-16, 1e-16, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);

    Ok(())
}

#[test]
fn high_tol_exits() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new()
        .set_abs_fx(1.0)?
        .set_max_iter(5)?;

    let res = bisection(f, -5.0, 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
    assert!(res.iterations < 3);

    Ok(())
}

#[test]
fn max_iter_1() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new().set_max_iter(1)?;
    let res = bisection(f, -5.0, 1.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    assert_eq!(res.iterations, 1);

    Ok(())
}

#[test]
fn identical_bounds() -> RiverResult {
    let f   = |x: f64| x;
    let cfg = BisectionCfg::new();
    let err = bisection(f, 1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
            err, 
            BisectionError::InvalidBounds { a, b } 
            if a == 1.0 && b == 1.0
    ));

    Ok(())
}


#[test]
fn infinite_eval() -> RiverResult {
    let f   = |x: f64| 1.0 / x;
    let cfg = BisectionCfg::new().set_abs_fx(1e-12)?;
    let err = bisection(f, -1.0, 1.0, cfg).unwrap_err();

    assert!(matches!(
        err,
        BisectionError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
        if x == 0.0 && fx.is_infinite()
    ));
    Ok(())
}

#[test]
fn both_endpoints_roots() -> RiverResult {
    let f   = |_x: f64| 0.0;
    let cfg = BisectionCfg::new().set_abs_fx(1e-12)?;
    let res = bisection(f, 1.0, 2.0, cfg)?;

    assert_eq!(res.root, 1.0);
    assert_eq!(res.iterations, 0);

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);

    Ok(())
}

#[test]
fn high_rel_tol() -> RiverResult {
    let f = |x: f64| x - 10.0;

    let cfg = BisectionCfg::new()
        .set_abs_x(1e-12)?
        .set_rel_x(0.5)? 
        .set_max_iter(100)?;

    let res = bisection(f, 0.0, 21.0, cfg)?;

    assert_eq!(res.termination_reason, TerminationReason::ToleranceReached);
    assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);

    assert!(res.iterations < 5);

    Ok(())
}

