//! tests for the regula falsi root-finding algorithm(s) with each variant

use numena::root_finding::algorithms::{Algorithm, BracketFamily};
use numena::root_finding::errors::{RootFindingError, ToleranceError};
use numena::root_finding::regula_falsi::{regula_falsi, RegulaFalsiCfg, RegulaFalsiError};
use numena::root_finding::report::{RootFindingReport, TerminationReason, ToleranceSatisfied, Stencil};

type TestResult = Result<(), RegulaFalsiError>;

fn variants() -> Vec<(BracketFamily, &'static str, &'static str)> {
    use BracketFamily as BF;
    vec![
        (BF::RegulaFalsiPure,           "pure",     Algorithm::Bracket(BF::RegulaFalsiPure).algorithm_name()),
        (BF::RegulaFalsiIllinois,       "illinois", Algorithm::Bracket(BF::RegulaFalsiIllinois).algorithm_name()),
        (BF::RegulaFalsiPegasus,        "pegasus",  Algorithm::Bracket(BF::RegulaFalsiPegasus).algorithm_name()),
        (BF::RegulaFalsiAndersonBjorck, "ab",       Algorithm::Bracket(BF::RegulaFalsiAndersonBjorck).algorithm_name()),
    ]
}

fn base_cfg(v: BracketFamily) -> RegulaFalsiCfg {
    RegulaFalsiCfg::new().set_variant(v).unwrap()
}

fn assert_bracket_invariant<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) {
    let fa = f(a);
    let fb = f(b);
    assert!(fa.is_finite() && fb.is_finite());
    assert!(a <= b);
    assert!(fa * fb <= 0.0);
}

fn bounds(res: &RootFindingReport) -> (f64, f64) {
    match res.stencil {
        Stencil::Bracket { bounds } => (bounds[0], bounds[1]),
        _ => panic!("expected Bracket stencil"),
    }
}

// Generic tests 

#[test]
fn finds_sqrt_2_all_variants() -> TestResult {
    let f   = |x: f64| x * x - 2.0;
    let tol = 1e-10;

    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(tol)?
            .set_abs_x(tol)?
            .set_rel_x(1e-5)?;
        let res = regula_falsi(f, 0.0, 2.0, cfg)?;

        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached, "variant={tag}");
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached,    "variant={tag}");
        assert!((res.root - 2.0_f64.sqrt()).abs() <= tol, "variant={tag}");
        assert!(res.iterations > 0, "variant={tag}");
        let (a, b) = bounds(&res);
        assert_bracket_invariant(&f, a, b);
    }
    Ok(())
}

#[test]
fn finds_linear_root_all_variants() -> TestResult {
    let f   = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(tol)?
            .set_abs_x(tol)?
            .set_rel_x(0.0)?
            .set_max_iter(10)?;
        let res = regula_falsi(f, 0.0, 10.0, cfg)?;

        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached, "variant={tag}");
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached,    "variant={tag}");
        assert!((res.root - 3.0).abs() <= tol, "variant={tag}");
        let (a, b) = bounds(&res);
        assert_bracket_invariant(&f, a, b);
    }
    Ok(())
}

#[test]
fn no_sign_change_all_variants() {
    let f = |x: f64| x * x + 1.0;
    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant).set_abs_fx(1e-10).unwrap();
        let err = regula_falsi(f, -1.0, 1.0, cfg).unwrap_err();
        assert!(matches!(err, RegulaFalsiError::NoSignChange { a: -1.0, b: 1.0 }), "variant={tag}");
    }
}

#[test]
fn invalid_bounds_all_variants() {
    let f = |x: f64| x;
    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant);
        let err = regula_falsi(f, 2.0, 0.0, cfg).unwrap_err();
        assert!(matches!(err, RegulaFalsiError::InvalidBounds { .. }), "variant={tag}");
    }
}

#[test]
fn non_finite_eval_propagates_all_variants() {
    let f = |x: f64| x.sqrt() - 2.0;
    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant).set_abs_fx(1e-10).unwrap();
        let err = regula_falsi(f, -1.0, 5.0, cfg).unwrap_err();
        assert!(matches!(
            err,
            RegulaFalsiError::RootFinding(RootFindingError::NonFiniteEvaluation { x, fx })
            if x == -1.0 && fx.is_nan()
        ), "variant={tag}");
    }
}

#[test]
fn endpoint_a_is_root_iterations_0_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant).set_abs_fx(1e-12)?;
        let res = regula_falsi(f, 0.0, 5.0, cfg)?;

        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn endpoint_b_is_root_iterations_0_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant).set_abs_fx(1e-12)?;
        let res = regula_falsi(f, -5.0, 0.0, cfg)?;

        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn uses_max_iter_1_hits_limit_all_variants() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1e-30)?
            .set_abs_x(0.0)?
            .set_rel_x(1e-16)?
            .set_max_iter(1)?;
        let res = regula_falsi(f, 0.0, 2.0, cfg)?;

        assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
        assert_eq!(res.iterations, 1);
    }
    Ok(())
}

#[test]
fn narrow_interval_stops_on_width_all_variants() -> TestResult {
    let f = |x: f64| x + 1e-16;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_x(1e-12)?
            .set_rel_x(0.0)?
            .set_abs_fx(1e-20)?;
        let res = regula_falsi(f, -3e-16, 1e-16, cfg)?;

        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);
        let (a, b) = bounds(&res);
        assert_bracket_invariant(&f, a, b);
    }
    Ok(())
}

#[test]
fn high_function_tol_stops_quickly_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1.0)?
            .set_max_iter(5)?;
        let res = regula_falsi(f, -5.0, 1.0, cfg)?;

        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
        assert!(res.iterations < 3);
    }
    Ok(())
}

#[test]
fn high_rel_tol_ignores_abs_x_all_variants() -> TestResult {
    let f = |x: f64| (x - 11.0) + 1e-8 * (x - 11.0).powi(2);
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_x(1e-12)?
            .set_rel_x(0.5)? // huge relative tolerance
            .set_max_iter(100)?;
        let res = regula_falsi(f, 0.0, 20.0, cfg)?;

        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::WidthTolReached);
        assert!(res.iterations < 25);
        let (a, b) = bounds(&res);
        assert_bracket_invariant(&f, a, b);
    }
    Ok(())
}

#[test]
fn degenerate_secant_fallback_is_safe_all_variants() -> TestResult {
    let delta = 1e-18;
    let f = move |x: f64| x + delta;
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1e-30)?
            .set_abs_x(1e-30)?
            .set_rel_x(0.0)?
            .set_max_iter(10)?;
        let res = regula_falsi(f, -1e-16, 1e-16, cfg)?;
        assert!(
            matches!(res.termination_reason, TerminationReason::ToleranceReached | TerminationReason::IterationLimit),
            "unexpected termination: {:?}", res.termination_reason
        );
    }
    Ok(())
}

// Variant-specific behavior

#[test]
fn algorithm_field_contains_variant_tag() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    for (variant, _, expected_full) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1e-12)?
            .set_abs_x(1e-12)?;
        let res = regula_falsi(f, 0.0, 2.0, cfg)?;
        assert!(res.algorithm_name.contains("regula_falsi"));
        assert_eq!(res.algorithm_name, expected_full);
    }
    Ok(())
}

#[test]
fn negative_zero_is_treated_as_zero() -> TestResult {
    let f = |x: f64| if x == 0.0 { -0.0 } else { x };
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant).set_abs_fx(1e-12)?;
        let res = regula_falsi(f, 0.0, 1.0, cfg)?;
        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn convex_pathology_orders_variants_by_speed() -> TestResult {
    let f = |x: f64| f64::exp(-x) - x;
    let mut iters: Vec<(&'static str, usize)> = Vec::new();

    for (variant, tag, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1e-12)?
            .set_abs_x(1e-12)?
            .set_rel_x(0.0)?
            .set_max_iter(500)?;
        let res = regula_falsi(f, 0.0, 1.0, cfg)?;
        iters.push((tag, res.iterations));
        assert_eq!(res.termination_reason,  TerminationReason::ToleranceReached);
        assert_eq!(res.tolerance_satisfied, ToleranceSatisfied::AbsFxReached);
        assert!((res.root - 0.567_143_290_409_783_8).abs() < 1e-7);
    }

    let get = |lbl: &str| -> usize { iters.iter().find(|(t, _)| *t == lbl).unwrap().1 };
    let p  = get("pure");
    let il = get("illinois");
    let pg = get("pegasus");
    let ab = get("ab");

    assert!(il < p || pg < p || ab < p);
    let best_improved = il.min(pg);
    assert!(ab <= best_improved + 2);
    Ok(())
}

#[test]
fn final_bracket_still_straddles_root_all_variants() -> TestResult {
    let f = |x: f64| (x - 1.7) * (x - 0.2) * (x + 3.0);
    for (variant, _, _) in variants() {
        let cfg = base_cfg(variant)
            .set_abs_fx(1e-12)?
            .set_abs_x(1e-12)?;
        let a0 = -0.5;
        let b0 = 1.0;
        let res = regula_falsi(f, a0, b0, cfg)?;

        let (a, b) = bounds(&res);
        assert!(a >= a0 && b <= b0);
        assert_bracket_invariant(&f, a, b);
        assert!(res.root >= a && res.root <= b);
    }
    Ok(())
}

#[test]
fn invalid_tolerance_is_reported_via_setter_all_variants() {
    // With new API, invalid tolerance is caught at setter time.
    for (variant, tag, _) in variants() {
        let cfg_res: Result<RegulaFalsiCfg, ToleranceError> =
            base_cfg(variant).set_abs_x(f64::NAN);
        let err = cfg_res.expect_err(&format!("expected setter to fail, variant={tag}"));
        assert!(matches!(err, ToleranceError::InvalidAbsX { .. }), "variant={tag}");
        // If you prefer to assert via RegulaFalsiError:
        let as_rf: RegulaFalsiError = err.into();
        assert!(matches!(as_rf, RegulaFalsiError::Tolerance(ToleranceError::InvalidAbsX { .. })));
    }
}

#[test]
fn pure_variant_reaches_iter_limit_on_quadratic() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    let cfg = base_cfg(BracketFamily::RegulaFalsiPure)
        .set_abs_fx(1e-30)?
        .set_abs_x(0.0)?
        .set_rel_x(1e-16)?
        .set_max_iter(25)?;
    let res = regula_falsi(f, 0.0, 2.0, cfg)?;
    assert_eq!(res.termination_reason, TerminationReason::IterationLimit);
    Ok(())
}

