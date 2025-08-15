use numena::root_finding::common::{RootFindingError, Termination, ToleranceReason};
use numena::root_finding::regula_falsi::{
    regula_falsi, RegulaFalsiCfg, RegulaFalsiError, RegulaFalsiVariant,
};

type TestResult = Result<(), RegulaFalsiError>;

fn variants() -> Vec<(RegulaFalsiVariant, &'static str, &'static str)> {
    vec![
        (RegulaFalsiVariant::Pure, "pure", "regula_falsi_pure"),
        (RegulaFalsiVariant::Illinois, "illinois", "regula_falsi_illinois"),
        (RegulaFalsiVariant::Pegasus, "pegasus", "regula_falsi_pegasus"),
        (RegulaFalsiVariant::AndersonBjorck, "ab", "regula_falsi_anderson"),
    ]
}

fn base_cfg(v: RegulaFalsiVariant) -> RegulaFalsiCfg {
    RegulaFalsiCfg::new().with_variant(v)
}

fn assert_bracket_invariant<F: Fn(f64) -> f64>(f: &F, left: f64, right: f64) {
    let fa = f(left);
    let fb = f(right);
    assert!(fa.is_finite() && fb.is_finite());
    assert!(left <= right);
    assert!(fa * fb <= 0.0);
} 

// Generic Tests for every variant 
 
#[test]
fn finds_sqrt_2_all_variants() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    let tol = 1e-10;

    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(tol).with_abs_x(tol).with_rel_x(1e-5);
        let res = regula_falsi(&f, -1.0, 2.0, cfg)?;
        assert_eq!(res.termination, Termination::ToleranceReached, "variant={tag}");
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached, "variant={tag}");
        assert!((res.root - 2.0_f64.sqrt()).abs() <= tol, "variant={tag}");
        assert!(res.iterations > 0);
        assert_bracket_invariant(&f, res.left, res.right);
    }
    Ok(())
}

#[test]
fn finds_linear_root_all_variants() -> TestResult {
    let f = |x: f64| 2.0 * x - 6.0;
    let tol = 1e-10;

    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(tol).with_abs_x(tol).with_rel_x(0.0).with_max_iter(10);
        let res = regula_falsi(&f, 0.0, 10.0, cfg)?;
        assert_eq!(res.termination, Termination::ToleranceReached, "variant={tag}");
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached, "variant={tag}");
        assert!((res.root - 3.0).abs() <= tol);
        assert_bracket_invariant(&f, res.left, res.right);
    }
    Ok(())
}

#[test]
fn no_sign_change_all_variants() {
    let f = |x: f64| x * x + 1.0;
    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-10);
        let err = regula_falsi(&f, -1.0, 1.0, cfg).unwrap_err();
        assert!(matches!(err, RegulaFalsiError::NoSignChange { a: -1.0, b: 1.0 }), "variant={tag}");
    }
}

#[test]
fn invalid_bounds_all_variants() {
    let f = |x: f64| x;
    for (v, tag, _) in variants() {
        let cfg = base_cfg(v);
        let err = regula_falsi(&f, 2.0, 0.0, cfg).unwrap_err();
        assert!(matches!(err, RegulaFalsiError::InvalidBounds { .. }), "variant={tag}");
    }
}

#[test]
fn non_finite_eval_propagates_all_variants() {
    let f = |x: f64| x.sqrt() - 2.0;
    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-10);
        let err = regula_falsi(&f, -1.0, 5.0, cfg).unwrap_err();
        assert!(matches!(
            err,
            RegulaFalsiError::Common(RootFindingError::NonFiniteEvaluation { x, fx })
            if x == -1.0 && fx.is_nan()
        ), "variant={tag}");
    }
}

#[test]
fn endpoint_a_is_root_iterations_0_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12);
        let res = regula_falsi(&f, 0.0, 5.0, cfg)?;
        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn endpoint_b_is_root_iterations_0_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12);
        let res = regula_falsi(&f, -5.0, 0.0, cfg)?;
        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn uses_max_iter_1_hits_limit_all_variants() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-30).with_abs_x(0.0).with_rel_x(1e-16).with_max_iter(1);
        let res = regula_falsi(&f, 0.0, 2.0, cfg)?;
        assert_eq!(res.termination, Termination::IterationLimit);
        assert_eq!(res.iterations, 1);
    }
    Ok(())
}

#[test]
fn narrow_interval_stops_on_width_all_variants() -> TestResult {
    let f = |x: f64| x + 1e-16;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_x(1e-12).with_rel_x(0.0).with_abs_fx(1e-20);
        let res = regula_falsi(&f, -3e-16, 1e-16, cfg)?;
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::WidthTolReached);
        assert_bracket_invariant(&f, res.left, res.right);
    }
    Ok(())
}

#[test]
fn high_function_tol_stops_quickly_all_variants() -> TestResult {
    let f = |x: f64| x;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1.0).with_max_iter(5);
        let res = regula_falsi(&f, -5.0, 1.0, cfg)?;
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
        assert!(res.iterations < 3);
    }
    Ok(())
}

#[test]
fn high_rel_tol_ignores_abs_x_all_variants() -> TestResult {
    let f = |x: f64| (x - 11.0) + 1e-8 * (x - 11.0).powi(2);
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_x(1e-12).with_rel_x(0.5).with_max_iter(100);
        let res = regula_falsi(&f, 0.0, 20.0, cfg)?;
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::WidthTolReached);
        assert!(res.iterations < 25);
        assert_bracket_invariant(&f, res.left, res.right);
    }
    Ok(())
}

#[test]
fn degenerate_secant_fallback_is_safe_all_variants() -> TestResult {
    let delta = 1e-18;
    let f = move |x: f64| x + delta;
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-30).with_abs_x(1e-30).with_rel_x(0.0).with_max_iter(10);
        let res = regula_falsi(&f, -1e-16, 1e-16, cfg)?;
        assert!(matches!(res.termination, Termination::ToleranceReached | Termination::IterationLimit));
    }
    Ok(())
} 

// Variant-specific behavior 

#[test]
fn algorithm_field_contains_variant_tag() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    for (v, _, expected) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12).with_abs_x(1e-12);
        let res = regula_falsi(&f, 0.0, 2.0, cfg)?;
        assert!(res.algorithm.contains("regula_falsi"));
        assert!(res.algorithm.contains(expected));
    }
    Ok(())
}

#[test]
fn negative_zero_is_treated_as_zero() -> TestResult {
    let f = |x: f64| if x == 0.0 { -0.0 } else { x };
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12);
        let res = regula_falsi(&f, 0.0, 1.0, cfg)?;
        assert_eq!(res.iterations, 0);
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
        assert_eq!(res.root, 0.0);
    }
    Ok(())
}

#[test]
fn convex_pathology_orders_variants_by_speed() -> TestResult {
    let f = |x: f64| f64::exp(-x) - x;
    let mut iters = Vec::new();

    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12).with_abs_x(1e-12).with_rel_x(0.0).with_max_iter(500);
        let res = regula_falsi(&f, 0.0, 1.0, cfg)?;
        iters.push((tag, res.iterations));
        assert_eq!(res.termination, Termination::ToleranceReached);
        assert_eq!(res.tolerance, ToleranceReason::AbsFxReached);
        assert!((res.root - 0.5671432904097838).abs() < 1e-7);
    }

    let get = |lbl: &str| -> usize { iters.iter().find(|(t, _)| *t == lbl).unwrap().1 };
    let p = get("pure");
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
    for (v, _, _) in variants() {
        let cfg = base_cfg(v).with_abs_fx(1e-12).with_abs_x(1e-12);
        let a0 = -0.5;
        let b0 = 1.0;
        let res = regula_falsi(&f, a0, b0, cfg)?;
        assert!(res.left >= a0 && res.right <= b0);
        assert_bracket_invariant(&f, res.left, res.right);
        assert!(res.root >= res.left && res.root <= res.right);
    }
    Ok(())
}

#[test]
fn invalid_tolerance_is_reported_all_variants() {
    let f = |x: f64| x;
    for (v, tag, _) in variants() {
        let cfg = base_cfg(v).with_abs_x(f64::NAN);
        let err = regula_falsi(&f, -1.0, 1.0, cfg).unwrap_err();
        assert!(matches!(err, RegulaFalsiError::Common(RootFindingError::InvalidAbsX { .. })), "variant={tag}");
    }
}

#[test]
fn pure_variant_reaches_iteration_limit_on_quadratic() -> TestResult {
    let f = |x: f64| x * x - 2.0;
    let cfg = base_cfg(RegulaFalsiVariant::Pure)
        .with_abs_fx(1e-30)
        .with_abs_x(0.0)
        .with_rel_x(1e-16)
        .with_max_iter(25);

    let res = regula_falsi(&f, 0.0, 2.0, cfg)?;
    assert!(matches!(res.termination, Termination::IterationLimit));
    Ok(())
}

