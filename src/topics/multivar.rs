//! Multivariable calculus: partial derivatives on a surface / contour map.

use crate::core::expr::parse_expr_text;
use crate::scene::{Report, Row, SceneSpec};
use crate::theme;

pub fn partial(fn_text: &str, domain: (f64, f64)) -> Result<SceneSpec, String> {
    let f = parse_expr_text(fn_text, &["x", "y"]).map_err(|e| e.0)?;
    check_domain(domain)?;
    let report = Report {
        title: "Partial derivatives on a surface".into(),
        formulas: vec![
            "∂f/∂x = slope along x (y held fixed)".into(),
            "∂f/∂y = slope along y (x held fixed)".into(),
        ],
        body: vec![
            Row::plain(format!("f(x, y) = {fn_text}")),
            Row::colored(
                format!("domain: [{}, {}]²", domain.0, domain.1),
                theme::MUTED,
            ),
            Row::colored("top view = the contour map", theme::MUTED),
        ],
    };
    Ok(SceneSpec::Surface { f, domain, report })
}

pub fn contour(fn_text: &str, domain: (f64, f64)) -> Result<SceneSpec, String> {
    let f = parse_expr_text(fn_text, &["x", "y"]).map_err(|e| e.0)?;
    check_domain(domain)?;
    let report = Report {
        title: "Contour map & partial derivatives".into(),
        formulas: vec!["∇f = (∂f/∂x, ∂f/∂y)".into()],
        body: vec![
            Row::plain(format!("f(x, y) = {fn_text}")),
            Row::colored(
                format!("domain: [{}, {}]²", domain.0, domain.1),
                theme::MUTED,
            ),
            Row::colored("drag the x/y sliders to move the point", theme::MUTED),
        ],
    };
    Ok(SceneSpec::Contour { f, domain, report })
}

fn check_domain(domain: (f64, f64)) -> Result<(), String> {
    if domain.0 >= domain.1 {
        Err(format!(
            "domain must be 'a b' with a < b (got {} {})",
            domain.0, domain.1
        ))
    } else {
        Ok(())
    }
}
