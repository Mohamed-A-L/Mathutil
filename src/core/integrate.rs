//! Parse and evaluate iterated integrals for the sweep visualizations.
//!
//! An integral is entered as an integrand plus per-variable blocks, innermost
//! first: `x*y : y 0 x : x 0 1` means ∫₀¹ ∫₀ˣ x·y dy dx. The first block (`y`)
//! is the innermost variable — swept first — and its bounds may reference any
//! outer variable; the outermost block must have constant bounds.
//!
//! Where the Python version asked sympy for an exact value, this port uses
//! nested Gauss–Legendre quadrature (order 32 per axis), which is accurate to
//! ~1e-10 for textbook integrands. Riemann cells for the animation are
//! generated identically to the original.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::expr::{parse_expr_text, Expr, ExprError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegralError(pub String);

impl fmt::Display for IntegralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for IntegralError {}

fn err<T>(msg: impl Into<String>) -> Result<T, IntegralError> {
    Err(IntegralError(msg.into()))
}

/// One variable of integration and its (possibly dependent) bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Axis {
    pub var: String,
    pub lo: Expr,
    pub hi: Expr,
    /// Variables the bounds may depend on (the axes listed after this one).
    pub outer: Vec<String>,
}

impl Axis {
    pub fn bounds(&self, outer_values: &HashMap<String, f64>) -> (f64, f64) {
        let args: Vec<f64> = self.outer.iter().map(|v| outer_values[v]).collect();
        (self.lo.eval(&args), self.hi.eval(&args))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegralSpec {
    pub integrand: Expr,
    /// Innermost first.
    pub axes: Vec<Axis>,
    /// Innermost first; the order `integrand` expects its arguments.
    pub variables: Vec<String>,
}

impl IntegralSpec {
    pub fn dim(&self) -> usize {
        self.axes.len()
    }

    pub fn value_at(&self, values: &HashMap<String, f64>) -> f64 {
        let args: Vec<f64> = self.variables.iter().map(|v| values[v]).collect();
        self.integrand.eval(&args)
    }
}

/// A Riemann cell: its box in variable space, center, size, and f value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub ranges: HashMap<String, (f64, f64)>,
    pub centers: HashMap<String, f64>,
    pub dvol: f64,
    pub value: f64,
    /// (outer, ..., inner)
    pub index: Vec<usize>,
}

/// Build an [`IntegralSpec`] from an integrand and inner→outer blocks.
pub fn parse_integral(
    integrand_text: &str,
    block_texts: &[String],
) -> Result<IntegralSpec, IntegralError> {
    if block_texts.is_empty() {
        return err("no integration variables given (use ': var lo hi')");
    }
    let parsed: Vec<(String, String, String)> = block_texts
        .iter()
        .map(|b| parse_block(b))
        .collect::<Result<_, _>>()?; // innermost first
    let var_names: Vec<String> = parsed.iter().map(|(n, _, _)| n.clone()).collect();
    {
        let mut uniq = var_names.clone();
        uniq.sort();
        uniq.dedup();
        if uniq.len() != var_names.len() {
            return err(format!("repeated integration variable in {var_names:?}"));
        }
    }

    let var_refs: Vec<&str> = var_names.iter().map(String::as_str).collect();
    let integrand =
        parse_expr_text(integrand_text, &var_refs).map_err(|ExprError(e)| IntegralError(e))?;

    let mut axes = Vec::new();
    for (i, (name, lo_txt, hi_txt)) in parsed.iter().enumerate() {
        let outer: Vec<&str> = var_refs[i + 1..].to_vec(); // listed after = outer
        let lo = parse_expr_text(lo_txt, &outer)
            .map_err(|ExprError(e)| IntegralError(format!("bounds for '{name}': {e}")))?;
        let hi = parse_expr_text(hi_txt, &outer)
            .map_err(|ExprError(e)| IntegralError(format!("bounds for '{name}': {e}")))?;
        axes.push(Axis {
            var: name.clone(),
            lo,
            hi,
            outer: outer.iter().map(|s| s.to_string()).collect(),
        });
    }
    Ok(IntegralSpec {
        integrand,
        axes,
        variables: var_names,
    })
}

/// Parse `"var lo hi"` or `"var lo..hi"` into (var, lo_text, hi_text).
fn parse_block(text: &str) -> Result<(String, String, String), IntegralError> {
    let text = text.trim();
    if text.is_empty() {
        return err("empty integration block (expected 'var lo hi')");
    }
    let mut parts = text.splitn(2, char::is_whitespace);
    let var = parts.next().unwrap().to_string();
    let rest = match parts.next() {
        Some(r) => r.trim(),
        None => return err(format!("block '{text}' needs a variable and two bounds")),
    };
    if rest.is_empty() {
        return err(format!("block '{text}' needs a variable and two bounds"));
    }
    let (lo, hi) = if let Some((lo, hi)) = rest.split_once("..") {
        (lo, hi)
    } else {
        let toks: Vec<&str> = rest.split_whitespace().collect();
        if toks.len() != 2 {
            return err(format!(
                "bounds for '{var}' must be 'lo hi' or 'lo..hi' (got '{rest}'); \
                 use no spaces inside a bound, e.g. 2*x not '2 x'"
            ));
        }
        (toks[0], toks[1])
    };
    Ok((var, lo.trim().to_string(), hi.trim().to_string()))
}

// ------------------------------------------------------------- quadrature

/// 32-point Gauss–Legendre nodes/weights on [-1, 1], generated by Newton
/// iteration on the Legendre polynomial (standard Golub-free construction).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess: Chebyshev approximation of the i-th root.
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        loop {
            // Evaluate P_n(x) and P'_n(x) by recurrence.
            let (mut p0, mut p1) = (1.0f64, x);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            let dp = (n as f64) * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                let (mut q0, mut q1) = (1.0f64, x);
                for k in 2..=n {
                    let kf = k as f64;
                    let q2 = ((2.0 * kf - 1.0) * x * q1 - (kf - 1.0) * q0) / kf;
                    q0 = q1;
                    q1 = q2;
                }
                let dp = (n as f64) * (x * q1 - q0) / (x * x - 1.0);
                nodes[i] = -x;
                nodes[n - 1 - i] = x;
                let w = 2.0 / ((1.0 - x * x) * dp * dp);
                weights[i] = w;
                weights[n - 1 - i] = w;
                break;
            }
        }
    }
    (nodes, weights)
}

/// High-accuracy 1-D integral of `f` over [a, b] by Gauss–Legendre.
pub fn quad1d(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let (nodes, weights) = gauss_legendre(64);
    let half = 0.5 * (b - a);
    let mid = 0.5 * (b + a);
    let sum: f64 = nodes
        .iter()
        .zip(&weights)
        .map(|(x, w)| w * f(mid + half * x))
        .sum();
    sum * half
}

/// High-accuracy value of the iterated integral by nested Gauss–Legendre.
///
/// Plays the role of the Python version's sympy "exact" value; returns `None`
/// if the result is not finite.
pub fn quadrature_value(spec: &IntegralSpec) -> Option<f64> {
    let (nodes, weights) = gauss_legendre(32);
    let order: Vec<&Axis> = spec.axes.iter().rev().collect(); // outermost first

    fn recurse(
        depth: usize,
        order: &[&Axis],
        spec: &IntegralSpec,
        fixed: &mut HashMap<String, f64>,
        nodes: &[f64],
        weights: &[f64],
    ) -> f64 {
        let axis = order[depth];
        let (lo, hi) = axis.bounds(fixed);
        let half = 0.5 * (hi - lo);
        let mid = 0.5 * (hi + lo);
        let mut sum = 0.0;
        for (x, w) in nodes.iter().zip(weights) {
            let t = mid + half * x;
            fixed.insert(axis.var.clone(), t);
            let inner = if depth + 1 < order.len() {
                recurse(depth + 1, order, spec, fixed, nodes, weights)
            } else {
                spec.value_at(fixed)
            };
            sum += w * inner;
        }
        fixed.remove(&axis.var);
        sum * half
    }

    let mut fixed = HashMap::new();
    let v = recurse(0, &order, spec, &mut fixed, &nodes, &weights);
    v.is_finite().then_some(v)
}

/// Generate Riemann cells (outermost-major order) and their summed value.
///
/// `ns` is the subdivision count per axis in innermost-first order (so angular
/// variables can be diced finer than radial ones). Cells are ordered so that
/// iterating the list advances the innermost variable fastest and the
/// outermost slowest — the order the animation reveals them.
pub fn riemann_cells(spec: &IntegralSpec, ns: &[usize]) -> Result<(Vec<Cell>, f64), IntegralError> {
    if ns.len() != spec.dim() {
        return err(format!(
            "expected {} subdivision counts, got {}",
            spec.dim(),
            ns.len()
        ));
    }
    let counts: Vec<usize> = ns.iter().rev().copied().collect(); // outermost-first
    let order: Vec<&Axis> = spec.axes.iter().rev().collect(); // outermost first

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        depth: usize,
        order: &[&Axis],
        counts: &[usize],
        spec: &IntegralSpec,
        fixed: &HashMap<String, f64>,
        ranges: &HashMap<String, (f64, f64)>,
        idx: &[usize],
        dvol: f64,
        cells: &mut Vec<Cell>,
    ) {
        let axis = order[depth];
        let m = counts[depth];
        let (lo, hi) = axis.bounds(fixed);
        let step = (hi - lo) / m as f64;
        for k in 0..m {
            let c0 = lo + step * k as f64;
            let c1 = lo + step * (k + 1) as f64;
            let mid = 0.5 * (c0 + c1);
            let mut f2 = fixed.clone();
            f2.insert(axis.var.clone(), mid);
            let mut r2 = ranges.clone();
            r2.insert(axis.var.clone(), (c0, c1));
            let mut i2 = idx.to_vec();
            i2.push(k);
            let dv = dvol * (c1 - c0);
            if depth + 1 < order.len() {
                recurse(depth + 1, order, counts, spec, &f2, &r2, &i2, dv, cells);
            } else {
                cells.push(Cell {
                    value: spec.value_at(&f2),
                    ranges: r2,
                    centers: f2,
                    dvol: dv,
                    index: i2,
                });
            }
        }
    }

    let mut cells = Vec::new();
    recurse(
        0,
        &order,
        &counts,
        spec,
        &HashMap::new(),
        &HashMap::new(),
        &[],
        1.0,
        &mut cells,
    );
    let total: f64 = cells.iter().map(|c| c.value * c.dvol).sum();
    Ok((cells, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn spec(f: &str, blocks: &[&str]) -> IntegralSpec {
        parse_integral(f, &blocks.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn parse_blocks() {
        let s = spec("x*y", &["y 0 x", "x 0 1"]);
        assert_eq!(s.variables, vec!["y", "x"]);
        assert_eq!(s.axes[0].outer, vec!["x"]);
        assert!(s.axes[1].outer.is_empty());
        // dotted form
        let s2 = spec("x", &["y 0..x^2", "x 0..1"]);
        assert_eq!(s2.axes[0].var, "y");
    }

    #[test]
    fn parse_errors() {
        assert!(parse_integral("x", &[]).is_err());
        assert!(parse_integral("x", &["x 0".into()]).is_err());
        assert!(parse_integral("x", &["x 0 1".into(), "x 0 1".into()]).is_err());
        // Outermost bounds may not reference the inner variable.
        assert!(parse_integral("x*y", &["y 0 1".into(), "x 0 y".into()]).is_err());
    }

    #[test]
    fn quad_simple_double() {
        // ∫₀¹∫₀ˣ x·y dy dx = 1/8
        let s = spec("x*y", &["y 0 x", "x 0 1"]);
        let v = quadrature_value(&s).unwrap();
        assert!((v - 0.125).abs() < 1e-10, "got {v}");
    }

    #[test]
    fn quad_triple() {
        // ∫₀¹∫₀¹∫₀¹ (x+y+z) = 1.5
        let s = spec("x+y+z", &["z 0 1", "y 0 1", "x 0 1"]);
        let v = quadrature_value(&s).unwrap();
        assert!((v - 1.5).abs() < 1e-10, "got {v}");
    }

    #[test]
    fn quad_polar_disk_area() {
        // ∫₀^{2π}∫₀¹ r dr dθ = π (Jacobian already in integrand)
        let s = spec("r", &["r 0 1", "theta 0 2*pi"]);
        let v = quadrature_value(&s).unwrap();
        assert!((v - PI).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn riemann_matches_quadrature_roughly() {
        let s = spec("x*y", &["y 0 x", "x 0 1"]);
        let (cells, total) = riemann_cells(&s, &[24, 24]).unwrap();
        assert_eq!(cells.len(), 24 * 24);
        assert!((total - 0.125).abs() < 0.01, "got {total}");
        // Cells are outermost-major: first 24 cells share the first x slab.
        assert!(cells[..24].iter().all(|c| c.index[0] == 0));
    }

    #[test]
    fn gauss_nodes_integrate_polynomials_exactly() {
        let (n, w) = gauss_legendre(32);
        // ∫₋₁¹ x⁴ dx = 2/5
        let v: f64 = n.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum();
        assert!((v - 0.4).abs() < 1e-13);
        let total: f64 = w.iter().sum();
        assert!((total - 2.0).abs() < 1e-13);
    }
}
