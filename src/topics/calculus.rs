//! Double/triple integrals as Riemann-cell sweeps (Cartesian, polar,
//! cylindrical, spherical) plus volumes of revolution.
//!
//! The step from Cartesian to radial is a coordinate map applied to each
//! cell's corners, plus a Jacobian folded into the integrand; the sweep
//! renderer never knows which system it is drawing.

use std::collections::HashMap;

use crate::core::expr::{parse_expr_text, Node};
use crate::core::integrate::{
    parse_integral, quad1d, quadrature_value, riemann_cells, Cell, IntegralSpec,
};
use crate::scene::{trim_num, CellGeom, Report, Row, SceneSpec};
use crate::theme;

const N2: usize = 40; // subdivisions per axis, Cartesian double integrals
const N3: usize = 12; // subdivisions per axis, Cartesian triple integrals
const N_RADIAL: usize = 12; // r, rho, z (radial / linear axes)
const N_THETA: usize = 48; // theta (azimuth — many slices to look round)
const N_PHI: usize = 24; // phi (polar angle)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coord {
    Cartesian,
    Polar,
    Cylindrical,
    Spherical,
}

impl Coord {
    fn name(self) -> &'static str {
        match self {
            Coord::Cartesian => "Cartesian",
            Coord::Polar => "polar",
            Coord::Cylindrical => "cylindrical",
            Coord::Spherical => "spherical",
        }
    }

    fn required(self) -> &'static [&'static str] {
        match self {
            Coord::Cartesian => &[],
            Coord::Polar => &["r", "theta"],
            Coord::Cylindrical => &["r", "theta", "z"],
            Coord::Spherical => &["rho", "theta", "phi"],
        }
    }

    /// Integrand multiplier (the area/volume element the user never types).
    fn jacobian_text(self) -> &'static str {
        match self {
            Coord::Cartesian => "1",
            Coord::Polar | Coord::Cylindrical => "r",
            Coord::Spherical => "rho^2*sin(phi)",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Coord::Cartesian => "",
            Coord::Polar => "area element  r dr dθ  (auto)",
            Coord::Cylindrical => "volume element  r dr dθ dz  (auto)",
            Coord::Spherical => "volume element  ρ² sinφ dρ dθ dφ  (auto)",
        }
    }

    /// Map one point in variable space to Cartesian xyz.
    fn to_xyz(self, values: &HashMap<String, f64>, variables: &[String]) -> [f64; 3] {
        match self {
            Coord::Cartesian => {
                let mut out = [0.0; 3];
                let mut taken = [false; 3];
                let pref = |v: &str| match v {
                    "x" => Some(0),
                    "y" => Some(1),
                    "z" => Some(2),
                    _ => None,
                };
                for v in variables {
                    if let Some(i) = pref(v) {
                        out[i] = values[v];
                        taken[i] = true;
                    }
                }
                let mut free = (0..3).filter(|i| !taken[*i]);
                for v in variables {
                    if pref(v).is_none() {
                        if let Some(i) = free.next() {
                            out[i] = values[v];
                        }
                    }
                }
                out
            }
            Coord::Polar => {
                let (r, th) = (values["r"], values["theta"]);
                [r * th.cos(), r * th.sin(), 0.0]
            }
            Coord::Cylindrical => {
                let (r, th, z) = (values["r"], values["theta"], values["z"]);
                [r * th.cos(), r * th.sin(), z]
            }
            Coord::Spherical => {
                let (rho, th, ph) = (values["rho"], values["theta"], values["phi"]);
                [
                    rho * ph.sin() * th.cos(),
                    rho * ph.sin() * th.sin(),
                    rho * ph.cos(),
                ]
            }
        }
    }
}

/// Corner bit patterns: which end (lo/hi) of each variable's range a corner
/// takes. 2D: a quad; 3D: a hexahedron, bottom face then top in same order.
const BITS_2D: [[usize; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];
const BITS_3D: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

pub fn sweep_integral(
    integrand_text: &str,
    blocks: &[String],
    dim: usize,
    coord: Coord,
) -> Result<SceneSpec, String> {
    let mut spec = parse_integral(integrand_text, blocks).map_err(|e| e.0)?;
    if spec.dim() != dim {
        return Err(format!(
            "expected {dim} integration variables, got {}",
            spec.dim()
        ));
    }
    for need in coord.required() {
        if !spec.variables.iter().any(|v| v == need) {
            return Err(format!(
                "{} integrals expect variables named {}; missing {} (you gave {})",
                coord.name(),
                coord.required().join(", "),
                need,
                spec.variables.join(", ")
            ));
        }
    }

    // Fold the Jacobian into the integrand: cells and exact value both get it.
    let user_f = spec.integrand.clone();
    if coord != Coord::Cartesian {
        let var_refs: Vec<&str> = spec.variables.iter().map(String::as_str).collect();
        let jac = parse_expr_text(coord.jacobian_text(), &var_refs).map_err(|e| e.0)?;
        spec.integrand.node = Node::Mul(
            Box::new(spec.integrand.node.clone()),
            Box::new(jac.node),
        );
    }

    let ns: Vec<usize> = match coord {
        Coord::Cartesian => vec![if dim == 2 { N2 } else { N3 }; dim],
        _ => spec
            .variables
            .iter()
            .map(|v| match v.as_str() {
                "theta" => N_THETA,
                "phi" => N_PHI,
                _ => N_RADIAL,
            })
            .collect(),
    };
    let (cells, total) = riemann_cells(&spec, &ns).map_err(|e| e.0)?;
    let exact = quadrature_value(&spec);
    let geoms: Vec<CellGeom> = cells
        .iter()
        .map(|c| cell_geom(c, &spec, coord, dim))
        .collect();

    let mut bounds = vec![(f64::INFINITY, f64::NEG_INFINITY); 3];
    for g in &geoms {
        for corner in &g.corners {
            for (i, v) in corner.iter().enumerate() {
                bounds[i].0 = bounds[i].0.min(*v);
                bounds[i].1 = bounds[i].1.max(*v);
            }
        }
    }
    bounds.truncate(if dim == 2 { 2 } else { 3 });

    let report = integral_report(&spec, &user_f.text, coord, dim, exact, total, blocks);
    Ok(SceneSpec::Sweep {
        dim,
        front: ns[0],
        cells: geoms,
        exact,
        total,
        bounds,
        report,
    })
}

fn cell_geom(cell: &Cell, spec: &IntegralSpec, coord: Coord, dim: usize) -> CellGeom {
    let corners: Vec<Vec<f64>> = if dim == 2 {
        BITS_2D
            .iter()
            .map(|bits| corner_xyz(cell, spec, coord, bits))
            .map(|p| vec![p[0], p[1]])
            .collect()
    } else {
        BITS_3D
            .iter()
            .map(|bits| corner_xyz(cell, spec, coord, bits))
            .map(|p| p.to_vec())
            .collect()
    };
    CellGeom {
        corners,
        value: cell.value,
        contribution: cell.value * cell.dvol,
    }
}

fn corner_xyz(cell: &Cell, spec: &IntegralSpec, coord: Coord, bits: &[usize]) -> [f64; 3] {
    let mut values = HashMap::new();
    for (p, var) in spec.variables.iter().enumerate() {
        let (lo, hi) = cell.ranges[var];
        values.insert(var.clone(), if bits[p] == 0 { lo } else { hi });
    }
    coord.to_xyz(&values, &spec.variables)
}

fn integral_report(
    spec: &IntegralSpec,
    user_f: &str,
    coord: Coord,
    dim: usize,
    exact: Option<f64>,
    total: f64,
    blocks: &[String],
) -> Report {
    let sign = if dim == 2 { "∬" } else { "∭" };
    let elem = if dim == 2 { "dA" } else { "dV" };
    let heading = if coord == Coord::Cartesian {
        format!("{} integral", if dim == 2 { "Double" } else { "Triple" })
    } else {
        format!("{} integral", capitalize(coord.name()))
    };
    let mut formula = format!("{sign} {user_f} {elem}");
    // Show the iterated bounds, innermost last: ∫ x 0..1 ∫ y 0..x
    let iterated = spec
        .axes
        .iter()
        .rev()
        .map(|a| format!("∫ {}: {} → {}", a.var, a.lo.text, a.hi.text))
        .collect::<Vec<_>>()
        .join("  ");
    formula.push_str(&format!("      {iterated}"));

    let mut body = vec![Row::plain(format!("f = {user_f}"))];
    for b in blocks {
        body.push(Row::colored(format!("  : {b}"), theme::MUTED));
    }
    if coord != Coord::Cartesian {
        body.push(Row::colored(format!("{} coordinates", coord.name()), theme::MUTED));
        body.push(Row::colored(coord.note(), theme::MUTED));
    }
    if let Some(v) = exact {
        body.push(Row::bold(format!("value ≈ {}", trim_num_precise(v)), theme::GOOD));
    }
    body.push(Row::colored(
        format!("Riemann sum ({} cells) = {}", "swept", trim_num_precise(total)),
        theme::ACCENT,
    ));
    body.push(Row::colored(
        "the first ':' block is the innermost variable (swept first)",
        theme::MUTED,
    ));
    Report {
        title: heading,
        formulas: vec![formula],
        body,
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn trim_num_precise(x: f64) -> String {
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

// ------------------------------------------------------------- 1D integral

/// Definite integral of one variable: the on-ramp to the 2D/3D sweeps.
pub fn riemann1(fn_text: &str, var: &str, a: f64, b: f64) -> Result<SceneSpec, String> {
    if a >= b {
        return Err(format!(
            "integrate needs bounds with a < b (got {} → {})",
            trim_num(a),
            trim_num(b)
        ));
    }
    let f = parse_expr_text(fn_text, &[var]).map_err(|e| e.0)?;
    let exact = quad1d(|x| f.eval(&[x]), a, b);
    const N: usize = 48;
    let dx = (b - a) / N as f64;
    let total: f64 = (0..N)
        .map(|i| f.eval(&[a + (i as f64 + 0.5) * dx]) * dx)
        .sum();
    let report = Report {
        title: "Definite integral  (area under the curve)".into(),
        formulas: vec![format!(
            "∫ {fn_text} d{var},   {var}: {} → {}",
            trim_num(a),
            trim_num(b)
        )],
        body: vec![
            Row::plain(format!("f({var}) = {fn_text}")),
            Row::bold(format!("value ≈ {}", trim_num_precise(exact)), theme::GOOD),
            Row::colored(
                format!("midpoint sum ({N} rectangles) = {}", trim_num_precise(total)),
                theme::ACCENT,
            ),
            Row::colored("signed area: below the axis counts negative", theme::MUTED),
        ],
    };
    Ok(SceneSpec::Riemann1 {
        f,
        var: var.to_string(),
        a,
        b,
        n: N,
        exact,
        total,
        report,
    })
}

// ---------------------------------------------------------------- revolution

pub fn revolution(
    outer_text: &str,
    var: &str,
    a: f64,
    b: f64,
    inner_text: Option<&str>,
) -> Result<SceneSpec, String> {
    let f = parse_expr_text(outer_text, &[var]).map_err(|e| e.0)?;
    let inner = inner_text
        .map(|t| parse_expr_text(t, &[var]).map_err(|e| e.0))
        .transpose()?;

    let n = 160;
    let sample = |e: &crate::core::expr::Expr| -> Vec<(f64, f64)> {
        (0..=n)
            .map(|i| {
                let x = a + (b - a) * i as f64 / n as f64;
                (x, e.eval(&[x]))
            })
            .collect()
    };
    let outer_pts = sample(&f);
    let inner_pts = inner.as_ref().map(&sample);

    // Disk / washer method: V = π ∫ (R² − r²) dv, revolved about the v-axis.
    let volume = std::f64::consts::PI
        * quad1d(
            |x| {
                let big = f.eval(&[x]);
                let small = inner.as_ref().map_or(0.0, |e| e.eval(&[x]));
                big * big - small * small
            },
            a,
            b,
        );

    let method = if inner.is_some() { "washer" } else { "disk" };
    let formula = if inner.is_some() {
        format!(
            "V = π ∫ (R({var})² − r({var})²) d{var},  {var}: {} → {}",
            trim_num(a),
            trim_num(b)
        )
    } else {
        format!(
            "V = π ∫ R({var})² d{var},  {var}: {} → {}",
            trim_num(a),
            trim_num(b)
        )
    };
    let mut body = vec![Row::plain(format!("R({var}) = {outer_text}"))];
    if let Some(t) = inner_text {
        body.push(Row::plain(format!("r({var}) = {t}")));
    }
    body.push(Row::bold(
        format!("V ≈ {}", trim_num_precise(volume)),
        theme::GOOD,
    ));
    let report = Report {
        title: format!("Volume of revolution  ({method} method, about the {var}-axis)"),
        formulas: vec![formula],
        body,
    };
    Ok(SceneSpec::Revolution {
        outer: outer_pts,
        inner: inner_pts,
        shells: false,
        var: var.to_string(),
        volume,
        report,
    })
}

pub fn shells(fn_text: &str, var: &str, a: f64, b: f64) -> Result<SceneSpec, String> {
    // Shells revolve about the axis perpendicular to the bounds variable.
    let axis = if var == "y" { "x" } else { "y" };
    if a < 0.0 {
        return Err(format!(
            "shell needs 0 <= a < b (radii measured from the {axis}-axis)"
        ));
    }
    let f = parse_expr_text(fn_text, &[var]).map_err(|e| e.0)?;
    let n = 160;
    let outer_pts: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let x = a + (b - a) * i as f64 / n as f64;
            (x, f.eval(&[x]))
        })
        .collect();
    // Shell method: V = 2π ∫ v·f(v) dv
    let volume = std::f64::consts::TAU * quad1d(|x| x * f.eval(&[x]), a, b);
    let report = Report {
        title: format!("Volume by cylindrical shells  (about the {axis}-axis)"),
        formulas: vec![format!(
            "V = 2π ∫ {var}·f({var}) d{var},  {var}: {} → {}",
            trim_num(a),
            trim_num(b)
        )],
        body: vec![
            Row::plain(format!("f({var}) = {fn_text}")),
            Row::bold(format!("V ≈ {}", trim_num_precise(volume)), theme::GOOD),
        ],
    };
    Ok(SceneSpec::Revolution {
        outer: outer_pts,
        inner: None,
        shells: true,
        var: var.to_string(),
        volume,
        report,
    })
}
