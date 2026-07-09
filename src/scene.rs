//! Scene specifications: the serialized contract between the TUI/CLI process
//! and the spawned visualization window.
//!
//! A topic module turns parsed input into numbers, a text [`Report`] for the
//! terminal, and optionally a [`SceneSpec`]. The spec is written to a temp
//! file as JSON and a `mathutil-rs viz <file>` child process renders it, so
//! windows keep animating while the REPL stays responsive (the same behavior
//! the Python/Qt original got from timers on one event loop).
//!
//! Live functions travel as serialized expression ASTs ([`Expr`]), so scenes
//! like the contour explorer can evaluate f, ∂f/∂x, ∂f/∂y interactively.

use serde::{Deserialize, Serialize};

use crate::core::expr::Expr;
use crate::theme::Rgb;

/// One styled line of panel/terminal text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub text: String,
    pub color: Option<Rgb>,
    pub bold: bool,
}

impl Row {
    pub fn plain(text: impl Into<String>) -> Row {
        Row {
            text: text.into(),
            color: None,
            bold: false,
        }
    }

    pub fn colored(text: impl Into<String>, color: Rgb) -> Row {
        Row {
            text: text.into(),
            color: Some(color),
            bold: false,
        }
    }

    pub fn bold(text: impl Into<String>, color: Rgb) -> Row {
        Row {
            text: text.into(),
            color: Some(color),
            bold: true,
        }
    }
}

/// What a command prints in the terminal (and the static part of the window's
/// info panel): a heading, unicode formula lines, and styled body rows.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Report {
    pub title: String,
    pub formulas: Vec<String>,
    pub body: Vec<Row>,
}

/// A vector drawn as an arrow from the origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrowSpec {
    pub components: Vec<f64>,
    pub color: Rgb,
    pub label: String,
}

/// A real eigen-direction to draw as a full-width line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenLine {
    pub value: f64,
    pub direction: Vec<f64>,
}

/// How the grid-warp scenes phrase their verdict once the motion ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarpVerdict {
    /// Eigenvalue summary (transform).
    Eigen,
    /// Emphasize invariant directions (the `eigen` command): eigen lines from
    /// the start, arrows on/off the eigen-directions, no unit cell.
    EigenFocus,
    /// INVERTIBLE / SINGULAR verdict.
    Invertibility,
    /// Composition step labels only.
    Compose,
}

/// What a `Space` scene animates after drawing its vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpaceMode {
    /// Shade the span (line / plane / all of space).
    Span,
    /// Collapse the most-dependent vector onto the span of the others
    /// (or report independence).
    Independent { collapse_index: Option<usize> },
    /// Draw the grid of the new basis alongside the standard grid.
    Basis,
    /// Build c1·v1 + c2·v2 + … tip-to-tail.
    Lincomb { coeffs: Vec<f64> },
    /// Is `point` in the span? Show it plus its projection onto the span.
    Member {
        point: Vec<f64>,
        projection: Vec<f64>,
        inside: bool,
        coeffs: Option<Vec<f64>>,
    },
}

/// One Riemann cell mapped into Cartesian space, ready to draw.
/// 2D: `corners` is a quad (4 × xy). 3D: a hexahedron (8 × xyz, the two
/// z-faces in matching order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellGeom {
    pub corners: Vec<Vec<f64>>,
    pub value: f64,
    pub contribution: f64, // value × dvol (with any Jacobian already applied)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneSpec {
    /// Identity grid warping through one or more matrix stages.
    /// Covers `transform`, `compose`, `invertible`, `eigen`.
    Warp {
        dim: usize,
        /// Cumulative right-to-left partial products, identity excluded:
        /// the animation tweens I → stages[0] → stages[1] → …  (row-major).
        stages: Vec<Vec<f64>>,
        /// The individual input matrices (row-major), for the panel.
        inputs: Vec<Vec<f64>>,
        eigen: Vec<EigenLine>,
        verdict: WarpVerdict,
        span: f64,
        report: Report,
    },
    /// Vectors from the origin plus a span/dependence/basis/lincomb/member
    /// animation. Covers `span`, `independent`, `basis`, `lincomb`, `member`.
    Space {
        dim: usize,
        arrows: Vec<ArrowSpec>,
        /// Orthonormal basis for span{vectors}, as columns flattened row-major
        /// (rows = dim, cols = rank).
        span_basis: Vec<Vec<f64>>,
        rank: usize,
        mode: SpaceMode,
        span: f64,
        report: Report,
    },
    /// Kernel (collapses to 0) and image (where outputs land) of a matrix.
    KernelImage {
        dim_in: usize,
        dim_out: usize,
        matrix: Vec<f64>, // row-major
        kernel: Vec<Vec<f64>>,
        image: Vec<Vec<f64>>,
        span: f64,
        report: Report,
    },
    /// Dimension bars: rank + nullity = dim(domain).
    RankNullity {
        rank: usize,
        nullity: usize,
        ncols: usize,
        report: Report,
    },
    /// Riemann-cell sweep for double/triple integrals (any coordinates —
    /// cells arrive already mapped to Cartesian geometry, Jacobian applied).
    Sweep {
        dim: usize,
        cells: Vec<CellGeom>,
        /// How many cells form one leading "front" (the innermost run).
        front: usize,
        exact: Option<f64>,
        total: f64,
        /// Cartesian bounding box, [lo, hi] per Cartesian axis.
        bounds: Vec<(f64, f64)>,
        report: Report,
    },
    /// Single-variable definite integral: rectangles rising under the curve.
    Riemann1 {
        f: Expr, // over [var]
        var: String,
        a: f64,
        b: f64,
        n: usize,
        exact: f64,
        total: f64,
        report: Report,
    },
    /// Solid of revolution (disk / washer / shell). The bounds variable owns
    /// its axis: disks/washers revolve about the `var`-axis, shells about
    /// the perpendicular one; the y-axis renders vertical.
    Revolution {
        /// Sample points of the outer curve, (v, f(v)).
        outer: Vec<(f64, f64)>,
        /// Washer inner curve, if any.
        inner: Option<Vec<(f64, f64)>>,
        /// `false`: disk/washer about the `var`-axis;
        /// `true`: shells about the axis perpendicular to `var`.
        shells: bool,
        /// The integration variable, e.g. "x", "y", "t".
        var: String,
        volume: f64,
        report: Report,
    },
    /// Surface z = f(x, y) with an animated partial-derivative sweep.
    Surface {
        f: Expr, // over ["x", "y"]
        domain: (f64, f64),
        report: Report,
    },
    /// Flat contour map with draggable x/y; live f, ∂f/∂x, ∂f/∂y readout.
    Contour {
        f: Expr, // over ["x", "y"]
        domain: (f64, f64),
        report: Report,
    },
}

impl SceneSpec {
    pub fn report(&self) -> &Report {
        match self {
            SceneSpec::Warp { report, .. }
            | SceneSpec::Space { report, .. }
            | SceneSpec::KernelImage { report, .. }
            | SceneSpec::RankNullity { report, .. }
            | SceneSpec::Sweep { report, .. }
            | SceneSpec::Riemann1 { report, .. }
            | SceneSpec::Revolution { report, .. }
            | SceneSpec::Surface { report, .. }
            | SceneSpec::Contour { report, .. } => report,
        }
    }

    pub fn window_title(&self) -> String {
        format!("mathutil — {}", self.report().title)
    }
}

/// What `viz_spawn` writes and the `viz` subprocess reads: the scene plus
/// the command text that produced it, which seeds the window's editable
/// command box (re-parameterize without retyping in the REPL).
#[derive(Serialize, Deserialize)]
pub struct ScenePackage {
    pub command: String,
    pub spec: SceneSpec,
}

/// Everything a command returns: terminal output plus an optional window.
pub struct Outcome {
    pub report: Report,
    pub scene: Option<SceneSpec>,
}

impl Outcome {
    pub fn with_scene(scene: SceneSpec) -> Outcome {
        Outcome {
            report: scene.report().clone(),
            scene: Some(scene),
        }
    }

    pub fn text_only(report: Report) -> Outcome {
        Outcome {
            report,
            scene: None,
        }
    }
}

/// Format a matrix (row-major, `rows × cols`) as aligned text lines with a
/// `name =` prefix on the middle row, like the Python panel's `mat_html`.
pub fn matrix_rows(name: &str, data: &[f64], rows: usize, cols: usize) -> Vec<String> {
    let cells: Vec<String> = data.iter().map(|v| format!("{v:+.2}")).collect();
    let width = cells.iter().map(String::len).max().unwrap_or(1);
    let prefix = format!("{name} = ");
    let pad = " ".repeat(prefix.len());
    let mid = (rows - 1) / 2;
    (0..rows)
        .map(|r| {
            let row = (0..cols)
                .map(|c| format!("{:>width$}", cells[r * cols + c]))
                .collect::<Vec<_>>()
                .join("  ");
            let lead = if r == mid { &prefix } else { &pad };
            format!("{lead}⎡{row}⎤")
                .replace('⎡', if r == 0 { "⎡" } else if r + 1 == rows { "⎣" } else { "⎢" })
                .replace('⎤', if r == 0 { "⎤" } else if r + 1 == rows { "⎦" } else { "⎥" })
        })
        .collect()
}

/// Format a vector like `(1, 0, 3.5)` trimming trailing zeros.
pub fn vec_text(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| trim_num(*x)).collect();
    format!("({})", parts.join(", "))
}

/// Human-friendly number: drop trailing zeros, keep up to 3 decimals.
pub fn trim_num(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e12 {
        format!("{}", x as i64)
    } else {
        let s = format!("{x:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// `R^n` with a superscript digit when possible.
pub fn rn(n: usize) -> String {
    let sup = match n {
        1 => "¹",
        2 => "²",
        3 => "³",
        4 => "⁴",
        _ => return format!("R^{n}"),
    };
    format!("ℝ{sup}")
}
