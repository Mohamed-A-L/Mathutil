//! Span, linear (in)dependence, linear combinations, and span membership.

use nalgebra::{DMatrix, DVector};

use crate::core::linalg::{
    as_columns, least_squares, most_dependent, project_onto_span, rank, span_basis, TOL,
};
use crate::scene::{vec_text, ArrowSpec, Report, Row, SceneSpec, SpaceMode};
use crate::theme;

use super::transforms::subscript;

fn check_dim(vectors: &[DVector<f64>]) -> Result<usize, String> {
    let dim = vectors[0].len();
    if dim == 2 || dim == 3 {
        Ok(dim)
    } else {
        Err(format!("expected 2D or 3D vectors, got {dim}D"))
    }
}

fn arrows(vectors: &[DVector<f64>]) -> Vec<ArrowSpec> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| ArrowSpec {
            components: v.as_slice().to_vec(),
            color: theme::BASIS_COLORS[i % 3],
            label: format!("v{}", subscript(i + 1)),
        })
        .collect()
}

fn vec_rows(vectors: &[DVector<f64>]) -> Vec<Row> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Row::colored(
                format!("v{} = {}", subscript(i + 1), vec_text(v.as_slice())),
                theme::BASIS_COLORS[i % 3],
            )
        })
        .collect()
}

fn basis_cols(b: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..b.ncols())
        .map(|c| b.column(c).iter().copied().collect())
        .collect()
}

fn span_desc(r: usize, dim: usize) -> String {
    let shape = match (r, dim) {
        (0, _) => "the origin",
        (1, _) => "a line through the origin",
        (2, 2) => "all of R^2",
        (2, 3) => "a plane through the origin",
        (3, _) => "all of R^3",
        _ => "a subspace",
    };
    format!("dim(span) = {r}  →  {shape} in R^{dim}")
}

fn space_spec(
    dim: usize,
    vectors: &[DVector<f64>],
    mode: SpaceMode,
    report: Report,
) -> SceneSpec {
    let b = span_basis(vectors, TOL);
    SceneSpec::Space {
        dim,
        arrows: arrows(vectors),
        span_basis: basis_cols(&b),
        rank: b.ncols(),
        mode,
        span: if dim == 2 { theme::DEFAULT_SPAN } else { 5.0 },
        report,
    }
}

pub fn span(vectors: Vec<DVector<f64>>) -> Result<SceneSpec, String> {
    let dim = check_dim(&vectors)?;
    let r = rank(&vectors, TOL);
    let k = vectors.len();
    let terms = (0..k)
        .map(|i| format!("c{}·v{}", subscript(i + 1), subscript(i + 1)))
        .collect::<Vec<_>>()
        .join(" + ");
    let mut body = vec_rows(&vectors);
    body.push(Row::bold(span_desc(r, dim), theme::ACCENT));
    let report = Report {
        title: format!("Span of vectors in R^{dim}"),
        formulas: vec![format!("span = {{ {terms} : cᵢ ∈ ℝ }}")],
        body,
    };
    Ok(space_spec(dim, &vectors, SpaceMode::Span, report))
}

pub fn independent(vectors: Vec<DVector<f64>>) -> Result<SceneSpec, String> {
    let dim = check_dim(&vectors)?;
    let r = rank(&vectors, TOL);
    let k = vectors.len();
    let is_independent = r == k;
    let mut body = vec_rows(&vectors);
    body.push(if is_independent {
        Row::bold(format!("independent — rank = {r} = {k} vectors"), theme::GOOD)
    } else {
        Row::bold(format!("dependent — rank = {r} < {k} vectors"), theme::BAD)
    });
    let collapse_index = if is_independent {
        None
    } else {
        body.push(Row::colored(
            "gold vector = a combination of the others",
            theme::MUTED,
        ));
        Some(most_dependent(&vectors, TOL))
    };
    let report = Report {
        title: "Linear (in)dependence".into(),
        formulas: vec!["c₁v⃗₁ + ⋯ + cₖv⃗ₖ = 0⃗".into()],
        body,
    };
    Ok(space_spec(
        dim,
        &vectors,
        SpaceMode::Independent { collapse_index },
        report,
    ))
}

pub fn lincomb(
    vectors: Vec<DVector<f64>>,
    coeffs: Option<DVector<f64>>,
) -> Result<SceneSpec, String> {
    let dim = check_dim(&vectors)?;
    let k = vectors.len();
    let coeffs = coeffs.unwrap_or_else(|| DVector::from_element(k, 1.0));
    if coeffs.len() != k {
        return Err(format!("got {k} vectors but {} coefficients", coeffs.len()));
    }
    let mut tip = DVector::zeros(dim);
    for i in 0..k {
        tip += &vectors[i] * coeffs[i];
    }
    let formula = format!(
        "r⃗ = {}",
        (0..k)
            .map(|i| format!("{}·v{}", crate::scene::trim_num(coeffs[i]), subscript(i + 1)))
            .collect::<Vec<_>>()
            .join(" + ")
    );
    let mut body = vec_rows(&vectors);
    body.push(Row::bold(
        format!("= {}", vec_text(tip.as_slice())),
        theme::POINT,
    ));
    let report = Report {
        title: "Linear combination".into(),
        formulas: vec![formula],
        body,
    };
    Ok(space_spec(
        dim,
        &vectors,
        SpaceMode::Lincomb {
            coeffs: coeffs.as_slice().to_vec(),
        },
        report,
    ))
}

pub fn member(vectors: Vec<DVector<f64>>, point: DVector<f64>) -> Result<SceneSpec, String> {
    let dim = point.len();
    if !(dim == 2 || dim == 3) || vectors.iter().any(|v| v.len() != dim) {
        return Err("member expects same-dimension 2D or 3D vectors".into());
    }
    let proj = project_onto_span(&point, &vectors, TOL);
    let residual = (&point - &proj).norm();
    let inside = residual < 1e-7;

    let coeffs = least_squares(&as_columns(&vectors), &point)
        .filter(|(_, r)| *r < 1e-7)
        .map(|(c, _)| c.as_slice().to_vec());

    let mut body = vec_rows(&vectors);
    body.push(Row::colored(
        format!("p = {}", vec_text(point.as_slice())),
        theme::POINT,
    ));
    if inside {
        body.push(Row::bold("IN the span", theme::GOOD));
        if let Some(c) = &coeffs {
            let combo = c
                .iter()
                .enumerate()
                .map(|(i, ci)| format!("{}·v{}", crate::scene::trim_num(*ci), subscript(i + 1)))
                .collect::<Vec<_>>()
                .join(" + ");
            body.push(Row::plain(format!("p = {combo}")));
        }
        body.push(Row::colored(
            "the point is a linear combination of the vectors",
            theme::MUTED,
        ));
    } else {
        body.push(Row::bold("NOT in the span", theme::BAD));
        body.push(Row::plain(format!("distance to span = {residual:.2}")));
        body.push(Row::colored(
            "no combination of the vectors reaches this point",
            theme::MUTED,
        ));
    }
    let report = Report {
        title: "Is the point in the span?".into(),
        formulas: vec!["p⃗ ∈ span{v⃗₁, …, v⃗ₖ} ?".into()],
        body,
    };
    Ok(space_spec(
        dim,
        &vectors,
        SpaceMode::Member {
            point: point.as_slice().to_vec(),
            projection: proj.as_slice().to_vec(),
            inside,
            coeffs,
        },
        report,
    ))
}
