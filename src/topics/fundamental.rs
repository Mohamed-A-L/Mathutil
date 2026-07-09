//! Kernel & image of a matrix, and the rank-nullity theorem.

use nalgebra::DMatrix;

use crate::core::linalg::{kernel_basis, orth, TOL};
use crate::scene::{matrix_rows, Report, Row, SceneSpec};
use crate::theme;

use super::transforms::row_major;

pub fn kernel_image(a: DMatrix<f64>) -> Result<SceneSpec, String> {
    let dim = match (a.nrows(), a.ncols()) {
        (2, 2) => 2,
        (3, 3) => 3,
        (r, c) => return Err(format!("kernel expects a 2x2 or 3x3 matrix, got {r}x{c}")),
    };
    let ker = kernel_basis(&a, TOL);
    let img = orth(&a, TOL);
    let (nullity, rank) = (ker.ncols(), img.ncols());

    let mut body: Vec<Row> = matrix_rows("A", &row_major(&a), dim, dim)
        .into_iter()
        .map(Row::plain)
        .collect();
    body.push(Row::colored(
        format!("kernel (nullity) = {nullity} — collapses to 0"),
        theme::EIGEN,
    ));
    body.push(Row::colored(
        format!("image (rank) = {rank} — where outputs land"),
        theme::SPAN,
    ));
    body.push(Row::bold(format!("{rank} + {nullity} = {dim}"), theme::ACCENT));

    let report = Report {
        title: "Kernel & Image".into(),
        formulas: vec!["rank A + nullity A = n".into()],
        body,
    };
    Ok(SceneSpec::KernelImage {
        dim_in: dim,
        dim_out: dim,
        matrix: row_major(&a),
        kernel: (0..ker.ncols())
            .map(|c| ker.column(c).iter().copied().collect())
            .collect(),
        image: (0..img.ncols())
            .map(|c| img.column(c).iter().copied().collect())
            .collect(),
        span: if dim == 2 { theme::DEFAULT_SPAN } else { 5.0 },
        report,
    })
}

pub fn rank_nullity(a: DMatrix<f64>) -> Result<SceneSpec, String> {
    let (m, n) = (a.nrows(), a.ncols());
    let rank = orth(&a, TOL).ncols();
    let nullity = n - rank;

    let mut body: Vec<Row> = matrix_rows("A", &row_major(&a), m, n)
        .into_iter()
        .map(Row::plain)
        .collect();
    body.push(Row::bold(format!("rank = {rank}"), theme::SPAN));
    body.push(Row::bold(format!("nullity = {nullity}"), theme::EIGEN));
    body.push(Row::bold(
        format!("{rank} + {nullity} = {n} = dim(domain)"),
        theme::ACCENT,
    ));

    let report = Report {
        title: "Rank–Nullity theorem".into(),
        formulas: vec!["rank A + nullity A = dim(domain)".into()],
        body,
    };
    Ok(SceneSpec::RankNullity {
        rank,
        nullity,
        ncols: n,
        report,
    })
}
