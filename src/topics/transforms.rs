//! Linear transformations and their composition — the marquee visual.
//!
//! `transform(A)` animates the identity grid warping into `A` (2×2 in the
//! plane, 3×3 in space); `compose` plays one stage per matrix right-to-left;
//! `invertible` swaps the verdict; `eigen` highlights invariant directions;
//! `basis` is the same warp with the basis vectors as columns.

use nalgebra::{DMatrix, DVector};

use crate::core::linalg::{real_eigenpairs, EigenPair, TOL};
use crate::scene::{
    matrix_rows, EigenLine, Report, Row, SceneSpec, WarpVerdict,
};
use crate::theme;

pub fn transform(a: DMatrix<f64>) -> Result<SceneSpec, String> {
    let dim = square_dim(&a)?;
    // One warp scene covers what used to be separate `transform` and
    // `invertible` commands: the report carries both the eigen summary and
    // the INVERTIBLE / SINGULAR verdict.
    let report = transform_report(
        "Linear transformation",
        "y⃗ = A·x⃗",
        &a,
        &[eigen_summary(&a), invertibility_verdict(&a)],
    );
    Ok(warp_spec(dim, vec![a.clone()], vec![a], WarpVerdict::Eigen, report))
}

pub fn compose(mats: Vec<DMatrix<f64>>) -> Result<SceneSpec, String> {
    if mats.len() < 2 {
        return Err("compose expects two or more matrices".into());
    }
    let dim = square_dim(&mats[0])?;
    if mats.iter().any(|m| m.nrows() != dim || m.ncols() != dim) {
        return Err("compose expects matrices of the same 2x2 or 3x3 size".into());
    }
    // The rightmost matrix is applied first: partial products Mn, M(n-1)Mn, …
    let mut stages: Vec<DMatrix<f64>> = Vec::new();
    let mut acc = DMatrix::identity(dim, dim);
    for m in mats.iter().rev() {
        acc = m * &acc;
        stages.push(acc.clone());
    }
    let product = stages.last().unwrap().clone();

    let m = mats.len();
    let matrix_labels = matrix_label_sequence(m);
    let mut body = Vec::new();
    for (i, mat) in mats.iter().enumerate() {
        for line in matrix_rows(&format!("M{}", i + 1), mat.transpose().as_slice(), dim, dim) {
            body.push(Row::plain(line));
        }
    }
    let label = matrix_labels.join("·");
    for line in matrix_rows(&label, product.transpose().as_slice(), dim, dim) {
        body.push(Row::colored(line, theme::SPAN));
    }
    body.push(Row::colored(
        "the rightmost matrix is applied first",
        theme::MUTED,
    ));
    let report = Report {
        title: "Composition".into(),
        formulas: vec![format!(
            "y⃗ = {} x⃗",
            matrix_labels.join(" ")
        )],
        body,
    };
    Ok(warp_spec(dim, stages, mats, WarpVerdict::Compose, report))
}

pub fn eigen(a: DMatrix<f64>) -> Result<SceneSpec, String> {
    let dim = square_dim(&a)?;
    let pairs = real_eigenpairs(&a, TOL);
    let mut body = mat_body(&a);
    if pairs.is_empty() {
        body.push(Row::colored(
            "no real eigenvectors — every direction rotates",
            theme::MUTED,
        ));
    } else {
        body.push(Row::colored(
            "eigen-directions: only stretch, never rotate",
            theme::EIGEN,
        ));
        body.push(Row::colored("test directions: rotate away", theme::ACCENT));
        for (i, p) in pairs.iter().enumerate() {
            body.push(Row::colored(
                format!(
                    "λ{} = {:+.2},  v = {}",
                    subscript(i + 1),
                    p.value,
                    crate::scene::vec_text(p.vector.as_slice())
                ),
                theme::EIGEN,
            ));
        }
    }
    let report = Report {
        title: "Eigenvectors: unchanged directions".into(),
        formulas: vec!["A·v⃗ = λ·v⃗".into()],
        body,
    };
    Ok(warp_spec(
        dim,
        vec![a.clone()],
        vec![a],
        WarpVerdict::EigenFocus,
        report,
    ))
}

pub fn basis(vectors: Vec<DVector<f64>>) -> Result<SceneSpec, String> {
    let dim = vectors[0].len();
    if vectors.len() != dim || !(dim == 2 || dim == 3) {
        return Err("basis needs n vectors of dimension n (2 in R^2, 3 in R^3)".into());
    }
    let b = crate::core::linalg::as_columns(&vectors);
    let report = transform_report(
        "Change of basis  (columns = new basis vectors)",
        "x⃗ = B·[x⃗]_B",
        &b,
        &[invertibility_verdict(&b)],
    );
    Ok(warp_spec(
        dim,
        vec![b.clone()],
        vec![b],
        WarpVerdict::Invertibility,
        report,
    ))
}

// ------------------------------------------------------------------ helpers

fn square_dim(a: &DMatrix<f64>) -> Result<usize, String> {
    match (a.nrows(), a.ncols()) {
        (2, 2) => Ok(2),
        (3, 3) => Ok(3),
        (r, c) => Err(format!("expected a 2x2 or 3x3 matrix, got {r}x{c}")),
    }
}

/// Row-major flattening (nalgebra stores column-major, so transpose first).
pub fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
    m.transpose().as_slice().to_vec()
}

fn warp_spec(
    dim: usize,
    stages: Vec<DMatrix<f64>>,
    inputs: Vec<DMatrix<f64>>,
    verdict: WarpVerdict,
    report: Report,
) -> SceneSpec {
    let final_m = stages.last().unwrap();
    let eigen = real_eigenpairs(final_m, TOL)
        .into_iter()
        .map(|EigenPair { value, vector }| EigenLine {
            value,
            direction: vector.as_slice().to_vec(),
        })
        .collect();
    SceneSpec::Warp {
        dim,
        stages: stages.iter().map(row_major).collect(),
        inputs: inputs.iter().map(row_major).collect(),
        eigen,
        verdict,
        span: if dim == 2 { theme::DEFAULT_SPAN } else { 5.0 },
        report,
    }
}

pub fn mat_body(a: &DMatrix<f64>) -> Vec<Row> {
    matrix_rows("A", &row_major(a), a.nrows(), a.ncols())
        .into_iter()
        .map(Row::plain)
        .collect()
}

fn transform_report(title: &str, formula: &str, a: &DMatrix<f64>, footer: &[Row]) -> Report {
    let det = a.determinant();
    let det_color = if det.abs() < 1e-9 {
        theme::MUTED
    } else if det < 0.0 {
        theme::BAD
    } else {
        theme::GOOD
    };
    let flip = if det < -1e-9 {
        "  (orientation flipped)"
    } else {
        ""
    };
    let mut body = mat_body(a);
    body.push(Row::bold(format!("det A = {det:+.2}{flip}"), det_color));
    body.push(Row::colored(
        "columns = where the basis vectors land",
        theme::MUTED,
    ));
    body.extend(footer.iter().cloned());
    Report {
        title: title.into(),
        formulas: vec![formula.into()],
        body,
    }
}

pub fn invertibility_verdict(a: &DMatrix<f64>) -> Row {
    let n = a.nrows();
    if a.determinant().abs() < 1e-9 {
        Row::bold(
            format!(
                "SINGULAR — collapses {} to a lower dimension; not invertible",
                crate::scene::rn(n)
            ),
            theme::BAD,
        )
    } else {
        Row::bold(
            format!(
                "INVERTIBLE — A keeps {} full-dimensional; it can be undone",
                crate::scene::rn(n)
            ),
            theme::GOOD,
        )
    }
}

fn eigen_summary(a: &DMatrix<f64>) -> Row {
    let pairs = real_eigenpairs(a, TOL);
    if pairs.is_empty() {
        Row::colored("no real eigenvalues", theme::EIGEN)
    } else {
        let eig = pairs
            .iter()
            .map(|p| format!("λ = {:+.2}", p.value))
            .collect::<Vec<_>>()
            .join("  ");
        Row::colored(format!("eigenvalues:  {eig}"), theme::EIGEN)
    }
}

fn matrix_label_sequence(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("M{}", i + 1)).collect()
}

pub fn subscript(n: usize) -> String {
    const SUBS: [&str; 10] = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
    n.to_string()
        .chars()
        .map(|c| SUBS[c.to_digit(10).unwrap() as usize])
        .collect()
}
