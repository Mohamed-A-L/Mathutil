//! Pure numerical linear-algebra helpers shared by the topic modules.
//!
//! No TUI, no GUI — just nalgebra. Everything here is unit-tested headlessly
//! and reused by both the 2D and 3D visualizations. Mirrors the Python
//! `mathutil.core.linalg` (which used numpy/scipy).

use nalgebra::{DMatrix, DVector};

pub const TOL: f64 = 1e-9;

/// Stack a list of vectors as the columns of a matrix.
pub fn as_columns(vectors: &[DVector<f64>]) -> DMatrix<f64> {
    if vectors.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    let rows = vectors[0].len();
    DMatrix::from_fn(rows, vectors.len(), |r, c| vectors[c][r])
}

/// Numerical rank of a matrix via its singular values.
pub fn rank_of(m: &DMatrix<f64>, tol: f64) -> usize {
    if m.is_empty() {
        return 0;
    }
    m.clone()
        .svd(false, false)
        .singular_values
        .iter()
        .filter(|s| **s > tol)
        .count()
}

/// Numerical rank of a set of column vectors.
pub fn rank(vectors: &[DVector<f64>], tol: f64) -> usize {
    rank_of(&as_columns(vectors), tol)
}

/// Orthonormal basis (as columns) for the span of the given vectors.
pub fn span_basis(vectors: &[DVector<f64>], tol: f64) -> DMatrix<f64> {
    let m = as_columns(vectors);
    orth(&m, tol)
}

/// Orthonormal basis (as columns) for the column space of `m`.
pub fn orth(m: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let rows = m.nrows();
    if m.is_empty() {
        return DMatrix::zeros(rows, 0);
    }
    let svd = m.clone().svd(true, false);
    let u = svd.u.as_ref().expect("svd computed with u");
    let cols: Vec<usize> = svd
        .singular_values
        .iter()
        .enumerate()
        .filter(|(_, s)| **s > tol)
        .map(|(i, _)| i)
        .collect();
    DMatrix::from_fn(rows, cols.len(), |r, c| u[(r, cols[c])])
}

/// Orthonormal basis (as columns) for the null space of `a`.
///
/// The kernel is the orthogonal complement of the row space: take an
/// orthonormal row-space basis (SVD of Aᵀ), then complete it to a basis of
/// ℝⁿ by Gram–Schmidt over the standard basis vectors. The added directions
/// are the kernel. (nalgebra has no full/non-thin SVD to read it off from.)
pub fn kernel_basis(a: &DMatrix<f64>, tol: f64) -> DMatrix<f64> {
    let n = a.ncols();
    if a.is_empty() {
        return DMatrix::zeros(n, 0);
    }
    let row_basis = orth(&a.transpose(), tol * (1.0 + a.amax()));
    let r = row_basis.ncols();
    let mut have: Vec<DVector<f64>> = (0..r).map(|c| row_basis.column(c).into_owned()).collect();
    let mut kernel: Vec<DVector<f64>> = Vec::new();
    while have.len() < n {
        // Greedily take the standard basis vector with the largest residual
        // after projecting out everything found so far.
        let mut best: Option<DVector<f64>> = None;
        let mut best_norm = 0.0;
        for k in 0..n {
            let mut v = DVector::zeros(n);
            v[k] = 1.0;
            for _ in 0..2 {
                // Repeat the projection for numerical stability.
                for b in &have {
                    let coeff = b.dot(&v);
                    v -= b * coeff;
                }
            }
            let norm = v.norm();
            if norm > best_norm {
                best_norm = norm;
                best = Some(v / norm);
            }
        }
        match best {
            Some(v) if best_norm > 1e-7 => {
                have.push(v.clone());
                kernel.push(v);
            }
            _ => break,
        }
    }
    let mut out = DMatrix::zeros(n, kernel.len());
    for (c, v) in kernel.iter().enumerate() {
        out.set_column(c, v);
    }
    out
}

/// A real eigenvalue and its (real, unit) eigenvector direction.
#[derive(Debug, Clone)]
pub struct EigenPair {
    pub value: f64,
    pub vector: DVector<f64>,
}

/// Eigenpairs of a square matrix whose eigenvalue is (numerically) real.
///
/// Complex-conjugate pairs (e.g. from a pure rotation) are dropped, since
/// they have no invariant real direction to draw. Eigenvectors are recovered
/// as the kernel of (A - λI).
pub fn real_eigenpairs(a: &DMatrix<f64>, tol: f64) -> Vec<EigenPair> {
    assert_eq!(a.nrows(), a.ncols(), "eigen needs a square matrix");
    let n = a.nrows();
    let complex = a.clone().complex_eigenvalues();
    // Collect distinct real eigenvalues (a repeated λ yields a multi-column
    // kernel below, so dedup close values).
    let mut reals: Vec<f64> = Vec::new();
    let scale = 1.0 + a.amax();
    for lam in complex.iter() {
        if lam.im.abs() < tol * scale {
            let re = lam.re;
            if !reals.iter().any(|r| (r - re).abs() < 1e-7 * scale) {
                reals.push(re);
            }
        }
    }
    let mut pairs = Vec::new();
    for lam in reals {
        let shifted = a - DMatrix::identity(n, n) * lam;
        // The eigenvalue from the QR iteration is approximate; use a looser
        // tolerance when extracting the invariant direction.
        let kern = kernel_basis(&shifted, 1e-6 * scale);
        for c in 0..kern.ncols() {
            let v = kern.column(c).into_owned();
            let norm = v.norm();
            if norm > tol {
                pairs.push(EigenPair {
                    value: lam,
                    vector: v / norm,
                });
            }
        }
    }
    pairs.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
    pairs
}

/// Orthogonal projection of `v` onto span{others}.
pub fn project_onto_span(v: &DVector<f64>, others: &[DVector<f64>], tol: f64) -> DVector<f64> {
    let b = span_basis(others, tol);
    if b.ncols() == 0 {
        return DVector::zeros(v.len());
    }
    &b * (b.transpose() * v)
}

/// Index of the vector best explained by the span of the others.
///
/// Returns the vector with the smallest residual after projecting it onto the
/// span of the remaining vectors — i.e. the one that most nearly lies in their
/// span. Used to pick which vector to "collapse" in the dependence animation.
pub fn most_dependent(vectors: &[DVector<f64>], tol: f64) -> usize {
    let mut best_idx = 0;
    let mut best_residual = f64::INFINITY;
    for (i, v) in vectors.iter().enumerate() {
        let others: Vec<DVector<f64>> = vectors
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, w)| w.clone())
            .collect();
        let residual = (v - project_onto_span(v, &others, tol)).norm();
        if residual < best_residual {
            best_idx = i;
            best_residual = residual;
        }
    }
    best_idx
}

/// Least-squares coefficients c with `columns * c ≈ target`, plus residual.
pub fn least_squares(
    columns: &DMatrix<f64>,
    target: &DVector<f64>,
) -> Option<(DVector<f64>, f64)> {
    let svd = columns.clone().svd(true, true);
    let c = svd.solve(target, TOL).ok()?;
    let residual = (columns * &c - target).norm();
    Some((c, residual))
}

/// Determinant, delegating to nalgebra (square matrices only).
pub fn det(a: &DMatrix<f64>) -> f64 {
    a.determinant()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(s)
    }

    #[test]
    fn rank_and_span() {
        assert_eq!(rank(&[v(&[1.0, 0.0]), v(&[0.0, 1.0])], TOL), 2);
        assert_eq!(rank(&[v(&[1.0, 2.0]), v(&[2.0, 4.0])], TOL), 1);
        assert_eq!(rank(&[v(&[0.0, 0.0])], TOL), 0);
        let b = span_basis(&[v(&[1.0, 0.0, 0.0]), v(&[0.0, 1.0, 0.0])], TOL);
        assert_eq!(b.ncols(), 2);
        // Orthonormal columns
        let g = b.transpose() * &b;
        assert!((g - DMatrix::identity(2, 2)).amax() < 1e-9);
    }

    #[test]
    fn kernel() {
        // [[1,2],[2,4]] has kernel spanned by (2,-1)/sqrt(5)
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let k = kernel_basis(&a, TOL);
        assert_eq!(k.ncols(), 1);
        let dir = k.column(0);
        assert!((a.clone() * dir).norm() < 1e-8);
        // Full-rank matrix: trivial kernel
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(kernel_basis(&b, TOL).ncols(), 0);
        // Wide matrix: [[1,2,3],[2,4,6]] rank 1 -> nullity 2
        let c = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(kernel_basis(&c, TOL).ncols(), 2);
    }

    #[test]
    fn eigen_symmetric() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let pairs = real_eigenpairs(&a, TOL);
        assert_eq!(pairs.len(), 2);
        let mut vals: Vec<f64> = pairs.iter().map(|p| p.value).collect();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-7);
        assert!((vals[1] - 3.0).abs() < 1e-7);
        for p in &pairs {
            let av = &a * &p.vector;
            assert!((av - &p.vector * p.value).norm() < 1e-6);
        }
    }

    #[test]
    fn eigen_rotation_has_no_real_pairs() {
        // Pure rotation: complex eigenvalues only.
        let a = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        assert!(real_eigenpairs(&a, TOL).is_empty());
    }

    #[test]
    fn projection_and_dependence() {
        let p = project_onto_span(&v(&[1.0, 1.0]), &[v(&[1.0, 0.0])], TOL);
        assert!((p - v(&[1.0, 0.0])).norm() < 1e-9);
        // (2,4) is exactly 2*(1,2): most dependent
        let idx = most_dependent(&[v(&[1.0, 2.0]), v(&[2.0, 4.0]), v(&[0.0, 1.0])], TOL);
        assert!(idx == 0 || idx == 1); // both are in each other's span
    }

    #[test]
    fn membership_least_squares() {
        let cols = as_columns(&[v(&[1.0, 0.0, 0.0]), v(&[0.0, 1.0, 0.0])]);
        let (c, r) = least_squares(&cols, &v(&[2.0, 3.0, 0.0])).unwrap();
        assert!(r < 1e-9);
        assert!((c[0] - 2.0).abs() < 1e-9 && (c[1] - 3.0).abs() < 1e-9);
        let (_, r2) = least_squares(&cols, &v(&[1.0, 1.0, 5.0])).unwrap();
        assert!(r2 > 1.0); // (1,1,5) is not in the xy-plane
    }
}
