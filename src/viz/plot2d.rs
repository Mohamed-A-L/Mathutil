//! 2D scenes drawn with egui_plot: grid warps, spans, kernel/image,
//! rank-nullity bars, and Riemann sweeps.

use eframe::egui::{self, Ui};
use egui_plot::{Line, Plot, PlotPoints, PlotUi, Points, Polygon, Text};
use nalgebra::{DVector, Matrix2, Vector2};

use crate::core::linalg::{project_onto_span, TOL};
use crate::scene::{trim_num, Row, SceneSpec, SpaceMode, WarpVerdict};
use crate::theme;

use super::{
    color32, color32_a, ease, KERNEL_SECS, LINCOMB_SECS, RIEMANN1_SECS, STAGE_SECS, SWEEP2D_SECS,
};

fn mat2(row_major: &[f64]) -> Matrix2<f64> {
    Matrix2::new(row_major[0], row_major[1], row_major[2], row_major[3])
}

fn lerp2(a: &Matrix2<f64>, b: &Matrix2<f64>, t: f64) -> Matrix2<f64> {
    a * (1.0 - t) + b * t
}

fn base_plot(id: &str, span: f64) -> Plot<'_> {
    Plot::new(id.to_owned())
        .data_aspect(1.0)
        .default_x_bounds(-span, span)
        .default_y_bounds(-span, span)
        .auto_bounds(false) // warped gridlines must not inflate the view
        .show_grid(false)
        .show_axes(true)
}

fn line(plot: &mut PlotUi<'_>, p0: [f64; 2], p1: [f64; 2], color: egui::Color32, width: f32) {
    plot.line(
        Line::new("", PlotPoints::from(vec![p0, p1]))
            .color(color)
            .width(width),
    );
}

/// Warped background lattice: gridlines are straight, so two endpoints each.
fn warped_grid(plot: &mut PlotUi<'_>, m: &Matrix2<f64>, span: f64) {
    let n = span.ceil() as i64 + 2;
    for k in -n..=n {
        let k = k as f64;
        let color = if k == 0.0 {
            color32(theme::GRID_HI)
        } else {
            color32(theme::GRID_LINE)
        };
        let v0 = m * Vector2::new(k, -(n as f64));
        let v1 = m * Vector2::new(k, n as f64);
        line(plot, [v0.x, v0.y], [v1.x, v1.y], color, 1.0);
        let h0 = m * Vector2::new(-(n as f64), k);
        let h1 = m * Vector2::new(n as f64, k);
        line(plot, [h0.x, h0.y], [h1.x, h1.y], color, 1.0);
    }
}

/// An arrow with a filled head sized relative to its length.
fn arrow(plot: &mut PlotUi<'_>, tail: [f64; 2], tip: [f64; 2], color: egui::Color32, width: f32) {
    let d = Vector2::new(tip[0] - tail[0], tip[1] - tail[1]);
    let len = d.norm();
    line(plot, tail, tip, color, width);
    if len < 1e-9 {
        return;
    }
    let u = d / len;
    let head = (0.22 * len.min(1.4)).max(0.08);
    let perp = Vector2::new(-u.y, u.x);
    let tipv = Vector2::new(tip[0], tip[1]);
    let a = tipv - u * head + perp * head * 0.45;
    let b = tipv - u * head - perp * head * 0.45;
    plot.polygon(
        Polygon::new("", PlotPoints::from(vec![tip, [a.x, a.y], [b.x, b.y]]))
            .fill_color(color)
            .stroke(egui::Stroke::new(1.0, color)),
    );
}

fn full_plane(plot: &mut PlotUi<'_>, span: f64, color: theme::Rgb, alpha: u8) {
    let s = span * 4.0;
    plot.polygon(
        Polygon::new(
            "",
            PlotPoints::from(vec![[-s, -s], [s, -s], [s, s], [-s, s]]),
        )
        .fill_color(color32_a(color, alpha))
        .stroke(egui::Stroke::NONE),
    );
}

fn subspace_line(plot: &mut PlotUi<'_>, dir: &[f64], span: f64, color: theme::Rgb) {
    let d = Vector2::new(dir[0], dir[1]) * span * 4.0;
    line(
        plot,
        [-d.x, -d.y],
        [d.x, d.y],
        color32_a(color, 200),
        2.0,
    );
}

// -------------------------------------------------------------------- scenes

pub fn scene2d(ui: &mut Ui, spec: &SceneSpec, t: f64) -> Vec<Row> {
    match spec {
        SceneSpec::Warp {
            stages,
            eigen,
            verdict,
            span,
            ..
        } => warp2d(ui, stages, eigen, *verdict, *span, t),
        SceneSpec::Space {
            arrows: arr,
            span_basis,
            rank,
            mode,
            span,
            ..
        } => space2d(ui, arr, span_basis, *rank, mode, *span, t),
        SceneSpec::KernelImage {
            matrix,
            kernel,
            image,
            span,
            ..
        } => kernel2d(ui, matrix, kernel, image, *span, t),
        _ => Vec::new(),
    }
}

fn warp2d(
    ui: &mut Ui,
    stages: &[Vec<f64>],
    eigen: &[crate::scene::EigenLine],
    verdict: WarpVerdict,
    span: f64,
    t: f64,
) -> Vec<Row> {
    let mats: Vec<Matrix2<f64>> = stages.iter().map(|s| mat2(s)).collect();
    let nstages = mats.len();
    let total = STAGE_SECS * nstages as f64;
    let done = t >= total;

    // Which stage are we in, and how far through it?
    let (current, frac) = if done {
        (nstages - 1, 1.0)
    } else {
        let k = (t / STAGE_SECS) as usize;
        (k.min(nstages - 1), ease(t / STAGE_SECS - k as f64))
    };
    let prev = if current == 0 {
        Matrix2::identity()
    } else {
        mats[current - 1]
    };
    let m = lerp2(&prev, &mats[current], frac);
    let det = m.determinant();

    let mut rows = Vec::new();
    base_plot("warp2d", span).show(ui, |plot| {
        warped_grid(plot, &m, span);

        if verdict == WarpVerdict::EigenFocus {
            // Invariant lines from the start; arrows on/off the directions.
            for e in eigen {
                subspace_line(plot, &e.direction, span, theme::EIGEN);
            }
            for e in eigen {
                let v = m * Vector2::new(e.direction[0], e.direction[1]);
                arrow(plot, [0.0, 0.0], [v.x, v.y], color32(theme::EIGEN), 3.5);
            }
            for d in test_directions(eigen) {
                let v = m * d;
                arrow(plot, [0.0, 0.0], [v.x, v.y], color32(theme::ACCENT), 2.0);
            }
        } else {
            // Unit cell tinted by orientation, basis arrows.
            let cell = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            let warped: Vec<[f64; 2]> = cell
                .iter()
                .map(|p| {
                    let v = m * Vector2::new(p[0], p[1]);
                    [v.x, v.y]
                })
                .collect();
            let cell_color = if det < 0.0 { theme::FLIP } else { theme::SPAN };
            plot.polygon(
                Polygon::new("", PlotPoints::from(warped))
                    .fill_color(color32_a(cell_color, 110))
                    .stroke(egui::Stroke::new(1.5, color32(cell_color))),
            );
            let e1 = m * Vector2::new(1.0, 0.0);
            let e2 = m * Vector2::new(0.0, 1.0);
            arrow(plot, [0.0, 0.0], [e1.x, e1.y], color32(theme::ACCENT), 3.0);
            arrow(plot, [0.0, 0.0], [e2.x, e2.y], color32(theme::ACCENT2), 3.0);

            if done && verdict != WarpVerdict::Compose {
                for e in eigen {
                    subspace_line(plot, &e.direction, span, theme::EIGEN);
                }
            }
        }
    });

    // Live numbers under the static report.
    let det_color = if det.abs() < 1e-9 {
        theme::MUTED
    } else if det < 0.0 {
        theme::BAD
    } else {
        theme::GOOD
    };
    rows.push(Row::bold(format!("det = {det:+.2}  (live)"), det_color));
    if verdict == WarpVerdict::Compose {
        rows.push(if done {
            Row::bold("done: composite applied".to_string(), theme::ACCENT)
        } else {
            Row::bold(
                format!("step {}:  apply M{}", current + 1, nstages - current),
                theme::ACCENT,
            )
        });
    }
    rows
}

fn test_directions(eigen: &[crate::scene::EigenLine]) -> Vec<Vector2<f64>> {
    let candidates = [
        Vector2::new(1.0, 0.5).normalize(),
        Vector2::new(-0.5, 1.0).normalize(),
    ];
    let out: Vec<Vector2<f64>> = candidates
        .into_iter()
        .filter(|c| {
            eigen.iter().all(|e| {
                let d = Vector2::new(e.direction[0], e.direction[1]);
                (c.dot(&d).abs() - 1.0).abs() > 1e-3
            })
        })
        .collect();
    if out.is_empty() {
        vec![candidates[0]]
    } else {
        out
    }
}

fn space2d(
    ui: &mut Ui,
    arrows_spec: &[crate::scene::ArrowSpec],
    span_basis: &[Vec<f64>],
    rank: usize,
    mode: &SpaceMode,
    span: f64,
    t: f64,
) -> Vec<Row> {
    let vectors: Vec<DVector<f64>> = arrows_spec
        .iter()
        .map(|a| DVector::from_row_slice(&a.components))
        .collect();
    let mut rows = Vec::new();

    base_plot("space2d", span).show(ui, |plot| {
        // span shading
        if rank == 1 {
            subspace_line(plot, &span_basis[0], span, theme::SPAN);
        } else if rank >= 2 {
            full_plane(plot, span, theme::SPAN, 46);
        }

        // grid for lincomb (helps read coordinates)
        if matches!(mode, SpaceMode::Lincomb { .. } | SpaceMode::Basis) {
            warped_grid(plot, &Matrix2::identity(), span);
        }

        match mode {
            SpaceMode::Span | SpaceMode::Basis => {
                for (i, a) in arrows_spec.iter().enumerate() {
                    draw_vec_arrow(plot, &vectors[i], a.color);
                }
            }
            SpaceMode::Independent { collapse_index } => {
                let frac = ease(t / STAGE_SECS);
                for (i, a) in arrows_spec.iter().enumerate() {
                    if Some(i) == *collapse_index {
                        let others: Vec<DVector<f64>> = vectors
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| j != &i)
                            .map(|(_, v)| v.clone())
                            .collect();
                        let target = project_onto_span(&vectors[i], &others, TOL);
                        let pos = &vectors[i] * (1.0 - frac) + target * frac;
                        arrow(
                            plot,
                            [0.0, 0.0],
                            [pos[0], pos[1]],
                            color32(theme::EIGEN),
                            3.5,
                        );
                    } else {
                        draw_vec_arrow(plot, &vectors[i], a.color);
                    }
                }
            }
            SpaceMode::Lincomb { coeffs } => {
                let k = vectors.len();
                let segments: Vec<Vector2<f64>> = (0..k)
                    .map(|i| Vector2::new(vectors[i][0], vectors[i][1]) * coeffs[i])
                    .collect();
                let mut tails = vec![Vector2::zeros()];
                for s in &segments {
                    let last = *tails.last().unwrap();
                    tails.push(last + s);
                }
                let s = ease(t / LINCOMB_SECS) * k as f64;
                for i in 0..k {
                    let frac = (s - i as f64).clamp(0.0, 1.0);
                    let tip = tails[i] + segments[i] * frac;
                    arrow(
                        plot,
                        [tails[i].x, tails[i].y],
                        [tip.x, tip.y],
                        color32(theme::BASIS_COLORS[i % 3]),
                        3.0,
                    );
                }
                let active = (s as usize).min(k - 1);
                let frac = (s - active as f64).clamp(0.0, 1.0);
                let r = tails[active] + segments[active] * frac;
                arrow(plot, [0.0, 0.0], [r.x, r.y], color32(theme::POINT), 3.5);
            }
            SpaceMode::Member {
                point,
                projection,
                inside,
                ..
            } => {
                for (i, a) in arrows_spec.iter().enumerate() {
                    draw_vec_arrow(plot, &vectors[i], a.color);
                }
                let pc = if *inside { theme::GOOD } else { theme::BAD };
                plot.points(
                    Points::new("", PlotPoints::from(vec![[point[0], point[1]]]))
                        .radius(6.0)
                        .color(color32(pc)),
                );
                if !inside {
                    plot.points(
                        Points::new("", PlotPoints::from(vec![[projection[0], projection[1]]]))
                            .radius(5.0)
                            .color(color32(theme::SPAN)),
                    );
                    line(
                        plot,
                        [point[0], point[1]],
                        [projection[0], projection[1]],
                        color32(theme::BAD),
                        2.0,
                    );
                }
            }
        }

        // labels on the vector tips
        for (i, a) in arrows_spec.iter().enumerate() {
            let v = &vectors[i];
            plot.text(
                Text::new(
                    "",
                    [v[0] * 1.08, v[1] * 1.08].into(),
                    egui::RichText::new(a.label.clone()).size(14.0),
                )
                .color(color32(a.color)),
            );
        }
    });

    if let SpaceMode::Independent {
        collapse_index: Some(_),
    } = mode
    {
        rows.push(Row::colored(
            "watch the gold vector slide into the span of the others",
            theme::MUTED,
        ));
    }
    rows
}

fn draw_vec_arrow(plot: &mut PlotUi<'_>, v: &DVector<f64>, color: theme::Rgb) {
    arrow(plot, [0.0, 0.0], [v[0], v[1]], color32(color), 3.0);
}

fn kernel2d(
    ui: &mut Ui,
    matrix: &[f64],
    kernel: &[Vec<f64>],
    image: &[Vec<f64>],
    span: f64,
    t: f64,
) -> Vec<Row> {
    let a = mat2(matrix);
    let frac = ease(t / KERNEL_SECS);
    let m = lerp2(&Matrix2::identity(), &a, frac);

    base_plot("kernel2d", span).show(ui, |plot| {
        // image (purple): where every output lands
        match image.len() {
            1 => subspace_line(plot, &image[0], span, theme::SPAN),
            n if n >= 2 => full_plane(plot, span, theme::SPAN, 56),
            _ => {}
        }
        // kernel (gold): collapses to zero
        for k in kernel {
            subspace_line(plot, k, span, theme::EIGEN);
        }
        warped_grid(plot, &m, span);
        for k in kernel {
            let v0 = Vector2::new(k[0], k[1]) * (span * 0.6);
            let v = m * v0;
            arrow(plot, [0.0, 0.0], [v.x, v.y], color32(theme::EIGEN), 3.5);
        }
    });
    vec![Row::colored(
        "gold arrows shrink to 0 — purple is where outputs land",
        theme::MUTED,
    )]
}

pub fn rank_nullity(ui: &mut Ui, spec: &SceneSpec) -> Vec<Row> {
    let SceneSpec::RankNullity {
        rank,
        nullity,
        ncols,
        ..
    } = spec
    else {
        return Vec::new();
    };
    let (rank, nullity, n) = (*rank as f64, *nullity as f64, *ncols as f64);

    Plot::new("ranknullity")
        .include_x(-0.6)
        .include_x(n.max(1.0) + 0.6)
        .include_y(-2.6)
        .include_y(3.2)
        .show_grid(false)
        .show_axes(false)
        .show(ui, |plot| {
            let rect = |x0: f64, x1: f64, y0: f64, y1: f64, c: theme::Rgb| {
                Polygon::new(
                    "",
                    PlotPoints::from(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                )
                .fill_color(color32_a(c, 128))
                .stroke(egui::Stroke::new(1.0, color32(c)))
            };
            let label = |plot: &mut PlotUi<'_>, x: f64, y: f64, s: String, c: theme::Rgb| {
                plot.text(
                    Text::new("", [x, y].into(), egui::RichText::new(s).size(15.0))
                        .color(color32(c)),
                );
            };
            // top bar: the domain R^n split into rank + nullity
            if rank > 0.0 {
                plot.polygon(rect(0.0, rank, 1.4, 2.2, theme::SPAN));
                label(plot, rank / 2.0, 1.8, format!("rank = {rank}"), theme::FG);
            }
            if nullity > 0.0 {
                plot.polygon(rect(rank, n, 1.4, 2.2, theme::EIGEN));
                label(
                    plot,
                    (rank + n) / 2.0,
                    1.8,
                    format!("nullity = {nullity}"),
                    theme::FG,
                );
            }
            label(
                plot,
                n / 2.0,
                2.55,
                format!("domain  R^{ncols}  (dim {ncols})"),
                theme::FG,
            );
            // bottom bar: the image, dimension = rank
            if rank > 0.0 {
                plot.polygon(rect(0.0, rank, -1.4, -0.6, theme::SPAN));
                label(
                    plot,
                    rank / 2.0,
                    -1.0,
                    format!("image · dim = {rank}"),
                    theme::FG,
                );
            }
            label(
                plot,
                n.max(1.0) / 2.0,
                -2.15,
                format!("rank + nullity  =  {rank} + {nullity}  =  {n}  =  dim(domain)"),
                theme::ACCENT,
            );
        });
    Vec::new()
}

pub fn riemann1(ui: &mut Ui, spec: &SceneSpec, t: f64) -> Vec<Row> {
    let SceneSpec::Riemann1 { f, a, b, n, .. } = spec else {
        return Vec::new();
    };
    let (a, b, n) = (*a, *b, *n);
    let dx = (b - a) / n as f64;
    let revealed = (ease(t / RIEMANN1_SECS) * n as f64).round() as usize;

    let curve: Vec<[f64; 2]> = (0..=240)
        .map(|i| {
            let x = a + (b - a) * i as f64 / 240.0;
            [x, f.eval(&[x])]
        })
        .collect();
    let (ymin, ymax) = curve
        .iter()
        .fold((0.0f64, 0.0f64), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
    let pad_x = (b - a) * 0.08;
    let pad_y = (ymax - ymin).max(1e-9) * 0.15;

    let mut running = 0.0;
    Plot::new("riemann1")
        .default_x_bounds(a - pad_x, b + pad_x)
        .default_y_bounds(ymin - pad_y, ymax + pad_y)
        .auto_bounds(false)
        .show_grid(false)
        .show(ui, |plot| {
            for i in 0..revealed {
                let x0 = a + i as f64 * dx;
                let h = f.eval(&[x0 + dx / 2.0]); // midpoint rule
                running += h * dx;
                let color = if i + 1 == revealed && revealed < n {
                    theme::EIGEN // the advancing front
                } else if h >= 0.0 {
                    theme::SPAN
                } else {
                    theme::ACCENT2
                };
                plot.polygon(
                    Polygon::new(
                        "",
                        PlotPoints::from(vec![[x0, 0.0], [x0 + dx, 0.0], [x0 + dx, h], [x0, h]]),
                    )
                    .fill_color(color32_a(color, 110))
                    .stroke(egui::Stroke::new(1.0, color32_a(color, 200))),
                );
            }
            plot.line(
                Line::new("f", PlotPoints::from(curve.clone()))
                    .color(color32(theme::ACCENT))
                    .width(2.5),
            );
        });

    let mut rows = vec![Row::bold(
        format!("running sum ≈ {}", trim_num(running)),
        theme::EIGEN,
    )];
    if revealed >= n {
        rows.push(Row::colored(
            format!("all {n} rectangles placed — thinner rectangles → exact value"),
            theme::GOOD,
        ));
    }
    rows
}

pub fn sweep2d(ui: &mut Ui, spec: &SceneSpec, t: f64) -> Vec<Row> {
    let SceneSpec::Sweep {
        cells,
        front,
        exact,
        total,
        bounds,
        ..
    } = spec
    else {
        return Vec::new();
    };
    let ncells = cells.len();
    let duration = SWEEP2D_SECS;
    let revealed = ((ease(t / duration)) * ncells as f64).round() as usize;
    let vmax = cells
        .iter()
        .map(|c| c.value.abs())
        .fold(1e-12, f64::max);

    let mut running = 0.0;
    Plot::new("sweep2d")
        .data_aspect(1.0)
        .default_x_bounds(bounds[0].0 - 0.3, bounds[0].1 + 0.3)
        .default_y_bounds(bounds[1].0 - 0.3, bounds[1].1 + 0.3)
        .auto_bounds(false)
        .show_grid(false)
        .show(ui, |plot| {
            for (i, cell) in cells.iter().take(revealed).enumerate() {
                running += cell.contribution;
                let is_front = i + front >= revealed;
                let alpha = (40.0 + 160.0 * (cell.value.abs() / vmax)).min(255.0) as u8;
                let (fill, stroke) = if is_front {
                    (color32_a(theme::EIGEN, alpha.max(120)), egui::Stroke::NONE)
                } else if cell.value >= 0.0 {
                    (color32_a(theme::SPAN, alpha), egui::Stroke::NONE)
                } else {
                    (color32_a(theme::ACCENT2, alpha), egui::Stroke::NONE)
                };
                let pts: Vec<[f64; 2]> = cell.corners.iter().map(|c| [c[0], c[1]]).collect();
                plot.polygon(
                    Polygon::new("", PlotPoints::from(pts))
                        .fill_color(fill)
                        .stroke(stroke),
                );
            }
        });

    let mut rows = vec![Row::bold(
        format!("running sum ≈ {}", trim_num(running)),
        theme::EIGEN,
    )];
    if let Some(e) = exact {
        rows.push(Row::colored(
            format!("→ converges to {:.6}", e),
            theme::GOOD,
        ));
    } else {
        rows.push(Row::colored(
            format!("→ Riemann total {:.6}", total),
            theme::GOOD,
        ));
    }
    rows
}

