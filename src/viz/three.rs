//! A small software-projected 3D scene: orbit camera, depth-sorted lines and
//! convex polygons drawn straight onto the egui painter. More than enough for
//! wireframe lattices, shaded subspaces, and Riemann boxes — no shaders.

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use nalgebra::{DVector, Matrix3, Vector3};

use crate::core::linalg::{project_onto_span, TOL};
use crate::scene::{trim_num, Row, SceneSpec, SpaceMode, WarpVerdict};
use crate::theme;

use super::{
    color32, color32_a, ease, KERNEL_SECS, LINCOMB_SECS, REVOLVE_SECS, STAGE_SECS, SWEEP3D_SECS,
};

pub struct Camera {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            yaw: 0.65,
            pitch: 0.42,
            dist: 3.2,
        }
    }
}

struct Proj {
    rect: Rect,
    rot: Matrix3<f32>,
    dist: f32,
    scale: f32,
}

impl Proj {
    fn new(rect: Rect, cam: &Camera, world: f32) -> Proj {
        let (sy, cy) = cam.yaw.sin_cos();
        let (sp, cp) = cam.pitch.sin_cos();
        // Yaw about world z (up), then pitch toward the viewer; z stays up.
        let yaw = Matrix3::new(cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0);
        let pitch = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
        Proj {
            rect,
            rot: pitch * yaw,
            dist: cam.dist,
            scale: rect.width().min(rect.height()) / (2.4 * world),
        }
    }

    /// World point -> (screen pos, camera depth). Larger depth = farther.
    fn point(&self, p: &[f64]) -> (Pos2, f32) {
        let v = self.rot
            * Vector3::new(p[0] as f32, p[1] as f32, p.get(2).copied().unwrap_or(0.0) as f32);
        // After rotation: x right, y depth-ish, z up. Mild perspective on y.
        let depth = v.y;
        let persp = self.dist / (self.dist + depth * 0.18);
        let c = self.rect.center();
        (
            Pos2::new(
                c.x + v.x * self.scale * persp,
                c.y - v.z * self.scale * persp,
            ),
            depth,
        )
    }
}

enum Prim {
    Line {
        a: Vec<f64>,
        b: Vec<f64>,
        color: Color32,
        width: f32,
    },
    Poly {
        pts: Vec<Vec<f64>>,
        fill: Color32,
        stroke: Stroke,
    },
    Dot {
        p: Vec<f64>,
        color: Color32,
        r: f32,
    },
    Arrow {
        a: Vec<f64>,
        b: Vec<f64>,
        color: Color32,
        width: f32,
    },
    Label {
        p: Vec<f64>,
        text: String,
        color: Color32,
        size: f32,
    },
}

impl Prim {
    fn depth(&self, proj: &Proj) -> f32 {
        let mid = |pts: &[Vec<f64>]| {
            let mut d = 0.0;
            for p in pts {
                d += proj.point(p).1;
            }
            d / pts.len() as f32
        };
        match self {
            Prim::Line { a, b, .. } | Prim::Arrow { a, b, .. } => {
                (proj.point(a).1 + proj.point(b).1) * 0.5
            }
            Prim::Poly { pts, .. } => mid(pts),
            Prim::Dot { p, .. } | Prim::Label { p, .. } => proj.point(p).1,
        }
    }
}

pub struct Scene3 {
    prims: Vec<Prim>,
}

impl Scene3 {
    fn new() -> Scene3 {
        Scene3 { prims: Vec::new() }
    }

    fn line(&mut self, a: &[f64], b: &[f64], color: Color32, width: f32) {
        self.prims.push(Prim::Line {
            a: a.to_vec(),
            b: b.to_vec(),
            color,
            width,
        });
    }

    fn poly(&mut self, pts: Vec<Vec<f64>>, fill: Color32, stroke: Stroke) {
        self.prims.push(Prim::Poly { pts, fill, stroke });
    }

    fn dot(&mut self, p: &[f64], color: Color32, r: f32) {
        self.prims.push(Prim::Dot {
            p: p.to_vec(),
            color,
            r,
        });
    }

    fn arrow(&mut self, a: &[f64], b: &[f64], color: Color32, width: f32) {
        self.prims.push(Prim::Arrow {
            a: a.to_vec(),
            b: b.to_vec(),
            color,
            width,
        });
    }

    fn label(&mut self, p: &[f64], text: impl Into<String>, color: Color32, size: f32) {
        self.prims.push(Prim::Label {
            p: p.to_vec(),
            text: text.into(),
            color,
            size,
        });
    }

    fn draw(mut self, painter: &egui::Painter, proj: &Proj) {
        // Painter's algorithm: farthest first.
        self.prims
            .sort_by(|a, b| b.depth(proj).partial_cmp(&a.depth(proj)).unwrap());
        for prim in &self.prims {
            match prim {
                Prim::Line { a, b, color, width } => {
                    let (pa, _) = proj.point(a);
                    let (pb, _) = proj.point(b);
                    painter.line_segment([pa, pb], Stroke::new(*width, *color));
                }
                Prim::Poly { pts, fill, stroke } => {
                    let screen: Vec<Pos2> = pts.iter().map(|p| proj.point(p).0).collect();
                    painter.add(egui::Shape::convex_polygon(screen, *fill, *stroke));
                }
                Prim::Dot { p, color, r } => {
                    let (pos, _) = proj.point(p);
                    painter.circle_filled(pos, *r, *color);
                }
                Prim::Arrow { a, b, color, width } => {
                    let (pa, _) = proj.point(a);
                    let (pb, _) = proj.point(b);
                    painter.line_segment([pa, pb], Stroke::new(*width, *color));
                    let d = pb - pa;
                    let len = d.length();
                    if len > 4.0 {
                        let u = d / len;
                        let perp = Vec2::new(-u.y, u.x);
                        let head = (0.18 * len).clamp(6.0, 16.0);
                        let p1 = pb - u * head + perp * head * 0.45;
                        let p2 = pb - u * head - perp * head * 0.45;
                        painter.add(egui::Shape::convex_polygon(
                            vec![pb, p1, p2],
                            *color,
                            Stroke::NONE,
                        ));
                    }
                }
                Prim::Label { p, text, color, size } => {
                    let (pos, _) = proj.point(p);
                    painter.text(
                        pos,
                        egui::Align2::CENTER_BOTTOM,
                        text,
                        egui::FontId::proportional(*size),
                        *color,
                    );
                }
            }
        }
    }
}

/// Allocate the canvas, handle orbit/zoom input, and return painter + proj.
fn canvas(ui: &mut Ui, cam: &mut Camera, world: f32) -> (egui::Painter, Proj) {
    let size = ui.available_size();
    let (response, painter) = ui.allocate_painter(size, Sense::drag());
    if response.dragged() {
        let d = response.drag_delta();
        cam.yaw += d.x * 0.01;
        cam.pitch = (cam.pitch + d.y * 0.01).clamp(-1.5, 1.5);
    }
    if response.hovered() {
        // Exponential zoom: each wheel notch scales the distance ~28%, so
        // zooming out accelerates with distance instead of crawling linearly.
        let scroll = ui.input(|i| i.smooth_scroll_delta().y);
        if scroll != 0.0 {
            cam.dist = (cam.dist * (-scroll * 0.005).exp()).clamp(1.2, 12.0);
        }
    }
    painter.rect_filled(response.rect, 0.0, color32(theme::BG));
    let proj = Proj::new(response.rect, cam, world * cam.dist / 3.2);
    (painter, proj)
}

fn axes(scene: &mut Scene3, s: f64) {
    axes_labeled(scene, s, ["x", "y", "z"]);
}

/// Axes with caller-chosen names for (right, depth, up) — revolution scenes
/// relabel them so the axis of revolution carries the user's variable.
fn axes_labeled(scene: &mut Scene3, s: f64, names: [&str; 3]) {
    let axes = [
        ([s, 0.0, 0.0], names[0]),
        ([0.0, s, 0.0], names[1]),
        ([0.0, 0.0, s], names[2]),
    ];
    for (tip, name) in axes {
        scene.line(&[0.0, 0.0, 0.0], &tip, color32(theme::GRID_HI), 1.4);
        if !name.is_empty() {
            scene.label(&tip.map(|v| v * 1.06), name, color32(theme::MUTED), 14.0);
        }
    }

    // Tick marks + values at a "nice" step chosen from the axis length.
    let step = tick_step(s);
    let h = s * 0.025; // tick half-length
    let tick_c = color32_a(theme::GRID_HI, 170);
    let num_c = color32_a(theme::MUTED, 200);
    let mut v = step;
    while v < s * 0.97 {
        let txt = trim_num(v);
        // x and y ticks cross in the horizontal plane, numbers hang below;
        // z ticks cross along x, numbers sit beside the axis.
        scene.line(&[v, -h, 0.0], &[v, h, 0.0], tick_c, 1.0);
        scene.label(&[v, 0.0, -3.5 * h], &txt, num_c, 11.0);
        scene.line(&[-h, v, 0.0], &[h, v, 0.0], tick_c, 1.0);
        scene.label(&[0.0, v, -3.5 * h], &txt, num_c, 11.0);
        scene.line(&[-h, 0.0, v], &[h, 0.0, v], tick_c, 1.0);
        scene.label(&[-3.5 * h, 0.0, v], &txt, num_c, 11.0);
        v += step;
    }
}

/// Largest of 1/2/5 × 10^k giving roughly four ticks along a length `s`.
fn tick_step(s: f64) -> f64 {
    let raw = s / 4.0;
    let mag = 10f64.powf(raw.log10().floor());
    match raw / mag {
        r if r < 1.5 => mag,
        r if r < 3.5 => 2.0 * mag,
        r if r < 7.5 => 5.0 * mag,
        _ => 10.0 * mag,
    }
}

fn mat3(row_major: &[f64]) -> Matrix3<f64> {
    Matrix3::from_row_slice(row_major)
}

fn lerp3(a: &Matrix3<f64>, b: &Matrix3<f64>, t: f64) -> Matrix3<f64> {
    a * (1.0 - t) + b * t
}

/// Lattice lines on the three coordinate planes through the origin, warped.
fn warped_lattice(scene: &mut Scene3, m: &Matrix3<f64>, extent: i64) {
    let s = extent as f64;
    let color = color32_a(theme::GRID_LINE, 130);
    let mut seg = |a: Vector3<f64>, b: Vector3<f64>| {
        let ta = m * a;
        let tb = m * b;
        scene.line(&[ta.x, ta.y, ta.z], &[tb.x, tb.y, tb.z], color, 1.0);
    };
    for k in -extent..=extent {
        let k = k as f64;
        // xy-plane (z = 0)
        seg(Vector3::new(k, -s, 0.0), Vector3::new(k, s, 0.0));
        seg(Vector3::new(-s, k, 0.0), Vector3::new(s, k, 0.0));
        // xz-plane (y = 0)
        seg(Vector3::new(k, 0.0, -s), Vector3::new(k, 0.0, s));
        seg(Vector3::new(-s, 0.0, k), Vector3::new(s, 0.0, k));
    }
}

fn unit_cube_edges(scene: &mut Scene3, m: &Matrix3<f64>, color: Color32) {
    let corners: Vec<Vector3<f64>> = (0..8)
        .map(|i| {
            m * Vector3::new(
                (i & 1) as f64,
                ((i >> 1) & 1) as f64,
                ((i >> 2) & 1) as f64,
            )
        })
        .collect();
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            let j = i ^ bit;
            if j > i {
                scene.line(
                    &[corners[i].x, corners[i].y, corners[i].z],
                    &[corners[j].x, corners[j].y, corners[j].z],
                    color,
                    1.6,
                );
            }
        }
    }
}

fn subspace(scene: &mut Scene3, basis: &[Vec<f64>], s: f64, color: theme::Rgb, alpha: u8) {
    match basis.len() {
        1 => {
            let d: Vec<f64> = basis[0].iter().map(|v| v * s * 1.6).collect();
            let neg: Vec<f64> = d.iter().map(|v| -v).collect();
            scene.line(&neg, &d, color32_a(color, 220), 2.4);
        }
        2 => {
            let b0 = &basis[0];
            let b1 = &basis[1];
            let corner = |c0: f64, c1: f64| -> Vec<f64> {
                (0..3).map(|i| (b0[i] * c0 + b1[i] * c1) * s).collect()
            };
            scene.poly(
                vec![
                    corner(-1.0, -1.0),
                    corner(1.0, -1.0),
                    corner(1.0, 1.0),
                    corner(-1.0, 1.0),
                ],
                color32_a(color, alpha),
                Stroke::new(1.0, color32_a(color, 190)),
            );
        }
        3 => {
            // All of space: a translucent bounding cube.
            let s2 = s * 0.9;
            for (fixed_axis, side) in [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)]
            {
                let mut pts = Vec::new();
                for (u, v) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut p = [0.0; 3];
                    p[fixed_axis] = side * s2;
                    let others: Vec<usize> = (0..3).filter(|i| *i != fixed_axis).collect();
                    p[others[0]] = u * s2;
                    p[others[1]] = v * s2;
                    pts.push(p.to_vec());
                }
                scene.poly(pts, color32_a(color, alpha / 3), Stroke::NONE);
            }
        }
        _ => {}
    }
}

// ------------------------------------------------------------------- scenes

pub fn scene3d(ui: &mut Ui, spec: &SceneSpec, cam: &mut Camera, t: f64) -> Vec<Row> {
    match spec {
        SceneSpec::Warp {
            stages,
            eigen,
            verdict,
            span,
            ..
        } => warp3d(ui, cam, stages, eigen, *verdict, *span, t),
        SceneSpec::Space {
            arrows,
            span_basis,
            rank,
            mode,
            span,
            ..
        } => space3d(ui, cam, arrows, span_basis, *rank, mode, *span, t),
        SceneSpec::KernelImage {
            matrix,
            kernel,
            image,
            span,
            ..
        } => kernel3d(ui, cam, matrix, kernel, image, *span, t),
        SceneSpec::Sweep {
            cells,
            front,
            exact,
            total,
            bounds,
            ..
        } => sweep3d(ui, cam, cells, *front, *exact, *total, bounds, t),
        SceneSpec::Revolution {
            outer,
            inner,
            shells,
            var,
            volume,
            ..
        } => revolution(ui, cam, outer, inner.as_deref(), *shells, var, *volume, t),
        _ => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn warp3d(
    ui: &mut Ui,
    cam: &mut Camera,
    stages: &[Vec<f64>],
    eigen: &[crate::scene::EigenLine],
    verdict: WarpVerdict,
    span: f64,
    t: f64,
) -> Vec<Row> {
    let mats: Vec<Matrix3<f64>> = stages.iter().map(|s| mat3(s)).collect();
    let nstages = mats.len();
    let total = STAGE_SECS * nstages as f64;
    let done = t >= total;
    let (current, frac) = if done {
        (nstages - 1, 1.0)
    } else {
        let k = (t / STAGE_SECS) as usize;
        (k.min(nstages - 1), ease(t / STAGE_SECS - k as f64))
    };
    let prev = if current == 0 {
        Matrix3::identity()
    } else {
        mats[current - 1]
    };
    let m = lerp3(&prev, &mats[current], frac);
    let det = m.determinant();

    let (painter, proj) = canvas(ui, cam, span as f32 * 0.75);
    let mut scene = Scene3::new();
    axes(&mut scene, span * 0.9);
    warped_lattice(&mut scene, &m, 3);
    unit_cube_edges(
        &mut scene,
        &m,
        color32(if det < 0.0 { theme::FLIP } else { theme::SPAN }),
    );
    for (k, color) in theme::BASIS_COLORS.iter().enumerate() {
        let v = m.column(k);
        scene.arrow(&[0.0; 3], &[v[0], v[1], v[2]], color32(*color), 3.0);
    }
    let show_eigen = verdict == WarpVerdict::EigenFocus || (done && verdict != WarpVerdict::Compose);
    if show_eigen {
        for e in eigen {
            let d: Vec<f64> = e.direction.iter().map(|v| v * span).collect();
            let neg: Vec<f64> = d.iter().map(|v| -v).collect();
            scene.line(&neg, &d, color32(theme::EIGEN), 2.2);
        }
    }
    if verdict == WarpVerdict::EigenFocus {
        for e in eigen {
            let v = m * Vector3::new(e.direction[0], e.direction[1], e.direction[2]);
            scene.arrow(&[0.0; 3], &[v.x, v.y, v.z], color32(theme::EIGEN), 3.4);
        }
    }
    scene.draw(&painter, &proj);

    let det_color = if det.abs() < 1e-9 {
        theme::MUTED
    } else if det < 0.0 {
        theme::BAD
    } else {
        theme::GOOD
    };
    let mut rows = vec![Row::bold(format!("det = {det:+.2}  (live)"), det_color)];
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

#[allow(clippy::too_many_arguments)]
fn space3d(
    ui: &mut Ui,
    cam: &mut Camera,
    arrows: &[crate::scene::ArrowSpec],
    span_basis: &[Vec<f64>],
    rank: usize,
    mode: &SpaceMode,
    span: f64,
    t: f64,
) -> Vec<Row> {
    let vectors: Vec<DVector<f64>> = arrows
        .iter()
        .map(|a| DVector::from_row_slice(&a.components))
        .collect();
    let (painter, proj) = canvas(ui, cam, span as f32 * 0.75);
    let mut scene = Scene3::new();
    axes(&mut scene, span * 0.9);
    let _ = rank;
    subspace(&mut scene, span_basis, span * 0.8, theme::SPAN, 60);

    match mode {
        SpaceMode::Independent {
            collapse_index: Some(idx),
        } => {
            let frac = ease(t / STAGE_SECS);
            for (i, a) in arrows.iter().enumerate() {
                if i == *idx {
                    let others: Vec<DVector<f64>> = vectors
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| j != idx)
                        .map(|(_, v)| v.clone())
                        .collect();
                    let target = project_onto_span(&vectors[i], &others, TOL);
                    let pos = &vectors[i] * (1.0 - frac) + target * frac;
                    scene.arrow(&[0.0; 3], pos.as_slice(), color32(theme::EIGEN), 3.4);
                } else {
                    scene.arrow(&[0.0; 3], vectors[i].as_slice(), color32(a.color), 3.0);
                }
            }
        }
        SpaceMode::Lincomb { coeffs } => {
            let k = vectors.len();
            let segments: Vec<Vector3<f64>> = (0..k)
                .map(|i| Vector3::new(vectors[i][0], vectors[i][1], vectors[i][2]) * coeffs[i])
                .collect();
            let mut tails = vec![Vector3::zeros()];
            for s in &segments {
                let last = *tails.last().unwrap();
                tails.push(last + s);
            }
            let s = ease(t / LINCOMB_SECS) * k as f64;
            for i in 0..k {
                let frac = (s - i as f64).clamp(0.0, 1.0);
                let tip = tails[i] + segments[i] * frac;
                scene.arrow(
                    &[tails[i].x, tails[i].y, tails[i].z],
                    &[tip.x, tip.y, tip.z],
                    color32(theme::BASIS_COLORS[i % 3]),
                    3.0,
                );
            }
            let active = (s as usize).min(k - 1);
            let frac = (s - active as f64).clamp(0.0, 1.0);
            let r = tails[active] + segments[active] * frac;
            scene.arrow(&[0.0; 3], &[r.x, r.y, r.z], color32(theme::POINT), 3.4);
        }
        SpaceMode::Member {
            point,
            projection,
            inside,
            ..
        } => {
            for (i, a) in arrows.iter().enumerate() {
                scene.arrow(&[0.0; 3], vectors[i].as_slice(), color32(a.color), 3.0);
            }
            let pc = if *inside { theme::GOOD } else { theme::BAD };
            scene.dot(point, color32(pc), 6.0);
            if !inside {
                scene.dot(projection, color32(theme::SPAN), 5.0);
                scene.line(point, projection, color32(theme::BAD), 2.0);
            }
        }
        _ => {
            for (i, a) in arrows.iter().enumerate() {
                scene.arrow(&[0.0; 3], vectors[i].as_slice(), color32(a.color), 3.0);
            }
        }
    }

    for (i, a) in arrows.iter().enumerate() {
        let p: Vec<f64> = vectors[i].iter().map(|v| v * 1.1).collect();
        scene.label(&p, a.label.clone(), color32(a.color), 14.0);
    }
    scene.draw(&painter, &proj);
    Vec::new()
}

fn kernel3d(
    ui: &mut Ui,
    cam: &mut Camera,
    matrix: &[f64],
    kernel: &[Vec<f64>],
    image: &[Vec<f64>],
    span: f64,
    t: f64,
) -> Vec<Row> {
    let a = mat3(matrix);
    let frac = ease(t / KERNEL_SECS);
    let m = lerp3(&Matrix3::identity(), &a, frac);

    let (painter, proj) = canvas(ui, cam, span as f32 * 0.75);
    let mut scene = Scene3::new();
    axes(&mut scene, span * 0.9);
    subspace(&mut scene, image, span * 0.8, theme::SPAN, 60);
    subspace(&mut scene, kernel, span * 0.8, theme::EIGEN, 70);
    warped_lattice(&mut scene, &m, 3);
    for k in kernel {
        let v0 = Vector3::new(k[0], k[1], k[2]) * (span * 0.6);
        let v = m * v0;
        scene.arrow(&[0.0; 3], &[v.x, v.y, v.z], color32(theme::EIGEN), 3.4);
    }
    scene.draw(&painter, &proj);
    vec![Row::colored(
        "gold arrows shrink to 0 — purple is where outputs land",
        theme::MUTED,
    )]
}

#[allow(clippy::too_many_arguments)]
fn sweep3d(
    ui: &mut Ui,
    cam: &mut Camera,
    cells: &[crate::scene::CellGeom],
    front: usize,
    exact: Option<f64>,
    total: f64,
    bounds: &[(f64, f64)],
    t: f64,
) -> Vec<Row> {
    let world = bounds
        .iter()
        .map(|(lo, hi)| lo.abs().max(hi.abs()))
        .fold(1.0, f64::max);
    let duration = SWEEP3D_SECS;
    let ncells = cells.len();
    let revealed = (ease(t / duration) * ncells as f64).round() as usize;
    let vmax = cells.iter().map(|c| c.value.abs()).fold(1e-12, f64::max);

    let (painter, proj) = canvas(ui, cam, world as f32);
    let mut scene = Scene3::new();
    axes(&mut scene, world * 1.15);

    // Cube faces as corner indices (matching BITS_3D ordering: bottom, top).
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // bottom
        [4, 5, 6, 7], // top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let mut running = 0.0;
    for (i, cell) in cells.iter().take(revealed).enumerate() {
        running += cell.contribution;
        let is_front = i + front >= revealed;
        let alpha = (14.0 + 90.0 * (cell.value.abs() / vmax)).min(200.0) as u8;
        let fill = if is_front {
            color32_a(theme::EIGEN, alpha.max(80))
        } else if cell.value >= 0.0 {
            color32_a(theme::SPAN, alpha)
        } else {
            color32_a(theme::ACCENT2, alpha)
        };
        for face in FACES {
            scene.poly(
                face.iter().map(|k| cell.corners[*k].clone()).collect(),
                fill,
                Stroke::NONE,
            );
        }
    }
    scene.draw(&painter, &proj);

    let mut rows = vec![Row::bold(
        format!("running sum ≈ {}", trim_num(running)),
        theme::EIGEN,
    )];
    match exact {
        Some(e) => rows.push(Row::colored(format!("→ converges to {e:.6}"), theme::GOOD)),
        None => rows.push(Row::colored(format!("→ Riemann total {total:.6}"), theme::GOOD)),
    }
    rows
}

fn revolution(
    ui: &mut Ui,
    cam: &mut Camera,
    outer: &[(f64, f64)],
    inner: Option<&[(f64, f64)]>,
    shells: bool,
    var: &str,
    volume: f64,
    t: f64,
) -> Vec<Row> {
    // The bounds variable owns its axis: disks revolve about the var-axis,
    // shells about the perpendicular one. The y-axis is drawn vertical
    // (scene up); every other axis of revolution is drawn horizontal.
    let axis: &str = if shells {
        if var == "y" { "x" } else { "y" }
    } else {
        var
    };
    let radial: &str = if shells {
        var
    } else if var == "y" {
        "x"
    } else {
        "y"
    };
    let vertical = axis == "y";

    let (a, b) = (outer.first().unwrap().0, outer.last().unwrap().0);
    let rmax = outer.iter().map(|(_, y)| y.abs()).fold(1e-9, f64::max);
    let world = (b - a).max(rmax * 2.0) as f32 * 0.7;
    let duration = REVOLVE_SECS;
    let frac = ease(t / duration);

    let (painter, proj) = canvas(ui, cam, world);
    let mut scene = Scene3::new();
    let names: [&str; 3] = if vertical {
        [radial, "", axis] // up = axis of revolution
    } else {
        [axis, radial, ""] // right = axis of revolution
    };
    axes_labeled(&mut scene, world as f64 * 1.2, names);

    const NTHETA: usize = 28;
    // A curve sample (v, f(v)) becomes (position along the axis, radius).
    let place = |v: f64, f: f64| -> (f64, f64) { if shells { (f, v) } else { (v, f) } };
    let ring = |along: f64, radius: f64| -> Vec<Vec<f64>> {
        (0..NTHETA)
            .map(|j| {
                let th = std::f64::consts::TAU * j as f64 / NTHETA as f64;
                if vertical {
                    vec![radius * th.cos(), radius * th.sin(), along]
                } else {
                    vec![along, radius * th.cos(), radius * th.sin()]
                }
            })
            .collect()
    };
    // The generating curve sits on the θ = 0 line of its rings.
    let curve_pt = |v: f64, f: f64| -> [f64; 3] {
        let (along, radius) = place(v, f);
        if vertical {
            [radius, 0.0, along]
        } else {
            [along, radius, 0.0]
        }
    };

    // Wireframe of the full solid (faint), plus the revealed surface rings.
    let step = 6;
    let n = outer.len() - 1;
    let revealed = ((n as f64) * frac) as usize;
    for i in (0..n).step_by(step) {
        let (v, f) = outer[i];
        let (along, radius) = place(v, f);
        let pts = ring(along, radius.abs());
        for j in 0..NTHETA {
            let k = (j + 1) % NTHETA;
            let color = if i <= revealed {
                color32_a(theme::SPAN, 190)
            } else {
                color32_a(theme::GRID_LINE, 90)
            };
            scene.line(&pts[j], &pts[k], color, 1.2);
        }
    }
    // The generating curve itself.
    for w in outer.windows(2) {
        scene.line(
            &curve_pt(w[0].0, w[0].1),
            &curve_pt(w[1].0, w[1].1),
            color32(theme::ACCENT),
            2.2,
        );
    }
    if let Some(inner_pts) = inner {
        for w in inner_pts.windows(2) {
            scene.line(
                &curve_pt(w[0].0, w[0].1),
                &curve_pt(w[1].0, w[1].1),
                color32(theme::ACCENT2),
                2.2,
            );
        }
    }

    // Sweeping disk / washer / shell at the front.
    let (fv, ff) = outer[revealed.min(n)];
    let (along, radius) = place(fv, ff);
    scene.poly(
        ring(along, radius.abs()),
        color32_a(theme::EIGEN, 90),
        Stroke::new(1.5, color32(theme::EIGEN)),
    );

    scene.draw(&painter, &proj);
    vec![Row::bold(
        format!("V ≈ {}", trim_num(volume * frac)),
        theme::EIGEN,
    )]
}

/// Expose minimal drawing for func.rs surface rendering.
pub fn surface_mesh(
    ui: &mut Ui,
    cam: &mut Camera,
    world: f32,
    build: impl FnOnce(&mut Scene3),
) {
    let (painter, proj) = canvas(ui, cam, world);
    let mut scene = Scene3::new();
    build(&mut scene);
    scene.draw(&painter, &proj);
}

impl Scene3 {
    pub fn add_line(&mut self, a: &[f64], b: &[f64], color: Color32, width: f32) {
        self.line(a, b, color, width);
    }

    pub fn add_poly(&mut self, pts: Vec<Vec<f64>>, fill: Color32, stroke: Stroke) {
        self.poly(pts, fill, stroke);
    }

    pub fn add_arrow(&mut self, a: &[f64], b: &[f64], color: Color32, width: f32) {
        self.arrow(a, b, color, width);
    }

    pub fn add_dot(&mut self, p: &[f64], color: Color32, r: f32) {
        self.dot(p, color, r);
    }

    pub fn add_axes(&mut self, s: f64) {
        axes(self, s);
    }
}
