//! Interactive function scenes: the partial-derivative surface and the flat
//! contour explorer. Both carry the expression AST in the scene spec, so f,
//! ∂f/∂x and ∂f/∂y are evaluated live as the user drags the sliders.

use eframe::egui::{self, Color32, ColorImage, Stroke, TextureHandle, TextureOptions, Ui};
use egui_plot::{Line, Plot, PlotImage, PlotPoints, Points};

use crate::core::expr::Expr;
use crate::scene::{Row, SceneSpec};
use crate::theme;

use super::three::{surface_mesh, Camera};
use super::{color32, color32_a};

const H: f64 = 1e-5; // finite-difference step for the partials

pub struct FuncState {
    pub x0: f64,
    pub y0: f64,
    texture: Option<TextureHandle>,
}

impl FuncState {
    pub fn new(spec: &SceneSpec) -> FuncState {
        let (a, b) = match spec {
            SceneSpec::Surface { domain, .. } | SceneSpec::Contour { domain, .. } => *domain,
            _ => (-3.0, 3.0),
        };
        let mid = 0.5 * (a + b);
        // Nudge off-center so the tangent arrows are not degenerate at a
        // symmetric critical point.
        let off = 0.18 * (b - a);
        FuncState {
            x0: mid + off,
            y0: mid - off,
            texture: None,
        }
    }
}

fn slider_row(ui: &mut Ui, state: &mut FuncState, domain: (f64, f64)) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("x").color(color32(theme::ACCENT)).monospace());
        ui.add(egui::Slider::new(&mut state.x0, domain.0..=domain.1).fixed_decimals(2));
        ui.add_space(16.0);
        ui.label(egui::RichText::new("y").color(color32(theme::ACCENT2)).monospace());
        ui.add(egui::Slider::new(&mut state.y0, domain.0..=domain.1).fixed_decimals(2));
    });
}

fn live_rows(f: &Expr, x0: f64, y0: f64) -> Vec<Row> {
    let args = [x0, y0];
    let val = f.eval(&args);
    let fx = f.partial(&args, 0, H);
    let fy = f.partial(&args, 1, H);
    vec![
        Row::plain(format!("at (x, y) = ({x0:+.2}, {y0:+.2})")),
        Row::colored(format!("f       = {val:+.4}"), theme::FG),
        Row::bold(format!("∂f/∂x = {fx:+.4}"), theme::ACCENT),
        Row::bold(format!("∂f/∂y = {fy:+.4}"), theme::ACCENT2),
    ]
}

// ------------------------------------------------------------------ surface

pub fn surface(
    ui: &mut Ui,
    spec: &SceneSpec,
    state: &mut FuncState,
    cam: &mut Camera,
) -> Vec<Row> {
    let SceneSpec::Surface { f, domain, .. } = spec else {
        return Vec::new();
    };
    let (a, b) = *domain;
    slider_row(ui, state, (a, b));
    let (x0, y0) = (state.x0, state.y0);
    let rows = live_rows(f, x0, y0);

    // Normalize z so wild functions stay in frame.
    const N: usize = 30;
    let xs: Vec<f64> = (0..=N).map(|i| a + (b - a) * i as f64 / N as f64).collect();
    let mut zmax: f64 = 1e-9;
    let mut z = vec![vec![0.0; N + 1]; N + 1];
    for (i, x) in xs.iter().enumerate() {
        for (j, y) in xs.iter().enumerate() {
            z[i][j] = f.eval(&[*x, *y]);
            zmax = zmax.max(z[i][j].abs());
        }
    }
    let zscale = 0.45 * (b - a) / zmax;
    let world = (b - a) as f32 * 0.72;
    let fx = f.partial(&[x0, y0], 0, H);
    let fy = f.partial(&[x0, y0], 1, H);
    let z0 = f.eval(&[x0, y0]);

    surface_mesh(ui, cam, world, |scene| {
        scene.add_axes((b - a) * 0.62);
        // Surface quads colored by height.
        for i in 0..N {
            for j in 0..N {
                let quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
                let zavg = quad.iter().map(|(p, q)| z[*p][*q]).sum::<f64>() / 4.0;
                let s = ((zavg / zmax) as f32).clamp(-1.0, 1.0);
                let fill = mix_color(theme::ACCENT2, theme::ACCENT, (s + 1.0) / 2.0, 130);
                scene.add_poly(
                    quad.iter()
                        .map(|(p, q)| vec![xs[*p], xs[*q], z[*p][*q] * zscale])
                        .collect(),
                    fill,
                    Stroke::new(0.4, color32_a(theme::GRID_LINE, 90)),
                );
            }
        }
        // Slice curves through (x0, y0): f(x, y0) and f(x0, y).
        for w in xs.windows(2) {
            scene.add_line(
                &[w[0], y0, f.eval(&[w[0], y0]) * zscale],
                &[w[1], y0, f.eval(&[w[1], y0]) * zscale],
                color32(theme::ACCENT),
                2.4,
            );
            scene.add_line(
                &[x0, w[0], f.eval(&[x0, w[0]]) * zscale],
                &[x0, w[1], f.eval(&[x0, w[1]]) * zscale],
                color32(theme::ACCENT2),
                2.4,
            );
        }
        // Tangent arrows along each slice (slope = the partials).
        let l = (b - a) * 0.16;
        let p0 = [x0, y0, z0 * zscale];
        scene.add_arrow(
            &p0,
            &[x0 + l, y0, (z0 + fx * l) * zscale],
            color32(theme::ACCENT),
            3.2,
        );
        scene.add_arrow(
            &p0,
            &[x0, y0 + l, (z0 + fy * l) * zscale],
            color32(theme::ACCENT2),
            3.2,
        );
        scene.add_dot(&p0, color32(theme::POINT), 5.0);
    });
    rows
}

fn mix_color(a: theme::Rgb, b: theme::Rgb, t: f32, alpha: u8) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgba_unmultiplied(m(a[0], b[0]), m(a[1], b[1]), m(a[2], b[2]), alpha)
}

// ------------------------------------------------------------------ contour

pub fn contour(ui: &mut Ui, spec: &SceneSpec, state: &mut FuncState) -> Vec<Row> {
    let SceneSpec::Contour { f, domain, .. } = spec else {
        return Vec::new();
    };
    let (a, b) = *domain;
    slider_row(ui, state, (a, b));
    let (x0, y0) = (state.x0, state.y0);
    let rows = live_rows(f, x0, y0);

    // Rasterize f once into a background texture.
    if state.texture.is_none() {
        const RES: usize = 240;
        let mut vals = vec![0.0f64; RES * RES];
        let mut vmax: f64 = 1e-9;
        for j in 0..RES {
            for i in 0..RES {
                let x = a + (b - a) * i as f64 / (RES - 1) as f64;
                // Image rows go top-down; flip y so up is +y.
                let y = b - (b - a) * j as f64 / (RES - 1) as f64;
                let v = f.eval(&[x, y]);
                vals[j * RES + i] = v;
                vmax = vmax.max(v.abs());
            }
        }
        let mut img = ColorImage::new([RES, RES], vec![Color32::BLACK; RES * RES]);
        for (k, v) in vals.iter().enumerate() {
            let t = ((v / vmax) as f32 + 1.0) / 2.0;
            // Banded shading fakes contour lines: quantize, then darken edges.
            let bands = 12.0;
            let q = (t * bands).fract();
            let edge = if q < 0.08 { 0.55 } else { 1.0 };
            let base = mix_color(theme::ACCENT2, theme::ACCENT, t, 255);
            img.pixels[k] = Color32::from_rgb(
                (base.r() as f32 * edge) as u8,
                (base.g() as f32 * edge) as u8,
                (base.b() as f32 * edge) as u8,
            );
        }
        state.texture = Some(ui.ctx().load_texture(
            "contour",
            img,
            TextureOptions::LINEAR,
        ));
    }
    let texture = state.texture.as_ref().unwrap();

    let fx = f.partial(&[x0, y0], 0, H);
    let fy = f.partial(&[x0, y0], 1, H);
    let glen = (fx * fx + fy * fy).sqrt().max(1e-9);
    let l = (b - a) * 0.12;

    Plot::new("contour")
        .data_aspect(1.0)
        .include_x(a)
        .include_x(b)
        .include_y(a)
        .include_y(b)
        .show_grid(false)
        .show(ui, |plot| {
            plot.image(PlotImage::new(
                "f",
                texture.id(),
                [0.5 * (a + b), 0.5 * (a + b)].into(),
                [(b - a) as f32, (b - a) as f32],
            ));
            // Gradient arrow (steepest ascent) and its axis components.
            let tip = [x0 + fx / glen * l, y0 + fy / glen * l];
            plot.line(
                Line::new("grad", PlotPoints::from(vec![[x0, y0], tip]))
                    .color(color32(theme::POINT))
                    .width(3.0),
            );
            plot.line(
                Line::new(
                    "dfdx",
                    PlotPoints::from(vec![[x0, y0], [x0 + fx / glen * l, y0]]),
                )
                .color(color32(theme::ACCENT))
                .width(2.0),
            );
            plot.line(
                Line::new(
                    "dfdy",
                    PlotPoints::from(vec![[x0, y0], [x0, y0 + fy / glen * l]]),
                )
                .color(color32(theme::ACCENT2))
                .width(2.0),
            );
            plot.points(
                Points::new("p", PlotPoints::from(vec![[x0, y0]]))
                    .radius(6.0)
                    .color(color32(theme::POINT)),
            );
        });
    rows
}
