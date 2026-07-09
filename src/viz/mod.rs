//! The visualization window: an eframe/egui app rendering one [`SceneSpec`].
//!
//! Layout mirrors the Python original: the canvas fills the window and a
//! fixed info panel (title / formulas / live numbers with a color key) sits
//! on the right. Animated scenes get play/pause, replay, and a seek slider;
//! 3D scenes orbit with a mouse drag and zoom with the scroll wheel.

mod func;
mod plot2d;
mod three;

use eframe::egui;
use egui::{Color32, RichText};

use crate::registry;
use crate::scene::{Report, Row, ScenePackage, SceneSpec, SpaceMode};
use crate::theme;

pub fn color32(c: theme::Rgb) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

pub fn color32_a(c: theme::Rgb, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a)
}

/// Ease a linear 0..1 into smooth start/stop motion (smoothstep).
pub fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Animation lengths (seconds). The renderers in plot2d.rs / three.rs and
// [`duration`] below must agree, so the constants live here.
pub(crate) const STAGE_SECS: f64 = 1.6; // one warp stage / dependence collapse
pub(crate) const LINCOMB_SECS: f64 = STAGE_SECS + 0.4;
pub(crate) const KERNEL_SECS: f64 = STAGE_SECS + 0.2;
pub(crate) const SWEEP2D_SECS: f64 = 5.0;
pub(crate) const SWEEP3D_SECS: f64 = 6.0;
pub(crate) const REVOLVE_SECS: f64 = 4.0;
pub(crate) const RIEMANN1_SECS: f64 = 4.0;

/// Total animation length for a scene; `None` means the scene is static
/// (or purely input-driven) and gets no play/pause/seek controls.
fn duration(spec: &SceneSpec) -> Option<f64> {
    match spec {
        SceneSpec::Warp { stages, .. } => Some(STAGE_SECS * stages.len() as f64),
        SceneSpec::Space { mode, .. } => match mode {
            SpaceMode::Independent {
                collapse_index: Some(_),
            } => Some(STAGE_SECS),
            SpaceMode::Lincomb { .. } => Some(LINCOMB_SECS),
            _ => None,
        },
        SceneSpec::KernelImage { .. } => Some(KERNEL_SECS),
        SceneSpec::Sweep { dim, .. } => Some(if *dim == 2 { SWEEP2D_SECS } else { SWEEP3D_SECS }),
        SceneSpec::Riemann1 { .. } => Some(RIEMANN1_SECS),
        SceneSpec::Revolution { .. } => Some(REVOLVE_SECS),
        SceneSpec::RankNullity { .. } | SceneSpec::Surface { .. } | SceneSpec::Contour { .. } => {
            None
        }
    }
}

/// Run the window for a scene file written by `viz_spawn`. Deletes the file
/// once loaded so aborted runs don't litter the temp dir.
pub fn run(path: &std::path::Path) -> Result<(), String> {
    let json = std::fs::read(path).map_err(|e| format!("could not read {path:?}: {e}"))?;
    let _ = std::fs::remove_file(path);
    let package: ScenePackage =
        serde_json::from_slice(&json).map_err(|e| format!("bad scene spec: {e}"))?;
    run_spec(package.spec, package.command)
}

pub fn run_spec(spec: SceneSpec, command: String) -> Result<(), String> {
    let title = spec.window_title();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_title(&title),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| Ok(Box::new(VizApp::new(cc, spec, command)))),
    )
    .map_err(|e| e.to_string())
}

struct VizApp {
    spec: SceneSpec,
    /// Animation clock in seconds — owned state (not wall time) so it can
    /// be paused and scrubbed with the seek slider.
    t: f64,
    playing: bool,
    last_frame: std::time::Instant,
    three: three::Camera,
    func: func::FuncState,
    /// Extra live rows the renderer wants shown under the report body.
    live_rows: Vec<Row>,
    /// The editable command line; Enter re-runs it and swaps the scene.
    cmd_edit: String,
    cmd_error: Option<String>,
}

impl VizApp {
    fn new(cc: &eframe::CreationContext<'_>, spec: SceneSpec, command: String) -> VizApp {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        // The 2D plots zoom with ctrl/cmd + scroll (or trackpad pinch);
        // egui's default speed is sluggish, so zoom 2.5x per scroll unit.
        cc.egui_ctx
            .options_mut(|o| o.input_options.scroll_zoom_speed = 2.5 / 200.0);
        cc.egui_ctx.all_styles_mut(|style| {
            style.visuals.panel_fill = color32(theme::BG);
            style.visuals.window_fill = color32(theme::BG);
            style.visuals.extreme_bg_color = color32(theme::BG);
        });
        let func = func::FuncState::new(&spec);
        VizApp {
            spec,
            t: 0.0,
            playing: true,
            last_frame: std::time::Instant::now(),
            three: three::Camera::default(),
            func,
            live_rows: Vec::new(),
            cmd_edit: command,
            cmd_error: None,
        }
    }

    /// Re-run the edited command and swap in the new scene (same window,
    /// camera kept so before/after comparisons stay aligned).
    fn rebuild(&mut self, ctx: &egui::Context) {
        match registry::run_command(self.cmd_edit.trim()) {
            Ok(Some(outcome)) => match outcome.scene {
                Some(spec) => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(spec.window_title()));
                    self.func = func::FuncState::new(&spec);
                    self.spec = spec;
                    self.t = 0.0;
                    self.playing = true;
                    self.live_rows.clear();
                    self.cmd_error = None;
                }
                None => self.cmd_error = Some("that command has no visualization".into()),
            },
            Ok(None) => {}
            Err(e) => self.cmd_error = Some(e),
        }
    }
}

impl eframe::App for VizApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Advance the clock while playing; clamp dt so resuming after a long
        // idle stretch (no repaints scheduled) doesn't jump the animation.
        let duration = duration(&self.spec);
        let dt = self.last_frame.elapsed().as_secs_f64().min(0.1);
        self.last_frame = std::time::Instant::now();
        if let Some(dur) = duration {
            if self.playing && self.t < dur {
                self.t = (self.t + dt).min(dur);
                root.ctx().request_repaint(); // drive the tween
            }
        }

        let report = self.spec.report().clone();
        let live = std::mem::take(&mut self.live_rows);
        let hint = hint(&self.spec);
        let submitted = egui::Panel::right("info")
            .exact_size(360.0)
            .show(root, |ui| {
                info_panel(
                    ui,
                    &report,
                    &live,
                    duration,
                    &mut self.t,
                    &mut self.playing,
                    &mut self.cmd_edit,
                    self.cmd_error.as_deref(),
                    hint,
                )
            })
            .inner;
        if submitted {
            self.rebuild(root.ctx());
        }

        let t = self.t;
        egui::CentralPanel::default().show(root, |ui| {
            self.live_rows = match &self.spec {
                SceneSpec::Warp { .. } => plot2d_or_three(self, ui, t),
                SceneSpec::Space { .. } => plot2d_or_three(self, ui, t),
                SceneSpec::KernelImage { .. } => plot2d_or_three(self, ui, t),
                SceneSpec::RankNullity { .. } => plot2d::rank_nullity(ui, &self.spec),
                SceneSpec::Sweep { dim, .. } => {
                    if *dim == 2 {
                        plot2d::sweep2d(ui, &self.spec, t)
                    } else {
                        three::scene3d(ui, &self.spec, &mut self.three, t)
                    }
                }
                SceneSpec::Riemann1 { .. } => plot2d::riemann1(ui, &self.spec, t),
                SceneSpec::Revolution { .. } => {
                    three::scene3d(ui, &self.spec, &mut self.three, t)
                }
                SceneSpec::Surface { .. } => {
                    func::surface(ui, &self.spec, &mut self.func, &mut self.three)
                }
                SceneSpec::Contour { .. } => func::contour(ui, &self.spec, &mut self.func),
            };
        });
    }
}

fn plot2d_or_three(app: &mut VizApp, ui: &mut egui::Ui, t: f64) -> Vec<Row> {
    let dim = match &app.spec {
        SceneSpec::Warp { dim, .. }
        | SceneSpec::Space { dim, .. }
        | SceneSpec::KernelImage { dim_in: dim, .. } => *dim,
        _ => 2,
    };
    if dim == 2 {
        plot2d::scene2d(ui, &app.spec, t)
    } else {
        three::scene3d(ui, &app.spec, &mut app.three, t)
    }
}

/// Interaction hint matching how the scene actually responds to the mouse:
/// the software-projected 3D scenes orbit/zoom; the egui_plot 2D scenes pan
/// with drag/scroll and zoom with ctrl+scroll.
fn hint(spec: &SceneSpec) -> &'static str {
    let three_d = match spec {
        SceneSpec::Warp { dim, .. } | SceneSpec::Space { dim, .. } => *dim == 3,
        SceneSpec::KernelImage { dim_in, .. } => *dim_in == 3,
        SceneSpec::Sweep { dim, .. } => *dim == 3,
        SceneSpec::Revolution { .. } | SceneSpec::Surface { .. } => true,
        SceneSpec::RankNullity { .. }
        | SceneSpec::Riemann1 { .. }
        | SceneSpec::Contour { .. } => false,
    };
    if three_d {
        "drag = orbit · scroll = zoom"
    } else {
        "drag/scroll = pan · ctrl+scroll = zoom · double-click = reset"
    }
}

/// Returns true when the command box was submitted (Enter).
#[allow(clippy::too_many_arguments)]
fn info_panel(
    ui: &mut egui::Ui,
    report: &Report,
    live: &[Row],
    duration: Option<f64>,
    t: &mut f64,
    playing: &mut bool,
    cmd: &mut String,
    cmd_error: Option<&str>,
    hint: &'static str,
) -> bool {
    ui.add_space(10.0);
    ui.label(
        RichText::new(&report.title)
            .size(19.0)
            .strong()
            .color(color32(theme::FG)),
    );
    ui.add_space(6.0);
    for f in &report.formulas {
        ui.label(
            RichText::new(f)
                .size(16.0)
                .color(color32(theme::EIGEN))
                .monospace(),
        );
    }
    ui.add_space(8.0);
    ui.separator();
    ui.add_space(8.0);
    for row in report.body.iter().chain(live) {
        let mut text = RichText::new(&row.text)
            .size(14.0)
            .monospace()
            .color(row.color.map(color32).unwrap_or(color32(theme::FG)));
        if row.bold {
            text = text.strong();
        }
        ui.label(text);
    }
    if let Some(dur) = duration {
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            let done = *t >= dur;
            let label = if *playing && !done { "  Pause  " } else { "  Play  " };
            if ui.button(RichText::new(label).size(15.0)).clicked() {
                if done {
                    *t = 0.0; // play again from the start
                    *playing = true;
                } else {
                    *playing = !*playing;
                }
            }
            if ui.button(RichText::new("  Replay  ").size(15.0)).clicked() {
                *t = 0.0;
                *playing = true;
            }
        });
        ui.add_space(8.0);
        // Seek: dragging scrubs the animation (works while paused too).
        // The value label is drawn separately: Slider's built-in value box
        // rounds to its display precision and writes that back every frame,
        // which would zero a fresh clock.
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = ui.available_width() - 64.0;
            ui.add(egui::Slider::new(t, 0.0..=dur).show_value(false));
            ui.label(
                RichText::new(format!("{:>4.1} s", *t))
                    .monospace()
                    .color(color32(theme::MUTED)),
            );
        });
    }
    ui.add_space(10.0);
    // Bottom-up: hint line at the very bottom, the command box above it.
    let mut submitted = false;
    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        ui.add_space(8.0);
        ui.label(
            RichText::new(hint)
                .size(12.0)
                .color(color32(theme::MUTED)),
        );
        ui.add_space(10.0);
        let resp = ui.add(
            egui::TextEdit::singleline(cmd)
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        submitted = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if let Some(err) = cmd_error {
            ui.label(RichText::new(err).size(12.0).color(color32(theme::BAD)));
        }
        ui.label(
            RichText::new("command — edit & press Enter to re-run:")
                .size(12.0)
                .color(color32(theme::MUTED)),
        );
    });
    submitted
}
