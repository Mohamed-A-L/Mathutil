//! mathutil-rs — an interactive linear-algebra & calculus teaching aid.
//!
//! - no arguments: the ratatui REPL (`mathutil>` prompt)
//! - `mathutil-rs <command> <args…>`: one-shot — print the report, open the
//!   visualization window, exit when it closes
//! - `mathutil-rs viz <spec.json>`: internal — render a scene spec (spawned
//!   by the REPL so windows outlive individual commands)

pub mod core;
pub mod registry;
pub mod scene;
pub mod theme;
pub mod topics;
pub mod tui;
pub mod viz;
pub mod viz_spawn;

use scene::Row;

fn main() -> std::process::ExitCode {
    // Die quietly on a closed pipe (`mathutil help | head`) like a normal
    // CLI instead of panicking — Rust ignores SIGPIPE by default.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    // Internal: render one scene spec file in a window.
    if args.len() == 2 && args[0] == "viz" {
        return match viz::run(std::path::Path::new(&args[1])) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("viz error: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }

    // No arguments: the REPL.
    if args.is_empty() {
        return match tui::run() {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("terminal error: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }

    // One-shot: `mathutil-rs transform [[1,2],[0,1]]`
    let line = args.join(" ");
    if line == "help" || line == "--help" || line == "-h" {
        for row in registry::help_text() {
            println!("{}", row.text);
        }
        return std::process::ExitCode::SUCCESS;
    }
    if let Some(name) = line.strip_prefix("help ") {
        return match registry::command_help(name.trim()) {
            Ok(rows) => {
                for row in rows {
                    println!("{}", row.text);
                }
                std::process::ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    match registry::run_command(&line) {
        Ok(Some(outcome)) => {
            print_report(&outcome.report);
            if let Some(spec) = outcome.scene {
                // One-shot runs the window on the main thread and blocks
                // until the user closes it (like `mathutil transform …`).
                if let Err(e) = viz::run_spec(spec, line.clone()) {
                    eprintln!("could not open window: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            }
            std::process::ExitCode::SUCCESS
        }
        Ok(None) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn print_report(report: &scene::Report) {
    println!("\x1b[1;38;2;78;161;255m{}\x1b[0m", report.title);
    for f in &report.formulas {
        println!("  \x1b[38;2;255;209;102m{f}\x1b[0m");
    }
    for row in &report.body {
        println!("  {}", ansi_row(row));
    }
}

fn ansi_row(row: &Row) -> String {
    let mut prefix = String::new();
    if row.bold {
        prefix.push_str("\x1b[1m");
    }
    if let Some([r, g, b]) = row.color {
        prefix.push_str(&format!("\x1b[38;2;{r};{g};{b}m"));
    }
    if prefix.is_empty() {
        row.text.clone()
    } else {
        format!("{prefix}{}\x1b[0m", row.text)
    }
}
