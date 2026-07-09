//! Spawn a visualization window as a detached child process.
//!
//! The scene spec is serialized to a temp file and `mathutil-rs viz <file>`
//! renders it (deleting the file once loaded). Running windows in their own
//! processes keeps the TUI responsive and sidesteps winit's requirement to
//! own the main thread.

use std::io::Write;
use std::process::{Child, Command, Stdio};

use crate::scene::{ScenePackage, SceneSpec};

pub fn spawn(scene: &SceneSpec, command: &str) -> Result<Child, String> {
    let package = ScenePackage {
        command: command.to_string(),
        spec: scene.clone(),
    };
    let json = serde_json::to_vec(&package).map_err(|e| e.to_string())?;
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "mathutil-rs-scene-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    file.write_all(&json).map_err(|e| e.to_string())?;
    drop(file);

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    Command::new(exe)
        .arg("viz")
        .arg(&path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null()) // GPU/window-system chatter must not hit the TUI
        .spawn()
        .map_err(|e| e.to_string())
}
