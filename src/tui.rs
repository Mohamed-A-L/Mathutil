//! The terminal REPL: command input with history and tab completion, a
//! scrollback output pane for result tables, and a status bar.
//!
//! Visualization windows are spawned as detached `mathutil-rs viz <file>`
//! child processes, so they keep animating while the REPL stays responsive.

use std::collections::VecDeque;
use std::process::{Child, Command, Stdio};
use std::io::Write;
use std::time::Duration;

use crossterm::event::{
    self, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::registry;
use crate::scene::{Report, Row};
use crate::theme;
use crate::viz_spawn;

fn rgb(c: theme::Rgb) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

enum SelectionArea {
    Output { start_row: usize, start_col: u16, end_row: usize, end_col: u16 },
    Input { start_col: u16, end_col: u16 },
}

enum SelectionStart {
    Input(u16),
    Output(usize, u16), // (row, col)
}

struct App {
    input: String,
    cursor: usize, // byte offset into input (ASCII commands, so char == byte)
    history: Vec<String>,
    history_pos: Option<usize>,
    stash: String, // input saved while browsing history
    lines: Vec<Line<'static>>,
    scroll_up: u16, // lines scrolled up from the bottom
    status: String,
    completions: Option<(Vec<String>, usize, usize)>, // (matches, next, word_start)
    children: VecDeque<Child>,
    quit: bool,
    current_path: Option<String>, // None = root, Some(topic) = in a topic
    selection: Option<SelectionArea>, // text selection for copying
    selection_start: Option<SelectionStart>, // where selection started (input col or output row/col)
    output_area: ratatui::layout::Rect, // cached output area for mouse handling
    input_area: ratatui::layout::Rect, // cached input area for mouse handling
    input_hscroll: u16, // horizontal scroll of input line
}

impl App {
    fn new() -> App {
        let mut app = App {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_pos: None,
            stash: String::new(),
            lines: Vec::new(),
            scroll_up: 0,
            status: String::from("'ls' lists topics, 'cd <topic>' enters it, 'help <cmd>' shows usage, 'quit' exits"),
            completions: None,
            children: VecDeque::new(),
            quit: false,
            current_path: None,
            selection: None,
            selection_start: None,
            output_area: ratatui::layout::Rect::default(),
            input_area: ratatui::layout::Rect::default(),
            input_hscroll: 0,
        };
        app.push_line(Line::from(vec![Span::styled(
            "mathutil",
            Style::new().fg(rgb(theme::ACCENT)).bold(),
        )]));
        app.push_row(&Row::colored(
            "linear-algebra & calculus visualizer (Rust)",
            theme::MUTED,
        ));
        app.push_row(&Row::plain(""));
        app
    }

    fn push_line(&mut self, line: Line<'static>) {
        self.lines.push(line);
        self.scroll_up = 0; // new output snaps the view to the bottom
    }

    fn push_row(&mut self, row: &Row) {
        let mut style = Style::new();
        if let Some(c) = row.color {
            style = style.fg(rgb(c));
        } else {
            style = style.fg(rgb(theme::FG));
        }
        if row.bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        self.push_line(Line::from(Span::styled(row.text.clone(), style)));
    }

    fn push_report(&mut self, report: &Report) {
        self.push_line(Line::from(Span::styled(
            report.title.clone(),
            Style::new().fg(rgb(theme::ACCENT)).bold(),
        )));
        for f in &report.formulas {
            self.push_line(Line::from(Span::styled(
                format!("  {f}"),
                Style::new().fg(rgb(theme::EIGEN)),
            )));
        }
        for row in report.body.clone() {
            let indented = Row {
                text: format!("  {}", row.text),
                ..row
            };
            self.push_row(&indented);
        }
        self.push_row(&Row::plain(""));
    }

    fn run_line(&mut self) {
        let text = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.completions = None;
        self.history_pos = None;
        if text.is_empty() {
            return;
        }
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }

        // echo the command
        self.push_line(Line::from(vec![
            Span::styled("mathutil> ", Style::new().fg(rgb(theme::ACCENT3)).bold()),
            Span::styled(text.clone(), Style::new().fg(rgb(theme::FG))),
        ]));

        match text.as_str() {
            "quit" | "exit" => {
                self.quit = true;
                return;
            }
            "ls" => {
                self.show_ls();
                return;
            }
            "clear" => {
                self.lines.clear();
                return;
            }
            _ => {}
        }
        if let Some(path) = text.strip_prefix("cd ") {
            self.handle_cd(path.trim());
            return;
        }
        if let Some(name) = text.strip_prefix("help ") {
            match registry::command_help(name.trim()) {
                Ok(rows) => {
                    for row in rows {
                        self.push_row(&row);
                    }
                    self.push_row(&Row::plain(""));
                }
                Err(e) => self.push_error(&e),
            }
            return;
        }

        match registry::run_command(&text) {
            Ok(Some(outcome)) => {
                self.push_report(&outcome.report);
                if let Some(scene) = outcome.scene {
                    match viz_spawn::spawn(&scene, &text) {
                        Ok(child) => {
                            self.children.push_back(child);
                            self.status = format!("window opened — {}", outcome.report.title);
                        }
                        Err(e) => self.push_error(&format!("could not open window: {e}")),
                    }
                }
            }
            Ok(None) => {}
            Err(e) => self.push_error(&e),
        }
    }

    fn push_error(&mut self, msg: &str) {
        self.push_row(&Row::bold(format!("error: {msg}"), theme::BAD));
        self.push_row(&Row::plain(""));
        self.status = String::from("error — see above");
    }

    fn show_ls(&mut self) {
        match &self.current_path {
            None => {
                self.push_row(&Row::plain("Topics  (cd <topic> to enter):"));
                self.push_row(&Row::plain(""));
                for t in registry::topics() {
                    self.push_row(&Row::plain(format!(
                        "  {}{}  {}  ({} commands)",
                        t.name,
                        " ".repeat((10usize).saturating_sub(t.name.len())),
                        t.title,
                        t.commands.len()
                    )));
                }
            }
            Some(topic) => {
                if let Some(t) = registry::topics().iter().find(|t| t.name == topic) {
                    self.push_row(&Row::plain(format!("{} — {}", t.name, t.title)));
                    self.push_row(&Row::plain(""));
                    for cname in t.commands {
                        if let Some(cmd) = registry::commands().iter().find(|c| c.name == *cname) {
                            self.push_row(&Row::plain(format!(
                                "  {}{}  {}",
                                cmd.name,
                                " ".repeat((15usize).saturating_sub(cmd.name.len())),
                                cmd.summary
                            )));
                        }
                    }
                    self.push_row(&Row::plain(""));
                    self.push_row(&Row::plain("  type 'help <command>' for usage and an example"));
                } else {
                    self.push_error(&format!("unknown topic '{topic}'"));
                    return;
                }
            }
        }
        self.push_row(&Row::plain(""));
    }

    fn handle_cd(&mut self, path: &str) {
        if path == "/" {
            self.current_path = None;
            self.status = String::from("/ (root)");
            return;
        }
        if path == ".." {
            self.current_path = None;
            self.status = String::from("/ (root)");
            return;
        }
        if let Some(_t) = registry::topics().iter().find(|t| t.name == path) {
            self.current_path = Some(path.to_string());
            self.status = format!("/{}", path);
        } else {
            self.push_error(&format!("unknown topic '{path}'"));
        }
    }

    fn copy_selection(&mut self) {
        match &self.selection {
            Some(sel) => {
                let text = self.extract_selected_text(sel);
                if text.is_empty() {
                    self.status = String::from("nothing selected to copy");
                } else if let Err(e) = self.copy_to_clipboard(&text) {
                    self.status = format!("copy failed: {e}");
                } else {
                    self.status = String::from("copied to clipboard");
                }
            }
            None => {
                self.status = String::from("no selection — drag to select text");
            }
        }
    }

    fn paste_from_clipboard(&mut self) {
        match self.paste_from_clipboard_impl() {
            Ok(text) => {
                self.input.insert_str(self.cursor, &text);
                self.cursor += text.len();
                self.status = String::from("pasted from clipboard");
                self.completions = None;
            }
            Err(e) => {
                self.status = format!("paste failed: {e}");
            }
        }
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<(), String> {
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();

        if session_type == "wayland" {
            if let Ok(mut child) = Command::new("wl-copy")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                if let Some(mut stdin) = child.stdin.take() {
                    if let Ok(_) = stdin.write_all(text.as_bytes()) {
                        let _ = stdin.flush();
                        drop(stdin);
                        let _ = child.wait();
                        return Ok(());
                    }
                }
            }
        }

        // Fallback to arboard for X11 or if wl-copy fails
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => clipboard.set_text(text.to_string())
                .map_err(|e| format!("{e}")),
            Err(e) => Err(format!("{e}")),
        }
    }

    fn paste_from_clipboard_impl(&self) -> Result<String, String> {
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();

        if session_type == "wayland" {
            if let Ok(output) = Command::new("wl-paste")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
            {
                if output.status.success() {
                    if let Ok(text) = String::from_utf8(output.stdout) {
                        return Ok(text.trim_end().to_string());
                    }
                }
            }
        }

        // Fallback to arboard for X11 or if wl-paste fails
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => clipboard.get_text()
                .map_err(|e| format!("{e}")),
            Err(e) => Err(format!("{e}")),
        }
    }

    fn extract_selected_text(&self, sel: &SelectionArea) -> String {
        match sel {
            SelectionArea::Input { start_col, end_col } => {
                let prompt_len = match &self.current_path {
                    None => 4, // "/ $ "
                    Some(t) => 4 + t.len(), // "/<topic> $ "
                };
                let start_col = *start_col as usize;
                let end_col = *end_col as usize;
                let input_start = start_col.saturating_sub(prompt_len);
                let input_end = end_col.saturating_sub(prompt_len);

                let s = input_start.min(input_end);
                let e = input_start.max(input_end).min(self.input.len());

                if s < e {
                    self.input[s..e].to_string()
                } else {
                    String::new()
                }
            }
            SelectionArea::Output { start_row, start_col, end_row, end_col } => {
                let sr = *start_row.min(end_row);
                let er = *start_row.max(end_row);
                let sc = if start_row <= end_row { *start_col } else { *end_col };
                let ec = if start_row <= end_row { *end_col } else { *start_col };

                let mut result = String::new();
                for row in sr..=er {
                    if row >= self.lines.len() {
                        break;
                    }
                    let line = &self.lines[row];
                    let line_text: String = line.iter().map(|s| s.content.to_string()).collect();

                    if sr == er {
                        let col_start = sc.min(line_text.len() as u16) as usize;
                        let col_end = ec.min(line_text.len() as u16) as usize;
                        result.push_str(&line_text[col_start..col_end]);
                    } else if row == sr {
                        let col_start = sc.min(line_text.len() as u16) as usize;
                        result.push_str(&line_text[col_start..]);
                        result.push('\n');
                    } else if row == er {
                        let col_end = ec.min(line_text.len() as u16) as usize;
                        result.push_str(&line_text[..col_end]);
                    } else {
                        result.push_str(&line_text);
                        result.push('\n');
                    }
                }
                result
            }
        }
    }

    fn is_in_selection(&self, row: usize, col: u16) -> bool {
        if let Some(SelectionArea::Output { start_row, start_col, end_row, end_col }) = &self.selection {
            let sr = *start_row.min(end_row);
            let er = *start_row.max(end_row);
            let (sc, ec) = if start_row <= end_row {
                (*start_col, *end_col)
            } else {
                (*end_col, *start_col)
            };

            if sr == er {
                row == sr && col >= sc && col < ec
            } else if row == sr {
                col >= sc
            } else if row == er {
                col < ec
            } else {
                row > sr && row < er
            }
        } else {
            false
        }
    }

    fn is_input_char_selected(&self, col: u16) -> bool {
        if let Some(SelectionArea::Input { start_col, end_col }) = &self.selection {
            let s = (*start_col).min(*end_col);
            let e = (*start_col).max(*end_col);
            col >= s && col < e
        } else {
            false
        }
    }

    fn apply_selection_style(base_style: Style, is_selected: bool) -> Style {
        if is_selected {
            base_style.bg(Color::Rgb(100, 100, 100))
        } else {
            base_style
        }
    }

    fn build_display_lines(&self, offset: u16, inner_height: usize) -> Vec<Line<'static>> {
        let start_row = offset as usize;
        let end_row = (start_row + inner_height).min(self.lines.len());
        let mut display_lines = Vec::new();

        for line_idx in start_row..end_row {
            let original_line = &self.lines[line_idx];
            let mut new_spans = Vec::new();
            let mut col = 0u16;

            for span in original_line.iter() {
                let span_text = &span.content;
                let span_len = span_text.len() as u16;
                let mut new_content = String::new();
                let mut in_selection = false;
                let current_style = span.style;

                for (char_idx, ch) in span_text.chars().enumerate() {
                    let char_col = col + char_idx as u16;
                    let is_selected = self.is_in_selection(line_idx, char_col);

                    if is_selected != in_selection {
                        if !new_content.is_empty() {
                            let style = Self::apply_selection_style(current_style, in_selection);
                            new_spans.push(Span::styled(new_content.clone(), style));
                            new_content.clear();
                        }
                        in_selection = is_selected;
                    }
                    new_content.push(ch);
                }

                if !new_content.is_empty() {
                    let style = Self::apply_selection_style(current_style, in_selection);
                    new_spans.push(Span::styled(new_content, style));
                }

                col += span_len;
            }

            display_lines.push(Line::from(new_spans));
        }

        display_lines
    }

    fn build_input_spans(&self, prompt: &str, _hscroll: usize) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let mut col = 0u16;

        // Prompt span
        let prompt_len = prompt.len();
        let prompt_style = Style::new().fg(rgb(theme::ACCENT3)).bold();
        spans.push(Span::styled(prompt.to_string(), prompt_style));
        col += prompt_len as u16;

        // Input spans with selection highlighting
        let mut current_content = String::new();
        let mut in_selection = false;
        let input_style = Style::new().fg(rgb(theme::FG));

        for (idx, ch) in self.input.chars().enumerate() {
            let char_col = col + idx as u16;
            let is_selected = self.is_input_char_selected(char_col);

            if is_selected != in_selection {
                if !current_content.is_empty() {
                    let style = Self::apply_selection_style(input_style, in_selection);
                    spans.push(Span::styled(current_content.clone(), style));
                    current_content.clear();
                }
                in_selection = is_selected;
            }
            current_content.push(ch);
        }

        if !current_content.is_empty() {
            let style = Self::apply_selection_style(input_style, in_selection);
            spans.push(Span::styled(current_content, style));
        }

        spans
    }

    fn on_mouse(&mut self, m: event::MouseEvent) {
        match m.kind {
            event::MouseEventKind::Down(event::MouseButton::Left) => {
                if m.row >= self.input_area.top() && m.row < self.input_area.bottom()
                    && m.column >= self.input_area.left() && m.column < self.input_area.right()
                {
                    let col = m.column.saturating_sub(self.input_area.left() + 1) + self.input_hscroll;
                    self.selection_start = Some(SelectionStart::Input(col));
                    self.selection = None;
                } else if m.row >= self.output_area.top() && m.row < self.output_area.bottom()
                    && m.column >= self.output_area.left() && m.column < self.output_area.right()
                {
                    let inner_row = (m.row - self.output_area.top() - 1) as usize;
                    let inner_col = m.column.saturating_sub(self.output_area.left() + 1);
                    let inner_height = self.output_area.height.saturating_sub(2) as usize;
                    let total = self.lines.len();
                    let bottom_offset = total.saturating_sub(inner_height);
                    let offset = (bottom_offset as u16).saturating_sub(self.scroll_up);
                    let actual_row = offset as usize + inner_row;
                    self.selection_start = Some(SelectionStart::Output(actual_row, inner_col));
                    self.selection = None;
                }
            }
            event::MouseEventKind::Drag(event::MouseButton::Left) => {
                if let Some(start) = &self.selection_start {
                    match start {
                        SelectionStart::Input(start_col) => {
                            if m.row >= self.input_area.top() && m.row < self.input_area.bottom()
                                && m.column >= self.input_area.left() && m.column < self.input_area.right()
                            {
                                let col = m.column.saturating_sub(self.input_area.left() + 1) + self.input_hscroll;
                                self.selection = Some(SelectionArea::Input {
                                    start_col: *start_col,
                                    end_col: col,
                                });
                            }
                        }
                        SelectionStart::Output(start_row, start_col) => {
                            if m.row >= self.output_area.top() && m.row < self.output_area.bottom()
                                && m.column >= self.output_area.left() && m.column < self.output_area.right()
                            {
                                let inner_row = (m.row - self.output_area.top() - 1) as usize;
                                let inner_col = m.column.saturating_sub(self.output_area.left() + 1);
                                let inner_height = self.output_area.height.saturating_sub(2) as usize;
                                let total = self.lines.len();
                                let bottom_offset = total.saturating_sub(inner_height);
                                let offset = (bottom_offset as u16).saturating_sub(self.scroll_up);
                                let actual_row = offset as usize + inner_row;
                                self.selection = Some(SelectionArea::Output {
                                    start_row: *start_row,
                                    start_col: *start_col,
                                    end_row: actual_row,
                                    end_col: inner_col,
                                });
                            }
                        }
                    }
                }
            }
            event::MouseEventKind::Up(event::MouseButton::Left) => {
                self.selection_start = None;
            }
            event::MouseEventKind::ScrollUp => {
                self.scroll_up = self.scroll_up.saturating_add(3);
            }
            event::MouseEventKind::ScrollDown => {
                self.scroll_up = self.scroll_up.saturating_sub(3);
            }
            _ => {}
        }
    }

    /// Cycle tab-completion for the word before the cursor.
    fn complete(&mut self) {
        if let Some((matches, next, word_start)) = self.completions.take() {
            // Cycle to the next candidate.
            let m = &matches[next % matches.len()];
            self.input.replace_range(word_start.., m);
            self.cursor = self.input.len();
            self.completions = Some((matches.clone(), next + 1, word_start));
            return;
        }
        let upto = &self.input[..self.cursor];
        let word_start = upto.rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &upto[word_start..];
        let matches = registry::completion_matches(upto, word);
        match matches.len() {
            0 => {}
            1 => {
                self.input.replace_range(word_start.., &matches[0]);
                self.input.push(' ');
                self.cursor = self.input.len();
            }
            _ => {
                self.input.replace_range(word_start.., &matches[0]);
                self.cursor = self.input.len();
                self.status = matches.join("  ");
                self.completions = Some((matches, 1, word_start));
            }
        }
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let pos = match self.history_pos {
            None => {
                self.stash = self.input.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.history_pos = Some(pos);
        self.input = self.history[pos].clone();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        match self.history_pos {
            None => {}
            Some(p) if p + 1 < self.history.len() => {
                self.history_pos = Some(p + 1);
                self.input = self.history[p + 1].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.history_pos = None;
                self.input = self.stash.clone();
                self.cursor = self.input.len();
            }
        }
    }

    fn reap_children(&mut self) {
        self.children
            .retain_mut(|c| !matches!(c.try_wait(), Ok(Some(_))));
    }

    fn on_key(&mut self, key: event::KeyEvent) {
        if key.kind != KeyEventKind::Press && key.kind != KeyEventKind::Repeat {
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            KeyCode::Char('c') if ctrl && !shift => self.quit = true,
            KeyCode::Char('c') if ctrl && shift => self.copy_selection(),
            KeyCode::Char('v') if ctrl && shift => self.paste_from_clipboard(),
            KeyCode::Char('l') if ctrl => self.lines.clear(),
            KeyCode::Char('u') if ctrl => {
                self.input.clear();
                self.cursor = 0;
                self.completions = None;
            }
            KeyCode::Char(c) if !ctrl => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                self.completions = None;
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let prev = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.input.replace_range(prev..self.cursor, "");
                    self.cursor = prev;
                }
                self.completions = None;
            }
            KeyCode::Enter => self.run_line(),
            KeyCode::Tab => self.complete(),
            KeyCode::Up => self.history_up(),
            KeyCode::Down => self.history_down(),
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor += self.input[self.cursor..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(0);
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::PageUp => self.scroll_up = self.scroll_up.saturating_add(10),
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(10),
            KeyCode::Esc => {
                self.completions = None;
            }
            _ => {}
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let [output_area, input_area, status_area] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        self.output_area = output_area;
        self.input_area = input_area;

        // -- output pane
        let inner_height = output_area.height.saturating_sub(2) as usize;
        let total = self.lines.len();
        let bottom_offset = total.saturating_sub(inner_height) as u16;
        let offset = bottom_offset.saturating_sub(self.scroll_up);
        let display_lines = self.build_display_lines(offset, inner_height);
        let output = Paragraph::new(display_lines)
            .block(
                Block::new()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(rgb(theme::GRID_HI)))
                    .title(Span::styled(
                        " mathutil ",
                        Style::new().fg(rgb(theme::ACCENT)).bold(),
                    )),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(output, output_area);

        // -- input line
        // Scroll horizontally so the cursor never runs off the right edge
        // when the command is longer than the pane.
        let prompt = match &self.current_path {
            None => "/ $ ".to_string(),
            Some(topic) => format!("/{} $ ", topic),
        };
        let inner_w = input_area.width.saturating_sub(2) as usize; // minus borders
        let cursor_col = prompt.len() + self.cursor;
        let hscroll = cursor_col.saturating_sub(inner_w.saturating_sub(1));
        self.input_hscroll = hscroll as u16;

        let input_spans = self.build_input_spans(&prompt, hscroll);
        let input = Paragraph::new(Line::from(input_spans))
            .scroll((0, hscroll as u16))
            .block(
                Block::new()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(rgb(theme::GRID_HI))),
        );
        frame.render_widget(input, input_area);
        frame.set_cursor_position((
            input_area.x + 1 + (cursor_col - hscroll) as u16,
            input_area.y + 1,
        ));

        // -- status bar
        let windows = if self.children.is_empty() {
            String::new()
        } else {
            format!("  ·  {} window(s) open", self.children.len())
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {}{}", self.status, windows),
                Style::new().fg(rgb(theme::MUTED)),
            ),
            Span::styled(
                "   Tab completes · ↑↓ history · wheel/PgUp/PgDn scroll · drag to select · Ctrl+Shift+C copy · Ctrl+Shift+V paste · Ctrl+C quit",
                Style::new().fg(rgb(theme::GRID_HI))),
        ]));
        frame.render_widget(status, status_area);
    }
}

/// Run the REPL until the user quits. Returns any terminal error.
pub fn run() -> std::io::Result<()> {
    let mut terminal = ratatui::init();
    // Mouse capture lets the wheel scroll the output pane. (Trade-off: the
    // terminal's own click-drag text selection is disabled while running.)
    let _ = crossterm::execute!(std::io::stdout(), event::EnableMouseCapture);
    let mut app = App::new();
    let result = loop {
        if let Err(e) = terminal.draw(|f| app.draw(f)) {
            break Err(e);
        }
        match event::poll(Duration::from_millis(120)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) => app.on_key(key),
                Ok(Event::Mouse(m)) => app.on_mouse(m),
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => app.reap_children(),
            Err(e) => break Err(e),
        }
        if app.quit {
            break Ok(());
        }
    };
    let _ = crossterm::execute!(std::io::stdout(), event::DisableMouseCapture);
    ratatui::restore();
    result
}
