//! Shared visual styling so every scene looks consistent.
//!
//! Pure constants, mirroring the Python `core/theme.py`. Colors are stored as
//! RGB bytes; helpers convert to egui / ratatui color types at the edges.

pub type Rgb = [u8; 3];

pub const BG: Rgb = [0x1a, 0x1a, 0x24];
pub const FG: Rgb = [0xe8, 0xe8, 0xf0];
pub const GRID: Rgb = [0x3a, 0x3a, 0x4a]; // UI chrome tone: axis pens, contours
pub const GRID_LINE: Rgb = [0x48, 0x48, 0x5e]; // deforming background lattice
pub const GRID_HI: Rgb = [0x56, 0x56, 0x75]; // brighter grid line for the axes

// Semantic colors reused across every topic.
pub const ACCENT: Rgb = [0x4e, 0xa1, 0xff]; // primary vector / basis i
pub const ACCENT2: Rgb = [0xff, 0x6b, 0x6b]; // secondary vector / basis j
pub const ACCENT3: Rgb = [0x5f, 0xd8, 0xa0]; // basis k
pub const EIGEN: Rgb = [0xff, 0xd1, 0x66]; // eigenvectors / special directions
pub const SPAN: Rgb = [0x7c, 0x5c, 0xff]; // spans / subspaces / unit cell
pub const POINT: Rgb = [0xff, 0x9f, 0x1c]; // highlighted point / projection
pub const FLIP: Rgb = [0xff, 0x54, 0x70]; // orientation-reversed cell (det < 0)

pub const GOOD: Rgb = ACCENT3; // affirmative verdict
pub const BAD: Rgb = FLIP; // negative verdict
pub const MUTED: Rgb = [0x9a, 0xa0, 0xb5]; // de-emphasized panel text

pub const BASIS_COLORS: [Rgb; 3] = [ACCENT, ACCENT2, ACCENT3];

/// Half-width of the visible world in each axis.
pub const DEFAULT_SPAN: f64 = 6.0;
