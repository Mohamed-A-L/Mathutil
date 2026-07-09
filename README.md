# mathutil-rs

A Rust remake of [`mathutil`](../mathutil) — the interactive linear-algebra
teaching aid — in a permanent, compiled form. Type a matrix, some vectors, or
an equation and a native window opens with a smooth, GPU-accelerated
visualization: grids warping under a transformation, spans shaded as
lines/planes, dependence collapsing, Riemann cells sweeping out an integral.
The goal is an *instant visualizer* for textbook-style math: you type what
the problem says, the tool infers the rest.

The original Python/pyqtgraph version lives untouched in `../mathutil` and
served as the spec for this port.

## Stack

| layer      | crate |
|------------|-------|
| TUI        | `ratatui` + `crossterm` — command input, scrollback output, result tables, status bar |
| math       | `nalgebra` (SVD, rank, kernel, eigen) + a hand-rolled expression parser/evaluator and Gauss–Legendre quadrature (replaces sympy) |
| windows    | `eframe`/`egui`, spawned as a **separate process** per visualization so windows keep animating while you type |
| 2D plots   | `egui_plot` |
| 3D         | a small software-projected painter (orbit camera, depth-sorted primitives) on the egui canvas |

## Build & run

```bash
cd mathutil-rs
cargo build --release

# the REPL
./target/release/mathutil-rs
mathutil> transform [[1,1],[0,1]]
mathutil> span (1,0,0) (0,1,1)
mathutil> help

# one-shot (blocks until you close the window)
./target/release/mathutil-rs eigen [[2,1],[1,2]]
./target/release/mathutil-rs "iint x*y : y 0 x : x 0 1"
./target/release/mathutil-rs help calc
```

In the REPL: **Tab** completes command names, **↑/↓** browse history,
the **mouse wheel** or **PgUp/PgDn** scrolls the output pane, **Ctrl+L**
clears it, `quit` exits.
Help works like a directory: `help` shows the four topics (easiest first —
`vectors`, `linalg`, `calc`, `multivar`), `help <topic>` lists that topic's
commands in ascending complexity, and `help <command>` shows usage and an
example.
Every command echoes its result table into the terminal *and* opens a window;
windows are independent processes, so they survive the REPL and each other.
Animated windows have **Pause/Play**, **Replay**, and a **seek slider**;
3D axes carry tick marks at a scale chosen from the plot size. Every window
also has an editable **command box** (bottom of the info panel): change the
matrix / integrand / bounds and press Enter to rebuild the scene in place.

## Commands

Grouped like the in-app help topics, easiest first:

```
# vectors — spans & linear combinations              (help vectors)
lincomb (1,0) (1,1) : 2 3               # c1·v1 + c2·v2 tip-to-tail
span (1,0,0) (0,1,0)                    # vectors + shaded plane
independent (1,2) (2,4)                 # dependence collapse
member (1,0,0) (0,1,0) : (1,1,5)        # is a point in the span?

# linalg — matrices as transformations               (help linalg)
transform [[1,2],[0,1]]                 # 2D grid warp (shear); reports det,
transform [[2,0,0],[0,1,0],[0,0,1]]     #   eigenvalues + INVERTIBLE/SINGULAR
compose [[0,-1],[1,0]] [[1,1],[0,1]]    # apply shear, then rotate (right-to-left)
basis (1,1) (-1,1)                      # change-of-basis grid
kernel [[1,2],[2,4]]                    # what collapses to 0 / where outputs land
ranknullity [[1,2,3],[2,4,6]]           # rank + nullity = dim(domain)
eigen [[2,1],[1,2]]                     # eigen-directions only stretch

# calc — single-variable areas & volumes             (help calc)
integrate x^2 : x 0 2                   # definite integral, Riemann rectangles
revolve sin(x) : x 0 pi                 # disk method, about the x-axis
revolve sqrt(y) : y 0 4                 # x = √y about the y-axis (drawn vertical)
revolve x : x^2 : x 0 1                 # between two curves (washer)
shell x^2 : x 0 2                       # shells about the y-axis
shell y : y 0 1                         # shells about the x-axis

# multivar — partials & multiple integrals           (help multivar)
partial sin(x)*cos(y) : -3 3            # surface with sliders; tangent slopes
contour sin(x)*cos(y) : -3 3            # flat contour map; ∂f/∂x, ∂f/∂y live
iint x*y : y 0 x : x 0 1                # double integral ∬ f dA
iiint x+y+z : z 0 1 : y 0 1 : x 0 1     # triple integral ∭ f dV
# radial forms — true disk/wedge/shell geometry; the r, r, or ρ²sinφ
# volume element is applied automatically:
polar 1 : r 0 1 : theta 0 2*pi          # disk, area = π
cylindrical 1 : r 0 1 : theta 0 2*pi : z 0 1    # cylinder, vol = π
spherical 1 : rho 0 1 : phi 0 pi : theta 0 2*pi # ball, vol = 4π/3
```

Input is forgiving, exactly like the original:

| kind        | accepted forms |
|-------------|----------------|
| matrix      | `[[1,2],[3,4]]` · `1 2; 3 4` · `1,2 / 3,4` |
| vector      | `(1,0,0)` · `[1,0,0]` · `1 0 0` · `1,0,0` |
| vector list | `(1,0,0) (0,1,0)` · `[1,2] [3,4]` · `1,0 ; 0,1` |
| function    | `x^2 + y*sin(x)` · `2x` (implicit ×) · `x*y` |
| bound       | `y 0 x` · `y 0..x^2` (no spaces inside a bound: `2*x`) |

The first `:` block of an integral is the **innermost** variable (swept
first); its bounds may reference outer variables (e.g. `y 0 x`). For the
radial commands the Jacobian is added for you — type just the integrand `f`
(use `1` for plain area/volume).

**The bounds variable owns its axis**: `integrate`, `revolve`, and `shell`
accept any variable name, and all formulas/labels follow it. `revolve f(v) :
v a b` revolves about the *v*-axis (so `revolve sqrt(y) : y 0 4` is the
textbook "x = √y about the y-axis", drawn vertical); `shell` revolves about
the axis perpendicular to its variable.

## Differences from the Python original

- **No symbolic engine.** sympy's exact integrals are replaced by 32-point
  nested Gauss–Legendre quadrature (shown as `value ≈ …`, accurate to ~1e-10
  on textbook integrands). Partial derivatives are central differences.
- **Formulas are unicode text**, not rendered LaTeX (no matplotlib mathtext
  equivalent worth its weight in Rust; the info panel typography carries it).
- **Windows are processes, not threads** — `mathutil-rs viz <spec.json>` is
  spawned per scene with a serialized scene spec, which is what lets the TUI
  stay responsive without fighting winit for the main thread. Expression ASTs
  serialize with the spec, so the contour/surface windows evaluate f live.

## Layout

```
src/core/      pure math + parsers (no UI; fully unit-tested):
               parse.rs (matrices/vectors), expr.rs (Pratt parser AST),
               linalg.rs (rank/span/kernel/eigen), integrate.rs (cells + quadrature)
src/topics/    transforms, spaces, fundamental, calculus, multivar —
               turn numbers into a Report + SceneSpec
src/scene.rs   the serialized scene contract between REPL and window
               (ScenePackage = command text + SceneSpec)
src/registry.rs  command table, help topics, dispatch, tab completion
src/viz_spawn.rs squirrels the package to a temp file, spawns `… viz <file>`
src/tui.rs     the ratatui REPL (mouse-wheel scrollback, history, completion)
src/viz/       the egui window: plot2d.rs, three.rs (3D painter), func.rs;
               pause/seek clock, command box, per-scene interaction hints
```

New topics plug in as a `topics/` function returning a `SceneSpec` variant
plus one entry in `registry.rs::commands()`.

## Test

```bash
cargo test        # parser, expression engine, linalg, integrals, dispatch
```