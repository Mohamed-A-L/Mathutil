//! Command registry: maps text command names to topic functions.
//!
//! Each command declares how to parse its own arguments, so the REPL and the
//! one-shot CLI never hard-code parsing per command. Adding a future topic
//! (e.g. Gaussian elimination) is just another entry in [`commands`].

use crate::core::expr::eval_const;
use crate::core::parse::{parse_matrix, parse_vector, parse_vector_list, split_top_level};
use crate::scene::{Outcome, Report, Row};
use crate::topics::{calculus, fundamental, multivar, spaces, transforms};
use crate::topics::calculus::Coord;

pub struct Command {
    pub name: &'static str,
    pub summary: &'static str,
    pub usage: &'static str,
    run: fn(&str) -> Result<Outcome, String>,
}

/// A help "directory": commands grouped by subject. Topics are ordered
/// easiest first, and each topic lists its commands in ascending complexity.
pub struct Topic {
    pub name: &'static str,
    pub title: &'static str,
    pub commands: &'static [&'static str],
}

pub fn topics() -> &'static [Topic] {
    &[
        Topic {
            name: "vectors",
            title: "vectors, spans & linear combinations",
            commands: &["lincomb", "span", "independent", "member"],
        },
        Topic {
            name: "linalg",
            title: "matrices as linear transformations",
            commands: &[
                "transform",
                "compose",
                "basis",
                "kernel",
                "ranknullity",
                "eigen",
            ],
        },
        Topic {
            name: "calc",
            title: "single-var calc: areas & volumes",
            commands: &["integrate", "revolve", "shell"],
        },
        Topic {
            name: "multivar",
            title: "multivariate calculus: partials & integrals",
            commands: &[
                "partial",
                "contour",
                "iint",
                "polar",
                "iiint",
                "cylindrical",
                "spherical",
            ],
        },
    ]
}

pub fn commands() -> &'static [Command] {
    &[
        Command {
            name: "transform",
            summary: "Grid warps into A — det, eigenvalues, invertibility verdict",
            usage: "transform <matrix>      e.g. transform [[1,2],[0,1]]",
            run: |args| scene(transforms::transform(parse_matrix(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "compose",
            summary: "Apply a sequence of matrices right-to-left",
            usage: "compose <A> <B> [<C> …]  e.g. compose [[0,-1],[1,0]] [[1,1],[0,1]]",
            run: |args| {
                let groups = split_top_level(args.trim()).map_err(|e| e.0)?;
                if groups.len() < 2 {
                    return Err("compose needs at least two bracketed matrices, \
                                e.g. compose [[0,-1],[1,0]] [[1,1],[0,1]]"
                        .into());
                }
                let mats = groups
                    .iter()
                    .map(|g| parse_matrix(g).map_err(|e| e.0))
                    .collect::<Result<Vec<_>, _>>()?;
                scene(transforms::compose(mats))
            },
        },
        Command {
            name: "span",
            summary: "Plot vectors and shade their span (line/plane/space)",
            usage: "span <v1> <v2> ...      e.g. span (1,0,0) (0,1,0)",
            run: |args| scene(spaces::span(parse_vector_list(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "independent",
            summary: "Test/visualize linear (in)dependence",
            usage: "independent <v1> ...    e.g. independent (1,2) (2,4)",
            run: |args| scene(spaces::independent(parse_vector_list(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "basis",
            summary: "Change-of-basis grid for a new basis",
            usage: "basis <b1> <b2> [b3]    e.g. basis (1,1) (-1,1)",
            run: |args| scene(transforms::basis(parse_vector_list(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "kernel",
            summary: "Kernel (collapses to 0) and image (where outputs land)",
            usage: "kernel <matrix>         e.g. kernel [[1,2],[2,4]]",
            run: |args| scene(fundamental::kernel_image(parse_matrix(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "ranknullity",
            summary: "Dimension bars: rank + nullity = dim(domain)",
            usage: "ranknullity <matrix>    e.g. ranknullity [[1,2,3],[2,4,6]]",
            run: |args| scene(fundamental::rank_nullity(parse_matrix(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "eigen",
            summary: "Eigen-directions: lines only stretched, never rotated",
            usage: "eigen <matrix>          e.g. eigen [[2,1],[1,2]]",
            run: |args| scene(transforms::eigen(parse_matrix(args).map_err(|e| e.0)?)),
        },
        Command {
            name: "lincomb",
            summary: "Build c1·v1 + c2·v2 + ... tip-to-tail",
            usage: "lincomb <v1> <v2> ... : <c1> <c2> ...   e.g. lincomb (1,0) (1,1) : 2 3",
            run: |args| {
                let (left, right) = split_colon(args);
                let vectors = parse_vector_list(&left).map_err(|e| e.0)?;
                let coeffs = if right.is_empty() {
                    None
                } else {
                    Some(parse_vector(&right).map_err(|e| e.0)?)
                };
                scene(spaces::lincomb(vectors, coeffs))
            },
        },
        Command {
            name: "member",
            summary: "Is a point in the span of some vectors?",
            usage: "member <v1> ... : <point>   e.g. member (1,0,0) (0,1,0) : (1,1,5)",
            run: |args| {
                let (left, right) = split_colon(args);
                if right.is_empty() {
                    return Err("member needs 'vectors : point', \
                                e.g. member (1,0,0) (0,1,0) : (1,1,5)"
                        .into());
                }
                scene(spaces::member(
                    parse_vector_list(&left).map_err(|e| e.0)?,
                    parse_vector(&right).map_err(|e| e.0)?,
                ))
            },
        },
        Command {
            name: "iint",
            summary: "Double integral as a staged inner→outer sweep",
            usage: "iint <f> : <var lo hi> : <var lo hi>   e.g. iint x*y : y 0 x : x 0 1",
            run: |args| {
                let (f, blocks) = integral_parts(args)?;
                scene(calculus::sweep_integral(&f, &blocks, 2, Coord::Cartesian))
            },
        },
        Command {
            name: "iiint",
            summary: "Triple integral swept inner→middle→outer",
            usage: "iiint <f> : <v lo hi> : <v lo hi> : <v lo hi>   e.g. iiint 1 : z 0 1 : y 0 1 : x 0 1",
            run: |args| {
                let (f, blocks) = integral_parts(args)?;
                scene(calculus::sweep_integral(&f, &blocks, 3, Coord::Cartesian))
            },
        },
        Command {
            name: "polar",
            summary: "Double integral in polar coords (true disk/wedge geometry)",
            usage: "polar <f(r,theta)> : r a b : theta c d   e.g. polar 1 : r 0 1 : theta 0 2*pi",
            run: |args| {
                let (f, blocks) = integral_parts(args)?;
                scene(calculus::sweep_integral(&f, &blocks, 2, Coord::Polar))
            },
        },
        Command {
            name: "cylindrical",
            summary: "Triple integral in cylindrical coords (r, theta, z)",
            usage: "cylindrical <f> : r a b : theta c d : z e f",
            run: |args| {
                let (f, blocks) = integral_parts(args)?;
                scene(calculus::sweep_integral(&f, &blocks, 3, Coord::Cylindrical))
            },
        },
        Command {
            name: "spherical",
            summary: "Triple integral in spherical coords (rho, theta, phi)",
            usage: "spherical <f> : rho a b : phi c d : theta e f",
            run: |args| {
                let (f, blocks) = integral_parts(args)?;
                scene(calculus::sweep_integral(&f, &blocks, 3, Coord::Spherical))
            },
        },
        Command {
            name: "partial",
            summary: "Partial derivatives on a surface (top view = contour map)",
            usage: "partial <f(x,y)> [: a b]   e.g. partial sin(x)*cos(y) : -3 3",
            run: |args| {
                let (f, domain) = fn_and_domain(args, "partial")?;
                scene(multivar::partial(&f, domain))
            },
        },
        Command {
            name: "contour",
            summary: "Partial derivatives on a flat contour map",
            usage: "contour <f(x,y)> [: a b]   e.g. contour sin(x)*cos(y) : -3 3",
            run: |args| {
                let (f, domain) = fn_and_domain(args, "contour")?;
                scene(multivar::contour(&f, domain))
            },
        },
        Command {
            name: "integrate",
            summary: "Definite integral: Riemann rectangles sweep out the area",
            usage: "integrate <f(x)> : x a b   e.g. integrate x^2 : x 0 2",
            run: |args| {
                let (left, right) = split_colon(args);
                let (var, a, b) = bounds_block(&right, "integrate")?;
                scene(calculus::riemann1(&left, &var, a, b))
            },
        },
        Command {
            name: "revolve",
            summary: "Revolution about the axis; disk/washer methods",
            usage: "revolve <R(v)> [: <r(v)>] : v a b   e.g. revolve sin(x) : x 0 pi  ·  revolve sqrt(y) : y 0 4 (about y)",
            run: |args| {
                let parts: Vec<String> = args.split(':').map(|p| p.trim().to_string()).collect();
                match parts.as_slice() {
                    [f, bounds] if !f.is_empty() => {
                        let (var, a, b) = bounds_block(bounds, "revolve")?;
                        scene(calculus::revolution(f, &var, a, b, None))
                    }
                    [outer, inner, bounds] if !outer.is_empty() && !inner.is_empty() => {
                        let (var, a, b) = bounds_block(bounds, "revolve")?;
                        scene(calculus::revolution(outer, &var, a, b, Some(inner)))
                    }
                    _ => Err("revolve needs 'f(x) : x a b' (disk) or \
                              'R(x) : r(x) : x a b' (washer)"
                        .into()),
                }
            },
        },
        Command {
            name: "shell",
            summary: "Volume by cylindrical shells, about the perpendicular axis",
            usage: "shell <f(v)> : v a b   e.g. shell x^2 : x 0 2 (about y)  ·  shell y : y 0 1 (about x)",
            run: |args| {
                let (left, right) = split_colon(args);
                let (var, a, b) = bounds_block(&right, "shell")?;
                scene(calculus::shells(&left, &var, a, b))
            },
        },
    ]
}

fn scene(result: Result<crate::scene::SceneSpec, String>) -> Result<Outcome, String> {
    result.map(Outcome::with_scene)
}

/// Parse `"<name> <args>"` and dispatch. `Err` carries a user-facing message.
pub fn run_command(text: &str) -> Result<Option<Outcome>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let (name, rest) = match text.split_once(char::is_whitespace) {
        Some((n, r)) => (n, r.trim()),
        None => (text, ""),
    };
    let cmd = commands()
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| format!("unknown command '{name}' (type 'help' for a list)"))?;
    if rest == "help" {
        return Ok(Some(Outcome::text_only(Report {
            title: format!("{} — {}", cmd.name, cmd.summary),
            formulas: vec![],
            body: vec![Row::plain(format!("  {}", cmd.usage))],
        })));
    }
    (cmd.run)(rest).map(Some)
}

/// Overview: the topic directory, easiest topic first.
pub fn help_text() -> Vec<Row> {
    let width = topics()
        .iter()
        .map(|t| t.name.len())
        .max()
        .unwrap_or(0)
        .max("help <command>".len());
    let mut rows = vec![
        Row::plain("Topics  (cd <topic> to enter, or help <command>):"),
        Row::plain(""),
    ];
    for t in topics() {
        rows.push(Row::plain(format!(
            "  {:<width$}  {}  ({} commands)",
            t.name,
            t.title,
            t.commands.len()
        )));
    }
    rows.push(Row::plain(""));
    rows.push(Row::plain(format!(
        "  {:<width$}  usage and an example for one command",
        "help <command>"
    )));
    rows.push(Row::plain(format!(
        "  {:<width$}  clear the screen (REPL only)",
        "clear"
    )));
    rows.push(Row::plain(format!(
        "  {:<width$}  leave the REPL",
        "quit / exit"
    )));
    rows
}

/// Detailed help: usage for a single command.
pub fn command_help(name: &str) -> Result<Vec<Row>, String> {
    let cmd = commands()
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| format!("unknown command '{name}' (type 'help' for a list)"))?;
    Ok(vec![
        Row::plain(format!("{} — {}", cmd.name, cmd.summary)),
        Row::plain(format!("  {}", cmd.usage)),
    ])
}

/// Tab-completion candidates for `text`, given the whole `line` so far.
pub fn completion_matches(line: &str, text: &str) -> Vec<String> {
    let line = line.trim_start();
    let first = line.split_whitespace().next().unwrap_or("");
    let options: Vec<String> = if !line.contains(' ') {
        let mut o: Vec<String> = commands().iter().map(|c| c.name.to_string()).collect();
        o.extend(["cd".into(), "ls".into(), "help".into(), "quit".into(), "exit".into(), "clear".into()]);
        o.sort();
        o
    } else if first == "cd" {
        topics().iter().map(|t| t.name.to_string()).collect()
    } else if first == "help" {
        commands().iter().map(|c| c.name.to_string()).collect()
    } else {
        vec!["help".into()]
    };
    options.into_iter().filter(|o| o.starts_with(text)).collect()
}

// ------------------------------------------------------------------ helpers

fn split_colon(args: &str) -> (String, String) {
    match args.split_once(':') {
        Some((l, r)) => (l.trim().to_string(), r.trim().to_string()),
        None => (args.trim().to_string(), String::new()),
    }
}

fn integral_parts(args: &str) -> Result<(String, Vec<String>), String> {
    let parts: Vec<String> = args.split(':').map(|p| p.trim().to_string()).collect();
    if parts.len() < 2 || parts[0].is_empty() {
        return Err("need an integrand and at least one ': var lo hi' block, \
                    e.g. iint x*y : y 0 x : x 0 1"
            .into());
    }
    let mut parts = parts;
    let integrand = parts.remove(0);
    Ok((integrand, parts))
}

fn fn_and_domain(args: &str, cmd: &str) -> Result<(String, (f64, f64)), String> {
    let (left, right) = split_colon(args);
    if left.is_empty() {
        return Err(format!("{cmd} needs a function, e.g. {cmd} sin(x)*cos(y)"));
    }
    let mut domain = (-3.0, 3.0);
    if !right.is_empty() {
        let toks: Vec<&str> = right.split_whitespace().collect();
        if toks.len() != 2 {
            return Err(format!(
                "{cmd} domain must be 'a b', e.g. {cmd} x^2-y^2 : -3 3"
            ));
        }
        domain = (const_tok(toks[0])?, const_tok(toks[1])?);
    }
    Ok((left, domain))
}

fn bounds_block(text: &str, cmd: &str) -> Result<(String, f64, f64), String> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.len() != 3 {
        return Err(format!(
            "{cmd} bounds must be 'var a b', e.g. {cmd} … : x 0 1"
        ));
    }
    Ok((toks[0].to_string(), const_tok(toks[1])?, const_tok(toks[2])?))
}

/// Evaluate a numeric bound token, allowing constants like `pi`.
fn const_tok(tok: &str) -> Result<f64, String> {
    tok.parse::<f64>()
        .or_else(|_| eval_const(tok).map_err(|e| e.0))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_and_reports() {
        let out = run_command("transform [[1,2],[0,1]]").unwrap().unwrap();
        assert!(out.scene.is_some());
        assert!(out.report.body.iter().any(|r| r.text.contains("det A")));

        let out = run_command("span (1,0,0) (0,1,0)").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("dim(span) = 2")));

        let out = run_command("independent (1,2) (2,4)").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("dependent")));

        let out = run_command("kernel [[1,2],[2,4]]").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("1 + 1 = 2")));

        let out = run_command("ranknullity [[1,2,3],[2,4,6]]").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("1 + 2 = 3")));

        let out = run_command("eigen [[2,1],[1,2]]").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("λ")));
    }

    #[test]
    fn dispatch_integrals() {
        let out = run_command("iint x*y : y 0 x : x 0 1").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Sweep { exact, total, .. } => {
                assert!((exact.unwrap() - 0.125).abs() < 1e-9);
                assert!((total - 0.125).abs() < 0.01);
            }
            _ => panic!("expected sweep"),
        }
        // polar disk of radius 1: area = π (Jacobian auto-added)
        let out = run_command("polar 1 : r 0 1 : theta 0 2*pi").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Sweep { exact, .. } => {
                assert!((exact.unwrap() - std::f64::consts::PI).abs() < 1e-8);
            }
            _ => panic!("expected sweep"),
        }
        // spherical ball of radius 1: volume = 4π/3
        let out = run_command("spherical 1 : rho 0 1 : phi 0 pi : theta 0 2*pi")
            .unwrap()
            .unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Sweep { exact, .. } => {
                let want = 4.0 * std::f64::consts::PI / 3.0;
                assert!((exact.unwrap() - want).abs() < 1e-6);
            }
            _ => panic!("expected sweep"),
        }
    }

    #[test]
    fn dispatch_revolution() {
        // revolve sin(x) over [0, π]: V = π ∫ sin² = π²/2
        let out = run_command("revolve sin(x) : x 0 pi").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Revolution { volume, .. } => {
                let want = std::f64::consts::PI.powi(2) / 2.0;
                assert!((volume - want).abs() < 1e-8, "got {volume}");
            }
            _ => panic!("expected revolution"),
        }
        // washer form: revolve x : x^2 over [0,1]: V = π ∫ (x² − x⁴) = 2π/15
        let out = run_command("revolve x : x^2 : x 0 1").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Revolution { volume, inner, .. } => {
                let want = 2.0 * std::f64::consts::PI / 15.0;
                assert!((volume - want).abs() < 1e-8, "got {volume}");
                assert!(inner.is_some());
            }
            _ => panic!("expected revolution"),
        }
        // shell x^2 over [0,2]: V = 2π ∫ x·x² = 2π·4 = 8π
        let out = run_command("shell x^2 : x 0 2").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Revolution { volume, .. } => {
                let want = 8.0 * std::f64::consts::PI;
                assert!((volume - want).abs() < 1e-8, "got {volume}");
            }
            _ => panic!("expected revolution"),
        }
    }

    #[test]
    fn dispatch_integrate_1d() {
        // ∫ x² over [0,2] = 8/3
        let out = run_command("integrate x^2 : x 0 2").unwrap().unwrap();
        match out.scene.unwrap() {
            crate::scene::SceneSpec::Riemann1 { exact, total, .. } => {
                assert!((exact - 8.0 / 3.0).abs() < 1e-9, "got {exact}");
                assert!((total - 8.0 / 3.0).abs() < 0.01, "riemann {total}");
            }
            _ => panic!("expected riemann1"),
        }
        assert!(run_command("integrate x^2 : x 2 0").is_err()); // a >= b
        assert!(run_command("integrate x^2").is_err()); // no bounds
    }

    #[test]
    fn variable_aware_axes() {
        use crate::scene::SceneSpec;
        // Disk about the y-axis: x = √y for y in [0,4] → V = π ∫ y dy = 8π
        let out = run_command("revolve sqrt(y) : y 0 4").unwrap().unwrap();
        assert!(out.report.title.contains("about the y-axis"), "{}", out.report.title);
        match out.scene.unwrap() {
            SceneSpec::Revolution { volume, var, shells, .. } => {
                assert_eq!(var, "y");
                assert!(!shells);
                let want = 8.0 * std::f64::consts::PI;
                assert!((volume - want).abs() < 1e-8, "got {volume}");
            }
            _ => panic!("expected revolution"),
        }
        // Shells about the x-axis: x = y for y in [0,1] → V = 2π ∫ y·y dy = 2π/3
        let out = run_command("shell y : y 0 1").unwrap().unwrap();
        assert!(out.report.title.contains("about the x-axis"), "{}", out.report.title);
        match out.scene.unwrap() {
            SceneSpec::Revolution { volume, shells, .. } => {
                assert!(shells);
                let want = 2.0 * std::f64::consts::PI / 3.0;
                assert!((volume - want).abs() < 1e-8, "got {volume}");
            }
            _ => panic!("expected revolution"),
        }
        // Any variable name works and the labels follow it.
        let out = run_command("revolve t : t 0 1").unwrap().unwrap();
        assert!(out.report.title.contains("about the t-axis"));
        assert!(out.report.body.iter().any(|r| r.text.contains("R(t) = t")));
        let out = run_command("integrate u^2 : u 0 2").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("f(u) = u^2")));
    }

    #[test]
    fn transform_carries_invertibility_verdict() {
        let out = run_command("transform [[1,2],[2,4]]").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("SINGULAR")));
        let out = run_command("transform [[2,1],[1,2]]").unwrap().unwrap();
        assert!(out.report.body.iter().any(|r| r.text.contains("INVERTIBLE")));
        // the absorbed commands are gone
        assert!(run_command("invertible [[1,2],[2,4]]").is_err());
        assert!(run_command("washer x : x^2 : x 0 1").is_err());
    }

    #[test]
    fn errors_are_user_facing() {
        assert!(run_command("bogus 1 2 3").is_err());
        assert!(run_command("transform [[1,2],[3]]").is_err());
        assert!(run_command("member (1,0)").is_err());
        assert!(run_command("iint x*y").is_err());
        assert!(run_command("").unwrap().is_none());
    }

    #[test]
    fn completion() {
        assert!(completion_matches("tra", "tra").contains(&"transform".to_string()));
        assert!(completion_matches("help sp", "sp").contains(&"span".to_string()));
        assert!(completion_matches("cd lin", "lin").contains(&"linalg".to_string()));
        assert_eq!(completion_matches("span he", "he"), vec!["help".to_string()]);
    }

    #[test]
    fn topics_cover_every_command_once() {
        let mut listed: Vec<&str> = topics().iter().flat_map(|t| t.commands).copied().collect();
        listed.sort();
        let before = listed.len();
        listed.dedup();
        assert_eq!(before, listed.len(), "a command appears in two topics");
        let mut names: Vec<&str> = commands().iter().map(|c| c.name).collect();
        names.sort();
        assert_eq!(listed, names, "topics and the command table disagree");
        // no topic name shadows a command
        for t in topics() {
            assert!(commands().iter().all(|c| c.name != t.name));
        }
    }
}
