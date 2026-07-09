//! Expression parsing and evaluation for integrands and integration bounds.
//!
//! Replaces the Python version's sympy dependency with a small Pratt parser
//! producing a serializable AST. Human-friendly rules match the original:
//! `^` means power and adjacent factors multiply (`2x`, `x y`, `2(x+1)`).
//! Only the declared variables (plus known math functions/constants) are
//! allowed — anything else is a clear error.
//!
//! The AST is `serde`-serializable so scene specs can carry live functions
//! across the process boundary to the visualization window.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprError(pub String);

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ExprError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ExprError> {
    Err(ExprError(msg.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "asin" | "arcsin" => Func::Asin,
            "acos" | "arccos" => Func::Acos,
            "atan" | "arctan" => Func::Atan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "tanh" => Func::Tanh,
            "exp" => Func::Exp,
            "ln" | "log" => Func::Ln,
            "log10" => Func::Log10,
            "sqrt" => Func::Sqrt,
            "abs" => Func::Abs,
            _ => return None,
        })
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Asin => x.asin(),
            Func::Acos => x.acos(),
            Func::Atan => x.atan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Tanh => x.tanh(),
            Func::Exp => x.exp(),
            Func::Ln => x.ln(),
            Func::Log10 => x.log10(),
            Func::Sqrt => x.sqrt(),
            Func::Abs => x.abs(),
        }
    }
}

/// Expression AST. Variables are indices into the declaring `Expr::vars`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Num(f64),
    Var(usize),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
    Pow(Box<Node>, Box<Node>),
    Neg(Box<Node>),
    Call(Func, Box<Node>),
}

impl Node {
    pub fn eval(&self, args: &[f64]) -> f64 {
        match self {
            Node::Num(v) => *v,
            Node::Var(i) => args[*i],
            Node::Add(a, b) => a.eval(args) + b.eval(args),
            Node::Sub(a, b) => a.eval(args) - b.eval(args),
            Node::Mul(a, b) => a.eval(args) * b.eval(args),
            Node::Div(a, b) => a.eval(args) / b.eval(args),
            Node::Pow(a, b) => a.eval(args).powf(b.eval(args)),
            Node::Neg(a) => -a.eval(args),
            Node::Call(f, a) => f.apply(a.eval(args)),
        }
    }
}

/// A parsed expression: the AST plus its ordered variable names and the
/// original source text (used for display).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expr {
    pub node: Node,
    pub vars: Vec<String>,
    pub text: String,
}

impl Expr {
    /// Evaluate with argument values in the same order as `self.vars`.
    pub fn eval(&self, args: &[f64]) -> f64 {
        debug_assert_eq!(args.len(), self.vars.len());
        self.node.eval(args)
    }

    /// Central-difference partial derivative w.r.t. variable index `i`.
    pub fn partial(&self, args: &[f64], i: usize, h: f64) -> f64 {
        let mut lo = args.to_vec();
        let mut hi = args.to_vec();
        lo[i] -= h;
        hi[i] += h;
        (self.eval(&hi) - self.eval(&lo)) / (2.0 * h)
    }
}

// ---------------------------------------------------------------- tokenizer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn tokenize(text: &str) -> Result<Vec<Tok>, ExprError> {
    let mut toks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => i += 1,
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                // Accept python-style ** as power too.
                if chars.get(i + 1) == Some(&'*') {
                    toks.push(Tok::Caret);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                toks.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Exponent part like 1e-3.
                if i < chars.len()
                    && (chars[i] == 'e' || chars[i] == 'E')
                    && chars
                        .get(i + 1)
                        .is_some_and(|c| c.is_ascii_digit() || *c == '+' || *c == '-')
                    && chars
                        .get(if matches!(chars.get(i + 1), Some('+' | '-')) {
                            i + 2
                        } else {
                            i + 1
                        })
                        .is_some_and(|c| c.is_ascii_digit())
                {
                    i += 2;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let s: String = chars[start..i].iter().collect();
                match s.parse::<f64>() {
                    Ok(v) => toks.push(Tok::Num(v)),
                    Err(_) => return err(format!("could not read the number '{s}'")),
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                toks.push(Tok::Ident(chars[start..i].iter().collect()));
            }
            other => return err(format!("unexpected character '{other}'")),
        }
    }
    Ok(toks)
}

// -------------------------------------------------------------- Pratt parser

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    vars: &'a [String],
    unknown: Vec<String>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// True when the next token can begin a primary — i.e. an implicit
    /// multiplication boundary like `2x`, `x y`, `2(x+1)`, `x sin(y)`.
    fn at_implicit_operand(&self) -> bool {
        matches!(
            self.peek(),
            Some(Tok::Num(_)) | Some(Tok::Ident(_)) | Some(Tok::LParen)
        )
    }

    fn expr(&mut self, min_bp: u8) -> Result<Node, ExprError> {
        let mut lhs = self.prefix()?;
        loop {
            let (op, l_bp, r_bp) = match self.peek() {
                Some(Tok::Plus) => ('+', 1, 2),
                Some(Tok::Minus) => ('-', 1, 2),
                Some(Tok::Star) => ('*', 3, 4),
                Some(Tok::Slash) => ('/', 3, 4),
                Some(Tok::Caret) => ('^', 8, 7), // right-assoc
                _ if self.at_implicit_operand() => ('.', 3, 4), // implicit ×
                _ => break,
            };
            if l_bp < min_bp {
                break;
            }
            if op != '.' {
                self.next();
            }
            let rhs = self.expr(r_bp)?;
            lhs = match op {
                '+' => Node::Add(Box::new(lhs), Box::new(rhs)),
                '-' => Node::Sub(Box::new(lhs), Box::new(rhs)),
                '*' | '.' => Node::Mul(Box::new(lhs), Box::new(rhs)),
                '/' => Node::Div(Box::new(lhs), Box::new(rhs)),
                '^' => Node::Pow(Box::new(lhs), Box::new(rhs)),
                _ => unreachable!(),
            };
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<Node, ExprError> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Node::Num(v)),
            Some(Tok::Minus) => {
                // Bind tighter than +,-,* but looser than ^ so -x^2 = -(x^2).
                let inner = self.expr(5)?;
                Ok(Node::Neg(Box::new(inner)))
            }
            Some(Tok::Plus) => self.prefix(),
            Some(Tok::LParen) => {
                let inner = self.expr(0)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(inner),
                    _ => err("missing closing ')'"),
                }
            }
            Some(Tok::Ident(name)) => self.ident(name),
            Some(t) => err(format!("unexpected token {t:?}")),
            None => err("expression ended unexpectedly"),
        }
    }

    fn ident(&mut self, name: String) -> Result<Node, ExprError> {
        // Declared variable?
        if let Some(i) = self.vars.iter().position(|v| *v == name) {
            return Ok(Node::Var(i));
        }
        // Known constant?
        match name.as_str() {
            "pi" | "PI" | "Pi" => return Ok(Node::Num(std::f64::consts::PI)),
            "e" | "E" => return Ok(Node::Num(std::f64::consts::E)),
            "tau" => return Ok(Node::Num(std::f64::consts::TAU)),
            _ => {}
        }
        // Known function? Requires an argument: parenthesized or a tight
        // primary (`sin x` works like sympy's implicit application).
        if let Some(f) = Func::from_name(&name) {
            let arg = match self.peek() {
                Some(Tok::LParen) => {
                    self.next();
                    let inner = self.expr(0)?;
                    match self.next() {
                        Some(Tok::RParen) => inner,
                        _ => return err(format!("missing closing ')' after {name}(...")),
                    }
                }
                Some(Tok::Num(_)) | Some(Tok::Ident(_)) => self.expr(6)?,
                _ => return err(format!("function '{name}' needs an argument, e.g. {name}(x)")),
            };
            return Ok(Node::Call(f, Box::new(arg)));
        }
        // Maybe juxtaposed single-letter variables like "xy" for x*y.
        if name.chars().count() > 1 {
            let parts: Vec<String> = name.chars().map(|c| c.to_string()).collect();
            if parts.iter().all(|p| self.vars.contains(p)) {
                let mut node = Node::Var(self.vars.iter().position(|v| *v == parts[0]).unwrap());
                for p in &parts[1..] {
                    let i = self.vars.iter().position(|v| v == p).unwrap();
                    node = Node::Mul(Box::new(node), Box::new(Node::Var(i)));
                }
                return Ok(node);
            }
        }
        self.unknown.push(name.clone());
        // Return a placeholder; the caller reports all unknowns at once.
        Ok(Node::Num(f64::NAN))
    }
}

/// Parse `text` into an [`Expr`] over the given variable names.
///
/// Any free symbol that is not one of `variables` is an [`ExprError`].
pub fn parse_expr_text(text: &str, variables: &[&str]) -> Result<Expr, ExprError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return err("empty expression");
    }
    let toks = tokenize(trimmed)?;
    if toks.is_empty() {
        return err("empty expression");
    }
    let vars: Vec<String> = variables.iter().map(|s| s.to_string()).collect();
    let mut p = Parser {
        toks,
        pos: 0,
        vars: &vars,
        unknown: Vec::new(),
    };
    let node = p.expr(0).map_err(|e| {
        ExprError(format!("could not parse '{trimmed}': {e}"))
    })?;
    if p.pos < p.toks.len() {
        return err(format!(
            "could not parse '{trimmed}': unexpected trailing input"
        ));
    }
    if !p.unknown.is_empty() {
        let mut names = p.unknown.clone();
        names.sort();
        names.dedup();
        let list = if vars.is_empty() {
            "none".to_string()
        } else {
            vars.join(", ")
        };
        return err(format!(
            "unknown symbol(s) in '{trimmed}': {} (variables are {list})",
            names.join(", ")
        ));
    }
    Ok(Expr {
        node,
        vars,
        text: trimmed.to_string(),
    })
}

/// Evaluate a constant expression (no variables), e.g. a bound like `2*pi`.
pub fn eval_const(text: &str) -> Result<f64, ExprError> {
    let e = parse_expr_text(text, &[])?;
    let v = e.eval(&[]);
    if v.is_finite() {
        Ok(v)
    } else {
        err(format!("'{text}' does not evaluate to a finite number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ev(text: &str, vars: &[&str], args: &[f64]) -> f64 {
        parse_expr_text(text, vars).unwrap().eval(args)
    }

    #[test]
    fn basic_arithmetic() {
        assert_eq!(ev("1+2*3", &[], &[]), 7.0);
        assert_eq!(ev("(1+2)*3", &[], &[]), 9.0);
        assert_eq!(ev("2^3^2", &[], &[]), 512.0); // right-assoc
        assert_eq!(ev("2**3", &[], &[]), 8.0);
        assert_eq!(ev("-2^2", &[], &[]), -4.0); // -(2^2)
        assert_eq!(ev("7/2", &[], &[]), 3.5);
    }

    #[test]
    fn implicit_multiplication() {
        assert_eq!(ev("2x", &["x"], &[3.0]), 6.0);
        assert_eq!(ev("x y", &["x", "y"], &[2.0, 5.0]), 10.0);
        assert_eq!(ev("2(x+1)", &["x"], &[2.0]), 6.0);
        assert_eq!(ev("xy", &["x", "y"], &[2.0, 5.0]), 10.0);
        assert!((ev("2pi", &[], &[]) - 2.0 * PI).abs() < 1e-12);
    }

    #[test]
    fn functions_and_constants() {
        assert!((ev("sin(pi/2)", &[], &[]) - 1.0).abs() < 1e-12);
        assert!((ev("x*sin(x)", &["x"], &[PI / 2.0]) - PI / 2.0).abs() < 1e-12);
        assert!((ev("sqrt(9)", &[], &[]) - 3.0).abs() < 1e-12);
        assert!((ev("exp(0)", &[], &[]) - 1.0).abs() < 1e-12);
        assert!((ev("sin x", &["x"], &[PI / 2.0]) - 1.0).abs() < 1e-12);
        // sin x^2 should parse as sin(x^2) like sympy's implicit application
        assert!((ev("sin x^2", &["x"], &[2.0]) - (4.0_f64).sin()).abs() < 1e-12);
    }

    #[test]
    fn unknown_symbols() {
        let e = parse_expr_text("x + q", &["x"]).unwrap_err();
        assert!(e.0.contains("unknown symbol"));
        assert!(e.0.contains('q'));
        assert!(parse_expr_text("", &["x"]).is_err());
    }

    #[test]
    fn integrand_examples_from_readme() {
        assert_eq!(ev("x*y", &["y", "x"], &[2.0, 3.0]), 6.0);
        assert_eq!(ev("x+y+z", &["z", "y", "x"], &[1.0, 2.0, 3.0]), 6.0);
        assert!((ev("sin(x)*cos(y)", &["x", "y"], &[1.0, 1.0])
            - 1.0_f64.sin() * 1.0_f64.cos())
        .abs()
            < 1e-12);
        assert_eq!(ev("x^2 - y^2", &["x", "y"], &[3.0, 2.0]), 5.0);
    }

    #[test]
    fn partial_derivative() {
        let e = parse_expr_text("x^2*y", &["x", "y"]).unwrap();
        let d = e.partial(&[3.0, 2.0], 0, 1e-5); // d/dx = 2xy = 12
        assert!((d - 12.0).abs() < 1e-6);
    }

    #[test]
    fn serde_roundtrip() {
        let e = parse_expr_text("sin(x)*cos(y) + 2x", &["x", "y"]).unwrap();
        let json = serde_json::to_string(&e).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(back.eval(&[0.5, 0.25]), e.eval(&[0.5, 0.25]));
    }
}
