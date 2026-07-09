//! Forgiving, injection-safe text parser for matrices and vectors.
//!
//! Direct port of the Python `mathutil.core.parse`. Input is whitelisted to
//! numeric characters and tokenized with a float regex, so nothing is ever
//! evaluated. On any problem a [`ParseError`] is returned carrying a short,
//! user-facing message the REPL can print verbatim.
//!
//! Accepted forms:
//! - matrix       `[[1,2],[3,4]]`   `1 2; 3 4`   `1,2 / 3,4`
//! - vector       `(1,0,0)`  `[1,0,0]`  `1 0 0`  `1,0,0`
//! - vector list  `(1,0,0) (0,1,0)`   `[1,2] [3,4]`   `1,0 ; 0,1`

use nalgebra::{DMatrix, DVector};
use regex::Regex;
use std::fmt;
use std::sync::LazyLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError(msg.into()))
}

/// A single floating-point number, e.g. -1, 2.5, .5, 1e-3, +4.
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?").unwrap());
/// Everything a numeric expression is allowed to contain.
const ALLOWED: &str = "0123456789.+-eE ,;/()[]\t\n";
static ROW_SEP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[;\n/]+").unwrap());

fn reject_bad_chars(text: &str) -> Result<(), ParseError> {
    let mut bad: Vec<char> = text.chars().filter(|c| !ALLOWED.contains(*c)).collect();
    bad.sort();
    bad.dedup();
    if !bad.is_empty() {
        let shown = bad
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>()
            .join(" ");
        return err(format!("unexpected character(s): {shown}"));
    }
    Ok(())
}

/// Extract the numbers from a flat token, rejecting leftover junk.
fn numbers(text: &str, context: &str) -> Result<Vec<f64>, ParseError> {
    let leftover = NUMBER.replace_all(text, " ");
    let junk = leftover.trim_matches([' ', ',', '(', ')', '[', ']', '\t', '\n']);
    if !junk.is_empty() {
        return err(format!(
            "could not read a number near '{}' in {context}",
            junk.trim()
        ));
    }
    let matches: Vec<_> = NUMBER.find_iter(text).collect();
    // Two numbers jammed together with no separator (e.g. "3.4.5") is malformed.
    for pair in matches.windows(2) {
        if pair[0].end() == pair[1].start() {
            let bad = &text[pair[0].start()..pair[1].end()];
            return err(format!("could not read a number near '{bad}' in {context}"));
        }
    }
    Ok(matches
        .iter()
        .map(|m| m.as_str().parse::<f64>().unwrap())
        .collect())
}

/// Split on bracket/paren groups at nesting depth 0.
///
/// `"(1,0,0) (0,1,0)"` -> `["1,0,0", "0,1,0"]`. Text with no brackets is
/// returned as a single chunk so callers can fall back to separator splitting.
pub fn split_top_level(text: &str) -> Result<Vec<String>, ParseError> {
    let mut groups = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut saw_bracket = false;
    for (i, ch) in text.char_indices() {
        if ch == '(' || ch == '[' {
            if depth == 0 {
                start = i + ch.len_utf8();
                saw_bracket = true;
            }
            depth += 1;
        } else if ch == ')' || ch == ']' {
            depth -= 1;
            if depth < 0 {
                return err("unbalanced brackets");
            }
            if depth == 0 {
                groups.push(text[start..i].to_string());
            }
        } else if depth == 0 && !" ,\t".contains(ch) && saw_bracket {
            // A stray token outside any bracket while we're grouping.
            return err(format!("stray text outside brackets near '{ch}'"));
        }
    }
    if depth != 0 {
        return err("unbalanced brackets");
    }
    Ok(if saw_bracket {
        groups
    } else {
        vec![text.to_string()]
    })
}

/// Split matrix text into row strings on `; / newline` or `],[` groups.
fn row_chunks(inner: &str) -> Result<Vec<String>, ParseError> {
    if inner.contains('[') || inner.contains('(') {
        let chunks: Vec<String> = split_top_level(inner)?
            .into_iter()
            .filter(|c| !c.trim().is_empty())
            .collect();
        if !chunks.is_empty() {
            return Ok(chunks);
        }
    }
    Ok(ROW_SEP
        .split(inner)
        .filter(|c| !c.trim().is_empty())
        .map(str::to_string)
        .collect())
}

/// Parse a 2-D matrix from forgiving text into an `(r, c)` float matrix.
pub fn parse_matrix(text: &str) -> Result<DMatrix<f64>, ParseError> {
    let text = text.trim();
    if text.is_empty() {
        return err("empty input; expected a matrix");
    }
    reject_bad_chars(text)?;

    let mut inner = text;
    // A fully bracketed matrix `[[...],[...]]` -> strip the outer brackets so
    // the inner `[..],[..]` splits into rows below.
    let first_char = text.chars().next();
    let last_char = text.chars().last();
    if let (Some(f), Some(l)) = (first_char, last_char) {
        if "([".contains(f) && ")]".contains(l) {
            let mut depth = 0i32;
            let mut fully_bracketed = true;
            for (i, ch) in text.char_indices() {
                if ch == '(' || ch == '[' {
                    depth += 1;
                } else if ch == ')' || ch == ']' {
                    depth -= 1;
                    if depth == 0 && i != text.len() - 1 {
                        fully_bracketed = false;
                        break;
                    }
                }
            }
            if fully_bracketed {
                inner = &text[1..text.len() - 1];
            }
        }
    }

    let rows_txt = row_chunks(inner)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, r) in rows_txt.iter().enumerate() {
        let nums = numbers(r, &format!("row {}", i + 1))?;
        if !nums.is_empty() {
            rows.push(nums); // drop blank rows from trailing separators
        }
    }
    if rows.is_empty() {
        return err("no numbers found; expected a matrix");
    }

    let width = rows[0].len();
    for (i, r) in rows.iter().enumerate() {
        if r.len() != width {
            return err(format!(
                "ragged matrix: row 1 has {width} entries but row {} has {}",
                i + 1,
                r.len()
            ));
        }
    }
    let nrows = rows.len();
    Ok(DMatrix::from_row_iterator(
        nrows,
        width,
        rows.into_iter().flatten(),
    ))
}

/// Parse a single vector into a length-n column vector.
pub fn parse_vector(text: &str) -> Result<DVector<f64>, ParseError> {
    let text = text.trim();
    if text.is_empty() {
        return err("empty input; expected a vector");
    }
    reject_bad_chars(text)?;
    if ROW_SEP.is_match(text) {
        return err("expected a single vector but found a row separator (; or /)");
    }
    let nums = numbers(text, "vector")?;
    if nums.is_empty() {
        return err("no numbers found; expected a vector");
    }
    Ok(DVector::from_vec(nums))
}

/// Parse several vectors into a list of equal-length column vectors.
pub fn parse_vector_list(text: &str) -> Result<Vec<DVector<f64>>, ParseError> {
    let text = text.trim();
    if text.is_empty() {
        return err("empty input; expected one or more vectors");
    }
    reject_bad_chars(text)?;

    let chunks: Vec<String> = if text.contains('[') || text.contains('(') {
        split_top_level(text)?
    } else {
        // No brackets: split on ; / newline, else treat as one vector.
        let split: Vec<String> = ROW_SEP
            .split(text)
            .filter(|c| !c.trim().is_empty())
            .map(str::to_string)
            .collect();
        if split.is_empty() {
            vec![text.to_string()]
        } else {
            split
        }
    };

    let mut vectors = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        if c.trim().is_empty() {
            continue;
        }
        let nums = numbers(c, &format!("vector {}", i + 1))?;
        if !nums.is_empty() {
            vectors.push(DVector::from_vec(nums));
        }
    }
    if vectors.is_empty() {
        return err("no vectors found");
    }

    let dim = vectors[0].len();
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return err(format!(
                "vectors have mixed dimensions: vector 1 is {dim}D but vector {} is {}D",
                i + 1,
                v.len()
            ));
        }
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_forms() {
        for form in ["[[1,2],[3,4]]", "1 2; 3 4", "1,2 / 3,4", "(1,2),(3,4)"] {
            let m = parse_matrix(form).unwrap();
            assert_eq!(m.nrows(), 2);
            assert_eq!(m.ncols(), 2);
            assert_eq!(m[(0, 0)], 1.0);
            assert_eq!(m[(0, 1)], 2.0);
            assert_eq!(m[(1, 0)], 3.0);
            assert_eq!(m[(1, 1)], 4.0);
        }
    }

    #[test]
    fn matrix_3x3_and_floats() {
        let m = parse_matrix("[[2,0,0],[0,1,0],[0,0,1.5]]").unwrap();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m[(2, 2)], 1.5);
    }

    #[test]
    fn matrix_errors() {
        assert!(parse_matrix("").is_err());
        assert!(parse_matrix("[[1,2],[3]]").is_err()); // ragged
        assert!(parse_matrix("[[1,2],[3,4]").is_err()); // unbalanced
        assert!(parse_matrix("1 2; 3 x").is_err()); // bad char
        assert!(parse_matrix("3.4.5").is_err()); // jammed numbers
    }

    #[test]
    fn vector_forms() {
        for form in ["(1,0,0)", "[1,0,0]", "1 0 0", "1,0,0"] {
            let v = parse_vector(form).unwrap();
            assert_eq!(v.as_slice(), &[1.0, 0.0, 0.0]);
        }
        let v = parse_vector("-1.5 2e3 .5").unwrap();
        assert_eq!(v.as_slice(), &[-1.5, 2000.0, 0.5]);
    }

    #[test]
    fn vector_rejects_separators() {
        assert!(parse_vector("1,0 ; 0,1").is_err());
    }

    #[test]
    fn vector_list_forms() {
        for form in ["(1,0) (0,1)", "[1,0] [0,1]", "1,0 ; 0,1"] {
            let vs = parse_vector_list(form).unwrap();
            assert_eq!(vs.len(), 2);
            assert_eq!(vs[0].as_slice(), &[1.0, 0.0]);
            assert_eq!(vs[1].as_slice(), &[0.0, 1.0]);
        }
    }

    #[test]
    fn vector_list_single_bare() {
        let vs = parse_vector_list("1 2 3").unwrap();
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0].len(), 3);
    }

    #[test]
    fn vector_list_mixed_dims() {
        assert!(parse_vector_list("(1,0) (1,2,3)").is_err());
    }

    #[test]
    fn stray_text_outside_brackets() {
        assert!(parse_vector_list("(1,0) x (0,1)").is_err());
    }
}
