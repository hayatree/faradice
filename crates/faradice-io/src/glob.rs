//! Minimal glob matching for net names: `*`, `?` and `[...]` classes (with `!` negation and ranges).

/// Returns `true` if `name` matches `pattern`, or equals it literally (so a
/// bus bit such as `bus[2]` selects itself; `[2]` as a class would only
/// match `bus2`). A pattern that uses an escape (`[*]`, `[?]`, `[[]`, as
/// produced by [`glob_escape`]) is a pure glob, so escaped names still match
/// only themselves.
#[must_use]
pub fn glob_match(pattern: &str, name: &str) -> bool {
    if pattern == name && !has_escape(pattern) {
        return true;
    }
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    matches(&p, &n)
}

fn has_escape(pattern: &str) -> bool {
    ["[*]", "[?]", "[[]"].iter().any(|e| pattern.contains(e))
}

/// Escapes `name` so that it matches only itself as a glob pattern (for
/// selecting exact net names such as `bus[3]`).
#[must_use]
pub fn glob_escape(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '*' | '?' | '[' => {
                out.push('[');
                out.push(c);
                out.push(']');
            }
            _ => out.push(c),
        }
    }
    out
}

fn matches(p: &[char], n: &[char]) -> bool {
    // Iterative backtracking on the last `*`.
    let (mut pi, mut ni) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ni < n.len() {
        if pi < p.len() {
            match p[pi] {
                '*' => {
                    star = Some((pi, ni));
                    pi += 1;
                    continue;
                }
                '?' => {
                    pi += 1;
                    ni += 1;
                    continue;
                }
                '[' => {
                    if let Some((ok, len)) = class(&p[pi..], n[ni]) {
                        if ok {
                            pi += len;
                            ni += 1;
                            continue;
                        }
                    } else if n[ni] == '[' {
                        pi += 1;
                        ni += 1;
                        continue;
                    }
                }
                c if c == n[ni] => {
                    pi += 1;
                    ni += 1;
                    continue;
                }
                _ => {}
            }
        }
        match star {
            Some((sp, sn)) => {
                pi = sp + 1;
                ni = sn + 1;
                star = Some((sp, sn + 1));
            }
            None => return false,
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

/// Parses a `[...]` class at the start of `p`; returns (matched, length) or `None` if unterminated.
fn class(p: &[char], c: char) -> Option<(bool, usize)> {
    let mut i = 1;
    let negate = p.get(1) == Some(&'!');
    if negate {
        i += 1;
    }
    let mut hit = false;
    let mut first = true;
    while i < p.len() {
        if p[i] == ']' && !first {
            return Some((hit != negate, i + 1));
        }
        if i + 2 < p.len() && p[i + 1] == '-' && p[i + 2] != ']' {
            hit |= p[i] <= c && c <= p[i + 2];
            i += 3;
        } else {
            hit |= p[i] == c;
            i += 1;
        }
        first = false;
    }
    None
}

#[cfg(test)]
mod tests {

    #[test]
    fn literal_names_match_themselves() {
        assert!(glob_match("bus[2]", "bus[2]"));
        assert!(glob_match("bus[2]", "bus2"));
        assert!(!glob_match("bus[2]", "bus[3]"));
        assert!(glob_match("bus[[]2]", "bus[2]"));
        // Escaped patterns never fall back to a literal match.
        assert!(!glob_match("bus[[]2]", "bus[[]2]"));
        for name in ["bus[2]", "a*b", "q?[x]"] {
            let e = glob_escape(name);
            assert!(glob_match(&e, name));
            assert!(!glob_match(&e, &e) || e == name, "{e}");
        }
    }
    use super::{glob_escape, glob_match};

    #[test]
    fn escaped_names_match_only_themselves() {
        for name in ["bus[3]", "a*b", "q?", "x[!0-9]", "plain", "[", "]"] {
            let p = glob_escape(name);
            assert!(glob_match(&p, name), "{p} must match {name}");
        }
        assert!(!glob_match(&glob_escape("bus[3]"), "bus3"));
        assert!(!glob_match(&glob_escape("a*b"), "axxb"));
        assert!(!glob_match(&glob_escape("q?"), "qx"));
    }

    #[test]
    fn patterns() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("clk*", "clk_core"));
        assert!(!glob_match("clk*", "xclk"));
        assert!(glob_match("data[0-7]", "data5"));
        assert!(!glob_match("data[0-7]", "data8"));
        assert!(glob_match("data[!0-7]", "data8"));
        assert!(glob_match("v?d*", "vdd1"));
        assert!(glob_match("a*b*c", "aXXbYYc"));
        assert!(!glob_match("a*b*c", "aXXbYY"));
        assert!(glob_match("VSS", "VSS"));
        assert!(!glob_match("VSS", "VSS1"));
    }
}
