use std::fmt;

use super::{DiceExpression, Evaluator};

/// Displays an expression slightly normalized, with parenthesis and spaces.
/// ```
/// use diceystats::DiceExpression;
/// let x: DiceExpression = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
/// assert_eq!(x.to_string(), "(d5 + d20xd5) * max(max(d4 * d4, d5), d10)x(d4 * d8)");
/// ```
impl fmt::Display for DiceExpression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut e = StringEvaluator { precedence: Vec::new() };
        let res = self.evaluate_generic(&mut e);
        write!(f, "{}", res)
    }
}

struct StringEvaluator {
    precedence: Vec<usize>,
}

impl Evaluator<String> for StringEvaluator {
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> String {
        self.precedence.push(8);
        format!("d{}", d)
    }

    fn repeat_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(3);
        *a = match (aa >= 3, bb > 3) {
            (true, true) => format!("{a}x{b}"),
            (true, false) => format!("{a}x({b})"),
            (false, true) => format!("({a})x{b}"),
            (false, false) => format!("({a})x({b})"),
        };
    }

    fn constant(&mut self, n: isize) -> String {
        self.precedence.push(7);
        format!("{}", n)
    }

    fn negate_inplace(&mut self, a: &mut String) {
        let _aa = self.precedence.pop().unwrap();
        self.precedence.push(6);
        *a = format!("-({a})");
    }

    fn add_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(0);
        if bb == 0 { *a = format!("{a} + ({b})") } else { *a = format!("{a} + {b}") }
    }

    fn mul_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(2);
        *a = match (aa >= 2, bb > 2) {
            (true, true) => format!("{a} * {b}"),
            (true, false) => format!("{a} * ({b})"),
            (false, true) => format!("({a}) * {b}"),
            (false, false) => format!("({a}) * ({b})"),
        };
    }

    fn sub_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(1);
        *a = match (aa >= 1, bb > 1) {
            (true, true) => format!("{a} - {b}"),
            (true, false) => format!("{a} - ({b})"),
            (false, true) => format!("({a}) - {b}"),
            (false, false) => format!("({a}) - ({b})"),
        };
    }

    fn max_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let _bb = self.precedence.pop().unwrap();
        self.precedence.push(9);
        *a = format!("max({a}, {b})");
    }

    fn min_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let _bb = self.precedence.pop().unwrap();
        self.precedence.push(8);
        *a = format!("min({a}, {b})");
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    extern crate test;

    macro_rules! test {
        ($f:ident, $right:expr, $($wrong:expr),+) => {
            #[test]
            fn $f() {
                $(
                    let a = DiceExpression::from_str($wrong).unwrap();
                    assert_eq!($right, a.to_string());
                )+
            }
        };
    }

    test!(add_mul, "(1 + 2) * 3", "((1+2)*3)");
    test!(sub_sub, "-10 - -20", "-10--20");
    test!(many_ops, "1xd2 - 3 + 4 * -2", "1xd2-3+4*-2", "(((1xd2)-3)+4*-2)");
}