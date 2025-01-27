use std::fmt;

use super::{DiceFormula, Evaluator};

/// Displays an expression slightly normalized, with parenthesis and spaces.
///
/// ```
/// use diceystats::DiceFormula;
/// let x: DiceFormula = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
/// assert_eq!(x.to_string(), "(d5 + d20xd5) * max(max(d4 * d4, d5), d10)x(d4 * d8)");
/// ```
impl fmt::Display for DiceFormula {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut e = StringEvaluator {};
        let res = self.traverse(&mut e);
        write!(f, "{}", res.0)
    }
}

struct StringEvaluator {}

impl Evaluator<(String, usize)> for StringEvaluator {
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> (String, usize) {
        (format!("d{}", d), 8)
    }

    fn multi_add_inplace(&mut self, (a, aa): &mut (String, usize), (b, bb): &(String, usize)) {
        *a = match (*aa >= 3, *bb > 3) {
            (true, true) => format!("{a}x{b}"),
            (true, false) => format!("{a}x({b})"),
            (false, true) => format!("({a})x{b}"),
            (false, false) => format!("({a})x({b})"),
        };
        *aa = 3;
    }

    fn constant(&mut self, n: isize) -> (String, usize) {
        (format!("{}", n), 7)
    }

    fn negate_inplace(&mut self, (a, aa): &mut (String, usize)) {
        *a = format!("-({a})");
        *aa = 6;
    }

    fn add_inplace(&mut self, (a, aa): &mut (String, usize), (b, bb): &(String, usize)) {
        *a = if *bb == 0 { format!("{a} + ({b})") } else { format!("{a} + {b}") };
        *aa = 0;
    }

    fn mul_inplace(&mut self, (a, aa): &mut (String, usize), (b, bb): &(String, usize)) {
        *a = match (*aa >= 2, *bb > 2) {
            (true, true) => format!("{a} * {b}"),
            (true, false) => format!("{a} * ({b})"),
            (false, true) => format!("({a}) * {b}"),
            (false, false) => format!("({a}) * ({b})"),
        };
        *aa = 2;
    }

    fn sub_inplace(&mut self, (a, aa): &mut (String, usize), (b, bb): &(String, usize)) {
        *a = if *bb > 0 { format!("{a} - {b}") } else { format!("{a} - ({b})") };
        *aa = 0;
    }

    fn max_inplace(&mut self, (a, aa): &mut (String, usize), (b, _): &(String, usize)) {
        *a = format!("max({a}, {b})");
        *aa = 9;
    }

    fn min_inplace(&mut self, (a, aa): &mut (String, usize), (b, _): &(String, usize)) {
        *a = format!("min({a}, {b})");
        *aa = 8;
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::dist::DenseDist;

    use super::*;
    extern crate test;

    // Given the name of a test, a formatted expression and multiple input
    // expressions, create a test which checks that the input expressions are
    // equal to the formatted expression
    macro_rules! test {
        ($f:ident, $right:expr, $($wrong:expr),+) => {
            #[test]
            fn $f() {
                $(
                    let a = DiceFormula::from_str($wrong).unwrap();
                    assert_eq!($right, a.to_string());
                )+
            }
        };
    }

    test!(add_mul, "(1 + 2) * 3", "((1+2)*3)");
    test!(sub_sub, "-10 - -20", "-10--20");
    test!(many_ops, "1xd2 - 3 + 4 * -2", "1xd2-3+4*-2", "(((1xd2)-3)+4*-2)");
    test!(multi_add, "4x10", "4 x 10");
    test!(multi_add_mul, "1 * 2x3 * 4", "1 * 2 x 3 * 4", "1 * (2 x 3) * 4", "((1 * (2 x 3)) * 4)");
    test!(add_sub, "1 + 2 - 3 + 4 - 5", "((((1+2)-3)+4)-5)");
    test!(sub_add, "1 - 2 + 3 - 4 + 5", "((((1-2)+3)-4)+5)");

    #[test]
    fn print_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..200 {
            // Generate a random expression and check if its distribution is the same after
            // being formatted and parsed into a new expression.
            let a = DiceFormula::random(&mut rng, 2, 10);
            let a_formatted = DiceFormula::from_str(&a.to_string()).unwrap();
            let dist: DenseDist<f64> = a.dist();
            let dist_formatted: DenseDist<f64> = a_formatted.dist();
            assert!(dist.distance(&dist_formatted) <= 0.01, "{a} = {a_formatted}");
        }
    }
}
