use std::str::FromStr;

use peg::str::LineCol;

use super::DiceExpression;

#[derive(Debug)]
pub enum DiceParseError {
    Parse(peg::error::ParseError<LineCol>),
    NegativeRolls,
}

impl FromStr for DiceExpression {
    type Err = DiceParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match list_parser::arithmetic(s) {
            Ok(expr) => {
                if expr.could_be_negative() != 0 {
                    Err(DiceParseError::NegativeRolls)
                } else {
                    Ok(expr)
                }
            }
            Err(err) => Err(DiceParseError::Parse(err)),
        }
    }
}

peg::parser! {
    grammar list_parser() for str {
        rule number() -> usize = n:$(['0'..='9']+) {? n.parse::<usize>().or(Err("u32")) }
        rule list_part() -> DiceExpression = " "* e:arithmetic() " "* { e }
        pub rule arithmetic() -> DiceExpression = precedence!{
            x:(@) " "* "+" " "* y:@ { x + &y }
            x:(@) " "* "-" " "* y:@ { x - &y }
            --
            x:(@) " "* "*" " "* y:@ { x * &y }
            --
            x:(@) " "* "x" " "* y:@ { x.multi_add(&y) }
            --
            n1:number() "d" n2:number() {
                let nn1 = DiceExpression::constant(n1 as isize);
                let nn2 = DiceExpression::dice(n2);
                nn1.multi_add(&nn2)
            }
            "-" n:number() { DiceExpression::constant(-(n as isize)) }
            "-" e:arithmetic() { e.negate() }
            n:number() { DiceExpression::constant(n as isize) }
            "d" n:number() { DiceExpression::dice(n) }
            "min(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.min(&b)).unwrap() }
            "max(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.max(&b)).unwrap() }
            "(" " "* e:arithmetic() " "* ")" { e }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    extern crate test;

    use test::Bencher;

    #[test]
    fn add() {
        let a = DiceExpression::from_str("1 * 2").unwrap();
        let b = DiceExpression::from_str("3 * 4").unwrap();
        let c = DiceExpression::from_str("1 * 2 + 3 * 4").unwrap();
        assert_eq!(a + &b, c);
    }

    #[test]
    fn sub() {
        let a = DiceExpression::from_str("1 * 2").unwrap();
        let b = DiceExpression::from_str("3 * 4").unwrap();
        let c = DiceExpression::from_str("1 * 2 - 3 * 4").unwrap();
        assert_eq!(a - &b, c);
    }

    #[test]
    fn mul() {
        let a = DiceExpression::from_str("1 * 2").unwrap();
        let b = DiceExpression::from_str("3 * 4").unwrap();
        let c = DiceExpression::from_str("(1 * 2) * (3 * 4)").unwrap();
        assert_eq!(a * &b, c);
    }

    #[test]
    fn max() {
        let a = DiceExpression::from_str("1 * 2").unwrap();
        let b = DiceExpression::from_str("3 * 4").unwrap();
        let c = DiceExpression::from_str("max(1 * 2, 3 * 4)").unwrap();
        assert_eq!(a.max(&b), c);
    }

    #[test]
    fn min() {
        let a = DiceExpression::from_str("1 * 2").unwrap();
        let b = DiceExpression::from_str("3 * 4").unwrap();
        let c = DiceExpression::from_str("min(1 * 2, 3 * 4)").unwrap();
        assert_eq!(a.min(&b), c);
    }

    #[test]
    fn precedence() {
        let a = DiceExpression::from_str("1").unwrap();
        let b = DiceExpression::from_str("2").unwrap();
        let c = DiceExpression::from_str("3").unwrap();
        let d = DiceExpression::from_str("4").unwrap();
        let e = DiceExpression::from_str("5").unwrap();
        let f = DiceExpression::from_str("(1 + 2 * 3 x 4) * 5").unwrap();
        assert_eq!((a + &(b * &(c.multi_add(&d)))) * &e, f);
    }

    #[test]
    fn wrong_parenthesis() {
        let wrong = [
            "( ) 4",
            "(4))",
            "(4 + ) 4",
            "d(1)",
            "(d1x)10",
            "d1(x10)",
        ];
        for s in wrong {
            let f = DiceExpression::from_str(s);
            assert!(f.is_err(), "{}", s);
        }
    }

    #[test]
    fn negative_repeat() {
        let goal = DiceExpression::from_str("d1x(-(d2))");
        assert!(goal.is_ok());
    }

    #[bench]
    fn parsing(b: &mut Bencher) {
        b.iter(|| {
            let yep: DiceExpression = "d30 + (d20xd30*d43423x(d20 + d4*d32 + 43))".parse().unwrap();
            test::black_box(yep);
        });
    }
}