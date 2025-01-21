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
            "-" e:arithmetic() { e.negate() }
            n:number() { DiceExpression::constant(n as isize) }
            "d" n:number() { DiceExpression::dice(n) }
            "min(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.min(&b)).unwrap() }
            "max(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.max(&b)).unwrap() }
            "(" " "* e:arithmetic() " "* ")" { e }
        }
    }
}
