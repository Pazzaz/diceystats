use dicers::{DiceExpression, Dist};
use num::BigRational;

fn main() {
    let test: DiceExpression = "d20".parse().unwrap();
    let p: Dist<BigRational> = test.evaluate();
    debug_assert_eq!("21/2".parse::<BigRational>().unwrap(), p.mean());
    println!("{}", p.mean());
}