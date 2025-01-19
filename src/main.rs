use dicers::{DiceExpression, Dist};
use num::BigRational;
use rand::thread_rng;

fn main() {
    let x: DiceExpression = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
    assert_eq!(x.to_string(), "(d5 + d20xd5) * max(d4 * d4, d5, d10)x(d4 * d8)")
}