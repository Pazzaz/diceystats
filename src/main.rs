use dicers::{DiceExpression, Dist};

fn main() {
    let x: DiceExpression = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
    let res: Dist<f64> = x.dist();
}
