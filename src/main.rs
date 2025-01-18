use dicers::{DiceExpression, Dist};
use num::BigRational;
use rand::thread_rng;

fn main() {
    let test: DiceExpression = "d20x(d7+d9)".parse().unwrap();
    let mut rng = thread_rng();
    let mean = (0..100000).map(|_| test.sample(&mut rng) as f64).sum::<f64>() / 100000.0;
    println!("{}", mean);
    println!("{}", test.dist::<f64>().mean());
}