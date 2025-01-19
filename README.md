# Dicers
This crate is used to simulate dice rolls using [dice notation](https://en.wikipedia.org/wiki/Dice_notation), in the style of Dungeons and Dragons.


## Usage
```rust
use dicers::{DiceExpression, Dist};
use num::BigRational;
use rand::thread_rng;

fn main() {
    // Roll a four side die and a five sided die, sum the result.
    // Then roll that number of six sided dice, and sum those rolls.
    let example = "(d4 + d5)xd6";

    // We first parse it into a `DiceExpression`
    let d4_d5_d6: DiceExpression = example.parse().unwrap();

    // Then we can sample from it
    let mut rng = thread_rng();
    let result = d4_d5_d6.sample(&mut rng);
    assert!(2 <= result && result <= 54);

    // Or calculate the whole distribution
    let dist: Dist<f64> = d4_d5_d6.dist();
    assert!((dist.mean() - 19.25).abs() <= 0.01);

    // If we want to be more precise we can use arbitrary precision numbers
    let dist: Dist<BigRational> = d4_d5_d6.dist();
    assert_eq!(dist.mean(), "77/4".parse().unwrap());
}
```

## Supported operations
### Addition
### Subtraction
### Multiplication
### Multiaddition