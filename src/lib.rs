//! This crate is used to simulate dice rolls using [dice notation](https://en.wikipedia.org/wiki/Dice_notation), in the style of Dungeons and Dragons.
//! 
//! The main types are
//! - [DiceExpression], a sequence of interacting dice rolls
//! - [Dist], a discrete distribution of outcomes
//! 
//! # Usage
//! ```
//! use dicers::{DiceExpression, Dist};
//! use num::BigRational;
//! use rand::thread_rng;
//! 
//! // Roll a four side die and a five sided die, sum the result.
//! // Then roll that number of six sided dice, and sum those rolls.
//! let example = "(d4 + d5)xd6";
//! 
//! // We first parse it into a `DiceExpression`
//! let d4_d5_d6: DiceExpression = example.parse().unwrap();
//! 
//! // Then we can sample from it
//! let mut rng = thread_rng();
//! let result = d4_d5_d6.sample(&mut rng);
//! assert!(2 <= result && result <= 54);
//! 
//! // Or calculate the whole distribution
//! let dist: Dist<f64> = d4_d5_d6.dist();
//! assert!((dist.mean() - 19.25).abs() <= 0.01);
//! 
//! // If we want to be more precise we can use arbitrary precision numbers
//! let dist: Dist<BigRational> = d4_d5_d6.dist();
//! assert_eq!(dist.mean(), "77/4".parse().unwrap());
//! ```

#![feature(test)]
mod dist;
mod dice;
pub use dist::Dist;
pub use dice::DiceExpression;


#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use num::BigInt;
    use test::Bencher;
    use num::BigRational;

    #[bench]
    fn eval_6dx6d(b: &mut Bencher) {
        let yep: DiceExpression = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res: Dist<BigRational> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn f64_30dx30d(b: &mut Bencher) {
        let yep: DiceExpression = "d30xd30".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_additions(b: &mut Bencher) {
        let yep: DiceExpression = "d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_multiplications(b: &mut Bencher) {
        let yep: DiceExpression = "d20*d20*d20".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn parsing(b: &mut Bencher) {
        b.iter(|| {
            let yep: DiceExpression = "d30 + (d20xd30*d43423x(d20 + d4*d32 + 43))".parse().unwrap();
            test::black_box(yep);
        });
    }

    #[test]
    fn repeat_simple() {
        let yep: DiceExpression = "d9xd10".parse().unwrap();
        assert_eq!(
            yep.dist::<BigRational>().mean(),
            BigRational::new(BigInt::from(55), BigInt::from(2))
        );
    }

    #[test]
    fn rational_30dx30d() {
        let yep: DiceExpression = "d30xd30".parse().unwrap();
        assert_eq!(
            yep.dist::<BigRational>().mean(),
            BigRational::new(BigInt::from(961), BigInt::from(4))
        );
    }
}
