//! This crate is used to simulate dice rolls using [dice notation](https://en.wikipedia.org/wiki/Dice_notation), in the style of Dungeons and Dragons.
//!
//! The main types are
//! - [DiceExpression], a sequence of interacting dice rolls
//! - [Dist], a discrete distribution of outcomes
//!
//! # Usage
//! ```
//! use diceystats::{DiceExpression, Dist, roll};
//! use num::BigRational;
//! use rand::thread_rng;
//!
//! // Roll a four side die and a five sided die, sum the result.
//! // Then roll that number of six sided dice, and sum those rolls.
//! let example = "(d4 + d5)xd6";
//!
//! // We can roll the dice and calculate the result
//! let result = roll(example, &mut thread_rng()).unwrap();
//! assert!(2 <= result && result <= 54);
//!
//! // Or to do more advanced things we can parse it into a `DiceExpression`
//! let d4_d5_d6: DiceExpression = example.parse().unwrap();
//!
//! // Then we can calculate its probability distribution
//! let dist: Dist<f64> = d4_d5_d6.dist();
//! assert!((dist.mean() - 19.25).abs() <= 0.01);
//!
//! // If we want to be more precise we can use arbitrary precision numbers
//! let dist: Dist<BigRational> = d4_d5_d6.dist();
//! assert_eq!(dist.mean(), "77/4".parse().unwrap());
//! ```
//!
//! The library also supports "negative dice", but the library will fail if there's a chance of rolling a negative *amount* of dice
//! ```
//! use diceystats::{DiceExpression, Dist, roll};
//! use rand::thread_rng;
//!
//! roll("(d4 - d4)xd20", &mut thread_rng()).is_err();
//! ```

#![feature(test)]
mod dices;
mod dist;
use std::str::FromStr;

pub use dices::DiceExpression;
use dices::parse::DiceParseError;
pub use dist::Dist;
use rand::Rng;

/// Roll a set of dice.
///
/// Returns `None` if the expression is invalid.
/// ```
/// use rand::thread_rng;
///
/// let x = diceystats::roll("d10 + d5", &mut thread_rng());
/// ```
pub fn roll<R: Rng + ?Sized>(s: &str, rng: &mut R) -> Result<isize, DiceParseError> {
    DiceExpression::from_str(s).map(|x| x.sample(rng))
}
