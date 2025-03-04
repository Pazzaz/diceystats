//! This crate is used to sample from and analyze dice formulas which use [dice
//! notation](https://en.wikipedia.org/wiki/Dice_notation), in the style of
//! Dungeons and Dragons. It can parse formulas as strings, sample from
//! formulas, analayze the distribution of formulas, simplify formulas, randomly
//! generate formulas, exhaustavely generate classes of formulas, etc.
//!
//! # Usage
//!
//! ```
//! use diceystats::{
//!     dices::DiceFormula,
//!     dist::{DenseDist, Dist},
//!     roll,
//! };
//!
//! // Roll a four side die and a five sided die, sum the result.
//! // Then roll that number of six sided dice, and sum those rolls.
//! let example = "(d4 + d5)xd6";
//!
//! // We can roll the dice and calculate the result
//! let result: isize = roll(example, &mut rand::rng()).unwrap();
//! assert!((2..=54).contains(&result));
//!
//! // Or to do more advanced things we can parse it into a `DiceFormula`
//! let d4_d5_d6: DiceFormula = example.parse().unwrap();
//!
//! // Then we can calculate its probability distribution
//! let dist: DenseDist<f64> = d4_d5_d6.dist();
//! assert!((dist.mean() - 19.25).abs() <= 0.01);
//!
//! // If we want to be more precise we can use arbitrary precision numbers
//! use num::BigRational;
//!
//! let dist: DenseDist<BigRational> = d4_d5_d6.dist();
//! assert_eq!(dist.mean().to_string(), "77/4");
//! ```
//!
//! There are multiple ways to represent distributions which are useful in
//! different situations, e.g. [SparseDist](dist::SparseDist) will attempt to
//! only store non-zero values in the distribution's support.
//!
//! ```
//! # use diceystats::{
//! #     dices::DiceFormula,
//! #     dist::{Dist, DenseDist},
//! # };
//! # use num::BigRational;
//! # let d4_d5_d6: DiceFormula = "(d4 + d5)xd6".parse().unwrap();
//! use diceystats::dist::SparseDist;
//!
//! let sparse_dist: SparseDist<BigRational> = d4_d5_d6.dist();
//! assert!(sparse_dist.chance(1).is_none());
//! ```
//!
//! The library also supports "negative dice", but will detect if
//! there's a chance of rolling a negative *amount* of dice
//!
//! ```
//! use diceystats::{dices::DiceParseError, roll};
//!
//! let result = roll("(d4 - d4)xd20", &mut rand::rng());
//! assert_eq!(result, Err(DiceParseError::NegativeRolls));
//! ```
//!
//! # Performance
//! An attempt has been made to make the library reasonably fast, though
//! the use of generics limits how much we can overoptimize. Calculating
//! probability distributions, the most expensive operation, can be done in
//! multiple ways with different performance implications. See [dist] for more
//! information.

#![feature(test)]

pub mod dices;
pub mod dist;
use std::str::FromStr;

use dices::{DiceFormula, DiceParseError};
use rand::{Rng, prelude::Distribution};

/// Roll dice and evaluate a dice formula.
///
/// Returns `None` if the expression is invalid.
///
/// ```
/// let x: isize = diceystats::roll("d10 + d5", &mut rand::rng()).unwrap();
/// ```
pub fn roll<R: Rng + ?Sized>(s: &str, rng: &mut R) -> Result<isize, DiceParseError> {
    DiceFormula::from_str(s).map(|x| x.sample(rng))
}
