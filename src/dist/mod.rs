//! Finite discrete probability distributions
//!
//! We have 3 different representations of distributions:
//! - [`DenseDist`], uses a [`Vec`] to store probabilities, can be slow when the
//!   distribution is sparse, e.g. `d2*10000`.
//! - [`SparseDist`], uses a [`HashMap`](std::collections::HashMap) to store
//!   probabilites.
//! - [`SortedDist`]. uses a [`Vec`] to store probabilites, in a sparse way.

use std::{
    cmp::Ordering,
    ops::{AddAssign, MulAssign, SubAssign},
};

use num::{FromPrimitive, Num};

mod dense;
mod sorted;
mod sparse;
pub use dense::DenseDist;
use rand::{
    distr::{
        uniform::SampleUniform,
        weighted::{Weight, WeightedIndex},
    },
    prelude::Distribution,
};
pub use sorted::SortedDist;
pub use sparse::SparseDist;

use crate::dices::Evaluator;

#[cfg(test)]
pub mod tests;

/// Methods for finite discrete probability distributions
pub trait Dist<'a, T>
where
    for<'b> T: 'a
        + Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
    Self: Sized,
{
    /// New uniform distribution from `min` to, and including, `max`.
    fn new_uniform(min: isize, max: isize) -> Self;

    /// New constant distribution, `n` has a 100% chance of occuring.
    fn new_constant(n: isize) -> Self;

    fn evaluator() -> impl Evaluator<Self>;

    /// Iteratete through the distribution's support, in order.
    fn iter_enumerate(&'a self) -> impl Iterator<Item = (isize, &'a T)>;

    fn min_value(&self) -> isize;
    fn max_value(&self) -> isize;

    /// The chance of outcome `n`. Returns `None` if `n` is not part of `self`,
    /// which means its chance is 0.
    fn chance(&'a self, n: isize) -> Option<&'a T> {
        self.iter_enumerate()
            .take_while(|x| x.0 <= n)
            .find_map(|x| if x.0 == n { Some(x.1) } else { None })
    }

    /// The expected value of the distribution.
    fn mean(&'a self) -> T {
        let mut out = T::zero();
        for (i, v) in self.iter_enumerate() {
            let mut thing = T::from_isize(i).unwrap();
            thing *= v;
            out += &thing;
        }
        out
    }

    /// The [variance](https://en.wikipedia.org/wiki/Variance) of the distribution
    fn variance(&'a self) -> T {
        let mean = self.mean();
        let mut total = T::zero();
        for (i, v) in self.iter_enumerate() {
            let mut v_i = T::from_isize(i).unwrap();
            v_i -= &mean;
            v_i *= v;
            total += &v_i;
        }
        total
    }

    /// The [median](https://en.wikipedia.org/wiki/Median#Probability_distributions) of the distribution; an `x` such that `P(X <= m) >= 0.5` and `P(m <= X) >= 0.5`.
    fn median(&'a self) -> isize {
        let half = T::from_f64(0.5).unwrap();
        let mut total = T::zero();
        for (i, v) in self.iter_enumerate() {
            total += v;
            if total >= half {
                return i;
            }
        }
        unreachable!()
    }

    /// The modes of the distribution, i.e. the values of the distribution with
    /// the highest chance of occuring.
    fn modes(&'a self) -> Vec<isize> {
        let mut out = Vec::new();
        let mut best: Option<&T> = None;
        for (i, v) in self.iter_enumerate() {
            match best {
                Some(x) if x < v => {
                    best = Some(v);
                    out.clear();
                    out.push(i)
                }
                Some(x) if x == v => out.push(i),
                Some(_) => {}
                None => out.push(i),
            }
        }
        out
    }

    fn distance(&'a self, other: &'a Self) -> T {
        let mut iter_a = self.iter_enumerate();
        let mut iter_b = other.iter_enumerate();
        let mut total: T = T::zero();
        let mut tmp: T = T::zero();
        'outer: loop {
            let mut current_a = iter_a.next();
            let mut current_b = iter_b.next();
            let (next_a, next_b) = loop {
                match (current_a, current_b) {
                    (None, _) | (_, None) => break 'outer,
                    (Some(a), Some(b)) => match a.0.cmp(&b.0) {
                        Ordering::Equal => break (a.1, b.1),
                        Ordering::Less => current_a = iter_a.next(),
                        Ordering::Greater => current_b = iter_b.next(),
                    },
                }
            };
            if next_a == next_b {
                continue;
            }
            tmp.set_zero();
            if next_a > next_b {
                tmp += next_a;
                tmp -= next_b;
            } else {
                tmp += next_b;
                tmp -= next_a;
            }
            total += &tmp;
        }
        total
    }

    fn distance_max(&'a self, other: &'a Self) -> T {
        let mut iter_a = self.iter_enumerate();
        let mut iter_b = other.iter_enumerate();
        let mut max: T = T::zero();
        let mut tmp: T = T::zero();
        'outer: loop {
            let mut current_a = iter_a.next();
            let mut current_b = iter_b.next();
            let (next_a, next_b) = loop {
                match (current_a, current_b) {
                    (None, _) | (_, None) => break 'outer,
                    (Some(a), Some(b)) => match a.0.cmp(&b.0) {
                        Ordering::Equal => break (a.1, b.1),
                        Ordering::Less => current_a = iter_a.next(),
                        Ordering::Greater => current_b = iter_b.next(),
                    },
                }
            };
            if next_a == next_b {
                continue;
            }
            tmp.set_zero();
            if next_a > next_b {
                tmp += next_a;
                tmp -= next_b;
            } else {
                tmp += next_b;
                tmp -= next_a;
            }
            if tmp > max {
                max = tmp.clone();
            }
        }
        max
    }
}

/// Trait to convert a [`Dist`] distribution to a [`Distribution`] (from
/// [`rand`]).
///
/// In contrast to `Dist`, this trait requires the probability type `T` to
/// implement [`SampleUniform`].
pub trait AsRand<'a, T>: Dist<'a, T>
where
    for<'b> T: 'a
        + Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + Weight
        + SampleUniform
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    /// Convert to a [`Distribution`]. Useful for sampling.
    #[must_use]
    fn to_rand_distribution(&'a self) -> impl Distribution<isize> {
        let mut values = Vec::new();
        let mut chances = Vec::new();
        for a in self.iter_enumerate() {
            values.push(a.0);
            chances.push(a.1);
        }
        WeightedIndex::new(chances).unwrap().map(move |x| values[x])
    }
}

// We don't do a blanket implementation of `AsRand` for `Dist`, because we want
// to specialize it and Rust doesn't support specialization yet.
