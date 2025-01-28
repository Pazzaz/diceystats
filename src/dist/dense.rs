use core::panic;
use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use num::{FromPrimitive, Num};
use rand::{
    distr::{
        uniform::SampleUniform,
        weighted::{Weight, WeightedIndex},
    },
    prelude::Distribution,
};

use crate::dices::Evaluator;

use super::{AsRand, Dist};

/// A discrete distribution of outcomes, stored densely in a [`Vec`].
///
/// Probabilities have type `T`, e.g. [`f32`], [`f64`], `BigRational` etc.
/// All distributions have finite support, represented by a [`Vec<T>`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenseDist<T> {
    values: Vec<T>,
    offset: isize,
}

impl<T> DenseDist<T> {
    #[must_use]
    pub fn map<U, F: Fn(&T) -> U>(&self, f: F) -> DenseDist<U> {
        DenseDist { values: self.values.iter().map(f).collect(), offset: self.offset }
    }
}

impl<'a, T> AsRand<'a, T> for DenseDist<T>
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
    #[must_use]
    fn to_rand_distribution(&'a self) -> impl Distribution<isize> {
        WeightedIndex::new(&self.values).unwrap().map(move |x| x as isize + self.offset)
    }
}

impl<'a, T> Dist<'a, T> for DenseDist<T>
where
    for<'b> T: 'a
        + Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    fn evaluator() -> impl Evaluator<Self> {
        DistEvaluator { buffer: Vec::new() }
    }

    fn new_constant(n: isize) -> Self {
        let values = vec![T::one()];
        DenseDist { values, offset: n }
    }

    fn iter_enumerate(&self) -> impl Iterator<Item = (isize, &T)> {
        self.values.iter().enumerate().map(|(x_i, x)| (x_i as isize + self.offset, x))
    }

    fn min_value(&self) -> isize {
        self.offset
    }

    fn max_value(&self) -> isize {
        self.offset + (self.values.len() as isize) - 1
    }

    /// The chance that `n` will be sampled from the distribution. Returns
    /// `None` if outside the distribution's support.
    ///
    /// ```
    /// use diceystats::{
    ///     dices::DiceFormula,
    ///     dist::{DenseDist, Dist},
    /// };
    ///
    /// let expr: DiceFormula = "d10".parse().unwrap();
    /// let dist: DenseDist<f64> = expr.dist();
    /// let p = dist.chance(3).unwrap_or(&0.0);
    /// assert_eq!(1.0 / 10.0, *p);
    /// ```
    fn chance(&self, n: isize) -> Option<&T> {
        usize::try_from(n - self.offset).ok().and_then(|x| self.values.get(x))
    }

    /// Distance between two distributions, measured by total elementwise
    /// difference of probabilities.
    ///
    /// ```
    /// use diceystats::{
    ///     dices::DiceFormula,
    ///     dist::{DenseDist, Dist},
    /// };
    /// use num::BigRational;
    ///
    /// let expr1: DenseDist<_> = "d4xd3".parse::<DiceFormula>().unwrap().dist();
    /// let expr2: DenseDist<_> = "d4+d6".parse::<DiceFormula>().unwrap().dist();
    /// let d: BigRational = expr1.distance(&expr2);
    /// assert_eq!(d, "25/54".parse().unwrap())
    /// ```
    fn distance(&self, other: &DenseDist<T>) -> T {
        let mut d = T::zero();
        let a = self.min_value().min(other.min_value());
        let b = self.max_value().max(other.max_value());
        for i in a..b {
            let in_self = self.min_value() <= i && i <= self.max_value();
            let in_other = other.min_value() <= i && i <= other.max_value();
            match (in_self, in_other) {
                (true, true) => {
                    let i1 = self.chance(i).unwrap();
                    let i2 = other.chance(i).unwrap();
                    if i1 < i2 {
                        d += i2;
                        d -= i1;
                    } else if i2 < i1 {
                        d += i1;
                        d -= i2;
                    };
                }
                (true, false) => {
                    d += self.chance(i).unwrap();
                }
                (false, true) => {
                    d += other.chance(i).unwrap();
                }
                (false, false) => {}
            }
        }
        d
    }

    /// Distance between two distributions, measured by maximum elementwise
    /// difference of probabilities
    ///
    /// ```
    /// use diceystats::{
    ///     dices::DiceFormula,
    ///     dist::{DenseDist, Dist},
    /// };
    /// use num::BigRational;
    ///
    /// let expr1: DenseDist<_> = "d4xd3".parse::<DiceFormula>().unwrap().dist();
    /// let expr2: DenseDist<_> = "d4+d6".parse::<DiceFormula>().unwrap().dist();
    /// let d: BigRational = expr1.distance_max(&expr2);
    /// assert_eq!(d, "1/12".parse().unwrap())
    /// ```
    fn distance_max(&self, other: &DenseDist<T>) -> T {
        let mut d: T = T::zero();
        let a = self.min_value().min(other.min_value());
        let b = self.max_value().max(other.max_value());
        for i in a..b {
            match (&self.chance(i), &other.chance(i)) {
                (None, None) => {}
                (None, &Some(bb)) => {
                    if *bb > d {
                        d = bb.clone();
                    }
                }
                (&Some(aa), None) => {
                    if *aa > d {
                        d = aa.clone();
                    }
                }
                (&Some(aa), &Some(bb)) => match aa.partial_cmp(bb) {
                    Some(Ordering::Equal) => {}
                    Some(Ordering::Greater) => {
                        let mut res = aa.clone();
                        res -= bb;
                        if res > d {
                            d = res;
                        }
                    }
                    Some(Ordering::Less) => {
                        let mut res = bb.clone();
                        res -= aa;
                        if res > d {
                            d = res;
                        }
                    }
                    None => panic!("Non-comparable"),
                },
            }
        }
        d.clone()
    }

    fn new_uniform(min: isize, max: isize) -> Self {
        debug_assert!(min <= max);
        let choices = (max - min + 1) as usize;
        DenseDist {
            values: (min..=max).map(|_| T::one() / T::from_usize(choices).unwrap()).collect(),
            offset: min,
        }
    }
}

impl<T> DenseDist<T>
where
    for<'b> T: Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    fn op_inplace<F: Fn(isize, isize) -> isize>(
        &mut self,
        other: &DenseDist<T>,
        buffer: &mut Vec<T>,
        f: F,
    ) {
        debug_assert!(buffer.is_empty());

        let sl = self.max_value();
        let sm = self.min_value();
        let ol = other.max_value();
        let om = other.min_value();
        let max_value = f(sl, ol).max(f(sl, om)).max(f(sm, ol)).max(f(sm, om));
        let min_value = f(sl, ol).min(f(sl, om)).min(f(sm, ol)).min(f(sm, om));

        buffer.resize((max_value - min_value + 1) as usize, T::zero());
        for (a_i, a) in self.iter_enumerate().filter(|x| !x.1.is_zero()) {
            for (b_i, b) in other.iter_enumerate().filter(|x| !x.1.is_zero()) {
                let new_value = f(a_i, b_i);
                let mut res: T = a.clone();
                res.mul_assign(b);
                buffer[(new_value - min_value) as usize] += &res;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        self.offset = min_value;
        buffer.clear();
    }

    pub(crate) fn add_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::add);
    }

    pub(crate) fn mul_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::mul);
    }

    pub(crate) fn sub_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::sub);
    }

    pub(crate) fn max_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::max);
    }

    pub(crate) fn min_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::min);
    }

    pub(crate) fn negate_inplace(&mut self) {
        self.offset = -self.max_value();
        self.values.reverse();
    }

    pub(crate) fn multi_add_inplace(&mut self, other: &DenseDist<T>, buffer: &mut Vec<T>) {
        debug_assert!(buffer.is_empty());
        debug_assert!(0 <= self.offset);

        //  starting_case  min_value                         max_value
        //  - 0 +
        // [ ]             self.max_value * other.min_value  self.min_value *
        // other.max_value   [ ]           self.max_value * other.min_value
        // self.max_value * other.max_value     [ ]         self.min_value *
        // other.min_value  self.max_value * other.max_value

        let min_value = (self.max_value() * other.min_value())
            .min(self.max_value() * other.min_value())
            .min(self.min_value() * other.min_value());

        let max_value = (self.min_value() * other.max_value())
            .max(self.max_value() * other.max_value())
            .max(self.max_value() * other.max_value());

        if max_value == min_value {
            *self = DenseDist::new_constant(max_value);
            return;
        }

        let new_len = (max_value - min_value + 1) as usize;
        buffer.resize(new_len, T::zero());

        let min_value_tmp = 0.min(min_value).min(other.min_value());
        let max_value_tmp = 0.max(max_value).max(other.max_value());
        let tmp_len = (max_value_tmp - min_value_tmp + 1) as usize;

        // We have a second buffer which tracks the chance of getting X with "current_i"
        // iterations
        let mut source = vec![T::zero(); tmp_len];
        let mut dest = vec![T::zero(); tmp_len];

        // Empty sum is alsways zero
        dest[(-min_value_tmp) as usize] = T::one();

        // First we iterate through the self.offset
        for _ in 0..self.offset {
            source.swap_with_slice(&mut dest);

            for d in dest.iter_mut() {
                d.set_zero();
            }
            for (b_i, b) in other.values.iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
                let real_b_i = b_i as isize + other.offset;
                for (c_i, c) in source.iter().enumerate() {
                    if c.is_zero() {
                        continue;
                    }
                    let real_c_i = c_i as isize + min_value_tmp;
                    let real_d_i = real_b_i + real_c_i;
                    let d_i = (real_d_i - min_value_tmp) as usize;
                    let mut res = b.clone();
                    res *= c;
                    dest[d_i] += &res;
                }
            }
        }

        // We handle first (non-offset) case seperately
        for (d_i, d) in dest.iter().enumerate() {
            if d.is_zero() {
                continue;
            }
            let real_d_i = d_i as isize + min_value_tmp;
            let p_i = (real_d_i - min_value) as usize;
            let mut res = d.clone();
            res *= &self.values[0];
            buffer[p_i] = res;
        }

        // Then the rest
        for a in self.values.iter().skip(1) {
            source.swap_with_slice(&mut dest);

            for d in dest.iter_mut() {
                d.set_zero();
            }
            for (b_i, b) in other.values.iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
                let real_b_i = b_i as isize + other.offset;
                for (c_i, c) in source.iter().enumerate() {
                    if c.is_zero() {
                        continue;
                    }
                    let real_c_i = c_i as isize + min_value_tmp;
                    let real_d_i = real_b_i + real_c_i;
                    let d_i = (real_d_i - min_value_tmp) as usize;
                    let mut res = b.clone();
                    res *= c;
                    dest[d_i] += &res;
                }
            }
            if a.is_zero() {
                continue;
            }

            for (d_i, d) in dest.iter().enumerate() {
                if d.is_zero() {
                    continue;
                }
                let real_d_i = d_i as isize + min_value_tmp;
                let mut res = d.clone();
                res *= a;
                let p_i = (real_d_i - min_value) as usize;
                buffer[p_i] += &res;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        self.offset = min_value;
        buffer.clear();
    }
}

pub(crate) struct DistEvaluator<T> {
    pub(crate) buffer: Vec<T>,
}

impl<T> Evaluator<DenseDist<T>> for DistEvaluator<T>
where
    for<'b> T: Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> DenseDist<T> {
        DenseDist::new_uniform(1, d as isize)
    }

    fn constant(&mut self, n: isize) -> DenseDist<T> {
        DenseDist::new_constant(n)
    }

    fn multi_add_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.multi_add_inplace(b, &mut self.buffer);
    }

    fn negate_inplace(&mut self, a: &mut DenseDist<T>) {
        a.negate_inplace();
    }

    fn add_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.add_inplace(b, &mut self.buffer);
    }

    fn mul_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.mul_inplace(b, &mut self.buffer);
    }

    fn sub_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.sub_inplace(b, &mut self.buffer);
    }

    fn max_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.max_inplace(b, &mut self.buffer);
    }

    fn min_inplace(&mut self, a: &mut DenseDist<T>, b: &DenseDist<T>) {
        a.min_inplace(b, &mut self.buffer);
    }
}
