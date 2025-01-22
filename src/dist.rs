use core::panic;
use std::{
    cmp::Ordering, fmt::Debug, ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign}
};

use num::{FromPrimitive, Num};
use rand::{
    distributions::{WeightedIndex, uniform::SampleUniform},
    prelude::Distribution,
};

/// A discrete distribution of outcomes.
///
/// Probabilities have type `T`, e.g. [`f32`], [`f64`], `BigRational` etc.
/// All distributions have finite support, represented by a [`Vec<T>`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dist<T> {
    values: Vec<T>,
    offset: isize,
}

impl<T: SampleUniform + PartialOrd + Clone + Default> Dist<T>
where
    for<'a> T: AddAssign<&'a T>,
{
    /// Convert to a [`Distribution`]. Useful for sampling.
    pub fn to_distribution(self) -> impl Distribution<isize> {
        WeightedIndex::new(self.values).unwrap().map(move |x| x as isize + self.offset)
    }
}

impl<T: num::Zero + PartialOrd + Debug + Clone> Dist<T>
where
    for<'a> T: AddAssign<&'a T> + SubAssign<&'a T>,
{
    /// Distance between two distributions, measured by total elementwise difference of probabilities
    /// ```
    /// use diceystats::{DiceExpression, Dist};
    /// use num::BigRational;
    ///
    /// let expr1 = "d4xd3".parse::<DiceExpression>().unwrap();
    /// let expr2 = "d4+d6".parse::<DiceExpression>().unwrap();
    /// let d: BigRational = expr1.dist().distance(&expr2.dist());
    /// assert_eq!(d, "25/54".parse().unwrap())
    /// ```
    pub fn distance(&self, other: &Dist<T>) -> T {
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

    /// Distance between two distributions, measured by maximum elementwise difference of probabilities
    /// ```
    /// use diceystats::{DiceExpression, Dist};
    /// use num::BigRational;
    ///
    /// let expr1 = "d4xd3".parse::<DiceExpression>().unwrap();
    /// let expr2 = "d4+d6".parse::<DiceExpression>().unwrap();
    /// let d: BigRational = expr1.dist().distance_max(&expr2.dist());
    /// assert_eq!(d, "1/12".parse().unwrap())
    /// ```
    pub fn distance_max(&self, other: &Dist<T>) -> T {
        let mut d: T = T::zero();
        let a = self.min_value().min(other.min_value());
        let b = self.max_value().max(other.max_value());
        for i in a..b {
            match (&self.chance(i), &other.chance(i)) {
                (None, None) => {},
                (None, &Some(bb)) => {
                    if *bb > d {
                        d = bb.clone();
                    }
                },
                (&Some(aa), None) => {
                    if *aa > d {
                        d = aa.clone();
                    }
                },
                (&Some(aa), &Some(bb)) => {
                    match aa.partial_cmp(&bb) {
                        Some(Ordering::Equal) => {}
                        Some(Ordering::Greater) => {
                            let mut res = aa.clone();
                            res -= bb;
                            if res > d {
                                d = res;
                            }
                        },
                        Some(Ordering::Less) => {
                            let mut res = bb.clone();
                            res -= aa;
                            if res > d {
                                d = res;
                            }
                        },
                        None => panic!("Non-comparable"),
                    }
                },
            }
        }
        d.clone()
    }
}

impl<T> Dist<T> {
    /// The minimum value in the distributions support.
    pub fn min_value(&self) -> isize {
        self.offset
    }

    /// The maximum value in the distributions support.
    pub fn max_value(&self) -> isize {
        self.offset + (self.values.len() as isize) - 1
    }

    /// Iterate through the distributions support, with each outcomes probability.
    pub fn iter_enumerate(&self) -> impl Iterator<Item = (isize, &T)> {
        self.values.iter().enumerate().map(|(x_i, x)| (x_i as isize + self.offset, x))
    }

    /// The chance that `n` will be sampled from the distribution. Returns `None` if ouside the distributions support.
    /// ```
    /// use diceystats::{Dist, DiceExpression};
    ///
    /// let expr: DiceExpression = "d10".parse().unwrap();
    /// let dist: Dist<f64> = expr.dist();
    /// let p = dist.chance(3).unwrap_or(&0.0);
    /// assert_eq!(1.0 / 10.0, *p);
    /// ```
    pub fn chance(&self, n: isize) -> Option<&T> {
        usize::try_from(n - self.offset).ok().and_then(|x| self.values.get(x))
    }

    pub(crate) fn negate_inplace(&mut self) {
        self.offset = -self.max_value();
        self.values.reverse();
    }
}

impl<T: Num + FromPrimitive> Dist<T> {
    pub(crate) fn uniform(min: isize, max: isize) -> Self {
        debug_assert!(min <= max);
        let choices = (max - min + 1) as usize;
        Dist {
            values: (min..=max).map(|_| T::one() / T::from_usize(choices).unwrap()).collect(),
            offset: min,
        }
    }
}

impl<T: Num + Clone> Dist<T> {
    pub(crate) fn constant(n: isize) -> Self {
        let values = vec![T::one()];

        Dist { values, offset: n }
    }
}

impl<T: Num + FromPrimitive + AddAssign> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    pub fn mean(&self) -> T {
        let mut out = T::zero();
        for (i, v) in self.values.iter().enumerate() {
            let mut thing = T::from_usize(i).unwrap();
            thing *= v;
            out += thing;
        }
        T::from_isize(self.offset).unwrap() + out
    }
}

impl<T: Num + Clone + AddAssign + std::fmt::Debug> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn op_inplace<F: Fn(isize, isize) -> isize>(
        &mut self,
        other: &Dist<T>,
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
                buffer[(new_value - min_value) as usize] += res;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        self.offset = min_value;
        buffer.clear();
    }

    pub(crate) fn add_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::add);
    }

    pub(crate) fn mul_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::mul);
    }

    pub(crate) fn sub_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::sub);
    }

    pub(crate) fn max_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::max);
    }

    pub(crate) fn min_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, isize::min);
    }
}

impl<T: Num + Clone + AddAssign + std::fmt::Debug> Dist<T>
where
    for<'a> T: MulAssign<&'a T> + AddAssign<&'a T>,
{
    pub(crate) fn repeat(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        debug_assert!(buffer.is_empty());
        debug_assert!(0 <= self.offset);

        //  starting_case  min_value                         max_value
        //  - 0 +
        // [ ]             self.max_value * other.min_value  self.min_value * other.max_value
        //   [ ]           self.max_value * other.min_value  self.max_value * other.max_value
        //     [ ]         self.min_value * other.min_value  self.max_value * other.max_value

        let min_value = (self.max_value() * other.min_value())
            .min(self.max_value() * other.min_value())
            .min(self.min_value() * other.min_value());

        let max_value = (self.min_value() * other.max_value())
            .max(self.max_value() * other.max_value())
            .max(self.max_value() * other.max_value());

        if max_value == min_value {
            *self = Dist::constant(max_value);
            return;
        }

        let new_len = (max_value - min_value + 1) as usize;
        buffer.resize(new_len, T::zero());

        let min_value_tmp = 0.min(min_value).min(other.min_value());
        let max_value_tmp = 0.max(max_value).max(other.max_value());
        let tmp_len = (max_value_tmp - min_value_tmp + 1) as usize;

        // We have a second buffer which tracks the chance of getting X with "current_i" iterations
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
                    dest[d_i] += res;
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
                    dest[d_i] += res;
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
                buffer[p_i] += res;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        self.offset = min_value;
        buffer.clear();
    }
}
