use std::{
    mem,
    ops::{AddAssign, MulAssign, SubAssign},
};

use num::{FromPrimitive, Num};
use rand::distr::{uniform::SampleUniform, weighted::Weight};

use crate::dices::Evaluator;

use super::{AsRand, Dist};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A discrete distribution of outcomes, stored sparsely in a sorted [`Vec`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct SortedDist<T> {
    values: Vec<(isize, T)>,
}

impl<T> SortedDist<T> {
    fn correct(&self) -> bool {
        for i in self.values.windows(2) {
            if i[0].0 >= i[1].0 {
                return false;
            }
        }
        true
    }

    fn new() -> Self {
        SortedDist { values: Vec::new() }
    }

    fn get_index(&self, i: isize) -> Result<usize, usize> {
        if self.values.is_empty() {
            Err(0)
        } else {
            let lower = self.values[0].0;
            let upper = self.values.last().unwrap().0;
            if i > upper {
                Err(self.values.len())
            } else if lower <= i && i <= upper {
                let end = ((i - lower) as usize).min(self.values.len() - 1);
                let start = (self.values.len() - 1).saturating_sub((upper - i) as usize);
                match self.values[start..=end].binary_search_by_key(&i, |x| x.0) {
                    Ok(x) => Ok(start + x),
                    Err(u) => Err(start + u),
                }
            } else if i < lower {
                Err(0)
            } else {
                unreachable!()
            }
        }
    }

    fn get_mut_or_insert(&mut self, i: isize) -> Result<&mut T, usize> {
        self.get_index(i).map(|x| &mut self.values[x].1)
    }

    fn insert(&mut self, i: isize, v: T) {
        match self.get_mut_or_insert(i) {
            Ok(x) => *x = v,
            Err(x) => self.values.insert(x, (i, v)),
        }
    }
}

impl<T> Dist<T> for SortedDist<T>
where
    for<'b> T: Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    fn evaluator() -> impl Evaluator<Self> {
        SortedDistEvaluator {}
    }

    fn iter_enumerate<'a>(&'a self) -> impl Iterator<Item = (isize, &'a T)>
    where
        T: 'a,
    {
        self.values.iter().map(|x| (x.0, &x.1))
    }

    fn min_value(&self) -> isize {
        self.values.first().unwrap().0
    }

    fn max_value(&self) -> isize {
        self.values.last().unwrap().0
    }

    fn chance(&self, n: isize) -> Option<&T> {
        self.get_index(n).map(|x| &self.values[x].1).ok()
    }

    fn new_uniform(min: isize, max: isize) -> Self {
        debug_assert!(min <= max);
        let values = (max - min + 1) as usize;
        let mut out = Vec::with_capacity(values);
        for i in min..=max {
            out.push((i, T::one() / T::from_usize(values).unwrap()));
        }
        let out = SortedDist { values: out };
        debug_assert!(out.correct());
        out
    }

    fn new_constant(n: isize) -> Self {
        let out = SortedDist { values: vec![(n, T::one())] };
        debug_assert!(out.correct());
        out
    }
}

impl<T> AsRand<T> for SortedDist<T> where
    for<'b> T: Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + Weight
        + SampleUniform
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>
{
}

impl<T> SortedDist<T>
where
    for<'a> T: Num + MulAssign<&'a T>,
{
    fn mul_constant(&mut self, c: isize) {
        if c == 0 {
            self.values.clear();
            self.values.push((0, T::one()));
        }
        for (a_k, _) in self.values.iter_mut() {
            *a_k *= c;
        }
        if c < 0 {
            self.values.reverse();
        }
    }
}

pub(crate) struct SortedDistEvaluator;

impl<T> Evaluator<SortedDist<T>> for SortedDistEvaluator
where
    for<'b> T: Num
        + FromPrimitive
        + PartialOrd
        + Clone
        + MulAssign<&'b T>
        + SubAssign<&'b T>
        + AddAssign<&'b T>,
{
    const CUSTOM_MULTI_ADD: bool = true;

    fn dice(&mut self, d: usize) -> SortedDist<T> {
        SortedDist::new_uniform(1, d as isize)
    }

    fn constant(&mut self, n: isize) -> SortedDist<T> {
        SortedDist::new_constant(n)
    }

    fn multi_add_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        let mut out1: SortedDist<T> = SortedDist::new();
        let mut out2: SortedDist<T> = SortedDist::new();
        let mut out_final: SortedDist<T> = SortedDist::new();

        out2.insert(0, T::one());

        let mut previous: usize = 0;
        for (a_k_i, a_v) in a.values.drain(..) {
            debug_assert!(a_k_i >= 0);
            let a_k = a_k_i as usize;

            let new_offset = a_k - previous;
            previous = a_k;

            for _ in 0..new_offset {
                mem::swap(&mut out1, &mut out2);
                out2.values.clear();
                for (b_k, b_v) in &b.values {
                    for (c_k, c_v) in &out1.values {
                        let mut tmp = b_v.clone();
                        tmp *= c_v;
                        match out2.get_mut_or_insert(b_k + c_k) {
                            Ok(entry) => {
                                *entry += &tmp;
                            }
                            Err(x) => out2.values.insert(x, (b_k + c_k, tmp)),
                        }
                    }
                }
            }
            if !a_v.is_zero() {
                for (c_k, c_v) in &out2.values {
                    let mut tmp = a_v.clone();
                    tmp *= c_v;
                    match out_final.get_mut_or_insert(*c_k) {
                        Ok(entry) => {
                            *entry += &tmp;
                        }
                        Err(x) => out_final.values.insert(x, (*c_k, tmp)),
                    }
                }
            }
        }
        debug_assert!(out_final.correct());
        *a = out_final;
    }

    fn negate_inplace(&mut self, a: &mut SortedDist<T>) {
        a.values = a.values.drain(..).map(|(k, v)| (-k, v)).collect();
        a.values.reverse();
        debug_assert!(a.correct());
    }

    fn add_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        let mut out = SortedDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let mut tmp = a_v.clone();
                tmp *= b_v;
                match out.get_mut_or_insert(a_k + b_k) {
                    Ok(entry) => {
                        *entry += &tmp;
                    }
                    Err(x) => out.values.insert(x, (a_k + b_k, tmp)),
                }
            }
        }
        *a = out;
        debug_assert!(a.correct());
    }

    fn mul_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        if b.values.len() == 1 {
            a.mul_constant(b.values[0].0);
        } else if a.values.len() == 1 {
            let a_k = a.values[0].0;
            *a = b.clone();
            a.mul_constant(a_k);
        } else {
            let mut out = SortedDist::new();
            for (a_k, a_v) in a.values.iter() {
                for (b_k, b_v) in b.values.iter() {
                    let mut tmp = a_v.clone();
                    tmp *= b_v;
                    match out.get_mut_or_insert(a_k * b_k) {
                        Ok(entry) => {
                            *entry += &tmp;
                        }
                        Err(x) => out.values.insert(x, (a_k * b_k, tmp)),
                    }
                }
            }
            *a = out;
            debug_assert!(a.correct());
        }
    }

    fn sub_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        let mut out = SortedDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let mut tmp = a_v.clone();
                tmp *= b_v;
                match out.get_mut_or_insert(a_k - b_k) {
                    Ok(entry) => {
                        *entry += &tmp;
                    }
                    Err(x) => out.values.insert(x, (a_k - b_k, tmp)),
                }
            }
        }
        *a = out;
        debug_assert!(a.correct());
    }

    // Same as `min_inplace` but uses `take_while`
    fn max_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        let mut out = SortedDist::new();
        let mut tmp = T::zero();
        let mut seen = 0;
        for (a_k, a_v) in a.values.iter() {
            for (_, b_v) in b.values.iter().skip(seen).take_while(|x| x.0 <= *a_k) {
                tmp += b_v;
                seen += 1;
            }
            if seen == 0 {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= a_v;
            match out.get_mut_or_insert(*a_k) {
                Ok(entry) => *entry += &tmp2,
                Err(x) => out.values.insert(x, (*a_k, tmp2)),
            }
        }
        tmp.set_zero();
        seen = 0;
        for (b_k, b_v) in b.values.iter() {
            for (_, a_v) in a.values.iter().skip(seen).take_while(|x| x.0 < *b_k) {
                tmp += a_v;
                seen += 1;
            }
            if seen == 0 {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= b_v;
            match out.get_mut_or_insert(*b_k) {
                Ok(entry) => *entry += &tmp2,
                Err(x) => out.values.insert(x, (*b_k, tmp2)),
            }
        }
        *a = out;
        debug_assert!(a.correct());
    }

    // Same as `max_inplace` but uses `skip_while`
    fn min_inplace(&mut self, a: &mut SortedDist<T>, b: &SortedDist<T>) {
        let sl = a.max_value();
        let ol = b.max_value();
        let max_value = isize::min(sl, ol);

        let mut out = SortedDist::new();
        let mut tmp = T::zero();
        tmp.set_one();
        let mut seen = 0;
        for (a_k, a_v) in a.values.iter() {
            if *a_k > max_value {
                break;
            }
            for (_, b_v) in b.values.iter().skip(seen).take_while(|x| x.0 <= *a_k) {
                tmp -= b_v;
                seen += 1;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= a_v;
            match out.get_mut_or_insert(*a_k) {
                Ok(entry) => *entry += &tmp2,
                Err(x) => out.values.insert(x, (*a_k, tmp2)),
            }
        }
        tmp.set_one();
        seen = 0;
        for (b_k, b_v) in b.values.iter() {
            if *b_k > max_value {
                break;
            }
            for (_, a_v) in a.values.iter().skip(seen).take_while(|x| x.0 < *b_k) {
                tmp -= a_v;
                seen += 1;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= b_v;
            match out.get_mut_or_insert(*b_k) {
                Ok(entry) => *entry += &tmp2,
                Err(x) => out.values.insert(x, (*b_k, tmp2)),
            }
        }
        *a = out;
        debug_assert!(a.correct());
    }
}
