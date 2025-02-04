use std::{
    collections::HashMap, mem, ops::{AddAssign, MulAssign, SubAssign}
};

use num::{FromPrimitive, Num};
use rand::distr::{uniform::SampleUniform, weighted::Weight};

use crate::dices::Evaluator;

use super::{AsRand, Dist};

/// A discrete distribution of outcomes, stored sparsely in a
/// [`HashMap`](std::collections::HashMap).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseDist<T> {
    values: HashMap<isize, T>,
}

impl<'a, T: 'a + Num + FromPrimitive + PartialOrd + Clone> Dist<'a, T> for SparseDist<T>
where
    for<'b> T: MulAssign<&'b T> + SubAssign<&'b T> + AddAssign<&'b T>,
{
    fn evaluator() -> impl Evaluator<Self> {
        SparseDistEvaluator {}
    }

    fn iter_enumerate(&self) -> impl Iterator<Item = (isize, &T)> {
        let mut values: Vec<(isize, &T)> = self.values.iter().map(|x| (*x.0, x.1)).collect();
        values.sort_by_key(|x| x.0);
        values.into_iter()
    }

    fn min_value(&self) -> isize {
        *self.values.keys().min().unwrap()
    }

    fn max_value(&self) -> isize {
        *self.values.keys().max().unwrap()
    }

    fn chance(&'a self, n: isize) -> Option<&'a T> {
        self.values.get(&n)
    }

    fn new_uniform(min: isize, max: isize) -> Self {
        debug_assert!(min <= max);
        let values = (max - min + 1) as usize;
        let mut out = HashMap::with_capacity(values);
        for i in min..=max {
            out.insert(i, T::one() / T::from_usize(values).unwrap());
        }
        SparseDist { values: out }
    }

    fn new_constant(n: isize) -> Self {
        let mut out = HashMap::with_capacity(1);
        out.insert(n, T::one());
        SparseDist { values: out }
    }
}

impl<'a, T: 'a + Num + FromPrimitive + PartialOrd + Clone + Weight + SampleUniform> AsRand<'a, T>
    for SparseDist<T>
where
    for<'b> T: MulAssign<&'b T> + SubAssign<&'b T> + AddAssign<&'b T>,
{
}

pub(crate) struct SparseDistEvaluator;
impl<T: Num + FromPrimitive + PartialOrd + Clone> Evaluator<SparseDist<T>> for SparseDistEvaluator
where
    for<'b> T: MulAssign<&'b T> + SubAssign<&'b T> + AddAssign<&'b T>,
{
    const CUSTOM_MULTI_ADD: bool = true;

    fn dice(&mut self, d: usize) -> SparseDist<T> {
        SparseDist::new_uniform(1, d as isize)
    }

    fn constant(&mut self, n: isize) -> SparseDist<T> {
        SparseDist::new_constant(n)
    }

    fn multi_add_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut a_values: Vec<(usize, T)> = a
            .values
            .drain()
            .map(|x| {
                debug_assert!(x.0 >= 0);
                (x.0 as usize, x.1)
            })
            .collect();
        a_values.sort_by_key(|x| x.0);

        let mut out1: HashMap<isize, T> = HashMap::with_capacity(a_len * b_len);
        let mut out2: HashMap<isize, T> = HashMap::with_capacity(a_len * b_len);
        let mut out_final: HashMap<isize, T> = HashMap::with_capacity(a_len * b_len);

        out2.insert(0, T::one());

        let mut previous: usize = 0;
        for (a_k, a_v) in a_values {
            debug_assert!(a_v != T::zero());

            let new_offset = a_k - previous;
            previous = a_k;

            for _ in 0..new_offset {
                mem::swap(&mut out1, &mut out2);
                out2.clear();
                for (&b_k, b_v) in &b.values {
                    for (&c_k, c_v) in &out1 {
                        let mut tmp = b_v.clone();
                        tmp *= c_v;
                        let entry = out2.entry(b_k + c_k).or_insert(T::zero());
                        *entry += &tmp;
                    }
                }
            }

            for (&c_k, c_v) in &out2 {
                let mut tmp = a_v.clone();
                tmp *= c_v;
                let entry = out_final.entry(c_k).or_insert(T::zero());
                *entry += &tmp;
            }
        }
        a.values = out_final;
    }

    fn negate_inplace(&mut self, a: &mut SparseDist<T>) {
        a.values = a.values.drain().map(|(k, v)| (-k, v)).collect();
    }

    fn add_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut out = HashMap::with_capacity(a_len + b_len);
        for (&a_k, a_v) in a.values.iter() {
            for (&b_k, b_v) in b.values.iter() {
                let entry = out.entry(a_k + b_k).or_insert(T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += &tmp;
            }
        }
        a.values = out;
    }

    fn mul_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut out = HashMap::with_capacity(a_len * b_len);
        for (&a_k, a_v) in a.values.iter() {
            for (&b_k, b_v) in b.values.iter() {
                let entry = out.entry(a_k * b_k).or_insert(T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += &tmp;
            }
        }
        a.values = out;
    }

    fn sub_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut out = HashMap::with_capacity(a_len + b_len);
        for (&a_k, a_v) in a.values.iter() {
            for (&b_k, b_v) in b.values.iter() {
                let entry = out.entry(a_k - b_k).or_insert(T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += &tmp;
            }
        }
        a.values = out;
    }

    fn max_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let a_sorted: Vec<(isize, &T)> = a.iter_enumerate().collect();
        let b_sorted: Vec<(isize, &T)> = b.iter_enumerate().collect();
        let mut out = HashMap::with_capacity(a_len.min(b_len));
        let mut tmp = T::zero();
        let mut seen = 0;
        for &(a_k, a_v) in &a_sorted {
            if a_v.is_zero() {
                continue;
            }
            for (_, b_v) in b_sorted.iter().skip(seen).take_while(|x| x.0 <= a_k) {
                tmp += b_v;
                seen += 1;
            }
            if seen == 0 {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= a_v;

            let entry = out.entry(a_k).or_insert(T::zero());
            *entry += &tmp2;
        }
        tmp.set_zero();
        seen = 0;
        for &(b_k, b_v) in &b_sorted {
            if b_v.is_zero() {
                continue;
            }
            for (_, a_v) in a_sorted.iter().skip(seen).take_while(|x| x.0 < b_k) {
                tmp += a_v;
                seen += 1;
            }
            if seen == 0 {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= b_v;

            let entry = out.entry(b_k).or_insert(T::zero());
            *entry += &tmp2;
        }
        a.values = out;
    }

    fn min_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let a_sorted: Vec<(isize, &T)> = a.iter_enumerate().collect();
        let b_sorted: Vec<(isize, &T)> = b.iter_enumerate().collect();
        let max_value = isize::min(a_sorted.last().unwrap().0, b_sorted.last().unwrap().0);
        let mut out = HashMap::with_capacity(a_len.min(b_len));
        let mut tmp = T::one();
        let mut seen = 0;
        for &(a_k, a_v) in &a_sorted {
            if a_k > max_value || a_v.is_zero() {
                continue;
            }
            for (_, b_v) in b_sorted.iter().skip(seen).take_while(|x| x.0 <= a_k) {
                tmp -= b_v;
                seen += 1;
            }
            if tmp.is_zero() {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= a_v;

            let entry = out.entry(a_k).or_insert(T::zero());
            *entry += &tmp2;
        }
        tmp.set_one();
        seen = 0;
        for &(b_k, b_v) in &b_sorted {
            if b_k > max_value || b_v.is_zero() {
                continue;
            }
            for (_, a_v) in a_sorted.iter().skip(seen).take_while(|x| x.0 < b_k) {
                tmp -= a_v;
                seen += 1;
            }
            if tmp.is_zero() {
                continue;
            }
            let mut tmp2 = tmp.clone();
            tmp2 *= b_v;

            let entry = out.entry(b_k).or_insert(T::zero());
            *entry += &tmp2;
        }
        a.values = out;
    }
}
