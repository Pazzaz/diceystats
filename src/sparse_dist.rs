use std::{collections::HashMap, mem, ops::{AddAssign, MulAssign, SubAssign}};

use num::{FromPrimitive, Num};

use crate::dices::Evaluator;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseDist<T> {
    values: HashMap<isize, T>,
}

impl<T: Num + FromPrimitive + AddAssign + PartialOrd> SparseDist<T>
where
    for<'a> T: MulAssign<&'a T> + SubAssign<&'a T>,
{
    pub fn mean(&self) -> T {
        let mut out = T::zero();
        for (&a_k, a_v) in self.values.iter() {
            let mut thing = T::from_isize(a_k).unwrap();
            thing *= a_v;
            out += thing;
        }
        out
    }
}

pub(crate) struct SparseDistEvaluator;

impl<T: Num + Clone + AddAssign + FromPrimitive + std::fmt::Debug> Evaluator<SparseDist<T>> for SparseDistEvaluator
where
    for<'a> T: MulAssign<&'a T> + AddAssign<&'a T>,
{
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> SparseDist<T> {
        let mut out = HashMap::with_capacity(d);
        for i in 1..=d {
            out.insert(i as isize, T::one() / T::from_usize(d).unwrap());
        }
        SparseDist {values: out}
    }

    fn constant(&mut self, n: isize) -> SparseDist<T> {
        let mut out = HashMap::with_capacity(1);
            out.insert(n, T::one());
        SparseDist {values: out}
    }

    fn multi_add_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut a_values: Vec<(usize, T)> = a.values.drain().map(|x| {
            debug_assert!(x.0 >= 0);
            (x.0 as usize, x.1)
        }).collect();
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
                        *entry += tmp;
                    }
                }
            }

            for (&c_k, c_v) in &out2 {
                let mut tmp = a_v.clone();
                tmp *= c_v;
                let entry = out_final.entry(c_k).or_insert(T::zero());
                *entry += tmp;
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
                *entry += tmp;
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
                *entry += tmp;
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
                *entry += tmp;
            }
        }
        a.values = out;
    }

    fn max_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut out = HashMap::with_capacity(a_len.max(b_len));
        for (&a_k, a_v) in a.values.iter() {
            for (&b_k, b_v) in b.values.iter() {
                let entry = out.entry(a_k.max(b_k)).or_insert(T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        a.values = out;
    }

    fn min_inplace(&mut self, a: &mut SparseDist<T>, b: &SparseDist<T>) {
        let a_len = a.values.len();
        let b_len = b.values.len();
        let mut out = HashMap::with_capacity(a_len.min(b_len));
        for (&a_k, a_v) in a.values.iter() {
            for (&b_k, b_v) in b.values.iter() {
                let entry = out.entry(a_k.min(b_k)).or_insert(T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        a.values = out;
    }
}