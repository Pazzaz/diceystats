use std::{collections::HashMap, mem, ops::{AddAssign, MulAssign, SubAssign}};

use num::{FromPrimitive, Num};

use crate::dices::Evaluator;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeirdDist<T> {
    values: Vec<(isize, T)>,
}

impl<T> WeirdDist<T> {
    fn new() -> Self {
        WeirdDist { values: Vec::new() }
    }

    fn iter(&self) -> impl Iterator<Item = &(isize, T)> {
        self.values.iter()
    }

    fn min_value(&self) -> isize {
        self.values.first().unwrap().0
    }

    fn max_value(&self) -> isize {
        self.values.last().unwrap().0
    }

    fn get_mut_or_insert(&mut self, i: isize, f: impl FnOnce() -> T) -> &mut T {
        let last = match self.values.last() {
            Some(x) => x,
            None => {
                self.values.push((i, f()));
                return &mut self.values[0].1;
            },
        };
        let last = self.values.last().unwrap();
        match i.cmp(&last.0) {
            std::cmp::Ordering::Greater => {
                self.values.push((i, f()));
                &mut self.values.last_mut().unwrap().1
            },
            std::cmp::Ordering::Equal => &mut self.values.last_mut().unwrap().1,
            std::cmp::Ordering::Less => {
                // TODO: Our binary guesses should be smarter
                match self.values.binary_search_by_key(&i, |x| x.0) {
                    Ok(x) => &mut self.values[x].1,
                    Err(u) => {
                        self.values.insert(u, (i, f()));
                        &mut self.values[u].1
                    },
                }
            },
        }
    }

    fn insert(&mut self, i: isize, v: T) {
        let last = match self.values.last_mut() {
            Some(x) => x,
            None => {
                self.values.push((i, v));
                return;
            },
        };
        match i.cmp(&last.0) {
            std::cmp::Ordering::Greater => {
                self.values.push((i, v));
            },
            std::cmp::Ordering::Equal => last.1 = v,
            std::cmp::Ordering::Less => {
                // TODO: Our binary guesses should be smarter
                match self.values.binary_search_by_key(&i, |x| x.0) {
                    Ok(x) => self.values[x].1 = v,
                    Err(u) => self.values.insert(u, (i, v)),
                }
            },
        }
    }
}

impl<T: Num + FromPrimitive + AddAssign + PartialOrd> WeirdDist<T>
where
    for<'a> T: MulAssign<&'a T> + SubAssign<&'a T>,
{
    pub fn mean(&self) -> T {
        let mut out = T::zero();
        for (a_k, a_v) in self.iter() {
            let mut thing = T::from_isize(*a_k).unwrap();
            thing *= a_v;
            out += thing;
        }
        out
    }
}

pub(crate) struct WeirdDistEvaluator;

impl<T: Num + Clone + AddAssign + FromPrimitive + std::fmt::Debug> Evaluator<WeirdDist<T>> for WeirdDistEvaluator
where
    for<'a> T: MulAssign<&'a T> + AddAssign<&'a T>,
{
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> WeirdDist<T> {
        let mut out = Vec::new();
        for i in 1..=d {
            out.push((i as isize, T::one() / T::from_usize(d).unwrap()));
        }
        WeirdDist {values: out}
    }

    fn constant(&mut self, n: isize) -> WeirdDist<T> {
        let mut out = Vec::new();
        out.push((n, T::one()));
        WeirdDist {values: out}
    }

    fn multi_add_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut a_values: Vec<(usize, T)> = a.values.drain(..).map(|x| {
            debug_assert!(x.0 >= 0);
            (x.0 as usize, x.1)
        }).collect();
        a_values.sort_by_key(|x| x.0);

        let mut out1: WeirdDist<T> = WeirdDist::new();
        let mut out2: WeirdDist<T> = WeirdDist::new();
        let mut out_final: WeirdDist<T> = WeirdDist::new();

        out2.insert(0, T::one());

        let mut previous: usize = 0;
        for (a_k, a_v) in a_values {
            debug_assert!(a_v != T::zero());

            let new_offset = a_k - previous;
            previous = a_k;

            for _ in 0..new_offset {
                mem::swap(&mut out1, &mut out2);
                out2.values.clear();
                for (b_k, b_v) in &b.values {
                    for (c_k, c_v) in &out1.values {
                        let mut tmp = b_v.clone();
                        tmp *= c_v;
                        let entry = out2.get_mut_or_insert(b_k + c_k, || T::zero());
                        *entry += tmp;
                    }
                }
            }

            for (c_k, c_v) in &out2.values {
                let mut tmp = a_v.clone();
                tmp *= c_v;
                let entry = out_final.get_mut_or_insert(*c_k, || T::zero());
                *entry += tmp;
            }
        }
        *a = out_final;
    }

    fn negate_inplace(&mut self, a: &mut WeirdDist<T>) {
        a.values = a.values.drain(..).map(|(k, v)| (-k, v)).collect();
        a.values.reverse();
    }

    fn add_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut out = WeirdDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let entry = out.get_mut_or_insert(a_k + b_k, || T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        *a = out;
    }

    fn mul_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut out = WeirdDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let entry = out.get_mut_or_insert(a_k * b_k, || T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        *a = out;
    }

    fn sub_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut out = WeirdDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let entry = out.get_mut_or_insert(a_k - b_k, || T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        *a = out;
    }

    fn max_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut out = WeirdDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let entry = out.get_mut_or_insert(*a_k.max(b_k), || T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        *a = out;
    }

    fn min_inplace(&mut self, a: &mut WeirdDist<T>, b: &WeirdDist<T>) {
        let mut out = WeirdDist::new();
        for (a_k, a_v) in a.values.iter() {
            for (b_k, b_v) in b.values.iter() {
                let entry = out.get_mut_or_insert(*a_k.min(b_k), || T::zero());
                let mut tmp = a_v.clone();
                tmp *= b_v;
                *entry += tmp;
            }
        }
        *a = out;
    }
}