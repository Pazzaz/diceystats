use std::{collections::HashMap, ops::{Add, Mul, Sub}};

use num::BigRational;
use rand::{Rng, distributions::Uniform, prelude::Distribution, seq::SliceRandom};

use crate::{Dist, dices::InvalidNegative};

use super::{DiceExpression, Evaluator, Part};

fn random_none<R: Rng + ?Sized>(rng: &mut R, n: usize) -> Part {
    let choices = [Part::Const(n as isize), Part::Dice(n)];
    *choices.choose(rng).unwrap()
}

fn random_dual<R: Rng + ?Sized>(rng: &mut R, a: usize, b: usize) -> Part {
    let choices = [
        Part::Add(a, b),
        Part::Sub(a, b),
        Part::Mul(a, b),
        Part::Min(a, b),
        Part::Max(a, b),
        Part::MultiAdd(a, b),
    ];
    *choices.choose(rng).unwrap()
}

impl Distribution<isize> for DiceExpression {
    /// Evaluate the expression into a single number, rolling dice using `rng`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> isize {
        let mut e = SampleEvaluator { rng };
        self.traverse(&mut e)
    }
}

pub struct Config<'a> {
    pub height: usize,
    pub dice: &'a [usize],
    pub constants: &'a [isize],
    pub bounds: (isize, isize),
}

pub fn make_all(config: &Config) -> Vec<(Dist<BigRational>, DiceExpression)> {
    let mut expressions: HashMap<Dist<BigRational>, DiceExpression> = HashMap::default();
    let Config { height, dice, constants, bounds } = *config;
    for &i in dice {
        let d = DiceExpression::dice(i);
        let d_dist = d.dist::<BigRational>();
        expressions.entry(d_dist).or_insert(d);
    }
    for &i in constants {
        let c = DiceExpression::constant(i);
        let c_dist = c.dist::<BigRational>();
        expressions.entry(c_dist).or_insert(c);
    }
    for i in 0..height {
        let mut sorted: Vec<(DiceExpression, (isize, isize), Dist<BigRational>)> = expressions
            .drain()
            .map(|x| {
                let bounds = x.1.bounds();
                (x.1, bounds, x.0)
            })
            .collect();
        println!("{}", sorted.len());
        sorted.sort();
        let mut buffer: Vec<BigRational> = Vec::new();
        let mut s = InvalidNegative::new();
        macro_rules! add_if_bounded {
            ($f1:expr, $f2:expr, $f3:expr, $a_bounds:expr, $b_bounds:expr, $a_dist:expr, $b_dist:expr, $a:expr, $b:expr) => {
                let mut c_bounds = $a_bounds;
                $f1(&mut s, &mut c_bounds, $b_bounds);
                if i < (height - 1) || (bounds.0 <= c_bounds.0 && c_bounds.1 <= bounds.1) {
                    let mut c_dist = $a_dist.clone();
                    $f2(&mut c_dist, $b_dist, &mut buffer);
                    expressions.entry(c_dist).or_insert_with(|| $f3($a.clone(), $b).simplified());
                }
            };
        }
        // Commutative expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("A: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            for (b_i, (b, b_bounds, b_dist)) in sorted.iter().enumerate() {
                if b_i > a_i {
                    break;
                }
                add_if_bounded!(InvalidNegative::add_inplace, Dist::add_inplace, DiceExpression::add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(InvalidNegative::mul_inplace, Dist::mul_inplace, DiceExpression::mul, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(InvalidNegative::max_inplace, Dist::max_inplace, DiceExpression::max, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(InvalidNegative::min_inplace, Dist::min_inplace, DiceExpression::min, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
            }
        }
        // Non-commutative expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("B: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            for (b, b_bounds, b_dist) in &sorted {
                add_if_bounded!(InvalidNegative::sub_inplace, Dist::sub_inplace, DiceExpression::sub, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
            }
        }
        // Special expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("C: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            if a_bounds.1 < 0 {
                continue;
            }
            for (b, b_bounds, b_dist) in sorted.iter() {
                add_if_bounded!(InvalidNegative::multi_add_inplace, Dist::multi_add_inplace, DiceExpression::multi_add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
            }
        }
    }
    expressions.drain().collect()
}

impl DiceExpression {
    /// Create a random expression, modeleted as a tree with some `height` and maximum die / constant `value_size`.
    pub fn make_random<R: Rng + ?Sized>(rng: &mut R, height: usize, value_size: usize) -> Self {
        let dist = Uniform::new_inclusive(1, value_size);
        let bottom = 2usize.pow(height as u32);
        loop {
            let mut parts = Vec::new();
            for _ in 0..bottom {
                let a1 = dist.sample(rng);
                parts.push(random_none(rng, a1));
            }
            if height != 0 {
                let mut i = 0;
                for j in bottom.. {
                    parts.push(random_dual(rng, i, i + 1));
                    i += 2;
                    if i >= j {
                        break;
                    }
                }
            }
            let exp = DiceExpression { parts };
            if exp.could_be_negative() == 0 {
                return exp;
            }
        }
    }
}

struct SampleEvaluator<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
}

impl<R: Rng + ?Sized> Evaluator<isize> for SampleEvaluator<'_, R> {
    const LOSSY: bool = true;
    fn to_usize(x: isize) -> usize {
        x.try_into().unwrap_or_else(|_| panic!("Can't roll negative amount of dice"))
    }
    fn dice(&mut self, d: usize) -> isize {
        (self.rng.gen_range(1..=d)) as isize
    }

    fn constant(&mut self, n: isize) -> isize {
        n
    }

    fn negate_inplace(&mut self, a: &mut isize) {
        *a = -*a;
    }

    fn add_inplace(&mut self, a: &mut isize, b: &isize) {
        *a += b;
    }

    fn mul_inplace(&mut self, a: &mut isize, b: &isize) {
        *a *= b;
    }

    fn sub_inplace(&mut self, a: &mut isize, b: &isize) {
        *a -= b;
    }

    fn max_inplace(&mut self, a: &mut isize, b: &isize) {
        *a = isize::max(*a, *b);
    }

    fn min_inplace(&mut self, a: &mut isize, b: &isize) {
        *a = isize::min(*a, *b);
    }
}
