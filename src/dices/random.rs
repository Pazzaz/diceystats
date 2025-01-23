use std::collections::HashMap;

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
        self.evaluate_generic(&mut e)
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
    let Config { height, dice, constants, bounds: (min_value, max_value) } = *config;
    for &i in dice {
        // let c = DiceExpression::constant(i as isize);
        // let c_dist = c.dist::<BigRational>();
        // expressions.entry(c_dist).or_insert(c);
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
        // Commutative expressions
        let mut buffer: Vec<BigRational> = Vec::new();
        let mut s = InvalidNegative::new();
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("A: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            for (b_i, (b, b_bounds, b_dist)) in sorted.iter().enumerate() {
                if b_i > a_i {
                    break;
                }
                {
                    let mut c_bounds = a_bounds.clone();
                    s.add_inplace(&mut c_bounds, &b_bounds);
                    if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                        let mut c_dist = a_dist.clone();
                        c_dist.add_inplace(&b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert((a.clone() + b).simplified());
                    }
                }
                {
                    let mut c_bounds = a_bounds.clone();
                    s.mul_inplace(&mut c_bounds, &b_bounds);
                    if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                        let mut c_dist = a_dist.clone();
                        c_dist.mul_inplace(&b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert((a.clone() * b).simplified());
                    }
                }
                {
                    let mut c_bounds = a_bounds.clone();
                    s.max_inplace(&mut c_bounds, &b_bounds);
                    if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                        let mut c_dist = a_dist.clone();
                        c_dist.max_inplace(&b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert((a.clone().max(b)).simplified());
                    }
                }
                {
                    let mut c_bounds = a_bounds.clone();
                    s.min_inplace(&mut c_bounds, &b_bounds);
                    if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                        let mut c_dist = a_dist.clone();
                        c_dist.min_inplace(&b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert((a.clone().min(b)).simplified());
                    }
                }
            }
        }
        // Non-commutative expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("B: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            for (b, b_bounds, b_dist) in &sorted {
                {
                    let mut c_bounds = a_bounds.clone();
                    s.sub_inplace(&mut c_bounds, &b_bounds);
                    if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                        let mut c_dist = a_dist.clone();
                        c_dist.sub_inplace(&b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert((a.clone() - b).simplified());
                    }
                }
            }
        }
        // Special expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            println!("C: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            if a_bounds.1 < 0 {
                continue;
            }
            for (b, b_bounds, b_dist) in sorted.iter() {
                let mut c_bounds = a_bounds.clone();
                s.repeat_inplace(&mut c_bounds, &b_bounds);
                if i < (height - 1) || (min_value <= c_bounds.0 && c_bounds.1 <= max_value) {
                    let mut c_dist = a_dist.clone();
                    c_dist.repeat(&b_dist, &mut buffer);
                    expressions.entry(c_dist).or_insert((a.clone().multi_add(b)).simplified());
                }
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
