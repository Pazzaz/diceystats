use super::{DiceFormula, Evaluator, Part};
use rand::{
    Rng,
    distr::Uniform,
    prelude::{Distribution, IndexedRandom},
};

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

impl Distribution<isize> for DiceFormula {
    /// Evaluate the expression into a single number, rolling dice using `rng`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> isize {
        let mut e = SampleEvaluator { rng };
        self.traverse(&mut e)
    }
}

pub fn random_formula<R: Rng + ?Sized>(
    rng: &mut R,
    height: usize,
    value_size: usize,
) -> DiceFormula {
    let dist = Uniform::new_inclusive(1, value_size).unwrap();
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
        let exp = DiceFormula { parts };
        if !exp.could_be_negative() {
            return exp;
        }
    }
}

struct SampleEvaluator<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
}

impl<R: Rng + ?Sized> Evaluator<isize> for SampleEvaluator<'_, R> {
    const CUSTOM_MULTI_ADD: bool = false;

    fn to_usize(x: isize) -> usize {
        x.try_into().unwrap_or_else(|_| panic!("Can't roll negative amount of dice"))
    }

    fn dice(&mut self, d: usize) -> isize {
        (self.rng.random_range(1..=d)) as isize
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
