use std::ops::{AddAssign, Mul, MulAssign, SubAssign};

use super::{DiceFormula, Evaluator, Part};

// Traverses a `DiceFormula` and creates a simplified `DiceFormula`
// using local/peephole optimizations.
pub(crate) struct Simplifier {}

impl Evaluator<DiceFormula> for Simplifier {
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> DiceFormula {
        if d == 1 { DiceFormula::constant(1) } else { DiceFormula::dice(d) }
    }

    fn constant(&mut self, n: isize) -> DiceFormula {
        DiceFormula::constant(n)
    }

    fn negate_inplace(&mut self, a: &mut DiceFormula) {
        match a.top_part() {
            // Double negate => remove negate
            Part::Const(n) => {
                a.parts.pop();
                a.parts.push(Part::Const(-n));
            }
            Part::Negate(_) => {
                a.parts.pop();
            }
            // Else we just negate
            Part::Dice(_)
            | Part::Add(_, _)
            | Part::Mul(_, _)
            | Part::Sub(_, _)
            | Part::Max(_, _)
            | Part::Min(_, _)
            | Part::MultiAdd(_, _) => {
                a.negate_inplace();
            }
        }
    }

    fn multi_add_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceFormula::constant(aa * bb);
            }
            (Part::Const(1), _) => {
                *a = b.clone();
            }
            (_, Part::Const(1)) => {}
            (_, Part::Const(_)) => {
                *a = DiceFormula::mul(a.clone(), b);
            }
            _ => {
                a.multi_add_assign(b);
            }
        }
    }

    fn add_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceFormula::constant(aa + bb);
            }
            _ => {
                a.add_assign(b);
            }
        }
    }

    fn mul_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceFormula::constant(aa * bb);
            }
            (Part::Const(1), _) => {
                *a = b.clone();
            }
            (_, Part::Const(1)) => {}
            _ => {
                a.mul_assign(b);
            }
        }
    }

    fn sub_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceFormula::constant(aa - bb);
            }
            _ => {
                a.sub_assign(b);
            }
        }
    }

    fn max_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        let a_bounds = a.bounds();
        let b_bounds = b.bounds();
        if a_bounds.1 <= b_bounds.0 {
            *a = b.clone();
        } else if !(b_bounds.1 <= a_bounds.0) {
            a.max_assign(b);
        }
    }

    fn min_inplace(&mut self, a: &mut DiceFormula, b: &DiceFormula) {
        let a_bounds = a.bounds();
        let b_bounds = b.bounds();
        if b_bounds.1 <= a_bounds.0 {
            *a = b.clone();
        } else if !(a_bounds.1 <= b_bounds.0) {
            a.min_assign(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn simplify_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..200 {
            // Generate a random expression and check if its
            // distribution is the same after being simplified.
            let a = DiceFormula::random(&mut rng, 2, 10);
            let a_simple = a.simplified();
            let dist = a.dist::<f64>();
            let simple_dist = a_simple.dist::<f64>();
            assert!(dist.distance(&simple_dist) <= 0.01, "{a} = {a_simple}");
        }
    }
}
