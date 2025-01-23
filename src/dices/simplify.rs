use std::ops::{AddAssign, Mul, MulAssign, SubAssign};

use super::{DiceExpression, Evaluator, Part};

// Traverses a `DiceExpression` and creates a simplified `DiceExpression`
// using local/peephole optimizations.
struct Simplifier {}

impl Evaluator<DiceExpression> for Simplifier {
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> DiceExpression {
        if d == 1 { DiceExpression::constant(1) } else { DiceExpression::dice(d) }
    }

    fn constant(&mut self, n: isize) -> DiceExpression {
        DiceExpression::constant(n)
    }

    fn negate_inplace(&mut self, a: &mut DiceExpression) {
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

    fn multi_add_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceExpression::constant(aa * bb);
            }
            (Part::Const(1), _) => {
                *a = b.clone();
            }
            (_, Part::Const(1)) => {}
            (_, Part::Const(_)) => {
                *a = DiceExpression::mul(a.clone(), b);
            }
            _ => {
                a.multi_add_assign(b);
            }
        }
    }

    fn add_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceExpression::constant(aa + bb);
            }
            _ => {
                a.add_assign(b);
            }
        }
    }

    fn mul_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceExpression::constant(aa * bb);
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

    fn sub_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceExpression::constant(aa - bb);
            }
            _ => {
                a.sub_assign(b);
            }
        }
    }

    fn max_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        let a_bounds = a.bounds();
        let b_bounds = b.bounds();
        if a_bounds.1 <= b_bounds.0 {
            *a = b.clone();
        } else if !(b_bounds.1 <= a_bounds.0) {
            a.max_assign(b);
        }
    }

    fn min_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        let a_bounds = a.bounds();
        let b_bounds = b.bounds();
        if b_bounds.1 <= a_bounds.0 {
            *a = b.clone();
        } else if !(a_bounds.1 <= b_bounds.0) {
            a.min_assign(b);
        }
    }
}

impl DiceExpression {
    /// Simplify the expression using simple rewriting rules
    /// ```
    /// use diceystats::DiceExpression;
    ///
    /// let complicated: DiceExpression = "min((d4+d5)*5, d5x2)".parse().unwrap();
    /// let simple = complicated.simplified();
    /// assert_eq!(simple.to_string(), "d5 * 2");
    /// ```
    pub fn simplified(&self) -> DiceExpression {
        let mut s = Simplifier {};
        self.traverse(&mut s)
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
            let a = DiceExpression::make_random(&mut rng, 2, 10);
            let a_simple = a.simplified();
            let dist = a.dist::<f64>();
            let simple_dist = a_simple.dist::<f64>();
            assert!(dist.distance(&simple_dist) <= 0.01, "{a} = {a_simple}");
        }
    }
}
