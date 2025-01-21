use std::ops::{AddAssign, Mul, MulAssign, SubAssign};

use super::{DiceExpression, Evaluator, Part};

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
                if n < 0 {
                    a.parts.pop();
                    a.parts.push(Part::Const(-n));
                } else {
                    a.negate_inplace()
                }
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

    fn repeat_inplace(&mut self, a: &mut DiceExpression, b: &DiceExpression) {
        match (a.top_part(), b.top_part()) {
            (Part::Const(aa), Part::Const(bb)) => {
                *a = DiceExpression::constant(aa * bb);
            }
            (Part::Const(1), _) => {}
            (_, Part::Const(1)) => {
                *a = b.clone();
            }
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
        self.evaluate_generic(&mut s)
    }
}
