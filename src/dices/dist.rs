use std::ops::{AddAssign, MulAssign};

use num::{FromPrimitive, Num};

use crate::Dist;

use super::{DiceExpression, Evaluator};

impl DiceExpression {
    /// Calculate the probability distribution of the outcomes of the expression.
    ///
    /// The function is generic over the number type used to represent probabilities.
    ///
    /// # Example
    /// ```
    /// use diceystats::{DiceExpression, Dist};
    /// use num::BigRational;
    ///
    /// let expr: DiceExpression = "d10 * d4".parse().unwrap();
    /// let fast_dist: Dist<f64> = expr.dist();
    /// let exact_dist: Dist<BigRational> = expr.dist();
    /// assert_eq!(exact_dist.mean(), "55/4".parse().unwrap());
    /// ```
    pub fn dist<T>(&self) -> Dist<T>
    where
        for<'a> T: MulAssign<&'a T>
            + AddAssign<&'a T>
            + Num
            + Clone
            + AddAssign
            + std::fmt::Debug
            + FromPrimitive,
    {
        let mut e = DistEvaluator { buffer: Vec::new() };
        self.evaluate_generic(&mut e)
    }
}

struct DistEvaluator<T> {
    buffer: Vec<T>,
}

impl<T: Num + Clone + AddAssign + std::fmt::Debug + FromPrimitive> Evaluator<Dist<T>>
    for DistEvaluator<T>
where
    for<'a> T: MulAssign<&'a T> + AddAssign<&'a T>,
{
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> Dist<T> {
        Dist::uniform(1, d as isize)
    }

    fn constant(&mut self, n: isize) -> Dist<T> {
        Dist::constant(n)
    }

    fn repeat_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.repeat(b, &mut self.buffer);
    }

    fn negate_inplace(&mut self, a: &mut Dist<T>) {
        a.negate_inplace();
    }

    fn add_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.add_inplace(b, &mut self.buffer);
    }

    fn mul_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.mul_inplace(b, &mut self.buffer);
    }

    fn sub_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.sub_inplace(b, &mut self.buffer);
    }

    fn max_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.max_inplace(b, &mut self.buffer);
    }

    fn min_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.min_inplace(b, &mut self.buffer);
    }
}
