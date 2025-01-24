use std::ops::{AddAssign, MulAssign};

use num::{FromPrimitive, Num};

use crate::Dist;

use super::Evaluator;

pub(crate) struct DistEvaluator<T> {
    pub(crate) buffer: Vec<T>,
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

    fn multi_add_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.multi_add_inplace(b, &mut self.buffer);
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

#[cfg(test)]
mod tests {
    use num::BigRational;

    use crate::DiceFormula;

    use super::*;
    extern crate test;
    use test::Bencher;

    #[test]
    fn repeat_simple() {
        let yep: DiceFormula = "d9xd10".parse().unwrap();
        assert_eq!(yep.dist::<BigRational>().mean(), "55/2".parse().unwrap(),);
    }

    #[bench]
    fn eval_6dx6d(b: &mut Bencher) {
        let yep: DiceFormula = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res: Dist<BigRational> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn f64_30dx30d(b: &mut Bencher) {
        let yep: DiceFormula = "d30xd30".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_additions(b: &mut Bencher) {
        let yep: DiceFormula = "d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_multiplications(b: &mut Bencher) {
        let yep: DiceFormula = "d20*d20*d20".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }
}
