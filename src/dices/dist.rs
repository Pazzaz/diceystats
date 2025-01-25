use std::ops::{AddAssign, MulAssign};

use num::{FromPrimitive, Num};

use crate::Dist;

use super::Evaluator;

pub(crate) struct DistEvaluator<T> {
    pub(crate) buffer: Vec<T>,
}

impl<T: Num + Clone + AddAssign + FromPrimitive> Evaluator<Dist<T>> for DistEvaluator<T>
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

    use crate::{sparse_dist::SparseDist, DiceFormula};

    use super::*;
    extern crate test;
    use test::Bencher;

    #[test]
    fn repeat_simple() {
        let yep: DiceFormula = "d9xd10".parse().unwrap();
        assert_eq!(yep.dist::<BigRational>().mean(), "55/2".parse().unwrap(),);
    }

    #[test]
    fn sparse_equal() {
        let yep: DiceFormula = "min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))".parse().unwrap();
        assert_eq!(yep.dist::<BigRational>().mean(), yep.dist_sparse::<BigRational>().mean());
    }

    #[test]
    fn sparse_equal_weird() {
        let yep: DiceFormula = "min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))".parse().unwrap();
        assert_eq!(yep.dist::<BigRational>().mean(), yep.dist_weird::<BigRational>().mean());
    }

    #[bench]
    fn eval_6dx6d(b: &mut Bencher) {
        let yep: DiceFormula = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res = yep.dist::<BigRational>();
            test::black_box(res);
        });
    }

    #[bench]
    fn eval_6dx6d_sparse(b: &mut Bencher) {
        let yep: DiceFormula = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res = yep.dist_sparse::<BigRational>();
            test::black_box(res);
        });
    }

    #[bench]
    fn eval_6dx6d_weird(b: &mut Bencher) {
        let yep: DiceFormula = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res = yep.dist_weird::<BigRational>();
            test::black_box(res);
        });
    }

    #[bench]
    fn large(b: &mut Bencher) {
        let yep: DiceFormula = "d2*10000".parse().unwrap();
        b.iter(|| {
            let res: Dist<BigRational> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn large_sparse(b: &mut Bencher) {
        let yep: DiceFormula = "d2*10000".parse().unwrap();
        b.iter(|| {
            let res: SparseDist<BigRational> = yep.dist_sparse();
            test::black_box(res);
        });
    }

    #[bench]
    fn large_weird(b: &mut Bencher) {
        let yep: DiceFormula = "d2*10000".parse().unwrap();
        b.iter(|| {
            let res = yep.dist_weird::<BigRational>();
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
    fn many_additions_sparse(b: &mut Bencher) {
        let yep: DiceFormula = "d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20".parse().unwrap();
        b.iter(|| {
            let res: SparseDist<f64> = yep.dist_sparse();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_additions_weird(b: &mut Bencher) {
        let yep: DiceFormula = "d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20".parse().unwrap();
        b.iter(|| {
            let res = yep.dist_weird::<f64>();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_multiplications(b: &mut Bencher) {
        let yep: DiceFormula = "d20*d20*d20*d20*d20".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.dist();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_multiplications_sparse(b: &mut Bencher) {
        let yep: DiceFormula = "d20*d20*d20*d20*d20".parse().unwrap();
        b.iter(|| {
            let res: SparseDist<f64> = yep.dist_sparse();
            test::black_box(res);
        });
    }

    #[bench]
    fn many_multiplications_weird(b: &mut Bencher) {
        let yep: DiceFormula = "d20*d20*d20*d20*d20".parse().unwrap();
        b.iter(|| {
            let res = yep.dist_weird::<f64>();
            test::black_box(res);
        });
    }
}
