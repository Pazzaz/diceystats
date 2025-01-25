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


    macro_rules! dist_bench {
        ($s:expr, $t:ty, $f1:ident, space) => {
            #[bench]
            fn $f1(b: &mut Bencher)  {
                let yep: DiceFormula = $s.parse().unwrap();
                b.iter(|| {
                    let res = yep.dist_sparse::<$t>();
                    test::black_box(res);
                });
            }
        };
        ($s:expr, $t:ty, $f1:ident, dense) => {
            #[bench]
            fn $f1(b: &mut Bencher)  {
                let yep: DiceFormula = $s.parse().unwrap();
                b.iter(|| {
                    let res = yep.dist::<$t>();
                    test::black_box(res);
                });
            }
        };
        ($s:expr, $t:ty, $f1:ident, weird) => {
            #[bench]
            fn $f1(b: &mut Bencher)  {
                let yep: DiceFormula = $s.parse().unwrap();
                b.iter(|| {
                    let res = yep.dist_weird::<$t>();
                    test::black_box(res);
                });
            }
        };
    }

    dist_bench!("min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))", BigRational, complicated_dense, dense);
    dist_bench!("min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))", BigRational, complicated_space, space);
    dist_bench!("min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))", BigRational, complicated_weird, weird);

    dist_bench!("d6xd6", BigRational, eval_6dx6d_dense, dense);
    dist_bench!("d6xd6", BigRational, eval_6dx6d_space, space);
    dist_bench!("d6xd6", BigRational, eval_6dx6d_weird, weird);

    dist_bench!("d2*10000", BigRational, large_dense, dense);
    dist_bench!("d2*10000", BigRational, large_space, space);
    dist_bench!("d2*10000", BigRational, large_weird, weird);

    dist_bench!("d30xd30", f64, f64_d30xd30_dense, dense);
    dist_bench!("d30xd30", f64, f64_d30xd30_space, space);
    dist_bench!("d30xd30", f64, f64_d30xd30_weird, weird);

    dist_bench!("d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20", f64, many_additions_dense, dense);
    dist_bench!("d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20", f64, many_additions_space, space);
    dist_bench!("d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20", f64, many_additions_weird, weird);

    dist_bench!("d20*d20*d20*d20*d20", f64, many_multiplications_dense, dense);
    dist_bench!("d20*d20*d20*d20*d20", f64, many_multiplications_space, space);
    dist_bench!("d20*d20*d20*d20*d20", f64, many_multiplications_weird, weird);

    dist_bench!("(-d20)*(-d20)*(-d20)", f64, negative_mul_dense, dense);
    dist_bench!("(-d20)*(-d20)*(-d20)", f64, negative_mul_space, space);
    dist_bench!("(-d20)*(-d20)*(-d20)", f64, negative_mul_weird, weird);
}
