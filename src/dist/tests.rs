use num::BigRational;

extern crate test;
use test::Bencher;

use crate::DiceFormula;


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
