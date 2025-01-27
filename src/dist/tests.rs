use num::BigRational;

extern crate test;
use crate::{
    DiceFormula,
    dist::{DenseDist, Dist, SparseDist, WeirdDist},
};
use test::Bencher;

macro_rules! check_eq {
    ($a:expr, $b:expr, BigRational) => {
        assert_eq!($a, $b);
    };
    ($a:expr, $b:expr, f64) => {
        assert!(($a - $b).abs() < 0.01);
    };
}

macro_rules! dist_bench {
    // We need to send $f1, $f2, $f3 and $f4 because `concat_idents!` doesn't work in function
    // signatures
    ($s:expr, $t:tt, $f1:ident, $f2:ident, $f3:ident, $f4:ident) => {
        #[bench]
        fn $f1(b: &mut Bencher) {
            let yep: DiceFormula = $s.parse().unwrap();
            b.iter(|| {
                let res: DenseDist<$t> = yep.dist();
                test::black_box(res);
            });
        }
        #[bench]
        fn $f2(b: &mut Bencher) {
            let yep: DiceFormula = $s.parse().unwrap();
            b.iter(|| {
                let res: SparseDist<$t> = yep.dist();
                test::black_box(res);
            });
        }
        #[bench]
        fn $f3(b: &mut Bencher) {
            let yep: DiceFormula = $s.parse().unwrap();
            b.iter(|| {
                let res: WeirdDist<$t> = yep.dist();
                test::black_box(res);
            });
        }

        #[test]
        fn $f4() {
            let yep: DiceFormula = $s.parse().unwrap();
            let res1: DenseDist<$t> = yep.dist();
            let res2: WeirdDist<$t> = yep.dist();
            let res3: SparseDist<$t> = yep.dist();
            check_eq!(res1.mean(), res2.mean(), $t);
            check_eq!(res2.mean(), res3.mean(), $t);
        }
    };
}

dist_bench!(
    "min(4*max(d6xd5, d5xd7), 5*max(d2xd6, d3xd5))",
    BigRational,
    complicated_dense,
    complicated_space,
    complicated_weird,
    complicated_eq
);
dist_bench!(
    "d6xd6",
    BigRational,
    eval_6dx6d_dense,
    eval_6dx6d_space,
    eval_6dx6d_weird,
    eval_6dx6d_eq
);
dist_bench!("d2*10000", BigRational, large_dense, large_space, large_weird, large_eq);
dist_bench!(
    "d30xd30",
    f64,
    f64_d30xd30_dense,
    f64_d30xd30_space,
    f64_d30xd30_weird,
    f64_d30xd30_eq
);
dist_bench!(
    "d20+d20+d20+d20+d20+d20+d20+d20+d20+d20+d20",
    f64,
    many_additions_dense,
    many_additions_space,
    many_additions_weird,
    many_additions_eq
);
dist_bench!(
    "d20*d20*d20*d20*d20",
    f64,
    many_multiplications_dense,
    many_multiplications_space,
    many_multiplications_weird,
    many_multiplications_eq
);
dist_bench!(
    "(-d20)*(-d20)*(-d20)",
    f64,
    negative_mul_dense,
    negative_mul_space,
    negative_mul_weird,
    negative_mul_eq
);
