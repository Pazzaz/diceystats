    //! Exhaustive search of dice expressions

    use std::{collections::HashMap, ops::{Add, Mul, Sub}};
    use num::BigRational;
    use crate::{dices::{Bounds, Evaluator}, Dist};
    use super::DiceFormula;

    /// Parameters for [`every_tree`].
    pub struct ConfigEveryTree<'a> {
        /// Height of the resulting tree of the generated [`DiceFormula`]. Probably 2 at max.
        pub height: usize,
        /// Set of dice that the generated trees can choose from.
        pub dice: &'a [usize],
        /// Set of constants that the generated trees can choose from.
        pub constants: &'a [isize],
        /// Bound that the expressions possible values will be contained in.
        pub bounds: (isize, isize),
        /// Print progress using [`println!`]
        pub print_progress: bool,
    }
    
    impl<'a> Default for ConfigEveryTree<'a> {
        fn default() -> Self {
            Self { height: 2, dice: &[4, 6, 8, 10, 12, 20], constants: &[], bounds: (1, 20), print_progress: true }
        }
    }
    
    /// Generate every [`DiceFormula`] with parameters chosen using [`ConfigEveryTree`].
    pub fn every_tree(&ConfigEveryTree { height, dice, constants, bounds, print_progress }: &ConfigEveryTree) -> Vec<(Dist<BigRational>, DiceFormula)> {
        let mut expressions: HashMap<Dist<BigRational>, DiceFormula> = HashMap::default();
        for &i in dice {
            let d = DiceFormula::dice(i);
            let d_dist = d.dist::<BigRational>();
            expressions.entry(d_dist).or_insert(d);
        }
        for &i in constants {
            let c = DiceFormula::constant(i);
            let c_dist = c.dist::<BigRational>();
            expressions.entry(c_dist).or_insert(c);
        }
        for i in 0..height {
            let mut sorted: Vec<(DiceFormula, (isize, isize), Dist<BigRational>)> = expressions
                .drain()
                .map(|x| {
                    let bounds = x.1.bounds();
                    (x.1, bounds, x.0)
                })
                .collect();
            sorted.sort();
            let mut buffer: Vec<BigRational> = Vec::new();
            let mut s = Bounds::new();
            macro_rules! add_if_bounded {
                ($f1:expr, $f2:expr, $f3:expr, $a_bounds:expr, $b_bounds:expr, $a_dist:expr, $b_dist:expr, $a:expr, $b:expr) => {
                    let mut c_bounds = $a_bounds;
                    $f1(&mut s, &mut c_bounds, $b_bounds);
                    if i < (height - 1) || (bounds.0 <= c_bounds.0 && c_bounds.1 <= bounds.1) {
                        let mut c_dist = $a_dist.clone();
                        $f2(&mut c_dist, $b_dist, &mut buffer);
                        expressions.entry(c_dist).or_insert_with(|| $f3($a.clone(), $b).simplified());
                    }
                };
            }
    
            // Commutative expressions
            for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
                if print_progress {
                    println!("A: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
                }
                for (b_i, (b, b_bounds, b_dist)) in sorted.iter().enumerate() {
                    if b_i > a_i {
                        break;
                    }
                    add_if_bounded!(Bounds::add_inplace, Dist::add_inplace, DiceFormula::add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                    add_if_bounded!(Bounds::mul_inplace, Dist::mul_inplace, DiceFormula::mul, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                    add_if_bounded!(Bounds::max_inplace, Dist::max_inplace, DiceFormula::max, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                    add_if_bounded!(Bounds::min_inplace, Dist::min_inplace, DiceFormula::min, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                }
            }
    
            // Non-commutative expressions
            for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
                if print_progress {
                    println!("B: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
                }
                for (b, b_bounds, b_dist) in &sorted {
                    add_if_bounded!(Bounds::sub_inplace, Dist::sub_inplace, DiceFormula::sub, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                }
            }
    
            // Special expressions
            for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
                if print_progress {
                    println!("C: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
                }
                if a_bounds.1 < 0 {
                    continue;
                }
                for (b, b_bounds, b_dist) in sorted.iter() {
                    add_if_bounded!(Bounds::multi_add_inplace, Dist::multi_add_inplace, DiceFormula::multi_add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                }
            }
        }
        expressions.drain().collect()
    }