use std::{collections::HashMap, ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign}};

use num::BigRational;

use crate::Dist;

mod dist;
pub mod parse;
mod print;
pub mod random;
mod simplify;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    Dice(usize),
    Const(isize),
    Negate(usize),
    Add(usize, usize),
    Mul(usize, usize),
    Sub(usize, usize),
    Max(usize, usize),
    Min(usize, usize),
    // Left value is how many times we should evaluate the right expression
    MultiAdd(usize, usize),
}

impl Part {
    fn increased_offset(&self, n: usize) -> Self {
        match *self {
            Part::Dice(dice) => Part::Dice(dice),
            Part::Const(n) => Part::Const(n),
            Part::Negate(a) => Part::Negate(a + n),
            Part::Add(a, b) => Part::Add(a + n, b + n),
            Part::Mul(a, b) => Part::Mul(a + n, b + n),
            Part::Sub(a, b) => Part::Sub(a + n, b + n),
            Part::Max(a, b) => Part::Max(a + n, b + n),
            Part::Min(a, b) => Part::Min(a + n, b + n),
            Part::MultiAdd(a, b) => Part::MultiAdd(a + n, b + n),
        }
    }
}

/// A sequence of interacting dice rolls
///
/// You can also display a `DiceExpression` in a simplified form
/// ```
/// # use diceystats::DiceExpression;
/// let x: DiceExpression = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
/// assert_eq!(x.to_string(), "(d5 + d20xd5) * max(max(d4 * d4, d5), d10)x(d4 * d8)")
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiceExpression {
    parts: Vec<Part>,
}

// Used when traversing the tree of a `DiceExpression`
trait Evaluator<T> {
    // Used for `multi_add`: some evaluators have to reevaluate the right side
    // expression multiple times (LOSSY = true) while some don't (LOSSY = false)
    const LOSSY: bool;
    fn to_usize(_x: T) -> usize {
        unreachable!("Only used if operations are LOSSY");
    }
    fn dice(&mut self, d: usize) -> T;
    fn constant(&mut self, n: isize) -> T;
    fn multi_add_inplace(&mut self, _a: &mut T, _b: &T) {
        unreachable!("Only used if operations are not LOSSY");
    }
    fn negate_inplace(&mut self, a: &mut T);
    fn add_inplace(&mut self, a: &mut T, b: &T);
    fn mul_inplace(&mut self, a: &mut T, b: &T);
    fn sub_inplace(&mut self, a: &mut T, b: &T);
    fn max_inplace(&mut self, a: &mut T, b: &T);
    fn min_inplace(&mut self, a: &mut T, b: &T);
}

// Finds the minimum and maximum value of a `DiceExpression`.
struct Bounds {
    multi_add_negative: usize,
}

impl Bounds {
    fn new() -> Self {
        Bounds { multi_add_negative: 0 }
    }
}

impl Evaluator<(isize, isize)> for Bounds {
    const LOSSY: bool = false;

    fn dice(&mut self, d: usize) -> (isize, isize) {
        (1, d as isize)
    }

    fn constant(&mut self, n: isize) -> (isize, isize) {
        (n, n)
    }

    fn multi_add_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        if a.0 < 0 {
            self.multi_add_negative += 1;
            a.0 = 0;
            if a.1 < 0 {
                a.1 = 0;
            }
        }
        let extremes = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap())
    }

    fn negate_inplace(&mut self, a: &mut (isize, isize)) {
        *a = (-a.1, -a.0);
    }

    fn add_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        *a = (a.0 + b.0, a.1 + b.1)
    }

    fn mul_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap())
    }

    fn sub_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0 - b.0, a.0 - b.1, a.1 - b.0, a.1 - b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap())
    }

    fn max_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0.max(b.0), a.0.max(b.1), a.1.max(b.0), a.1.max(b.1)];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap())
    }

    fn min_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0.min(b.0), a.0.min(b.1), a.1.min(b.0), a.1.min(b.1)];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap())
    }
}

// A state machine used in `DiceExpression::traverse`
enum EvaluateStage {
    Dice(usize),
    Const(isize),
    MultiAddCreate(usize, usize),
    MultiAddCollect,
    MultiAddCollectPartial(usize),
    MultiAddExtra(usize),

    NegateCreate(usize),
    AddCreate(usize, usize),
    SubCreate(usize, usize),
    MulCreate(usize, usize),
    MaxCreate(usize, usize),
    MinCreate(usize, usize),
    NegateCollect,
    AddCollect,
    SubCollect,
    MulCollect,
    MaxCollect,
    MinCollect,
}

impl EvaluateStage {
    fn collect_from(part: Part) -> Self {
        match part {
            Part::Dice(dice) => EvaluateStage::Dice(dice),
            Part::Const(n) => EvaluateStage::Const(n),
            Part::Negate(a) => EvaluateStage::NegateCreate(a),
            Part::Add(a, b) => EvaluateStage::AddCreate(a, b),
            Part::Sub(a, b) => EvaluateStage::SubCreate(a, b),
            Part::Mul(a, b) => EvaluateStage::MulCreate(a, b),
            Part::Min(a, b) => EvaluateStage::MinCreate(a, b),
            Part::Max(a, b) => EvaluateStage::MaxCreate(a, b),
            Part::MultiAdd(a, b) => EvaluateStage::MultiAddCreate(a, b),
        }
    }
}

impl DiceExpression {
    // The last item in `self.parts` should always be the top node in the tree
    fn top_part(&self) -> Part {
        *self.parts.last().unwrap()
    }

    // Traverse the tree with an Evaluator.
    fn traverse<T, Q: Evaluator<T>>(&self, state: &mut Q) -> T {
        let mut stack: Vec<EvaluateStage> = vec![EvaluateStage::collect_from(self.top_part())];
        let mut values: Vec<T> = Vec::new();
        while let Some(x) = stack.pop() {
            match x {
                EvaluateStage::Dice(dice) => values.push(state.dice(dice)),
                EvaluateStage::Const(n) => values.push(state.constant(n)),
                EvaluateStage::MultiAddCreate(a, b) => {
                    if Q::LOSSY {
                        stack.push(EvaluateStage::MultiAddCollectPartial(b));
                        stack.push(EvaluateStage::collect_from(self.parts[a]));
                    } else {
                        stack.push(EvaluateStage::MultiAddCollect);
                        stack.push(EvaluateStage::collect_from(self.parts[a]));
                        stack.push(EvaluateStage::collect_from(self.parts[b]));
                    }
                }
                EvaluateStage::MultiAddCollectPartial(b) => {
                    assert!(Q::LOSSY);
                    let aa = Q::to_usize(values.pop().unwrap());
                    stack.push(EvaluateStage::MultiAddExtra(aa));
                    for _ in 0..aa {
                        stack.push(EvaluateStage::collect_from(self.parts[b]));
                    }
                }
                EvaluateStage::MultiAddCollect => {
                    assert!(!Q::LOSSY);
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.multi_add_inplace(&mut aa, &bb);
                    values.push(aa);
                }
                EvaluateStage::MultiAddExtra(aa) => {
                    debug_assert!(aa != 0);
                    let mut res = values.pop().unwrap();
                    for _ in 1..aa {
                        let v = values.pop().unwrap();
                        state.add_inplace(&mut res, &v);
                    }
                    values.push(res);
                }
                EvaluateStage::NegateCreate(a) => {
                    stack.push(EvaluateStage::NegateCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                }
                EvaluateStage::AddCreate(a, b) => {
                    stack.push(EvaluateStage::AddCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                    stack.push(EvaluateStage::collect_from(self.parts[b]));
                }
                EvaluateStage::MulCreate(a, b) => {
                    stack.push(EvaluateStage::MulCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                    stack.push(EvaluateStage::collect_from(self.parts[b]));
                }
                EvaluateStage::SubCreate(a, b) => {
                    stack.push(EvaluateStage::SubCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                    stack.push(EvaluateStage::collect_from(self.parts[b]));
                }
                EvaluateStage::MinCreate(a, b) => {
                    stack.push(EvaluateStage::MinCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                    stack.push(EvaluateStage::collect_from(self.parts[b]));
                }
                EvaluateStage::MaxCreate(a, b) => {
                    stack.push(EvaluateStage::MaxCollect);
                    stack.push(EvaluateStage::collect_from(self.parts[a]));
                    stack.push(EvaluateStage::collect_from(self.parts[b]));
                }
                EvaluateStage::NegateCollect => {
                    let mut aa = values.pop().unwrap();
                    state.negate_inplace(&mut aa);
                    values.push(aa);
                }
                EvaluateStage::AddCollect => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.add_inplace(&mut aa, &bb);
                    values.push(aa);
                }
                EvaluateStage::MulCollect => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.mul_inplace(&mut aa, &bb);
                    values.push(aa);
                }
                EvaluateStage::SubCollect => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.sub_inplace(&mut aa, &bb);
                    values.push(aa);
                }
                EvaluateStage::MinCollect => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.min_inplace(&mut aa, &bb);
                    values.push(aa);
                }
                EvaluateStage::MaxCollect => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
                    state.max_inplace(&mut aa, &bb);
                    values.push(aa);
                }
            };
        }
        values.pop().unwrap()
    }

    fn dice(d: usize) -> Self {
        Self { parts: vec![Part::Dice(d)] }
    }

    fn constant(n: isize) -> Self {
        Self { parts: vec![Part::Const(n)] }
    }

    fn negate_inplace(&mut self) {
        let orig_len = self.parts.len();
        self.parts.push(Part::Negate(orig_len - 1))
    }

    fn negate(mut self) -> Self {
        self.negate_inplace();
        self
    }

    fn op_double_inplace(&mut self, other: &DiceExpression) -> (usize, usize) {
        self.parts.reserve(other.parts.len() + 1);

        let orig_len = self.parts.len();
        let first_element = orig_len - 1;
        self.parts.extend(other.parts.iter().map(|x| x.increased_offset(orig_len)));
        let second_element = self.parts.len() - 1;
        (first_element, second_element)
    }

    fn min_assign(&mut self, other: &DiceExpression) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::Min(a, b));
    }

    fn min(mut self, other: &DiceExpression) -> Self {
        self.min_assign(other);
        self
    }

    fn max_assign(&mut self, other: &DiceExpression) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::Max(a, b));
    }

    fn max(mut self, other: &DiceExpression) -> Self {
        self.max_assign(other);
        self
    }

    fn multi_add_assign(&mut self, other: &DiceExpression) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::MultiAdd(a, b));
    }

    fn multi_add(mut self, other: &DiceExpression) -> Self {
        self.multi_add_assign(other);
        self
    }

    fn could_be_negative(&self) -> bool {
        let mut s = Bounds { multi_add_negative: 0 };
        self.traverse(&mut s);
        s.multi_add_negative != 0
    }

    pub fn bounds(&self) -> (isize, isize) {
        let mut s = Bounds { multi_add_negative: 0 };
        let (a, b) = self.traverse(&mut s);
        debug_assert!(a <= b);
        (a, b)
    }
}

impl AddAssign<&Self> for DiceExpression {
    fn add_assign(&mut self, other: &Self) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::Add(a, b));
    }
}

impl Add<&Self> for DiceExpression {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}

impl Add<Self> for DiceExpression {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self.add_assign(&other);
        self
    }
}

impl MulAssign<&Self> for DiceExpression {
    fn mul_assign(&mut self, other: &Self) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::Mul(a, b));
    }
}

impl Mul<&Self> for DiceExpression {
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}

impl Mul<Self> for DiceExpression {
    type Output = Self;

    fn mul(mut self, other: Self) -> Self {
        self.mul_assign(&other);
        self
    }
}

impl SubAssign<&Self> for DiceExpression {
    fn sub_assign(&mut self, other: &Self) {
        let (a, b) = self.op_double_inplace(other);
        self.parts.push(Part::Sub(a, b));
    }
}

impl Sub<&Self> for DiceExpression {
    type Output = Self;

    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}

impl Sub<Self> for DiceExpression {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self.sub_assign(&other);
        self
    }
}


/// Parameters for [`every_tree`].
pub struct ConfigEveryTree<'a> {
    /// Height of the resulting tree of the generated [`DiceExpression`]. Probably 2 at max.
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

/// Generate every [`DiceExpression`] with parameters chosen using [`ConfigEveryTree`].
pub fn every_tree(&ConfigEveryTree { height, dice, constants, bounds, print_progress }: &ConfigEveryTree) -> Vec<(Dist<BigRational>, DiceExpression)> {
    let mut expressions: HashMap<Dist<BigRational>, DiceExpression> = HashMap::default();
    for &i in dice {
        let d = DiceExpression::dice(i);
        let d_dist = d.dist::<BigRational>();
        expressions.entry(d_dist).or_insert(d);
    }
    for &i in constants {
        let c = DiceExpression::constant(i);
        let c_dist = c.dist::<BigRational>();
        expressions.entry(c_dist).or_insert(c);
    }
    for i in 0..height {
        let mut sorted: Vec<(DiceExpression, (isize, isize), Dist<BigRational>)> = expressions
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
                add_if_bounded!(Bounds::add_inplace, Dist::add_inplace, DiceExpression::add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(Bounds::mul_inplace, Dist::mul_inplace, DiceExpression::mul, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(Bounds::max_inplace, Dist::max_inplace, DiceExpression::max, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
                add_if_bounded!(Bounds::min_inplace, Dist::min_inplace, DiceExpression::min, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
            }
        }

        // Non-commutative expressions
        for (a_i, (a, a_bounds, a_dist)) in sorted.iter().enumerate() {
            if print_progress {
                println!("B: {} / {} : {} : {}", a_i + 1, sorted.len(), expressions.len(), a);
            }
            for (b, b_bounds, b_dist) in &sorted {
                add_if_bounded!(Bounds::sub_inplace, Dist::sub_inplace, DiceExpression::sub, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
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
                add_if_bounded!(Bounds::multi_add_inplace, Dist::multi_add_inplace, DiceExpression::multi_add, *a_bounds, b_bounds, a_dist, &b_dist, a, b);
            }
        }
    }
    expressions.drain().collect()
}
