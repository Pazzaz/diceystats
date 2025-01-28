use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::{FromPrimitive, Num};
use rand::Rng;
use simplify::Simplifier;

use crate::dist::{DenseDist, Dist};

pub mod list;
mod parse;
pub use parse::DiceParseError;
mod print;
mod random;
pub use random::random_formula;
mod simplify;

/// A sequence of interacting dice rolls
///
/// You can also display a `DiceFormula` in a simplified form
///
/// ```
/// use diceystats::dices::DiceFormula;
///
/// let x: DiceFormula = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
/// assert_eq!(x.to_string(), "(d5 + d20xd5) * max(max(d4 * d4, d5), d10)x(d4 * d8)")
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiceFormula {
    parts: Vec<Part>,
}

// Each part of a dice expression is a node in a tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    // Leaf nodes
    Dice(usize),
    Const(isize),

    // Nodes with children. The integers are used
    // as indexes into the `DiceFormula.parts`.
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

/// Evaluates a [`DiceFormula`].
///
/// Evaluators are used by [`DiceFormula::traverse`] to evaluate a formula
/// recursively[^note] as a tree, depth-first, returning some `T`. For each
/// node, it calls the associated `Evaluator` function, giving the child nodes
/// as arguments.
///
/// This trait is used internally for printing, sampling, calculating
/// distributions, and so on.
///
/// Which functions of the trait you should implement depends on whether the
/// evaluator has a custom `multi_add`. If `CUSTOM_MULTI_ADD` is:
/// - `true`:  Implement `multi_add_inplace`, not `to_usize`
/// - `false`:  Implement `to_usize`, not `multi_add_inplace`
///
/// [^note]: Not actually implemented as a recursive function.
pub trait Evaluator<T> {
    /// Whether to use a custom `multi_add` or just evaluate the left-hand side
    /// and use that to decide how many times to evaluate the right-hand side.
    const CUSTOM_MULTI_ADD: bool;

    /// Convert value to `usize`, used when `CUSTOM_MULTI_ADD = false` to count
    /// how many times to evaluate the right side of a `multi_add` (`x`)
    /// operation.
    fn to_usize(_x: T) -> usize {
        unreachable!("Used when no CUSTOM_MULTI_ADD");
    }

    /// Evaluate a dice as a leaf node (e.g. `d20`).
    fn dice(&mut self, d: usize) -> T;

    /// Evaluate a constant as a leaf node (e.g. `4`).
    fn constant(&mut self, n: isize) -> T;

    /// Perform a `multi_add` (`x`). The left argument decides how many times to
    /// evaluate the right argument, then the results from the right argument
    /// are summed together. This function is used when `CUSTOM_MULTI_ADD =
    /// true`.
    fn multi_add_inplace(&mut self, _a: &mut T, _b: &T) {
        unreachable!("CUSTOM_MULTI_ADD: missing implementation");
    }
    fn negate_inplace(&mut self, a: &mut T);
    fn add_inplace(&mut self, a: &mut T, b: &T);
    fn mul_inplace(&mut self, a: &mut T, b: &T);
    fn sub_inplace(&mut self, a: &mut T, b: &T);
    fn max_inplace(&mut self, a: &mut T, b: &T);
    fn min_inplace(&mut self, a: &mut T, b: &T);
}

// Finds the minimum and maximum value of a `DiceFormula`.
struct Bounds {
    multi_add_negative: usize,
}

impl Bounds {
    fn new() -> Self {
        Bounds { multi_add_negative: 0 }
    }
}

impl Evaluator<(isize, isize)> for Bounds {
    const CUSTOM_MULTI_ADD: bool = true;

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

// State machine used in `DiceFormula::traverse`
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

impl DiceFormula {
    // The last item in `self.parts` should always be the top node in the tree
    fn top_part(&self) -> Part {
        *self.parts.last().unwrap()
    }

    /// Traverse the tree with an [Evaluator].
    pub fn traverse<T, Q: Evaluator<T>>(&self, state: &mut Q) -> T {
        let mut stack: Vec<EvaluateStage> = vec![EvaluateStage::collect_from(self.top_part())];
        let mut values: Vec<T> = Vec::new();
        while let Some(x) = stack.pop() {
            match x {
                EvaluateStage::Dice(dice) => values.push(state.dice(dice)),
                EvaluateStage::Const(n) => values.push(state.constant(n)),
                EvaluateStage::MultiAddCreate(a, b) => {
                    if Q::CUSTOM_MULTI_ADD {
                        stack.push(EvaluateStage::MultiAddCollect);
                        stack.push(EvaluateStage::collect_from(self.parts[a]));
                        stack.push(EvaluateStage::collect_from(self.parts[b]));
                    } else {
                        stack.push(EvaluateStage::MultiAddCollectPartial(b));
                        stack.push(EvaluateStage::collect_from(self.parts[a]));
                    }
                }
                EvaluateStage::MultiAddCollectPartial(b) => {
                    assert!(!Q::CUSTOM_MULTI_ADD);
                    let aa = Q::to_usize(values.pop().unwrap());
                    stack.push(EvaluateStage::MultiAddExtra(aa));
                    for _ in 0..aa {
                        stack.push(EvaluateStage::collect_from(self.parts[b]));
                    }
                }
                EvaluateStage::MultiAddCollect => {
                    assert!(Q::CUSTOM_MULTI_ADD);
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

    /// Calculate the probability distribution of the outcomes of the
    /// expression.
    ///
    /// The function is generic over the number type used to represent
    /// probabilities, and the distribution representation used (see
    /// [dist](crate::dist)).
    ///
    /// # Example
    ///
    /// ```
    /// use diceystats::{
    ///     dices::DiceFormula,
    ///     dist::{DenseDist, Dist},
    /// };
    /// use num::BigRational;
    ///
    /// let expr: DiceFormula = "d10 * d4".parse().unwrap();
    /// let fast_dist: DenseDist<f64> = expr.dist();
    /// let exact_dist: DenseDist<BigRational> = expr.dist();
    /// assert_eq!(exact_dist.mean(), "55/4".parse().unwrap());
    /// ```
    pub fn dist<'a, T, D: Dist<'a, T>>(&self) -> D
    where
        for<'b> T: 'a
            + Num
            + FromPrimitive
            + PartialOrd
            + Clone
            + MulAssign<&'b T>
            + AddAssign<&'b T>
            + SubAssign<&'b T>,
    {
        let mut e = D::evaluator();
        self.traverse(&mut e)
    }

    /// Create a random expression, modeleted as a tree with some `height` and
    /// maximum die / constant `value_size`.
    pub fn random<R: Rng + ?Sized>(rng: &mut R, height: usize, value_size: usize) -> Self {
        random_formula(rng, height, value_size)
    }

    /// Simplify the expression using simple rewriting rules
    /// ```
    /// use diceystats::dices::DiceFormula;
    ///
    /// let complicated: DiceFormula = "min((d4+d5)*5, d5x2)".parse().unwrap();
    /// let simple = complicated.simplified();
    /// assert_eq!(simple.to_string(), "d5 * 2");
    /// ```
    #[must_use]
    pub fn simplified(&self) -> DiceFormula {
        let mut s = Simplifier {};
        self.traverse(&mut s)
    }

    pub fn negate_inplace(&mut self) {
        let orig_len = self.parts.len();
        self.parts.push(Part::Negate(orig_len - 1))
    }

    pub fn negate(mut self) -> Self {
        self.negate_inplace();
        self
    }

    // Appends `other` to `self`, creating a broken state. `self.parts` has to have
    // a `Part` appended to it to make sense. Returns indices of top node of
    // `self` and `other`.
    #[must_use]
    fn concat(&mut self, other: &DiceFormula) -> (usize, usize) {
        let orig_len = self.parts.len();
        self.parts.extend(other.parts.iter().map(|x| x.increased_offset(orig_len)));
        (orig_len - 1, self.parts.len() - 1)
    }

    pub fn min_assign(&mut self, other: &DiceFormula) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Min(a, b));
    }

    pub fn min(mut self, other: &DiceFormula) -> Self {
        self.min_assign(other);
        self
    }

    pub fn max_assign(&mut self, other: &DiceFormula) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Max(a, b));
    }

    pub fn max(mut self, other: &DiceFormula) -> Self {
        self.max_assign(other);
        self
    }

    pub fn multi_add_assign(&mut self, other: &DiceFormula) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::MultiAdd(a, b));
    }

    pub fn multi_add(mut self, other: &DiceFormula) -> Self {
        self.multi_add_assign(other);
        self
    }

    #[must_use]
    fn could_be_negative(&self) -> bool {
        let mut s = Bounds { multi_add_negative: 0 };
        self.traverse(&mut s);
        s.multi_add_negative != 0
    }

    #[must_use]
    pub fn bounds(&self) -> (isize, isize) {
        let mut s = Bounds { multi_add_negative: 0 };
        let (a, b) = self.traverse(&mut s);
        debug_assert!(a <= b);
        (a, b)
    }
}

impl AddAssign<&Self> for DiceFormula {
    fn add_assign(&mut self, other: &Self) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Add(a, b));
    }
}

impl Add<&Self> for DiceFormula {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}

impl Add<Self> for DiceFormula {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self.add_assign(&other);
        self
    }
}

impl MulAssign<&Self> for DiceFormula {
    fn mul_assign(&mut self, other: &Self) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Mul(a, b));
    }
}

impl Mul<&Self> for DiceFormula {
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}

impl Mul<Self> for DiceFormula {
    type Output = Self;

    fn mul(mut self, other: Self) -> Self {
        self.mul_assign(&other);
        self
    }
}

impl SubAssign<&Self> for DiceFormula {
    fn sub_assign(&mut self, other: &Self) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Sub(a, b));
    }
}

impl Sub<&Self> for DiceFormula {
    type Output = Self;

    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}

impl Sub<Self> for DiceFormula {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self.sub_assign(&other);
        self
    }
}
