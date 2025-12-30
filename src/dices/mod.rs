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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct DiceFormula {
    parts: Vec<Part>,
}

// Each part of a dice expression is a node in a tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    // We also track whether we ever tried to `multi_add` with a negative argument, i.e. "throwing
    // a negative number of dice".
    multi_add_negative: bool,
}

impl Bounds {
    fn new() -> Self {
        Bounds { multi_add_negative: false }
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
            self.multi_add_negative = true;
            a.0 = 0;
            if a.1 < 0 {
                a.1 = 0;
            }
        }
        let extremes = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap());
    }

    fn negate_inplace(&mut self, a: &mut (isize, isize)) {
        *a = (-a.1, -a.0);
    }

    fn add_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        *a = (a.0 + b.0, a.1 + b.1);
    }

    fn mul_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap());
    }

    fn sub_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0 - b.0, a.0 - b.1, a.1 - b.0, a.1 - b.1];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap());
    }

    fn max_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0.max(b.0), a.0.max(b.1), a.1.max(b.0), a.1.max(b.1)];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap());
    }

    fn min_inplace(&mut self, a: &mut (isize, isize), b: &(isize, isize)) {
        let extremes = [a.0.min(b.0), a.0.min(b.1), a.1.min(b.0), a.1.min(b.1)];
        *a = (*extremes.iter().min().unwrap(), *extremes.iter().max().unwrap());
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

// Apply some operation `f` to some value(s) on the stack
macro_rules! apply {
    (unary, $stack:ident, $state:ident, $f:ident) => {{
        let aa = $stack.last_mut().unwrap();
        $state.$f(aa);
    }};

    (binary, $stack:ident, $state:ident, $f:ident) => {{
        let mut aa = $stack.pop().unwrap();
        let bb = $stack.pop().unwrap();
        $state.$f(&mut aa, &bb);
        $stack.push(aa);
    }};
}

// Push some evaluation stage(s) to the stack
macro_rules! add {
    (unary, $dice:ident, $stack:ident, $enum:ident, $a:ident) => {{
        $stack.push(EvaluateStage::$enum);
        $stack.push(EvaluateStage::collect_from($dice.parts[$a]));
    }};
    (unary_with, $dice:ident, $stack:ident, $enum:ident, $a:ident, $b:ident) => {{
        $stack.push(EvaluateStage::$enum($b));
        $stack.push(EvaluateStage::collect_from($dice.parts[$a]));
    }};
    (binary, $dice:ident, $stack:ident, $enum:ident, $a:ident, $b:ident) => {{
        $stack.push(EvaluateStage::$enum);
        $stack.push(EvaluateStage::collect_from($dice.parts[$a]));
        $stack.push(EvaluateStage::collect_from($dice.parts[$b]));
    }};
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
                        add!(binary, self, stack, MultiAddCollect, a, b);
                    } else {
                        add!(unary_with, self, stack, MultiAddCollectPartial, a, b);
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
                    apply!(binary, values, state, multi_add_inplace);
                }
                EvaluateStage::MultiAddExtra(aa) => {
                    assert!(!Q::CUSTOM_MULTI_ADD);
                    debug_assert!(aa != 0);
                    let mut res = values.pop().unwrap();
                    for _ in 1..aa {
                        let v = values.pop().unwrap();
                        state.add_inplace(&mut res, &v);
                    }
                    values.push(res);
                }
                EvaluateStage::NegateCreate(a) => add!(unary, self, stack, NegateCollect, a),
                EvaluateStage::AddCreate(a, b) => add!(binary, self, stack, AddCollect, a, b),
                EvaluateStage::MulCreate(a, b) => add!(binary, self, stack, MulCollect, a, b),
                EvaluateStage::SubCreate(a, b) => add!(binary, self, stack, SubCollect, a, b),
                EvaluateStage::MinCreate(a, b) => add!(binary, self, stack, MinCollect, a, b),
                EvaluateStage::MaxCreate(a, b) => add!(binary, self, stack, MaxCollect, a, b),
                EvaluateStage::NegateCollect => apply!(unary, values, state, negate_inplace),
                EvaluateStage::AddCollect => apply!(binary, values, state, add_inplace),
                EvaluateStage::MulCollect => apply!(binary, values, state, mul_inplace),
                EvaluateStage::SubCollect => apply!(binary, values, state, sub_inplace),
                EvaluateStage::MinCollect => apply!(binary, values, state, min_inplace),
                EvaluateStage::MaxCollect => apply!(binary, values, state, max_inplace),
            }
        }
        values.pop().unwrap()
    }

    #[must_use]
    fn dice(d: usize) -> Self {
        Self { parts: vec![Part::Dice(d)] }
    }

    #[must_use]
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
    /// assert_eq!(exact_dist.mean().to_string(), "55/4");
    /// ```
    #[must_use]
    pub fn dist<T, D: Dist<T>>(&self) -> D
    where
        for<'b> T: Num
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
        self.parts.push(Part::Negate(orig_len - 1));
    }

    #[must_use]
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

    #[must_use]
    pub fn min(mut self, other: &DiceFormula) -> Self {
        self.min_assign(other);
        self
    }

    pub fn max_assign(&mut self, other: &DiceFormula) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::Max(a, b));
    }

    #[must_use]
    pub fn max(mut self, other: &DiceFormula) -> Self {
        self.max_assign(other);
        self
    }

    pub fn multi_add_assign(&mut self, other: &DiceFormula) {
        let (a, b) = self.concat(other);
        self.parts.push(Part::MultiAdd(a, b));
    }

    #[must_use]
    pub fn multi_add(mut self, other: &DiceFormula) -> Self {
        self.multi_add_assign(other);
        self
    }

    #[must_use]
    fn could_be_negative(&self) -> bool {
        let mut s = Bounds::new();
        self.traverse(&mut s);
        s.multi_add_negative
    }

    #[must_use]
    pub fn bounds(&self) -> (isize, isize) {
        let mut s = Bounds::new();
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
