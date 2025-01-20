use num::{FromPrimitive, Num};
use peg::str::LineCol;
use rand::Rng;
use std::{
    fmt,
    ops::{Add, AddAssign, MulAssign},
    str::FromStr,
};

use crate::Dist;

#[derive(Debug, Clone, Copy)]
struct Dice {
    n: usize,
}

impl Dice {
    fn new(n: usize) -> Self {
        debug_assert!(n != 0);
        Self { n }
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> isize {
        (rng.gen_range(0..self.n) + 1) as isize
    }
    fn dist<T: Num + FromPrimitive>(&self) -> Dist<T> {
        Dist::uniform(1, self.n as isize)
    }
}

#[derive(Debug, Clone, Copy)]
enum Part {
    None(NoOp),
    Double(DoubleOp),
}

#[derive(Debug, Clone, Copy)]
enum NoOp {
    Dice(Dice),
    Const(isize),
}

#[derive(Debug, Clone, Copy)]
struct DoubleOp {
    name: DoubleOpName,
    a: usize,
    b: usize,
}

#[derive(Debug, Clone, Copy)]
enum DoubleOpName {
    Add,
    Mul,
    Sub,
    Max,
    Min,

    // Left value is how many times we should evaluate the right expression
    MultiAdd,
}

impl Part {
    fn increased_offset(&self, n: usize) -> Self {
        match *self {
            Part::None(x) => Part::None(x),
            Part::Double(DoubleOp { name, a, b }) => {
                Part::Double(DoubleOp { name, a: a + n, b: b + n })
            }
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
#[derive(Debug, Clone)]
pub struct DiceExpression {
    parts: Vec<Part>,
}

trait Evaluator<T> {
    const LOSSY: bool;
    fn to_usize(_x: T) -> usize {
        unreachable!("Only used if operations are LOSSY");
    }
    fn dice(&mut self, d: Dice) -> T;
    fn constant(&mut self, n: isize) -> T;
    fn repeat_inplace(&mut self, _a: &mut T, _b: &T) {
        unreachable!("Only used if operations are not LOSSY");
    }
    fn add_inplace(&mut self, a: &mut T, b: &T);
    fn mul_inplace(&mut self, a: &mut T, b: &T);
    fn sub_inplace(&mut self, a: &mut T, b: &T);
    fn max_inplace(&mut self, a: &mut T, b: &T);
    fn min_inplace(&mut self, a: &mut T, b: &T);
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

    fn dice(&mut self, d: Dice) -> Dist<T> {
        d.dist()
    }

    fn constant(&mut self, n: isize) -> Dist<T> {
        Dist::constant(n)
    }

    fn repeat_inplace(&mut self, a: &mut Dist<T>, b: &Dist<T>) {
        a.repeat(b, &mut self.buffer);
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

struct SampleEvaluator<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
}

impl<R: Rng + ?Sized> Evaluator<isize> for SampleEvaluator<'_, R> {
    const LOSSY: bool = true;
    fn to_usize(x: isize) -> usize {
        x.try_into().unwrap_or_else(|_| panic!("Can't roll negative amount of dice"))
    }
    fn dice(&mut self, d: Dice) -> isize {
        d.sample(self.rng)
    }

    fn constant(&mut self, n: isize) -> isize {
        n
    }

    fn add_inplace(&mut self, a: &mut isize, b: &isize) {
        *a += b;
    }

    fn mul_inplace(&mut self, a: &mut isize, b: &isize) {
        *a *= b;
    }

    fn sub_inplace(&mut self, a: &mut isize, b: &isize) {
        *a -= b;
    }

    fn max_inplace(&mut self, a: &mut isize, b: &isize) {
        *a = isize::max(*a, *b);
    }

    fn min_inplace(&mut self, a: &mut isize, b: &isize) {
        *a = isize::min(*a, *b);
    }
}

struct StringEvaluator {
    precedence: Vec<usize>,
}

impl Evaluator<String> for StringEvaluator {
    const LOSSY: bool = false;

    fn dice(&mut self, d: Dice) -> String {
        self.precedence.push(7);
        format!("d{}", d.n)
    }

    fn repeat_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(3);
        *a = match (aa >= 3, bb > 3) {
            (true, true) => format!("{a}x{b}"),
            (true, false) => format!("{a}x({b})"),
            (false, true) => format!("({a})x{b}"),
            (false, false) => format!("({a})x({b})"),
        };
    }

    fn constant(&mut self, n: isize) -> String {
        self.precedence.push(6);
        format!("{}", n)
    }

    fn add_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(0);
        if bb == 0 { *a = format!("{a} + ({b})") } else { *a = format!("{a} + {b}") }
    }

    fn mul_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(2);
        *a = match (aa >= 2, bb > 2) {
            (true, true) => format!("{a} * {b}"),
            (true, false) => format!("{a} * ({b})"),
            (false, true) => format!("({a}) * {b}"),
            (false, false) => format!("({a}) * ({b})"),
        };
    }

    fn sub_inplace(&mut self, a: &mut String, b: &String) {
        let aa = self.precedence.pop().unwrap();
        let bb = self.precedence.pop().unwrap();
        self.precedence.push(1);
        *a = match (aa >= 1, bb > 1) {
            (true, true) => format!("{a} - {b}"),
            (true, false) => format!("{a} - ({b})"),
            (false, true) => format!("({a}) - {b}"),
            (false, false) => format!("({a}) - ({b})"),
        };
    }

    fn max_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let _bb = self.precedence.pop().unwrap();
        self.precedence.push(9);
        *a = format!("max({a}, {b})");
    }

    fn min_inplace(&mut self, a: &mut String, b: &String) {
        let _aa = self.precedence.pop().unwrap();
        let _bb = self.precedence.pop().unwrap();
        self.precedence.push(8);
        *a = format!("max({a}, {b})");
    }
}

enum EvaluateStage {
    Dice(Dice),
    Const(isize),
    MultiAddCreate(usize, usize),
    MultiAddCollect,
    MultiAddCollectPartial(usize),
    MultiAddExtra(usize),

    AddCreate(usize, usize),
    SubCreate(usize, usize),
    MulCreate(usize, usize),
    MaxCreate(usize, usize),
    MinCreate(usize, usize),
    AddCollect,
    SubCollect,
    MulCollect,
    MaxCollect,
    MinCollect,
}

impl EvaluateStage {
    fn collect_from(part: Part) -> Self {
        match part {
            Part::None(NoOp::Dice(dice)) => EvaluateStage::Dice(dice),
            Part::None(NoOp::Const(n)) => EvaluateStage::Const(n),
            Part::Double(DoubleOp { name: DoubleOpName::Add, a, b }) => {
                EvaluateStage::AddCreate(a, b)
            }
            Part::Double(DoubleOp { name: DoubleOpName::Sub, a, b }) => {
                EvaluateStage::SubCreate(a, b)
            }
            Part::Double(DoubleOp { name: DoubleOpName::Mul, a, b }) => {
                EvaluateStage::MulCreate(a, b)
            }
            Part::Double(DoubleOp { name: DoubleOpName::Min, a, b }) => {
                EvaluateStage::MinCreate(a, b)
            }
            Part::Double(DoubleOp { name: DoubleOpName::Max, a, b }) => {
                EvaluateStage::MaxCreate(a, b)
            }
            Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }) => {
                EvaluateStage::MultiAddCreate(a, b)
            }
        }
    }
}

impl DiceExpression {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> isize {
        let mut e = SampleEvaluator { rng };
        self.evaluate_generic(&mut e)
    }

    /// Calculate the probability distribution of the outcomes of the expression.
    ///
    /// The function is generic over the number type used to represent probabilities.
    ///
    /// ## Example
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

    // Traverses the tree with an Evaluator.
    fn evaluate_generic<T, Q: Evaluator<T>>(&self, state: &mut Q) -> T {
        let mut stack: Vec<EvaluateStage> =
            vec![EvaluateStage::collect_from(*self.parts.last().unwrap())];
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
                    state.repeat_inplace(&mut aa, &bb);
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

    fn new(start: NoOp) -> Self {
        Self { parts: vec![Part::None(start)] }
    }

    fn op_double_inplace(&mut self, other: &DiceExpression, name: DoubleOpName) {
        self.parts.reserve(other.parts.len() + 1);

        let orig_len = self.parts.len();
        let first_element = orig_len - 1;
        self.parts.extend(other.parts.iter().map(|x| x.increased_offset(orig_len)));
        let second_element = self.parts.len() - 1;
        self.parts.push(Part::Double(DoubleOp { name, a: first_element, b: second_element }));
    }

    fn mul_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::Mul);
    }

    fn mul(mut self, other: &DiceExpression) -> Self {
        self.mul_assign(other);
        self
    }

    fn sub_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::Sub);
    }

    fn sub(mut self, other: &DiceExpression) -> Self {
        self.sub_assign(other);
        self
    }

    fn min_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::Min);
    }

    fn min(mut self, other: &DiceExpression) -> Self {
        self.min_assign(other);
        self
    }

    fn max_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::Max);
    }

    fn max(mut self, other: &DiceExpression) -> Self {
        self.max_assign(other);
        self
    }

    fn multi_add_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::MultiAdd);
    }

    fn multi_add(mut self, other: &DiceExpression) -> Self {
        self.multi_add_assign(other);
        self
    }
}

impl AddAssign<&Self> for DiceExpression {
    fn add_assign(&mut self, other: &Self) {
        self.op_double_inplace(other, DoubleOpName::Add);
    }
}

impl Add<&Self> for DiceExpression {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.op_double_inplace(other, DoubleOpName::Add);
        self
    }
}

impl FromStr for DiceExpression {
    type Err = peg::error::ParseError<LineCol>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        list_parser::arithmetic(s)
    }
}

/// Displays an expression slightly normalized, with parenthesis and spaces.
/// ```
/// use diceystats::DiceExpression;
/// let x: DiceExpression = "((d5) + d20xd5)* max(d4 *d4,d5, d10)x(d4*d8)".parse().unwrap();
/// assert_eq!(x.to_string(), "(d5 + d20xd5) * max(max(d4 * d4, d5), d10)x(d4 * d8)");
/// ```
impl fmt::Display for DiceExpression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut e = StringEvaluator { precedence: Vec::new() };
        let res = self.evaluate_generic(&mut e);
        write!(f, "{}", res)
    }
}

peg::parser! {
    grammar list_parser() for str {
        rule number() -> usize = n:$(['0'..='9']+) {? n.parse::<usize>().or(Err("u32")) }
        rule list_part() -> DiceExpression = " "* e:arithmetic() " "* { e }
        pub rule arithmetic() -> DiceExpression = precedence!{
            x:(@) " "* "+" " "* y:@ { x.add(&y) }
            x:(@) " "* "-" " "* y:@ { x.sub(&y) }
            --
            x:(@) " "* "*" " "* y:@ { x.mul(&y) }
            --
            x:(@) " "* "x" " "* y:@ { x.multi_add(&y) }
            --
            n1:number() "d" n2:number() {
                let nn1 = DiceExpression::new(NoOp::Const(n1 as isize));
                let nn2 = DiceExpression::new(NoOp::Dice(Dice::new(n2)));
                nn1.multi_add(&nn2)
            }
            n:number() { DiceExpression::new(NoOp::Const(n as isize)) }
            "-" n:number() { DiceExpression::new(NoOp::Const(-(n as isize))) }
            "d" n:number() { DiceExpression::new(NoOp::Dice(Dice::new(n))) }
            "min(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.min(&b)).unwrap() }
            "max(" l:(list_part() ++ ",") ")" { l.into_iter().reduce(|a, b| a.max(&b)).unwrap() }
            "(" " "* e:arithmetic() " "* ")" { e }
        }
    }
}
