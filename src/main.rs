use std::{
    collections::btree_map::Keys,
    ops::{Add, AddAssign},
};

use rand::{Rng, thread_rng};

#[derive(Debug, Clone, Copy)]
struct Dice {
    n: usize,
}

impl Dice {
    fn new(n: usize) -> Self {
        debug_assert!(n != 0);
        Self { n }
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        rng.gen_range(0..self.n) + 1
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
    Const(usize),
}

impl NoOp {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        match self {
            &NoOp::Dice(dice) => dice.sample(rng),
            &NoOp::Const(x) => x,
        }
    }
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
    Div,
    Max,
    Min,

    // Left value is how many times we should evaluate the right expression
    MultiAdd,
}

impl Part {
    fn increased_offset(&self, n: usize) -> Self {
        match self {
            &Part::None(x) => Part::None(x),
            &Part::Double(DoubleOp { name, a, b }) => {
                Part::Double(DoubleOp { name, a: a + n, b: b + n })
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DiceExpressionSlice<'a> {
    parts: &'a [Part],
}

impl<'a> DiceExpressionSlice<'a> {
    fn slice(&self, n: usize) -> DiceExpressionSlice<'a> {
        let parts = match self.parts[n] {
            Part::None(_) => &self.parts[0..=n],
            Part::Double(DoubleOp { a, b, .. }) => {
                debug_assert!(a < b);
                debug_assert!(b < n);
                &self.parts[0..=n]
            }
        };
        DiceExpressionSlice { parts }
    }

    fn evaluate<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        enum Stage {
            First,
            Second,
            Third(usize),
        }
        let mut stack: Vec<(Part, Stage)> = vec![(*self.parts.last().unwrap(), Stage::First)];
        let mut values: Vec<usize> = Vec::new();
        while let Some(x) = stack.pop() {
            match x {
                (Part::None(NoOp::Dice(dice)), _) => values.push(dice.sample(rng)),
                (Part::None(NoOp::Const(n)), _) => values.push(n),
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }), Stage::First) => {
                    stack.push((
                        Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }),
                        Stage::Second,
                    ));
                    stack.push((self.parts[a], Stage::First));
                }
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }), Stage::Second) => {
                    let repetitions = values.pop().unwrap();

                    stack.push((
                        Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }),
                        Stage::Third(repetitions),
                    ));
                    for _ in 0..repetitions {
                        stack.push((self.parts[b], Stage::First));
                    }
                }
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, .. }), Stage::Third(n)) => {
                    let mut sum = 0;
                    for _ in 0..n {
                        sum += values.pop().unwrap();
                    }
                    values.push(sum);
                }
                (Part::Double(double_op), Stage::First) => {
                    stack.push((Part::Double(double_op), Stage::Second));
                    stack.push((self.parts[double_op.a], Stage::First));
                    stack.push((self.parts[double_op.b], Stage::First));
                }
                (Part::Double(DoubleOp { name, .. }), Stage::Second) => {
                    let bb = values.pop().unwrap();
                    let aa = values.pop().unwrap();
                    let n = match name {
                        DoubleOpName::Add => aa + bb,
                        DoubleOpName::Mul => aa * bb,
                        DoubleOpName::Div => aa / bb,
                        DoubleOpName::Max => aa.max(bb),
                        DoubleOpName::Min => aa.min(bb),
                        DoubleOpName::MultiAdd => unreachable!(),
                    };
                    values.push(n);
                }
                (Part::Double(_), Stage::Third(_)) => {
                    unreachable!();
                }
            };
        }
        *values.last().unwrap()
    }
}

#[derive(Debug, Clone)]
struct DiceExpression {
    parts: Vec<Part>,
}

impl DiceExpression {
    fn slice<'a>(&'a self) -> DiceExpressionSlice<'a> {
        DiceExpressionSlice { parts: &self.parts[..] }
    }

    fn new(start: NoOp) -> Self {
        Self { parts: vec![Part::None(start)] }
    }

    fn new_d(n: usize) -> Self {
        Self::new(NoOp::Dice(Dice::new(n)))
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

    fn div_assign(&mut self, other: &DiceExpression) {
        self.op_double_inplace(other, DoubleOpName::Div);
    }

    fn div(mut self, other: &DiceExpression) -> Self {
        self.div_assign(other);
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
        self.op_double_inplace(&other, DoubleOpName::Add);
        self
    }
}

fn main() {
    let mut rng = thread_rng();
    let yep = DiceExpression::new_d(20)
        .add(&DiceExpression::new_d(10))
        .multi_add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10));

    let mut total: f64 = 0.0;
    let count = 10;
    for _ in 0..count {
        total += yep.slice().evaluate(&mut rng) as f64;
    }
    println!("{:?}", total / (count as f64));
}
