use std::ops::{Add, AddAssign};

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
    Max,
    Min,

    // Left value is how many times we should evaluate the right expression
    MultiAdd,
}

impl Part {
    fn increased_offset(&self, n: usize) -> Self {
        match self {
            &Part::None(x) => Part::None(x),
            &Part::Double(DoubleOp { name, a, b }) => Part::Double(DoubleOp {
                name,
                a: a + n,
                b: b + n,
            }),
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

    // Recursive :/
    fn evaluate<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        match *self.parts.last().unwrap() {
            Part::None(NoOp::Dice(dice)) => dice.sample(rng),
            Part::None(NoOp::Const(n)) => n,
            Part::Double(DoubleOp { name, a, b }) => {
                let aa = self.slice(a).evaluate(rng);
                match name {
                    DoubleOpName::Add => {
                        let bb = self.slice(b).evaluate(rng);
                        aa + bb
                    }
                    DoubleOpName::Max => {
                        let bb = self.slice(b).evaluate(rng);
                        aa.max(bb)
                    }
                    DoubleOpName::Min => {
                        let bb = self.slice(b).evaluate(rng);
                        aa.min(bb)
                    }
                    DoubleOpName::MultiAdd => (0..aa).map(|_| self.slice(b).evaluate(rng)).sum(),
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DiceExpression {
    parts: Vec<Part>,
}

impl DiceExpression {
    fn slice<'a>(&'a self) -> DiceExpressionSlice<'a> {
        DiceExpressionSlice {
            parts: &self.parts[..],
        }
    }

    fn new(start: NoOp) -> Self {
        Self {
            parts: vec![Part::None(start)],
        }
    }

    fn new_d(n: usize) -> Self {
        Self::new(NoOp::Dice(Dice::new(n)))
    }

    fn op_double_inplace(&mut self, other: &DiceExpression, name: DoubleOpName) {
        self.parts.reserve(other.parts.len() + 1);

        let orig_len = self.parts.len();
        let first_element = orig_len - 1;
        self.parts
            .extend(other.parts.iter().map(|x| x.increased_offset(orig_len)));
        let second_element = self.parts.len() - 1;
        self.parts.push(Part::Double(DoubleOp {
            name,
            a: first_element,
            b: second_element,
        }));
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
        .add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10))
        .add(&DiceExpression::new_d(10));

    let mut total: f64 = 0.0;
    let count = 10000;
    for _ in 0..count {
        total += yep.slice().evaluate(&mut rng) as f64;
    }
    println!("{:?}", total / (count as f64));
}
