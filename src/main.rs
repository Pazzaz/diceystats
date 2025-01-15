use rand::{Rng, thread_rng};
use std::{
    cmp::max,
    ops::{Add, AddAssign, Mul},
};

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
    fn dist(&self) -> ProbDist {
        ProbDist::uniform(self.n)
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
    // Div,
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
struct DiceExpression {
    parts: Vec<Part>,
}

#[derive(Debug, Clone)]
struct ProbDist {
    values: Vec<f64>,
}

impl ProbDist {
    fn uniform(n: usize) -> Self {
        ProbDist { values: (0..n).map(|x| 1.0 / (n as f64)).collect() }
    }

    fn constant(n: usize) -> Self {
        debug_assert!(n != 0);
        let mut values = vec![0.0; n];
        values[n - 1] = 1.0;

        ProbDist { values }
    }

    fn mean(&self) -> f64 {
        let mut out = 0.0;
        for (i, v) in self.values.iter().enumerate() {
            out += ((i + 1) as f64) * v;
        }
        out
    }

    fn op_inplace<F: Fn(usize, usize) -> usize>(
        &mut self,
        other: &ProbDist,
        buffer: &mut Vec<f64>,
        f: F,
    ) {
        debug_assert!(buffer.len() == 0);

        // Assumes f is monotone
        let new_len = f(self.values.len() + 1, other.values.len() + 1) - 1;
        buffer.resize(new_len, 0.0);
        for (a_i, a) in self.values.iter().enumerate() {
            for (b_i, b) in other.values.iter().enumerate() {
                let new_value = f(a_i + 1, b_i + 1);
                buffer[new_value - 1] += a * b;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        buffer.clear();
    }

    fn add_inplace(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
        self.op_inplace(other, buffer, usize::add);
    }

    fn mul_inplace(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
        self.op_inplace(other, buffer, usize::mul);
    }

    fn min_inplace(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
        self.op_inplace(other, buffer, usize::min);
    }

    fn max_inplace(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
        self.op_inplace(other, buffer, usize::max);
    }

    fn repeat(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
        debug_assert!(buffer.len() == 0);

        let new_len = (self.values.len() + 1) * (other.values.len() + 1) - 1;
        buffer.resize(new_len, 0.0);
        // We have a second buffer which tracks the chance of getting X with "current_i" iterations
        let mut buffer_2 = vec![0.0; new_len];
        let mut buffer_3 = vec![0.0; new_len];
        let mut first = true;
        for a in &self.values {
            for c in &mut buffer_3 {
                *c = 0.0;
            }
            if first {
                first = false;
                for (b_i, b) in other.values.iter().enumerate() {
                    buffer_3[b_i] = *b;
                }
            } else {
                for (b_i, &b) in other.values.iter().enumerate() {
                    for (c_i, &c) in buffer_2.iter().enumerate() {
                        if c == 0.0 {
                            continue;
                        }
                        let new_i = (b_i + 1) + (c_i + 1) - 1;
                        buffer_3[new_i] += b * c;
                    }
                }
            }
            for (c_i, c) in buffer_3.iter().enumerate() {
                buffer[c_i] += c * a;
            }
            buffer_2.copy_from_slice(&buffer_3[..]);
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        buffer.clear();
    }

    // fn div_inplace(&mut self, other: &ProbDist, buffer: &mut Vec<f64>) {
    //     debug_assert!(buffer.len() == 0);
    //     let new_len = max(other.values.len(), self.values.len());
    //     buffer.resize(new_len, 0.0);
    //     for (a_i, a) in self.values.iter().enumerate() {
    //         for (b_i, b) in other.values.iter().enumerate() {
    //             let new_value = (a_i + 1) / (b_i + 1);
    //             buffer[new_value - 1] += a * b;
    //         }
    //     }
    //     self.values.clear();
    //     self.values.copy_from_slice(&buffer[..]);
    //     buffer.clear();
    // }
}

impl DiceExpression {
    fn evaluate(&self) -> ProbDist {
        enum Stage {
            First,
            Second,
        }
        let mut stack: Vec<(Part, Stage)> = vec![(*self.parts.last().unwrap(), Stage::First)];
        let mut values: Vec<ProbDist> = Vec::new();
        let mut buffer = Vec::new();
        while let Some(x) = stack.pop() {
            match x {
                (Part::None(NoOp::Dice(dice)), _) => values.push(dice.dist()),
                (Part::None(NoOp::Const(n)), _) => values.push(ProbDist::constant(n)),
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }), Stage::First) => {
                    stack.push((
                        Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }),
                        Stage::Second,
                    ));
                    stack.push((self.parts[a], Stage::First));
                    stack.push((self.parts[b], Stage::First));
                }
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, .. }), Stage::Second) => {
                    let b = values.pop().unwrap();
                    let mut repetitions = values.pop().unwrap();
                    repetitions.repeat(&b, &mut buffer);
                    values.push(repetitions);
                }
                (Part::Double(double_op), Stage::First) => {
                    stack.push((Part::Double(double_op), Stage::Second));
                    stack.push((self.parts[double_op.a], Stage::First));
                    stack.push((self.parts[double_op.b], Stage::First));
                }
                (Part::Double(DoubleOp { name, .. }), Stage::Second) => {
                    let bb = values.pop().unwrap();
                    let mut aa = values.pop().unwrap();
                    match name {
                        DoubleOpName::Add => aa.add_inplace(&bb, &mut buffer),
                        DoubleOpName::Mul => aa.mul_inplace(&bb, &mut buffer),
                        // DoubleOpName::Div => aa / bb,
                        DoubleOpName::Max => aa.max_inplace(&bb, &mut buffer),
                        DoubleOpName::Min => aa.min_inplace(&bb, &mut buffer),
                        DoubleOpName::MultiAdd => unreachable!(),
                    }
                    values.push(aa);
                }
            };
        }
        values.last().unwrap().clone()
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

    // fn div_assign(&mut self, other: &DiceExpression) {
    //     self.op_double_inplace(other, DoubleOpName::Div);
    // }

    // fn div(mut self, other: &DiceExpression) -> Self {
    //     self.div_assign(other);
    //     self
    // }

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
    let yep = DiceExpression::new_d(30).multi_add(&DiceExpression::new_d(30));

    let res = yep.evaluate();
    let one: f64 = res.values.iter().sum();
    println!("{:?}", res);
    println!("mean: {}", res.mean());
    println!("one?: {}", one);
}
