#![feature(test)]
use num::{BigRational, FromPrimitive, Num};
use peg::str::LineCol;
use rand::Rng;
use std::{
    ops::{Add, AddAssign, Mul, MulAssign}, str::FromStr,
};

use num::Zero;

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
    fn dist<T: Num + FromPrimitive>(&self) -> Dist<T> {
        Dist::uniform(self.n)
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
        match *self {
            NoOp::Dice(dice) => dice.sample(rng),
            NoOp::Const(x) => x,
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
        match *self {
            Part::None(x) => Part::None(x),
            Part::Double(DoubleOp { name, a, b }) => {
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
struct Dist<T> {
    values: Vec<T>,
}
// <'a, 'b: 'a, T: Num + FromPrimitive + Clone + AddAssign
//      + Mul + MulAssign<&'a T> + 'b> Dist<T>

impl<T: Num + FromPrimitive> Dist<T> {
    fn uniform(n: usize) -> Self {
        Dist { values: (0..n).map(|_| T::one() / T::from_usize(n).unwrap()).collect() }
    }
}

impl<T: Num + Clone> Dist<T> {
    fn constant(n: usize) -> Self {
        debug_assert!(n != 0);
        let mut values = vec![T::zero(); n];
        values[n - 1] = T::one();

        Dist { values }
    }
}

impl<'a, 'b: 'a, T: Num + FromPrimitive + MulAssign<&'a T> + AddAssign + 'b> Dist<T> {
    fn mean(&'b self) -> T {
        let mut out = T::zero();
        for (i, v) in self.values.iter().enumerate() {
            let mut thing = T::from_usize(i + 1).unwrap();
            thing *= v;
            out += thing;
        }
        out
    }
}

impl<T: Num + Clone + AddAssign> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn op_inplace<F: Fn(usize, usize) -> usize>(
        &mut self,
        other: &Dist<T>,
        buffer: &mut Vec<T>,
        f: F,
    ) {
        debug_assert!(buffer.is_empty());

        // Assumes f is monotone
        let new_len = f(self.values.len(), other.values.len());
        buffer.resize(new_len, T::zero());
        for (a_i, a) in self.values.iter().enumerate() {
            if a.is_zero() { continue }
            for (b_i, b) in other.values.iter().enumerate() {
                if b.is_zero() { continue }
                let new_value = f(a_i + 1, b_i + 1);
                let mut res: T = a.clone();
                res.mul_assign(b);
                buffer[new_value - 1] += res;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        buffer.clear();
    }
}

impl<T: Num + Clone + AddAssign> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn add_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, usize::add);
    }
}

impl<T: Num + Clone + AddAssign + MulAssign> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn mul_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, usize::mul);
    }
}

impl<T: Num + Clone + AddAssign + PartialOrd> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn max_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, usize::max);
    }
}

impl<T: Num + Clone + AddAssign + PartialOrd> Dist<T>
where
    for<'a> T: MulAssign<&'a T>,
{
    fn min_inplace(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        self.op_inplace(other, buffer, usize::min);
    }
}

impl<T: Num + Clone + AddAssign + std::fmt::Debug> Dist<T>
where
    for<'a> T: MulAssign<&'a T> + AddAssign<&'a T>,
{
    fn repeat(&mut self, other: &Dist<T>, buffer: &mut Vec<T>) {
        debug_assert!(buffer.is_empty());

        let new_len = self.values.len() * other.values.len();
        buffer.resize(new_len, T::zero());
        // We have a second buffer which tracks the chance of getting X with "current_i" iterations
        let mut buffer_2 = vec![T::zero(); new_len - self.values.len() + 1];
        let mut buffer_3 = vec![T::zero(); new_len - self.values.len() + 1];

        for (b_i, b) in other.values.iter().enumerate() {
            buffer_3[b_i] += b;
            buffer[b_i] += b;
            buffer[b_i] *= &self.values[0];
        }

        let mut s = other.values.len();

        for (a_i, a) in self.values.iter().skip(1).enumerate() {
            if a.is_zero() {
                continue;
            }

            buffer_2[0..s].swap_with_slice(&mut buffer_3[0..s]);
            
            for ii in 0..(other.values.len() + s-1) {
                buffer_3[ii].set_zero();
            }
            for (b_i, b) in other.values.iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
                for c_i in 0..s {
                    let c = &buffer_2[c_i];
                    if c.is_zero() {
                        continue;
                    }
                    let new_i = b_i + c_i;
                    let mut res = b.clone();
                    res *= c;
                    buffer_3[new_i] += res;
                }
            }
            s += other.values.len() - 1;
            for c_i in 0..s {
                let c = &buffer_3[c_i];
                let mut aa = a.clone();
                aa.mul_assign(c);
                buffer[a_i+c_i+1] += aa;
            }
        }
        self.values.clear();
        self.values.extend_from_slice(&buffer[..]);
        buffer.clear();
    }

    // fn div_inplace(&mut self, other: &Dist, buffer: &mut Vec<f64>) {
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
    fn evaluate<T: Num + Clone + MulAssign + PartialOrd + FromPrimitive + AddAssign + std::fmt::Debug>(
        &self,
    ) -> Dist<T>
    where
        for<'a> T: AddAssign<&'a T> + MulAssign<&'a T>,
    {
        enum Stage {
            First,
            Second,
        }
        let mut stack: Vec<(Part, Stage)> = vec![(*self.parts.last().unwrap(), Stage::First)];
        let mut values: Vec<Dist<T>> = Vec::new();
        let mut buffer: Vec<T> = Vec::new();
        while let Some(x) = stack.pop() {
            match x {
                (Part::None(NoOp::Dice(dice)), _) => values.push(dice.dist()),
                (Part::None(NoOp::Const(n)), _) => values.push(Dist::constant(n)),
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }), Stage::First) => {
                    stack.push((
                        Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, a, b }),
                        Stage::Second,
                    ));
                    stack.push((self.parts[a], Stage::First));
                    stack.push((self.parts[b], Stage::First));
                }
                (Part::Double(DoubleOp { name: DoubleOpName::MultiAdd, .. }), Stage::Second) => {
                    let mut repetitions = values.pop().unwrap();
                    let b = values.pop().unwrap();
                    repetitions.repeat(&b, &mut buffer);
                    values.push(repetitions);
                }
                (Part::Double(double_op), Stage::First) => {
                    stack.push((Part::Double(double_op), Stage::Second));
                    stack.push((self.parts[double_op.a], Stage::First));
                    stack.push((self.parts[double_op.b], Stage::First));
                }
                (Part::Double(DoubleOp { name, .. }), Stage::Second) => {
                    let mut aa = values.pop().unwrap();
                    let bb = values.pop().unwrap();
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
        values.pop().unwrap()
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

peg::parser! {
    grammar list_parser() for str {
        rule number() -> usize = n:$(['0'..='9']+) {? n.parse::<usize>().or(Err("u32")) }
        pub rule arithmetic() -> DiceExpression = precedence!{
            x:(@) " "* "+" " "* y:@ { x + &y }
            --
            x:(@) " "* "*" " "* y:@ { x.mul(&y) }
            --
            x:(@) " "* "x" " "* y:@ { x.multi_add(&y) }
            --
            n:number() { DiceExpression::new(NoOp::Const(n)) }
            "d" n:number() { DiceExpression::new(NoOp::Dice(Dice::new(n))) }
            "(" " "* e:arithmetic() " "* ")" { e }
        }
    }
}

fn main() {
    match "d6xd4".parse::<DiceExpression>() {
        Ok(x) => {
            let res: Dist<f64> = x.evaluate();
            println!("{}", res.mean());
            // for (i, v) in res.values.iter().enumerate() {
            //     let min_precision = 20;
            //     let formatted = format_impl(v, String::new(), 0, min_precision, None);
            //     println!("{} : {}", i, formatted);
            // }
        }
        Err(x) => println!("ERR: {}", x),
    }

    // let res: Dist<BigRational> = yep.evaluate();
    // println!("mean 2: {}", res.mean());
}

const MAX_PRECISION: usize = 30;

fn format_impl(
    value: &BigRational,
    mut result: String,
    depth: usize,
    min_precision: usize,
    precision: Option<usize>,
) -> String {
    debug_assert!(min_precision <= precision.unwrap_or(MAX_PRECISION));

    let trunc = value.trunc().to_integer();
    result += &trunc.to_string();

    let numer = value.numer();
    let denom = value.denom();
    let value = numer - (trunc * denom);

    let at_min = depth >= min_precision;
    let at_max = depth >= precision.unwrap_or(MAX_PRECISION);
    // If the user specified a precision for the formatting then we
    // honor that by ensuring that we have that many decimals.
    // Otherwise we print as many as there are, up to `MAX_PRECISION`.
    if (value.is_zero() && precision.is_none() && at_min) || at_max {
        result
    } else {
        if depth == 0 {
            result += ".";
        }

        let value = BigRational::new(value * 10, denom.clone());
        format_impl(&value, result, depth + 1, min_precision, precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use num::BigInt;
    use test::Bencher;

    #[bench]
    fn eval_6dx6d(b: &mut Bencher) {
        let yep: DiceExpression = "d6xd6".parse().unwrap();
        b.iter(|| {
            let res: Dist<BigRational> = yep.evaluate();
            test::black_box(res);
        });
    }

    #[bench]
    fn f64_30dx30d(b: &mut Bencher) {
        let yep: DiceExpression = "d30xd30".parse().unwrap();
        b.iter(|| {
            let res: Dist<f64> = yep.evaluate();
            test::black_box(res);
        });
    }

    #[bench]
    fn parsing(b: &mut Bencher) {
        b.iter(|| {
            let yep: DiceExpression = "d30 + (d20xd30*d43423x(d20 + d4*d32 + 43))".parse().unwrap();
            test::black_box(yep);
        });
    }

    #[test]
    fn repeat_simple() {
        let yep: DiceExpression = "d9xd10".parse().unwrap();
        assert_eq!(yep.evaluate::<BigRational>().mean(), BigRational::new(BigInt::from(55), BigInt::from(2)));
    }

    #[test]
    fn rational_30dx30d() {
        let yep: DiceExpression = "d30xd30".parse().unwrap();
        assert_eq!(yep.evaluate::<BigRational>().mean(), BigRational::new(BigInt::from(961), BigInt::from(4)));
    }
}
