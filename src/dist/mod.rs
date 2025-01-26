use std::ops::{AddAssign, MulAssign, SubAssign};

use num::{FromPrimitive, Num};

mod complex;
mod dense;
mod sparse;
pub use complex::WeirdDist;
pub(crate) use complex::WeirdDistEvaluator;
pub use dense::Dist;
pub(crate) use dense::DistEvaluator;
pub use sparse::SparseDist;
pub(crate) use sparse::SparseDistEvaluator;

#[cfg(test)]
pub mod tests;

pub trait DistTrait<'a, T: 'a + Num + FromPrimitive + AddAssign + PartialOrd>
where
    for<'b> T: MulAssign<&'b T> + SubAssign<&'b T> + AddAssign<&'b T>,
{
    // Iteratete through the distribution's support, in order.
    fn iter_enumerate(&'a self) -> impl Iterator<Item = (isize, &'a T)>;

    fn mean(&'a self) -> T {
        let mut out = T::zero();
        for (i, v) in self.iter_enumerate() {
            let mut thing = T::from_isize(i).unwrap();
            thing *= v;
            out += thing;
        }
        out
    }

    fn variance(&'a self) -> T {
        let mean = self.mean();
        let mut total = T::zero();
        for (i, v) in self.iter_enumerate() {
            let mut v_i = T::from_isize(i).unwrap();
            v_i -= &mean;
            v_i *= v;
            total += v_i;
        }
        total
    }

    /// Returns a value `x` such that `P(X <= m) >= 0.5` and `P(m <= X) >= 0.5`.
    fn median(&'a self) -> isize {
        let half = T::from_f64(0.5).unwrap();
        let mut total = T::zero();
        for (i, v) in self.iter_enumerate() {
            total += v;
            if total >= half {
                return i;
            }
        }
        unreachable!()
    }

    fn modes(&'a self) -> Vec<isize> {
        let mut out = Vec::new();
        let mut best: Option<&T> = None;
        for (i, v) in self.iter_enumerate() {
            match best {
                Some(x) if x < v => {
                    best = Some(v);
                    out.clear();
                    out.push(i)
                }
                Some(x) if x == v => out.push(i),
                Some(_) => {}
                None => out.push(i),
            }
        }
        out
    }
}
